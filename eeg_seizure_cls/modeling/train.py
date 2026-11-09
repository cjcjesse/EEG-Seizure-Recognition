import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from tqdm.auto import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics.classification import MatthewsCorrCoef
from torchvision.models import resnet18
from eeg_seizure_cls.config import MODELS_DIR
from eeg_seizure_cls.modeling.classes import TimeSeriesModel, MLP, MultiModalModel

MODEL_TYPES = ['time-series', 'mlp', 'resnet', 'mult']

def build_model(model_type, train_dataset, num_classes, device, num_layers=None, num_nodes=None, dropout=None, out_channels=None, k=None, stride=None, nhead=4, d_model=256, resnet_state_dict_path=None, mlp_state_dict_path=None, mlp_best_params=None):
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Input model type is not one of {MODEL_TYPES}")

    if model_type == 'time-series':
        num_in_features = train_dataset.X.size(1) # (n_examples, num_in_features=1, seq_len=178)
        seq_len = train_dataset.X.size(-1) # (n_examples, num_in_features=1, seq_len=178)
        model = TimeSeriesModel(nhead, num_layers, d_model * 2, dropout, num_in_features, out_channels, k, stride, seq_len, num_classes, d_model=d_model)
    if model_type == 'mlp':
        num_in_features = train_dataset.X.size(-1)
        model = MLP(num_in_features, num_layers, num_nodes, num_classes, dropout)
    if model_type == 'resnet':
        model = resnet18(weights='IMAGENET1K_V1')
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes, bias=True) # replace classification head from native 1000-class to binary
    if model_type == 'mult':
        model = MultiModalModel(num_layers, num_nodes, num_classes, dropout, resnet_state_dict_path, mlp_state_dict_path, mlp_best_params)

    return model.to(device)

def train_init(train_dataset, val_dataset, weights_tensor, device, model_type, batch_size, num_classes, lr, weight_decay, **model_kwargs):
    drop_last = model_type != 'time-series' # drop_last=True prevents error from BatchNorm1d if last batch has only 1 example
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(model_type, train_dataset, num_classes, device, **model_kwargs)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor) # class weighting for training time mitigation of class imbalance effects
    optimizer = optim.AdamW(filter(lambda param:param.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    mcc_score = MatthewsCorrCoef(task='binary').to(device)

    return model, criterion, optimizer, scheduler, mcc_score, train_loader, val_loader

def unpack_batch(batch, device, model_type):
    if model_type == 'mult':
        numerical, image, labels = batch
        return (numerical.to(device), image.to(device)), labels.to(device)

    inputs, labels = batch
    return (inputs.to(device), ), labels.to(device)

def new_best():
    return {
        'eval_metric': float('-inf'), # higher better
        'stage': None,
        'epoch': None,
        'preds': None,
        'true': None,
        'path': None,
    }

def epoch_train(model, train_loader, criterion, optimizer, device, model_type, clip_grad=True):
    model.train()
    train_loss = 0.0
    train_total = 0

    for batch in train_loader:
        inputs, labels = unpack_batch(batch, device, model_type)
        optimizer.zero_grad()
        outputs = model(*inputs)
        loss = criterion(outputs, labels) # (batch_size, n_classes), (batch_size, )
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        train_total += labels.size(0)

    return train_loss, train_total

def epoch_eval(model, mcc_score, val_loader, criterion, device, model_type):
    model.eval()
    mcc_score.reset()
    val_loss = 0.0
    val_total = 0
    with torch.no_grad():
        true_list = []
        preds_list = []
        for batch in val_loader:
            inputs, labels = unpack_batch(batch, device, model_type)
            outputs = model(*inputs)
            loss = criterion(outputs, labels) # (batch_size, n_classes), (batch_size, )
            preds = outputs.argmax(1)

            val_loss += loss.item() * labels.size(0)
            val_total += labels.size(0)
            mcc_score.update(preds, labels)

            true_list.append(labels.detach().cpu().numpy())
            preds_list.append(preds.detach().cpu().numpy())

    return mcc_score, val_loss, val_total, preds_list, true_list

def epoch_checkpoint(model, mcc_score, preds_list, true_list, train_loss, train_total, train_epoch_losses, val_loss, val_total, val_epoch_losses, scheduler, epoch, epochs, state_dict_path, stage, model_type, best):
    epoch_eval_metric = mcc_score.compute().item()
    train_epoch_loss = train_loss / train_total
    val_epoch_loss = val_loss / val_total
    train_epoch_losses.append(train_epoch_loss)
    val_epoch_losses.append(val_epoch_loss)

    if (epoch == 0) or ((epoch + 1) % 10 == 0) or (epoch == (epochs - 1)):
        print(f'Epoch No.: {epoch + 1}\n  Epoch Train Loss: {train_epoch_loss:.3f}\n  Epoch Val Loss: {val_epoch_loss:.3f}')

    if epoch_eval_metric > best['eval_metric']:
        best.update(eval_metric=epoch_eval_metric, stage=stage, epoch=epoch, preds=np.concatenate(preds_list), true=np.concatenate(true_list), path=state_dict_path)
        # multi-modal model only trains the fusion head, the frozen MLP and ResNet18 backbones are loaded from their own state dicts
        torch.save(model.fusion_head.state_dict() if model_type == 'mult' else model.state_dict(), state_dict_path)

    scheduler.step(epoch_eval_metric)

    return train_epoch_losses, val_epoch_losses

def train_epochs(model, train_loader, val_loader, criterion, optimizer, scheduler, mcc_score, epochs, device, model_type, state_dict_path, best, train_epoch_losses, val_epoch_losses, desc='Epochs', stage=1, clip_grad=True):
    for epoch in tqdm(range(epochs), desc=desc, leave=False):
        train_loss, train_total = epoch_train(model, train_loader, criterion, optimizer, device, model_type, clip_grad)
        mcc_score, val_loss, val_total, preds_list, true_list = epoch_eval(model, mcc_score, val_loader, criterion, device, model_type)
        train_epoch_losses, val_epoch_losses = epoch_checkpoint(model, mcc_score, preds_list, true_list, train_loss, train_total, train_epoch_losses, val_loss, val_total, val_epoch_losses, scheduler, epoch, epochs, state_dict_path, stage, model_type, best)

    return train_epoch_losses, val_epoch_losses

def objective_epochs(trial, model, train_loader, val_loader, criterion, optimizer, scheduler, mcc_score, epochs, device, model_type):
    best_eval_metric = float('-inf') # higher better

    for epoch in tqdm(range(epochs), desc='Epochs', leave=False):
        epoch_train(model, train_loader, criterion, optimizer, device, model_type)
        mcc_score, _, _, _, _ = epoch_eval(model, mcc_score, val_loader, criterion, device, model_type)
        epoch_eval_metric = mcc_score.compute().item()

        if epoch_eval_metric > best_eval_metric:
            best_eval_metric = epoch_eval_metric

        scheduler.step(epoch_eval_metric)

        trial.report(epoch_eval_metric, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_eval_metric

def ts_objective(trial, train_dataset, val_dataset, weights_tensor, device, epochs=35, nhead=4, d_model=256, batch_size=64, num_classes=2, weight_decay=1e-3):
    model = optimizer = None
    try:
        num_layers = trial.suggest_int('num_layers', 2, 6)
        dropout = trial.suggest_float('dropout', 0.1, 0.4)
        out_channels = trial.suggest_int('out_channels', 8, 15)
        k = trial.suggest_int('k', 1, 7)
        stride = trial.suggest_int('stride', 1, 3)
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)

        model, criterion, optimizer, scheduler, mcc_score, train_loader, val_loader = train_init(train_dataset, val_dataset, weights_tensor, device, 'time-series', batch_size, num_classes, lr, weight_decay, num_layers=num_layers, dropout=dropout, out_channels=out_channels, k=k, stride=stride, nhead=nhead, d_model=d_model)

        return objective_epochs(trial, model, train_loader, val_loader, criterion, optimizer, scheduler, mcc_score, epochs, device, 'time-series')
    finally:
        del model
        del optimizer
        torch.cuda.empty_cache()

def mlp_objective(trial, train_dataset, val_dataset, weights_tensor, device, epochs=35, batch_size=64, num_classes=2, weight_decay=1e-3):
    model = optimizer = None
    try:
        num_layers = trial.suggest_int('num_layers', 2, 10)
        num_nodes = trial.suggest_int('num_nodes', 64, 128)
        dropout = trial.suggest_float('dropout', 0.05, 0.3)
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)

        model, criterion, optimizer, scheduler, mcc_score, train_loader, val_loader = train_init(train_dataset, val_dataset, weights_tensor, device, 'mlp', batch_size, num_classes, lr, weight_decay, num_layers=num_layers, num_nodes=num_nodes, dropout=dropout)

        return objective_epochs(trial, model, train_loader, val_loader, criterion, optimizer, scheduler, mcc_score, epochs, device, 'mlp')
    finally:
        del model
        del optimizer
        torch.cuda.empty_cache()

def mult_objective(trial, train_dataset, val_dataset, weights_tensor, device, epochs, resnet_state_dict_path, mlp_state_dict_path, mlp_best_params, batch_size=64, num_classes=2, weight_decay=1e-3):
    model = optimizer = None
    try:
        num_layers = trial.suggest_int('num_layers', 1, 3)
        num_nodes = trial.suggest_int('num_nodes', 64, 128)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)

        model, criterion, optimizer, scheduler, mcc_score, train_loader, val_loader = train_init(train_dataset, val_dataset, weights_tensor, device, 'mult', batch_size, num_classes, lr, weight_decay, num_layers=num_layers, num_nodes=num_nodes, dropout=dropout, resnet_state_dict_path=resnet_state_dict_path, mlp_state_dict_path=mlp_state_dict_path, mlp_best_params=mlp_best_params)

        return objective_epochs(trial, model, train_loader, val_loader, criterion, optimizer, scheduler, mcc_score, epochs, device, 'mult')
    finally:
        del model
        del optimizer
        torch.cuda.empty_cache()

def run_hpo(objective, n_trials=20):
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value

def save_best_params(json_path, best_mcc, best_params):
    with open(json_path, 'w') as file:
        json.dump({'mcc': best_mcc, 'best_params': best_params}, file, indent=4)

def load_best_params(json_path):
    with open(json_path, 'r') as file:
        hpo_dict = json.load(file)

    return hpo_dict['best_params'], hpo_dict['mcc']

def train_val_loop(model_type, state_dict_name, train_dataset, val_dataset, weights_tensor, device, epochs, batch_size, num_classes, lr, weight_decay, models_dir, **model_kwargs):
    models_dir.mkdir(parents=True, exist_ok=True)
    state_dict_path = models_dir / state_dict_name

    model, criterion, optimizer, scheduler, mcc_score, train_loader, val_loader = train_init(train_dataset, val_dataset, weights_tensor, device, model_type, batch_size, num_classes, lr, weight_decay, **model_kwargs)

    best = new_best()
    train_epoch_losses = []
    val_epoch_losses = []

    train_epoch_losses, val_epoch_losses = train_epochs(model, train_loader, val_loader, criterion, optimizer, scheduler, mcc_score, epochs, device, model_type, state_dict_path, best, train_epoch_losses, val_epoch_losses)

    if model_type == 'mult':
        model.fusion_head.load_state_dict(torch.load(best['path'], weights_only=True))
    else:
        model.load_state_dict(torch.load(best['path'], weights_only=True))

    out_dict = {
        'model':model,
        'train_epoch_losses':train_epoch_losses,
        'val_epoch_losses':val_epoch_losses,
        'best_eval_metric':best['eval_metric'],
        'best_true':best['true'],
        'best_preds':best['preds'],
        'best_epoch':best['epoch'],
        'state_dict_path':best['path']
    }

    return out_dict

def ts_train_val_loop(train_dataset, val_dataset, weights_tensor, device, num_layers, out_channels, k, stride, epochs=35, nhead=4, d_model=256, batch_size=64, num_classes=2, dropout=0.1, lr=1e-3, weight_decay=1e-3, models_dir=MODELS_DIR):
    return train_val_loop('time-series', 'eeg_ts.pt', train_dataset, val_dataset, weights_tensor, device, epochs, batch_size, num_classes, lr, weight_decay, models_dir, num_layers=num_layers, dropout=dropout, out_channels=out_channels, k=k, stride=stride, nhead=nhead, d_model=d_model)

def mlp_train_val_loop(train_dataset, val_dataset, weights_tensor, device, epochs=35, batch_size=64, num_layers=5, num_nodes=64, num_classes=2, dropout=0.1, lr=1e-3, weight_decay=1e-3, models_dir=MODELS_DIR):
    return train_val_loop('mlp', 'eeg_mlp.pt', train_dataset, val_dataset, weights_tensor, device, epochs, batch_size, num_classes, lr, weight_decay, models_dir, num_layers=num_layers, num_nodes=num_nodes, dropout=dropout)

def mult_train_val_loop(train_dataset, val_dataset, weights_tensor, device, epochs, resnet_state_dict_path, mlp_state_dict_path, mlp_best_params, batch_size=64, num_layers=5, num_nodes=64, num_classes=2, dropout=0.1, lr=1e-3, weight_decay=1e-3, models_dir=MODELS_DIR):
    return train_val_loop('mult', 'mult.pt', train_dataset, val_dataset, weights_tensor, device, epochs, batch_size, num_classes, lr, weight_decay, models_dir, num_layers=num_layers, num_nodes=num_nodes, dropout=dropout, resnet_state_dict_path=resnet_state_dict_path, mlp_state_dict_path=mlp_state_dict_path, mlp_best_params=mlp_best_params)

def resnet_train_val_loop(train_dataset, val_dataset, weights_tensor, device, epochs, batch_size=64, num_classes=2, lr=1e-3, weight_decay=1e-3, models_dir=MODELS_DIR):
    models_dir.mkdir(parents=True, exist_ok=True)
    state_dict_path_s1 = models_dir / 'eeg_resnet_s1.pt'
    state_dict_path_s2 = models_dir / 'eeg_resnet_s2.pt'

    model, criterion, optimizer, scheduler, mcc_score, train_loader, val_loader = train_init(train_dataset, val_dataset, weights_tensor, device, 'resnet', batch_size, num_classes, lr, weight_decay)

    best = new_best()
    train_epoch_losses_s1 = []
    val_epoch_losses_s1 = []

    print(f"{'='*15}\n{'Stage 1 Start'}\n{'='*15}")

    train_epoch_losses_s1, val_epoch_losses_s1 = train_epochs(model, train_loader, val_loader, criterion, optimizer, scheduler, mcc_score, epochs, device, 'resnet', state_dict_path_s1, best, train_epoch_losses_s1, val_epoch_losses_s1, desc='[STAGE 1] Epochs', stage=1, clip_grad=False)

    model.load_state_dict(torch.load(best['path'], weights_only=True)) # load best parameter set from stage 1 as start for stage 2

    print(f"{'='*15}\n{'Stage 2 Start'}\n{'='*15}")

    for param in model.layer4.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([
        {'params':model.layer4.parameters(), 'lr':(lr / 10)}, # 10-fold lower lr for last convolutional block to prevent unlearning from transferred ResNet18 model
        {'params':model.fc.parameters(), 'lr':lr}],
        weight_decay=weight_decay
        )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    train_epoch_losses_s2 = []
    val_epoch_losses_s2 = []

    train_epoch_losses_s2, val_epoch_losses_s2 = train_epochs(model, train_loader, val_loader, criterion, optimizer, scheduler, mcc_score, epochs, device, 'resnet', state_dict_path_s2, best, train_epoch_losses_s2, val_epoch_losses_s2, desc='[STAGE 2] Epochs', stage=2, clip_grad=False)

    model.load_state_dict(torch.load(best['path'], weights_only=True)) # load best from stage 2 (or stage 1 if stage 2 never improved on it)

    out_dict = {
        'model':model,
        'train_epoch_losses_s1':train_epoch_losses_s1,
        'val_epoch_losses_s1':val_epoch_losses_s1,
        'best_eval_metric':best['eval_metric'],
        'train_epoch_losses_s2':train_epoch_losses_s2,
        'val_epoch_losses_s2':val_epoch_losses_s2,
        'best_state_dict_path':best['path'],
        'best_true':best['true'],
        'best_preds':best['preds'],
        'best_epoch':best['epoch']
    }

    return out_dict

def save_out_dict(out_dict, json_path):
    serializable = {}
    for key, value in out_dict.items():
        if key == 'model':
            continue
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, Path):
            value = value.name
        serializable[key] = value

    with open(json_path, 'w') as file:
        json.dump(serializable, file, indent=4)

def load_out_dict(json_path, models_dir=MODELS_DIR):
    with open(json_path, 'r') as file:
        out_dict = json.load(file)

    for key in ['state_dict_path', 'best_state_dict_path']:
        if key in out_dict:
            out_dict[key] = models_dir / Path(out_dict[key]).name

    return out_dict

def load_trained_model(model_type, train_dataset, best_params, state_dict_path, device, num_classes=2, resnet_state_dict_path=None, mlp_state_dict_path=None, mlp_best_params=None):
    model_params = {key:value for key, value in (best_params or {}).items() if key != 'lr'} # lr is not a model architecture parameter

    if model_type == 'resnet':
        model = resnet18(weights=None)  # architecture only, weights will be overwritten
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # match final FC layer shape from trained model
        model.load_state_dict(torch.load(state_dict_path, weights_only=True))
        model = model.to(device)
    else:
        model = build_model(model_type, train_dataset, num_classes, device, resnet_state_dict_path=resnet_state_dict_path, mlp_state_dict_path=mlp_state_dict_path, mlp_best_params=mlp_best_params, **model_params)
        if model_type == 'mult':
            model.fusion_head.load_state_dict(torch.load(state_dict_path, weights_only=True))
        else:
            model.load_state_dict(torch.load(state_dict_path, weights_only=True))
    model.eval()

    return model
