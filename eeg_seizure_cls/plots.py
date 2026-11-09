import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torchvision.utils as vutils
import torchvision.transforms.functional as F

from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from eeg_seizure_cls.config import FIGURES_DIR
from eeg_seizure_cls.dataset import SpectrogramImageDataset

# model_type -> (display name, saved figure prefix)
MODEL_INFO = {
    'time-series':('Time-series', 'ts'),
    'mlp':('MLP', 'mlp'),
    'resnet':('ResNet18', 'resnet'),
    'mult':('Multi-Modal Model', 'mult')
}

def viz_sampled_eeg(df_dict):
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    for idx, ax in enumerate(axes.flat):
        df = df_dict[idx][0]
        indices = df_dict[idx][1]
        up_qmax = df.max(axis=1).quantile(0.9)
        lw_qmin = df.min(axis=1).quantile(0.1)
        #mean = df.mean(axis=0)
        median = df.median(axis=0)
        ax.plot(df.iloc[indices, :].T, c='lightgrey', alpha=0.6)
        #ax.plot(mean, c='y', lw=2)
        ax.plot(median, c='b', lw=1.5, alpha=0.8)
        ax.axhline(up_qmax, c='r', lw=2)
        ax.axhline(lw_qmin, c='r', lw=2)
        ax.set_title('Positive Classes' if idx == 0  else 'Negative Classes')
        ax.set_xlim(0, len(df.columns) - 1)
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Timestep')
        ticks = [tick + 1 for tick in ax.get_xticks()[::20]]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()

def viz_feature_boxplots(clean_df):
    # boxplot comparison of class-stratified dist to evaluate discriminability of features engineered
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    viz_df = clean_df.loc[:, ~clean_df.columns.isin(['patient_id'])]
    for idx, ax in enumerate(axes.flat):
        neg_class = viz_df.loc[viz_df['binary_y'] == 0, viz_df.columns[idx]]
        neg_class.name = f'Negative ({viz_df.columns[idx]})'
        pos_class = viz_df.loc[viz_df['binary_y'] == 1, viz_df.columns[idx]]
        pos_class.name = f'Positive ({viz_df.columns[idx]})'

        assert neg_class.isna().any() == False, 'NAs Present!'
        assert pos_class.isna().any() == False, 'NAs Present!'

        ax.boxplot([neg_class, pos_class])
        title_list = viz_df.columns[idx].split('_')
        ax.set_title(title_list[0].upper() if (title_list[0] == 'mad' or title_list[0] == 'iqr') else title_list[0].capitalize() if title_list[-1] == 'vals' else ' '.join(title_list).title())
        ax.set_ylabel('Values')
        ax.set_xlabel('Class')
        ax.set_xticklabels(['Negative', 'Positive'])
    plt.tight_layout()
    plt.show()
    plt.close()

def viz_feature_hists(features_df):
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    for idx, ax in enumerate(axes.flat):
        ax.hist(features_df.iloc[:, idx], bins='sturges')
        ax.set_title(features_df.columns[idx])
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Values')
    plt.tight_layout()
    plt.show()
    plt.close()

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axes = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(10, 6))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axes[0, i].imshow(np.asarray(img))
        axes[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def viz_spectrograms(raw_df, transform, batch_size=8, nrow=4, padding=5):
    # visualize transformation pipeline output
    X, y = raw_df.loc[:, ~(raw_df.columns.isin(['patient_id', 'binary_y']))].copy(), raw_df['binary_y'].copy()
    test_class = SpectrogramImageDataset(X, y, transform)
    loader = DataLoader(test_class, batch_size=batch_size, shuffle=False)
    for inputs, _ in loader:
        grid = vutils.make_grid(inputs, nrow=nrow, padding=padding)
        show(grid)
        break

def generate_val_report(out_dict, model_type, from_saved=False, figures_dir=FIGURES_DIR):
    if model_type not in MODEL_INFO:
        raise ValueError(f"Model type not in {list(MODEL_INFO.keys())}")

    model_name, prefix = MODEL_INFO[model_type]
    cm_path = figures_dir / f'{prefix}_confusion_matrix.png'

    best_mcc = out_dict['best_eval_metric']
    best_true = out_dict['best_true']
    best_pred = out_dict['best_preds']
    best_epoch = out_dict['best_epoch']

    if model_type == 'resnet':
        train_epoch_losses_s1 = out_dict['train_epoch_losses_s1']
        val_epoch_losses_s1 = out_dict['val_epoch_losses_s1']
        train_epoch_losses_s2 = out_dict['train_epoch_losses_s2']
        val_epoch_losses_s2 = out_dict['val_epoch_losses_s2']
        state_dict_path = out_dict['best_state_dict_path']
        X = np.arange(1, len(train_epoch_losses_s1) + 1)

        if from_saved:
            print(f"Last Epoch Train Loss [STAGE 1]: {train_epoch_losses_s1[-1]:.3f}, Last Epoch Val Loss [STAGE 1]: {val_epoch_losses_s1[-1]:.3f}")
            print(f"Last Epoch Train Loss [STAGE 2]: {train_epoch_losses_s2[-1]:.3f}, Last Epoch Val Loss [STAGE 2]: {val_epoch_losses_s2[-1]:.3f}")

        fig, axes = plt.subplots(2, 1, figsize=(10, 12))
        for idx, ax in enumerate(axes.flat):
            ax.plot(X, out_dict[f'val_epoch_losses_s{idx + 1}'], label='Val')
            ax.plot(X, out_dict[f'train_epoch_losses_s{idx + 1}'], label='Train')
            ax.set_title(f'{model_name} Stage {idx + 1} Learning Curves')
            ax.set_ylabel('Loss')
            ax.set_xlabel('Epochs')
            ax.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

        best_stage = Path(state_dict_path).stem[-1] # state dict file names end in _s1 / _s2
        best_val_loss = val_epoch_losses_s1[best_epoch] if best_stage == '1' else val_epoch_losses_s2[best_epoch]
        best_train_loss = train_epoch_losses_s1[best_epoch] if best_stage == '1' else train_epoch_losses_s2[best_epoch]
    else:
        train_epoch_losses = out_dict['train_epoch_losses']
        val_epoch_losses = out_dict['val_epoch_losses']
        X = np.arange(1, len(train_epoch_losses) + 1)
        best_val_loss = val_epoch_losses[best_epoch]
        best_train_loss = train_epoch_losses[best_epoch]

        if from_saved:
            print(f'Last Epoch Train Loss: {train_epoch_losses[-1]:.3f}, Last Epoch Val Loss: {val_epoch_losses[-1]:.3f}')

        plt.figure(figsize=(10, 6))
        plt.plot(X, val_epoch_losses, label='Val')
        plt.plot(X, train_epoch_losses, label='Train')
        plt.title(f'{model_name} Learning Curves')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()
        plt.close()

    print(f"{'=' * (23 + len(model_name))}\n{model_name} Validation Performance\n{'=' * (23 + len(model_name))}")

    if from_saved:
        img = mpimg.imread(cm_path)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img)
        ax.axis('off')
        plt.show()
        plt.close()
    else:
        figures_dir.mkdir(parents=True, exist_ok=True)
        ConfusionMatrixDisplay.from_predictions(best_true, best_pred, cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix' if model_type == 'mult' else f'{model_name} Model Confusion Matrix')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    print(classification_report(best_true, best_pred))
    print(f'Best {model_name} Val MCC Score: {best_mcc:.3f}')
    if model_type == 'resnet':
        print(f'Best {model_name} Val Loss ([STAGE {best_stage}] Epoch: {best_epoch + 1}): {best_val_loss:.3f}')
        print(f'Best {model_name} Train Loss ([STAGE {best_stage}] Epoch: {best_epoch + 1}): {best_train_loss:.3f}')
    else:
        print(f'Best {model_name} Val Loss (Epoch: {best_epoch + 1}): {best_val_loss:.3f}')
        print(f'Best {model_name} Train Loss (Epoch: {best_epoch + 1}): {best_train_loss:.3f}')
