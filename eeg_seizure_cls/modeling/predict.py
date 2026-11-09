import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchmetrics.classification import MatthewsCorrCoef
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from eeg_seizure_cls.modeling.train import unpack_batch

def test_eval_model(model_list, model_name, device, batch_size=32):
    temp_list = [(i[1].to(device), i[-1]) for i in model_list if i[0] == model_name]
    model, test_dataset = temp_list[0]
    model_type = 'mult' if model_name == 'Multi-Modal Model' else None

    mcc_score = MatthewsCorrCoef(task='binary').to(device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    preds_list = []
    labels_list = []

    model.eval()
    mcc_score.reset()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='[TEST]', leave=False):
            inputs, labels = unpack_batch(batch, device, model_type)
            outputs = model(*inputs)
            preds = outputs.argmax(1)
            mcc_score.update(preds, labels)

            preds_list.append(preds.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    final_score = mcc_score.compute().item()
    y_pred = np.concatenate(preds_list)
    y_true = np.concatenate(labels_list)

    print(f"{'=' * (32 + len(model_name))}\n{model_name} Test Set Evaluation Performance\n{'=' * (32 + len(model_name))}")
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues')
    plt.title(f'{model_name} Test Set Confusion Matrix')
    plt.show()
    plt.close()

    print(classification_report(y_true, y_pred))
    print(f'Final MCC Score: {final_score:.3f}')

def eval_model(model_list, device, batch_size=32):
    best_model_name = sorted(model_list, reverse=True, key=lambda x:x[2])[0][0]
    test_eval_model(model_list, best_model_name, device, batch_size)
