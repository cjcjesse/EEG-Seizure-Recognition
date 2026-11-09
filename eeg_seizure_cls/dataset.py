import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.utils.class_weight import compute_class_weight
from scipy.signal import spectrogram

def generate_splits(df, model_type, seed):
    if model_type not in ['time-series', 'mlp', 'resnet']:
        raise ValueError("Model type not in ['time-series', 'mlp', 'resnet']")

    unique_ids = df.groupby('patient_id')['binary_y'].first().reset_index() # drop=False default
    outer_ids, test_ids = train_test_split(unique_ids, train_size=0.8, test_size=0.2, stratify=unique_ids['binary_y'], random_state=seed)
    train_ids, val_ids = train_test_split(outer_ids, train_size=0.9, test_size=0.1, stratify=outer_ids['binary_y'], random_state=seed)

    train_ids = train_ids['patient_id']
    val_ids = val_ids['patient_id']
    test_ids = test_ids['patient_id']

    train_set = df.loc[df['patient_id'].isin(train_ids.values), ~df.columns.isin(['patient_id'])].reset_index(drop=True).copy() # (n_samples, seq_len)
    val_set = df.loc[df['patient_id'].isin(val_ids.values), ~df.columns.isin(['patient_id'])].reset_index(drop=True).copy() # (n_samples, seq_len)
    test_set = df.loc[df['patient_id'].isin(test_ids.values), ~df.columns.isin(['patient_id'])].reset_index(drop=True).copy() # (n_samples, seq_len)

    if model_type == 'time-series':
        train_arr = train_set.iloc[:, :-1].to_numpy(dtype=np.float64)
        val_arr = val_set.iloc[:, :-1].to_numpy(dtype=np.float64)
        test_arr = test_set.iloc[:, :-1].to_numpy(dtype=np.float64)

        X_train = (train_arr - np.mean(train_arr, axis=1, keepdims=True)) / (np.std(train_arr, axis=1, keepdims=True) + 1e-8)
        X_val = (val_arr - np.mean(val_arr, axis=1, keepdims=True)) / (np.std(val_arr, axis=1, keepdims=True) + 1e-8)
        X_test = (test_arr - np.mean(test_arr, axis=1, keepdims=True)) / (np.std(test_arr, axis=1, keepdims=True) + 1e-8)

        y_train = train_set['binary_y'].to_numpy()
        y_val = val_set['binary_y'].to_numpy()
        y_test = test_set['binary_y'].to_numpy()

    if model_type == 'mlp':
        X_train, y_train = train_set.loc[:, ~train_set.columns.isin(['binary_y'])].to_numpy(copy=True), train_set['binary_y'].copy().values # writable copy, scaled in place below
        X_val, y_val = val_set.loc[:, ~val_set.columns.isin(['binary_y'])].to_numpy(copy=True), val_set['binary_y'].copy().values
        X_test, y_test = test_set.loc[:, ~test_set.columns.isin(['binary_y'])].to_numpy(copy=True), test_set['binary_y'].copy().values

        pt_train_scaler = PowerTransformer(method='yeo-johnson').fit(X_train[:, :-1]) # fit on training set only EXCLUDING last feature (log-scaled length)

        X_train[:, :-1] = pt_train_scaler.transform(X_train[:, :-1]) # Yeo-Johnson transform all features EXCLUDING last feature (log-scaled length)
        X_val[:, :-1] = pt_train_scaler.transform(X_val[:, :-1]) # Yeo-Johnson transform all features EXCLUDING last feature (log-scaled length)
        X_test[:, :-1] = pt_train_scaler.transform(X_test[:, :-1]) # Yeo-Johnson transform all features EXCLUDING last feature (log-scaled length)

        z_train_scaler = StandardScaler().fit(X_train[:, -1:]) # fit on training set only, on only last feature (log-scaled length)

        X_train[:, -1:] = z_train_scaler.transform(X_train[:, -1:]) # z-norm last feature (log-scaled length)
        X_val[:, -1:] = z_train_scaler.transform(X_val[:, -1:]) # z-norm last feature (log-scaled length)
        X_test[:, -1:] = z_train_scaler.transform(X_test[:, -1:]) # z-norm last feature (log-scaled length)

    if model_type == 'resnet':
        X_train, y_train = train_set.drop(columns=['binary_y']).copy().values, train_set['binary_y'].copy().values
        X_val, y_val = val_set.drop(columns=['binary_y']).copy().values, val_set['binary_y'].copy().values
        X_test, y_test = test_set.drop(columns=['binary_y']).copy().values, test_set['binary_y'].copy().values

    return X_train, y_train, X_val, y_val, X_test, y_test

def generate_class_weights(y_train, device):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train) # deterministic once training set split is established
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    return weights_tensor

class TimeSeriesDataset(Dataset):
    def __init__(self, X_scaled, y):
        X_scaled = X_scaled.values if hasattr(X_scaled, 'values') else np.array(X_scaled) # (n_examples, seq_len=178)
        y = y.values if hasattr(y, 'values') else np.array(y) # (n_examples, )
        self.X = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(-1).permute(0, 2, 1) # (n_examples, num_in_features=1, seq_len=178)
        self.y = torch.tensor(y, dtype=torch.long) # (n_examples, )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ScaledMLPDataset(Dataset):
    def __init__(self, X_scaled, y):
        X_scaled = X_scaled.values if hasattr(X_scaled, 'values') else np.array(X_scaled) # (n_examples, 9 features)
        y = y.values if hasattr(y, 'values') else np.array(y) # (n_examples, )
        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) # (n_examples, )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SpectrogramTransform(nn.Module):
    def forward(self, example):
        _, _, Sxx = spectrogram(example, fs=173.61, nperseg=32, noverlap=24)
        Sxx = 10 * np.log10(Sxx + 1e-10) # convert PSD values into decibels to compress dynamic range
        Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min() + 1e-10) # min-max scale for resnet

        Sxx_tensor = torch.from_numpy(Sxx).float().unsqueeze(0) # (1, H, W)
        Sxx_tensor = Sxx_tensor.repeat(3, 1, 1) # (3, H, W)

        return Sxx_tensor

transform = transforms.Compose([
    SpectrogramTransform(),
    transforms.Resize((224, 224)), # resize spectrogram 'image' to be compatible with resnet input spatial resolution, default bilinear interpolation
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize to ImageNet mean and std
])

class SpectrogramImageDataset(Dataset):
    def __init__(self, X, y, transform):
        self.X = X.values if hasattr(X, 'values') else np.array(X)  # (n_examples, m_features) np array, tensor conversion part of transformation pipeline
        self.y = y.values if hasattr(y, 'values') else np.array(y)  # (n_examples,) np array, tensor conversion part of transformation pipeline
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        example, label = self.X[idx, :], self.y[idx] # retrieve single row example and its corresponding class label
        image = self.transform(example)

        return image, torch.tensor(label, dtype=torch.long)

class MultiModalDataset(Dataset):
    def __init__(self, X_scaled, X_raw, y, transform):
        X_scaled = X_scaled.values if hasattr(X_scaled, 'values') else np.array(X_scaled)
        X_raw = X_raw.values if hasattr(X_raw, 'values') else np.array(X_raw)
        y = y.values if hasattr(y, 'values') else np.array(y)

        assert len(X_scaled) == len(X_raw) == len(y), 'Row mismatch between numerical, raw signal, and labels'

        self.X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
        self.X_raw = X_raw # keep as np.ndarray, transform expects raw numpy per-row
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        numerical = self.X_scaled[idx] # (9,) tensor
        raw_signal = self.X_raw[idx, :] # (m,) np.ndarray
        image = self.transform(raw_signal) # (3, 224, 224) tensor
        label = self.y[idx]

        return numerical, image, label
