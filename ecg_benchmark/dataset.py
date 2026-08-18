"""
PyTorch Dataset & DataLoader for PTB-XL 5-Superclass Multi-Label Benchmark
ecg_benchmark/dataset.py
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import (
    extract_superclass_labels,
    load_ecg_record,
    handle_nans,
    resample_to_500hz,
    bandpass_filter,
    normalize_signal,
    fix_length,
    NUM_SUPERCLASSES
)


class ECGDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL 5-Superclass Multi-Label Classification.
    Robust path resolution supports both PhysioNet and Kaggle dataset structures.
    """

    def __init__(self, ptbxl_path, folds=None):
        self.ptbxl_path = ptbxl_path
        self.folds = folds

        csv_path = self._find_file('ptbxl_database.csv')
        scp_path = self._find_file('scp_statements.csv')

        df = pd.read_csv(csv_path)
        scp_df = pd.read_csv(scp_path, index_col=0)

        if folds is not None:
            df = df[df['strat_fold'].isin(folds)].copy()

        base_dir = os.path.dirname(csv_path)

        self.records = []
        for _, row in df.iterrows():
            rel_path = row['filename_hr']
            record_path = os.path.join(base_dir, rel_path)

            scp_codes = row['scp_codes']
            label_vec = extract_superclass_labels(scp_codes, scp_df)

            self.records.append({
                'path': record_path,
                'label': label_vec
            })

    def _find_file(self, filename):
        """Finds CSV file using os.walk across ptbxl_path, /kaggle/input, and parent directories."""
        # 1. Direct check
        direct = os.path.join(self.ptbxl_path, filename)
        if os.path.exists(direct):
            return direct

        # 2. Walk ptbxl_path
        if os.path.exists(self.ptbxl_path):
            for dirpath, _, filenames in os.walk(self.ptbxl_path):
                for f in filenames:
                    if f.lower() == filename.lower():
                        return os.path.join(dirpath, f)

        # 3. Walk /kaggle/input
        if os.path.exists('/kaggle/input'):
            for dirpath, _, filenames in os.walk('/kaggle/input'):
                for f in filenames:
                    if f.lower() == filename.lower():
                        return os.path.join(dirpath, f)

        # 4. Walk /content (Colab)
        if os.path.exists('/content'):
            for dirpath, _, filenames in os.walk('/content'):
                for f in filenames:
                    if f.lower() == filename.lower():
                        return os.path.join(dirpath, f)

        raise FileNotFoundError(f"Could not locate {filename} anywhere under {self.ptbxl_path} or /kaggle/input")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        record_path = item['path']
        label_vec = item['label']

        signal, fs = load_ecg_record(record_path)
        signal = handle_nans(signal)
        signal = resample_to_500hz(signal, fs)
        signal = bandpass_filter(signal)
        signal = normalize_signal(signal)
        signal = fix_length(signal, target_len=5000)

        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        label_tensor = torch.tensor(label_vec, dtype=torch.float32)

        return signal_tensor, label_tensor

    def get_labels(self):
        return np.array([r['label'] for r in self.records], dtype=np.float32)


def create_dataloaders(ptbxl_path, batch_size=32, num_workers=2):
    """
    Creates DataLoaders for Train (Folds 1-8), Validation (Fold 9), and Test (Fold 10).
    """
    train_dataset = ECGDataset(ptbxl_path, folds=[1, 2, 3, 4, 5, 6, 7, 8])
    val_dataset   = ECGDataset(ptbxl_path, folds=[9])
    test_dataset  = ECGDataset(ptbxl_path, folds=[10])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset
