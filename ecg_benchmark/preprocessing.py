"""
PTB-XL 5-Superclass Multi-Label Preprocessing Pipeline
ecg_benchmark/preprocessing.py
"""

import os
import ast
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
import wfdb

SUPERCLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
NUM_SUPERCLASSES = 5
SUPERCLASS_TO_IDX = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}


def extract_superclass_labels(scp_codes, scp_statements_df):
    """
    Extracts a 5-element binary multi-label vector [NORM, MI, STTC, CD, HYP]
    from a record's scp_codes dictionary using scp_statements.csv mapping.

    Literature Rule: NORM is active (1.0) ONLY if no disease superclass
    (MI, STTC, CD, HYP) is present.
    """
    if isinstance(scp_codes, str):
        try:
            scp_codes = ast.literal_eval(scp_codes)
        except Exception:
            scp_codes = {}

    labels = np.zeros(NUM_SUPERCLASSES, dtype=np.float32)

    if not isinstance(scp_codes, dict) or len(scp_codes) == 0:
        labels[SUPERCLASS_TO_IDX['NORM']] = 1.0
        return labels

    for code in scp_codes.keys():
        if code in scp_statements_df.index:
            diag_class = scp_statements_df.loc[code, 'diagnostic_class']
            if pd.notna(diag_class) and str(diag_class).strip() in SUPERCLASS_TO_IDX:
                idx = SUPERCLASS_TO_IDX[str(diag_class).strip()]
                labels[idx] = 1.0

    # Literature NORM Rule
    has_disease = np.any(labels[1:] > 0)
    if has_disease:
        labels[0] = 0.0
    else:
        labels[0] = 1.0

    return labels


def load_ecg_record(record_path):
    """Loads 12-lead ECG signal using WFDB -> (5000, 12) numpy array."""
    base_path = os.path.splitext(record_path)[0]
    record = wfdb.rdrecord(base_path)
    signal = record.p_signal.astype(np.float32)
    return signal, record.fs


def handle_nans(signal):
    """Replaces NaNs and Infs with 0.0."""
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)


def resample_to_500hz(signal, current_fs):
    """Resamples signal to 500Hz if necessary."""
    if current_fs == 500 or current_fs is None:
        return signal

    num_samples = int(signal.shape[0] * (500.0 / current_fs))
    resampled_signal = resample(signal, num_samples, axis=0)
    return resampled_signal.astype(np.float32)


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=500, order=4):
    """4th-order Butterworth bandpass filter (0.5 - 40 Hz) using zero-phase filtfilt."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal, axis=0)
    return filtered_signal.astype(np.float32)


def normalize_signal(signal):
    """Z-score normalizes each lead independently (zero mean, unit variance)."""
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    normalized = (signal - mean) / (std + 1e-8)
    return normalized.astype(np.float32)


def fix_length(signal, target_len=5000):
    """Trims or zero-pads signal along time axis to exactly target_len samples."""
    current_len = signal.shape[0]
    if current_len == target_len:
        return signal
    elif current_len > target_len:
        return signal[:target_len, :]
    else:
        pad_len = target_len - current_len
        pad = np.zeros((pad_len, signal.shape[1]), dtype=np.float32)
        return np.vstack([signal, pad])


def preprocess_record(record_path, scp_codes, scp_statements_df):
    """Executes full 7-step preprocessing pipeline on a single PTB-XL record."""
    signal, fs = load_ecg_record(record_path)
    signal = handle_nans(signal)
    signal = resample_to_500hz(signal, fs)
    signal = bandpass_filter(signal)
    signal = normalize_signal(signal)
    signal = fix_length(signal, target_len=5000)
    labels = extract_superclass_labels(scp_codes, scp_statements_df)
    return signal, labels
