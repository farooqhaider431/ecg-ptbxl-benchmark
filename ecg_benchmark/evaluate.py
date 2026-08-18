"""
Evaluation Script for PTB-XL 5-Superclass Multi-Label Benchmark
ecg_benchmark/evaluate.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score,
    hamming_loss, confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings('ignore')

SUPERCLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
STRODTHOFF_MACRO_AUC = 0.930
STRODTHOFF_MACRO_F1  = 0.833
STRODTHOFF_PER_AUC   = {'NORM': 0.957, 'MI': 0.937, 'STTC': 0.920, 'CD': 0.937, 'HYP': 0.898}


def run_inference(model, loader, device):
    model.eval()
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            logits  = model(signals)
            probs   = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_probs)


def optimise_thresholds(y_true, y_prob, low=0.10, high=0.90, step=0.01):
    thresholds = np.arange(low, high + step, step)
    opt_thresh = np.zeros(5)
    opt_f1     = np.zeros(5)

    print("\nThreshold Optimisation on Validation Fold 9:")
    print(f"  Search range: [{low:.2f}, {high:.2f}] step {step:.2f} - {len(thresholds)} candidates per class")
    print(f"  {'Class':<6}  {'Best Threshold':>14}  {'Val F1 @ Threshold':>18}")
    print(f"  {'-' * 44}")

    for c, name in enumerate(SUPERCLASS_NAMES):
        best_t  = 0.5
        best_f1 = 0.0
        for t in thresholds:
            preds = (y_prob[:, c] >= t).astype(int)
            f1    = f1_score(y_true[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t  = t
        opt_thresh[c] = best_t
        opt_f1[c]     = best_f1
        print(f"  {name:<6}  {best_t:>14.2f}  {best_f1:>18.4f}")

    print(f"\n  Optimised thresholds: {opt_thresh.round(2)}")
    return opt_thresh, opt_f1


def compute_full_metrics(y_true, y_prob, thresholds):
    y_pred = np.zeros_like(y_prob, dtype=int)
    for c in range(5):
        y_pred[:, c] = (y_prob[:, c] >= thresholds[c]).astype(int)

    try:
        macro_auc = roc_auc_score(y_true, y_prob, average='macro')
        per_auc   = roc_auc_score(y_true, y_prob, average=None)
    except ValueError:
        macro_auc = 0.0
        per_auc   = np.zeros(5)

    macro_f1  = f1_score(y_true, y_pred, average='macro',   zero_division=0)
    per_f1    = f1_score(y_true, y_pred, average=None,      zero_division=0)
    macro_pre = precision_score(y_true, y_pred, average='macro', zero_division=0)
    per_pre   = precision_score(y_true, y_pred, average=None,    zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    per_rec   = recall_score(y_true, y_pred, average=None,    zero_division=0)
    ham_loss  = hamming_loss(y_true, y_pred)

    return {
        'macro_auc':  macro_auc,
        'macro_f1':   macro_f1,
        'macro_pre':  macro_pre,
        'macro_rec':  macro_rec,
        'hamming':    ham_loss,
        'per_auc':    per_auc,
        'per_f1':     per_f1,
        'per_pre':    per_pre,
        'per_rec':    per_rec,
        'y_pred':     y_pred,
        'thresholds': thresholds,
    }


def print_benchmark_table(metrics, n_test):
    sep70 = '=' * 70
    sep60 = '-' * 60
    print(f"\n{sep70}")
    print(f"  PTB-XL 5-SUPERCLASS BENCHMARK - TEST FOLD 10 RESULTS")
    print(f"  Total test records: {n_test:,}")
    print(f"{sep70}")

    gap_auc = metrics['macro_auc'] - STRODTHOFF_MACRO_AUC
    gap_f1  = metrics['macro_f1']  - STRODTHOFF_MACRO_F1
    print(f"\n  OVERALL METRICS")
    print(f"  {'Metric':<22}  {'MediSense':>10}  {'SOTA (xResNet)':>14}  {'Gap':>8}")
    print(f"  {sep60}")
    print(f"  {'Macro AUC (primary)':<22}  {metrics['macro_auc']:>10.4f}  {STRODTHOFF_MACRO_AUC:>14.3f}  {gap_auc:>+8.4f}")
    print(f"  {'Macro F1 (threshold-opt)':<22}  {metrics['macro_f1']:>10.4f}  {STRODTHOFF_MACRO_F1:>14.3f}  {gap_f1:>+8.4f}")
    print(f"  {'Macro Precision':<22}  {metrics['macro_pre']:>10.4f}  {'N/A':>14}  {'N/A':>8}")
    print(f"  {'Macro Recall':<22}  {metrics['macro_rec']:>10.4f}  {'N/A':>14}  {'N/A':>8}")
    print(f"  {'Hamming Loss':<22}  {metrics['hamming']:>10.4f}  {'N/A':>14}  {'N/A':>8}")

    print(f"\n  PER-CLASS BREAKDOWN")
    print(f"  {'Class':<6}  {'AUC':>7}  {'SOTA AUC':>9}  {'Gap':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}  {'Thresh':>7}")
    print(f"  {sep70}")
    for i, name in enumerate(SUPERCLASS_NAMES):
        auc  = metrics['per_auc'][i]
        f1   = metrics['per_f1'][i]
        pre  = metrics['per_pre'][i]
        rec  = metrics['per_rec'][i]
        thr  = metrics['thresholds'][i]
        sota = STRODTHOFF_PER_AUC.get(name, 0.0)
        gap  = auc - sota
        print(f"  {name:<6}  {auc:>7.4f}  {sota:>9.3f}  {gap:>+7.4f}  {f1:>7.4f}  {pre:>7.4f}  {rec:>7.4f}  {thr:>7.2f}")

    print(f"\n{sep70}")

    if metrics['macro_auc'] >= STRODTHOFF_MACRO_AUC:
        print("  VERDICT: [EXCEEDS SOTA] Macro AUC meets or beats published benchmark!")
    elif metrics['macro_auc'] >= STRODTHOFF_MACRO_AUC - 0.01:
        print("  VERDICT: [COMPETITIVE] Within 1% of SOTA Macro AUC.")
    elif metrics['macro_auc'] >= STRODTHOFF_MACRO_AUC - 0.03:
        print("  VERDICT: [STRONG] Within 3% of SOTA Macro AUC -- solid result.")
    else:
        print(f"  VERDICT: [GAP {abs(gap_auc):.4f}] Review training run -- consider additional epochs or threshold tuning.")
    print(f"{sep70}\n")


def plot_roc_curves(y_true, y_prob, results_dir):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('PTB-XL 5-Superclass ROC Curves — Test Fold 10', fontsize=13, fontweight='bold')

    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
    for i, (name, color) in enumerate(zip(SUPERCLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        auc_val     = roc_auc_score(y_true[:, i], y_prob[:, i])
        sota_auc    = STRODTHOFF_PER_AUC.get(name, 0.0)

        axes[i].plot(fpr, tpr, color=color, lw=2, label=f'AUC = {auc_val:.4f}')
        axes[i].axhline(sota_auc, color='gray', lw=1, linestyle='--', alpha=0.7, label=f'SOTA = {sota_auc:.3f}')
        axes[i].plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4)
        axes[i].set_title(name, fontweight='bold')
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].legend(loc='lower right', fontsize=8)
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1.02])
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, 'roc_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def save_results_csv(metrics, thresholds, results_dir):
    rows = []
    for i, name in enumerate(SUPERCLASS_NAMES):
        rows.append({
            'superclass':      name,
            'auc':             metrics['per_auc'][i],
            'f1':              metrics['per_f1'][i],
            'precision':       metrics['per_pre'][i],
            'recall':          metrics['per_rec'][i],
            'threshold':       thresholds[i],
            'sota_auc':        STRODTHOFF_PER_AUC.get(name, None),
            'auc_gap_vs_sota': metrics['per_auc'][i] - STRODTHOFF_PER_AUC.get(name, 0.0),
        })
    rows.append({
        'superclass':      'MACRO_AVERAGE',
        'auc':             metrics['macro_auc'],
        'f1':              metrics['macro_f1'],
        'precision':       metrics['macro_pre'],
        'recall':          metrics['macro_rec'],
        'threshold':       None,
        'sota_auc':        STRODTHOFF_MACRO_AUC,
        'auc_gap_vs_sota': metrics['macro_auc'] - STRODTHOFF_MACRO_AUC,
    })
    df   = pd.DataFrame(rows)
    path = os.path.join(results_dir, 'test_fold10_results.csv')
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df


def run_evaluation(config=None):
    if config is None:
        config = {
            'ptbxl_path':     '/kaggle/input/ptb-xl-dataset/ptb-xl-1.0.3',
            'checkpoint_dir': '/kaggle/working/checkpoints',
            'results_dir':    '/kaggle/working/results',
        }

    # Dynamic path detection for Kaggle vs Drive vs local
    search_roots = [
        '/kaggle/input',
        config['ptbxl_path'],
        '/content',
    ]
    found = False
    for root in search_roots:
        if os.path.exists(root):
            for dirpath, _, filenames in os.walk(root):
                if any(f.lower() == 'ptbxl_database.csv' for f in filenames):
                    config['ptbxl_path'] = dirpath
                    found = True
                    break
            if found:
                break

    from .model import ECGTransformer
    from .dataset import create_dataloaders

    os.makedirs(config['results_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation device: {device}")

    swa_path  = os.path.join(config['checkpoint_dir'], 'best_model_swa.pth')
    best_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')

    model = ECGTransformer().to(device)
    if os.path.exists(swa_path):
        ckpt = torch.load(swa_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded SWA model: {swa_path}")
    elif os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded best model: {best_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found at:\n  {swa_path}\n  {best_path}")

    _, val_loader, test_loader, _ = create_dataloaders(config['ptbxl_path'], batch_size=64)

    print("\nStep 1/4: Running inference on Validation Fold 9...")
    val_true, val_prob = run_inference(model, val_loader, device)

    print("\nStep 2/4: Optimising per-class thresholds on Fold 9...")
    opt_thresholds, val_f1_per = optimise_thresholds(val_true, val_prob)

    print("\nStep 3/4: Running inference on Test Fold 10 (held out)...")
    test_true, test_prob = run_inference(model, test_loader, device)

    print("\nStep 4/4: Computing final metrics on Test Fold 10...")
    metrics = compute_full_metrics(test_true, test_prob, opt_thresholds)

    print_benchmark_table(metrics, n_test=test_true.shape[0])

    plot_roc_curves(test_true, test_prob, config['results_dir'])
    results_df = save_results_csv(metrics, opt_thresholds, config['results_dir'])

    return {
        'metrics':          metrics,
        'thresholds':       opt_thresholds,
        'val_f1_per_class': val_f1_per,
        'results_df':       results_df,
    }


if __name__ == '__main__':
    run_evaluation()
