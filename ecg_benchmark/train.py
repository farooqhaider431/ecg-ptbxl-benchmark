"""
Multi-Label BCE Training Loop for PTB-XL 5-Superclass Benchmark
ecg_benchmark/train.py
"""

import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import f1_score, roc_auc_score

# Default Configuration (Kaggle paths supported dynamically)
CONFIG = {
    'ptbxl_path':     '/kaggle/input/ptb-xl-dataset/ptb-xl-1.0.3',
    'checkpoint_dir': '/kaggle/working/checkpoints',
    'results_dir':    '/kaggle/working/results',
    'batch_size':     32,
    'learning_rate':  1e-4,
    'weight_decay':   0.01,
    'warmup_steps':   500,
    'max_epochs':     75,
    'phase1_end':     50,
    'grad_clip':      1.0,
    'min_lr':         1e-6,
    'patience_phase1':         15,
    'patience_phase2':         8,
    'min_delta_phase1':        0.001,
    'min_delta_phase2':        0.002,
    'plateau_patience_phase1': 7,
    'plateau_patience_phase2': 4,
    'plateau_factor_phase1':   0.5,
    'plateau_factor_phase2':   0.3,
    'snapshot_every':          3,
    'swa_top_k':              5,
}

SUPERCLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def compute_multilabel_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    try:
        macro_auc = roc_auc_score(y_true, y_prob, average='macro')
        per_auc   = roc_auc_score(y_true, y_prob, average=None)
    except ValueError:
        macro_auc = 0.0
        per_auc   = np.zeros(5)

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    per_f1   = f1_score(y_true, y_pred, average=None,    zero_division=0)

    return {
        'macro_auc': macro_auc,
        'macro_f1':  macro_f1,
        'per_auc':   per_auc,
        'per_f1':    per_f1,
    }


def train_one_epoch(model, loader, criterion, optimizer, device, config, global_step):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for signals, labels in loader:
        if global_step < config['warmup_steps']:
            warmup_factor = (global_step + 1) / config['warmup_steps']
            for pg in optimizer.param_groups:
                pg['lr'] = config['learning_rate'] * warmup_factor

        signals = signals.to(device)
        labels  = labels.to(device)

        optimizer.zero_grad()
        logits = model(signals)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
        global_step += 1

    return total_loss / n_batches, global_step


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            labels  = labels.to(device)

            logits = model(signals)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            n_batches  += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    metrics = compute_multilabel_metrics(y_true, y_prob)

    return total_loss / n_batches, metrics


def save_checkpoint_safe(state, path):
    tmp = path + '.tmp'
    torch.save(state, tmp)
    verify = torch.load(tmp, map_location='cpu', weights_only=False)
    assert verify['epoch'] == state['epoch']
    os.replace(tmp, path)


def load_checkpoint_safe(checkpoint_dir):
    rolling = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth')))
    for path in reversed(rolling[-2:]):
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            print(f"Loaded checkpoint: {path} (epoch {ckpt['epoch']})")
            return ckpt
        except Exception as e:
            print(f"Skipping corrupted checkpoint {path}: {e}")

    snapshots = sorted(glob.glob(os.path.join(checkpoint_dir, 'snapshot_epoch_*.pth')))
    if snapshots:
        try:
            ckpt = torch.load(snapshots[-1], map_location='cpu', weights_only=False)
            print(f"Loaded snapshot: {snapshots[-1]}")
            return ckpt
        except Exception:
            pass

    print("No valid checkpoint found — starting fresh.")
    return None


def save_training_log(log_row, results_dir):
    path = os.path.join(results_dir, 'training_log.csv')
    df   = pd.DataFrame([log_row])
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


def build_swa_model(model_class, best_epochs, checkpoint_dir, device, top_k=5):
    epochs_to_avg = best_epochs[-top_k:]
    print(f"\nSWA: Averaging checkpoints from epochs: {epochs_to_avg}")

    avg_state = None
    loaded    = 0

    for ep in epochs_to_avg:
        paths = [
            os.path.join(checkpoint_dir, f'checkpoint_epoch_{ep:03d}.pth'),
            os.path.join(checkpoint_dir, f'snapshot_epoch_{ep:03d}.pth'),
        ]
        for path in paths:
            if os.path.exists(path):
                try:
                    ckpt       = torch.load(path, map_location='cpu', weights_only=False)
                    state_dict = ckpt['model_state_dict']
                    if avg_state is None:
                        avg_state = {k: v.clone().float() for k, v in state_dict.items()}
                    else:
                        for k in avg_state:
                            avg_state[k] += state_dict[k].float()
                    loaded += 1
                    break
                except Exception as e:
                    print(f"  Skipping {path}: {e}")

    if loaded == 0:
        print("SWA: No checkpoints found to average.")
        return None

    for k in avg_state:
        avg_state[k] /= loaded

    model = model_class().to(device)
    model.load_state_dict(avg_state)
    print(f"SWA: Averaged {loaded} checkpoints successfully.")
    return model


def train(config=None):
    if config is None:
        config = CONFIG.copy()

    # Dynamic path detection for Kaggle vs Drive vs local
    if not os.path.exists(config['ptbxl_path']):
        possible_paths = [
            '/kaggle/input/ptb-xl-dataset/ptb-xl-1.0.3',
            '/kaggle/input/ptb-xl-dataset',
            '/kaggle/input/physionet-ptbxl',
            '/content/drive/MyDrive/MediSense/datasets/module3_ecg/ptbxl/physionet.org/files/ptb-xl/1.0.3',
        ]
        for p in possible_paths:
            if os.path.exists(p):
                config['ptbxl_path'] = p
                break

    from .model import ECGTransformer
    from .dataset import create_dataloaders

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['results_dir'],    exist_ok=True)

    print(f"Loading dataset from: {config['ptbxl_path']}")
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        config['ptbxl_path'], batch_size=config['batch_size']
    )
    print(f"Train records: {len(train_dataset):,} | Val records: {len(val_loader.dataset):,} | Test: {len(test_loader.dataset):,}")

    model = ECGTransformer().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=config['max_epochs'], eta_min=config['min_lr'])
    scheduler_p1 = ReduceLROnPlateau(optimizer, mode='max', patience=config['plateau_patience_phase1'], factor=config['plateau_factor_phase1'], min_lr=config['min_lr'])
    scheduler_p2 = ReduceLROnPlateau(optimizer, mode='max', patience=config['plateau_patience_phase2'], factor=config['plateau_factor_phase2'], min_lr=config['min_lr'])

    start_epoch = 1; best_auc = 0.0; patience_counter = 0
    training_log = []; best_epochs = []; global_step = 0

    ckpt = load_checkpoint_safe(config['checkpoint_dir'])
    if ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler_cosine.load_state_dict(ckpt['scheduler_cosine'])
        start_epoch      = ckpt['epoch'] + 1
        best_auc         = ckpt['best_auc']
        patience_counter = ckpt['patience_counter']
        training_log     = ckpt.get('training_log', [])
        best_epochs      = ckpt.get('best_epochs', [])
        global_step      = ckpt.get('global_step', start_epoch * len(train_loader))
        print(f"Resuming from epoch {ckpt['epoch']} — best Val AUC: {best_auc:.4f}")
    else:
        for pg in optimizer.param_groups:
            pg['lr'] = config['learning_rate'] / 100

    print(f"Starting training loop — epochs {start_epoch} to {config['max_epochs']}")

    for epoch in range(start_epoch, config['max_epochs'] + 1):
        t0     = time.time()
        phase  = 1 if epoch <= config['phase1_end'] else 2
        plimit = config['patience_phase1']  if phase == 1 else config['patience_phase2']
        delta  = config['min_delta_phase1'] if phase == 1 else config['min_delta_phase2']
        psched = scheduler_p1               if phase == 1 else scheduler_p2

        if epoch == config['phase1_end'] + 1:
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * config['plateau_factor_phase2']
            print("\n=== PHASE 2 STARTS ===")

        train_loss, global_step = train_one_epoch(model, train_loader, criterion, optimizer, device, config, global_step)
        val_loss, metrics = validate(model, val_loader, criterion, device)

        elapsed  = time.time() - t0
        lr       = optimizer.param_groups[0]['lr']
        improved = metrics['macro_auc'] > best_auc + delta

        print(f"Epoch {epoch:02d}/{config['max_epochs']} | Phase {phase} | Loss: {train_loss:.4f} (train), {val_loss:.4f} (val) | Val AUC: {metrics['macro_auc']:.4f} | Val F1: {metrics['macro_f1']:.4f} | LR: {lr:.2e} | {int(elapsed)}s")

        if improved:
            best_auc = metrics['macro_auc']
            patience_counter = 0
            best_epochs.append(epoch)
            save_checkpoint_safe({'epoch': epoch, 'model_state_dict': model.state_dict(), 'metrics': metrics, 'best_auc': best_auc}, os.path.join(config['checkpoint_dir'], 'best_model.pth'))
            print(f"  --> NEW BEST: Val Macro AUC = {best_auc:.4f}")
        else:
            patience_counter += 1

        if global_step >= config['warmup_steps']:
            scheduler_cosine.step()
        psched.step(metrics['macro_auc'])

        log_row = {
            'epoch': epoch, 'phase': phase, 'train_loss': train_loss, 'val_loss': val_loss,
            'val_auc_macro': metrics['macro_auc'], 'val_f1_macro': metrics['macro_f1'],
            'lr': lr, 'patience': patience_counter, 'time_s': elapsed
        }
        for i, name in enumerate(SUPERCLASS_NAMES):
            log_row[f'auc_{name}'] = metrics['per_auc'][i] if i < len(metrics['per_auc']) else 0.0
            log_row[f'f1_{name}']  = metrics['per_f1'][i]  if i < len(metrics['per_f1'])  else 0.0
        save_training_log(log_row, config['results_dir'])
        training_log.append(log_row)

        ckpt_state = {
            'epoch': epoch, 'phase': phase, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_cosine': scheduler_cosine.state_dict(),
            'best_auc': best_auc, 'patience_counter': patience_counter,
            'training_log': training_log, 'best_epochs': best_epochs, 'global_step': global_step,
        }
        save_checkpoint_safe(ckpt_state, os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch:03d}.pth'))

        old = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch-2:03d}.pth')
        if os.path.exists(old) and epoch % config['snapshot_every'] != 0:
            os.remove(old)

        if epoch % config['snapshot_every'] == 0:
            save_checkpoint_safe(ckpt_state, os.path.join(config['checkpoint_dir'], f'snapshot_epoch_{epoch:03d}.pth'))

        if patience_counter >= plimit:
            print(f"\nEARLY STOPPING triggered (patience {plimit} reached). Best Val AUC: {best_auc:.4f}")
            break

    print(f"\nTraining complete. Averaging best epochs: {best_epochs[-config['swa_top_k']:]}")
    swa_model = build_swa_model(ECGTransformer, best_epochs, config['checkpoint_dir'], device, top_k=config['swa_top_k'])
    if swa_model is not None:
        save_checkpoint_safe({'epoch': epoch, 'model_state_dict': swa_model.state_dict(), 'best_auc': best_auc, 'best_epochs': best_epochs}, os.path.join(config['checkpoint_dir'], 'best_model_swa.pth'))
        print("SWA Model Saved: best_model_swa.pth")

    return model, best_epochs


if __name__ == '__main__':
    train()
