# ecg-ptbxl-benchmark

**PTB-XL 5-Superclass Multi-Label ECG Classifier**
Built from scratch in PyTorch. Benchmarked against Strodthoff et al. (2020) — IEEE JBHI.

> **Student:** Abdullah (farooqhaider431) | **Institution:** COMSATS University Islamabad — Lahore Campus

---

## What This Is

A clean, reproducible implementation of a **multi-label ECG classifier** trained on the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/) targeting the standard 5 diagnostic superclasses used in published literature:

| Superclass | Code | Description |
|---|---|---|
| Normal | `NORM` | Normal sinus rhythm / normal ECG |
| Myocardial Infarction | `MI` | Heart attack variants (STEMI, NSTEMI, AMI, ASMI, IMI…) |
| ST/T-Wave Changes | `STTC` | ST depression, T-wave inversions, ischemic changes |
| Conduction Disturbance | `CD` | Bundle branch blocks (LBBB, RBBB, IVCD, LAFB…) |
| Hypertrophy | `HYP` | Left/right ventricular and atrial hypertrophy |

---

## Architecture

```
Input: (batch, 5000, 12)  ← 10-second 12-lead ECG at 500 Hz
    ↓
Conv1D Block 1: 12→64 channels, kernel=7, stride=2  → (B, 64, 2500)
Conv1D Block 2: 64→256 channels, kernel=7, stride=2 → (B, 256, 1250)
    ↓
Patch Embedding: 1250 frames → 25 patches → Linear projection → (B, 25, 256)
Positional Encoding: Learnable (1, 25, 256)
    ↓
Transformer Encoder: 6 layers, 8 heads, d_ff=1024, Pre-LayerNorm → (B, 25, 256)
Global Average Pooling → (B, 256)
    ↓
Classifier: LayerNorm → Dropout → Linear(256→128) → GELU → Dropout → Linear(128→5)
    ↓
Output: raw logits (B, 5)  ← Sigmoid applied at inference/evaluation
```

**Parameters:** 8,177,093 | **Loss:** BCEWithLogitsLoss | **Training platform:** Kaggle (T4×2)

---

## Published SOTA Targets (Strodthoff et al. 2020)

| Model | Macro AUC | Macro F1 |
|---|---|---|
| xResNet1d101 (SOTA) | 0.930 | 0.833 |
| Inception1D | 0.928 | 0.825 |
| ResNet1D-34 | 0.924 | 0.819 |
| **This model (target)** | **≥ 0.925** | **≥ 0.825** |

---

## Repository Structure

```
ecg-ptbxl-benchmark/
├── README.md
├── requirements.txt
├── ecg_benchmark/
│   ├── __init__.py
│   ├── preprocessing.py    ← 5-superclass label extraction + DSP pipeline
│   ├── dataset.py          ← ECGDataset + create_dataloaders()
│   ├── model.py            ← ECGTransformer (single-head, 5 classes)
│   ├── train.py            ← BCEWithLogitsLoss training loop + SWA
│   └── evaluate.py         ← Threshold optimisation + benchmark metrics + plots
└── notebooks/
    ├── 01_train.ipynb      ← Kaggle training notebook
    └── 02_evaluate.ipynb   ← Evaluation and benchmark comparison notebook
```

---

## Quick Start (Kaggle)

See the full setup guide in `SETUP_GUIDE.md`.

```python
# In Kaggle notebook after attaching PTB-XL dataset:
import subprocess
subprocess.run(['git', 'clone', 'https://github.com/farooqhaider431/ecg-ptbxl-benchmark.git'])

import sys
sys.path.insert(0, '/kaggle/working/ecg-ptbxl-benchmark')

from ecg_benchmark.train import train
model, best_epochs = train()
```

---

## References

- Wagner, P. et al. *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data 7, 154 (2020).
- Strodthoff, N. et al. *Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL.* IEEE JBHI (2020).
- Vaswani, A. et al. *Attention Is All You Need.* NeurIPS (2017).
