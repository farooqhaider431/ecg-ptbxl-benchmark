# PTB-XL 5-Superclass Benchmark — Setup & Execution Guide

Complete step-by-step guide for setting up the environment from scratch across **GitHub**, **Kaggle**, and **Google Drive**, attaching the dataset, running training, and generating benchmark evaluation results.

---

## Overview

- **Repository Name:** `ecg-ptbxl-benchmark`
- **Target Taxonomy:** 5 PTB-XL Superclasses (`NORM`, `MI`, `STTC`, `CD`, `HYP`)
- **Dataset:** PhysioNet PTB-XL (v1.0.3)
- **Primary Training Platform:** Kaggle Notebooks (Dual T4 or P100 GPU)
- **Reference Benchmark:** Strodthoff et al. (2020) — IEEE JBHI (Macro AUC ~0.930, Macro F1 ~0.833)

---

## Phase 1 — GitHub Setup (New Repository)

### 1. Create a New Repository on GitHub
1. Go to [GitHub New Repository](https://github.com/new).
2. Set **Repository name**: `ecg-ptbxl-benchmark`
3. Set **Description**: `PTB-XL 5-Superclass Multi-Label ECG Transformer Benchmark`
4. Set visibility: **Public** (recommended so Kaggle can clone it easily without personal access tokens).
5. Leave "Add a README file" unchecked (we already built `README.md`).
6. Click **Create repository**.

### 2. Initialize and Push Local Repository
In your local command prompt / terminal inside `d:\Medisense-ai\ecg-ptbxl-benchmark`:

```bash
# Navigate to the repo folder
cd d:\Medisense-ai\ecg-ptbxl-benchmark

# Initialize Git
git init
git branch -M main

# Add all files
git add .
git commit -m "feat: initial commit for PTB-XL 5-superclass benchmark"

# Link to your GitHub remote and push
git remote add origin https://github.com/farooqhaider431/ecg-ptbxl-benchmark.git
git push -u origin main
```

---

## Phase 2 — Kaggle Environment Setup

### 1. Locating the PTB-XL Dataset on Kaggle
You do **not** need to upload 2GB of dataset files to Kaggle yourself. PTB-XL is hosted publicly on Kaggle.

1. Open [Kaggle Datasets](https://www.kaggle.com/datasets).
2. Search for `ptb-xl-dataset` or `physionet ptbxl`.
3. Locate the dataset named **PTB-XL Dataset** (by PhysioNet / 1.0.3 version).
4. Direct Kaggle dataset link: `https://www.kaggle.com/datasets/khaledgohar/ptb-xl-dataset` or `https://www.kaggle.com/datasets/bjoernjostein/ptbxl-dataset`.
5. Bookmark / note this dataset.

### 2. Creating the Kaggle Training Notebook
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code) -> Click **New Notebook**.
2. Title your notebook: `PTB-XL 5-Superclass Benchmark Training`.
3. In the right-hand panel under **Input**:
   - Click **+ Add Data**.
   - Search `ptb-xl-dataset` and click **Add** next to the PTB-XL dataset.
   - It will attach to `/kaggle/input/ptb-xl-dataset/` or `/kaggle/input/physionet-ptbxl/`.
4. In the right-hand panel under **Notebook options**:
   - **ACCELERATOR**: Select **GPU T4 x2** (or GPU P100).
   - **PERSISTENCE**: Select **Variables and Files** (preserves output files across restarts).
   - **ENVIRONMENT**: Always use latest environment.
   - **INTERNET**: Toggle **ON** (required for `git clone` and package installations).

---

## Phase 3 — Training Execution on Kaggle

In your Kaggle Notebook, create and execute the following cells in order:

### Cell 1: Clone Repository & Setup Environment
```python
import os
import sys
import subprocess

# Clone your clean benchmark repo
repo_url = "https://github.com/farooqhaider431/ecg-ptbxl-benchmark.git"
repo_dir = "/kaggle/working/ecg-ptbxl-benchmark"

if not os.path.exists(repo_dir):
    print("Cloning repository...")
    subprocess.run(['git', 'clone', repo_url, repo_dir], check=True)
else:
    print("Pulling latest updates...")
    subprocess.run(['git', '-C', repo_dir, 'pull'], check=True)

# Add repo to python path
sys.path.insert(0, repo_dir)
print("Environment ready.")
```

### Cell 2: Verify GPU & Dataset Path
```python
import torch
import os

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")

# Check Kaggle input dataset path
kaggle_input = "/kaggle/input"
print("Attached Datasets in /kaggle/input:")
for d in os.listdir(kaggle_input):
    print(f" - {d}")
```

### Cell 3: Execute Model Training Loop
```python
from ecg_benchmark.train import train

# Custom config to ensure paths match Kaggle working directory
config = {
    'ptbxl_path':     '/kaggle/input/ptb-xl-dataset/ptb-xl-1.0.3', # Auto-detects if slightly different
    'checkpoint_dir': '/kaggle/working/checkpoints',
    'results_dir':    '/kaggle/working/results',
    'batch_size':     32,
    'learning_rate':  1e-4,
    'max_epochs':     75,
}

# Run multi-label BCE training loop
model, best_epochs = train(config)
```

---

## Phase 4 — Evaluation & Benchmark Table Generation

After training completes (or stopping early when convergence is reached), execute the evaluation cell in Kaggle to compute metrics on **Test Fold 10**:

### Cell 4: Execute Threshold Optimization & Final Evaluation
```python
from ecg_benchmark.evaluate import run_evaluation

# Run full evaluation protocol
results = run_evaluation({
    'ptbxl_path':     '/kaggle/input/ptb-xl-dataset/ptb-xl-1.0.3',
    'checkpoint_dir': '/kaggle/working/checkpoints',
    'results_dir':    '/kaggle/working/results',
})
```

---

## Phase 5 — Saving Outputs & Google Drive Backup (Optional)

Kaggle output files in `/kaggle/working/` persist for your session. To backup checkpoints and evaluation plots permanently:

### Option A: Direct Kaggle Download
You can download files (`best_model_swa.pth`, `test_fold10_results.csv`, `roc_curves.png`) directly from the Kaggle right-sidebar under **Output -> Output Files**.

### Option B: Back Up to Google Drive from Kaggle
If you wish to save checkpoints directly to Google Drive from Kaggle:

```python
from google.colab import drive # Only if running in Colab
# In Kaggle, use kagglehub or zip the results folder for download:
import shutil
shutil.make_archive('/kaggle/working/ptbxl_results', 'zip', '/kaggle/working/results')
shutil.make_archive('/kaggle/working/ptbxl_checkpoints', 'zip', '/kaggle/working/checkpoints')
print("Zipped results and checkpoints ready for download.")
```

---

## Summary Checklist

- [x] Local repo `ecg-ptbxl-benchmark` created & populated.
- [ ] Push code to GitHub: `git push -u origin main`
- [ ] Create Kaggle Notebook, attach `ptb-xl-dataset`, turn **GPU ON** and **Internet ON**.
- [ ] Run Cell 1 (git clone), Cell 2 (GPU check), Cell 3 (`train()`), Cell 4 (`run_evaluation()`).
- [ ] Inspect benchmark table comparing Macro AUC against Strodthoff et al. (0.930).
