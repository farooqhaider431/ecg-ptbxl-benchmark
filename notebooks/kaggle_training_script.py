"""
Kaggle Notebook Execution Script for PTB-XL 5-Superclass Benchmark
Paste into Kaggle Code Cell or run directly.
"""

import os
import sys
import subprocess

# 1. Clone repository
repo_url = "https://github.com/farooqhaider431/ecg-ptbxl-benchmark.git"
repo_dir = "/kaggle/working/ecg-ptbxl-benchmark"

if not os.path.exists(repo_dir):
    print("Cloning ecg-ptbxl-benchmark repository...")
    subprocess.run(['git', 'clone', repo_url, repo_dir], check=True)
else:
    print("Pulling latest repository updates...")
    subprocess.run(['git', '-C', repo_dir, 'pull'], check=True)

sys.path.insert(0, repo_dir)

# 2. Run Training
from ecg_benchmark.train import train
model, best_epochs = train()

# 3. Run Evaluation
from ecg_benchmark.evaluate import run_evaluation
results = run_evaluation()
