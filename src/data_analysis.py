import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from collections import Counter

def check_class_imbalance(csv_path):
    df = pd.read_csv(csv_path)
    # Binarize label at 0.5 threshold
    y_bin = (df['label'] >= 0.5).astype(int)
    counts = Counter(y_bin)
    total = len(y_bin)
    print(f"Class distribution in {csv_path} (threshold=0.5):")
    for k, v in counts.items():
        print(f"  Class {k}: {v} ({v/total:.2%})")
    return counts

def run_kfold_crossval(csv_path, n_splits=5):
    df = pd.read_csv(csv_path)
    y_bin = (df['label'] >= 0.5).astype(int)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"Fold {i+1}: train={len(train_idx)}, val={len(val_idx)}")
        y_train = y_bin.iloc[train_idx]
        y_val = y_bin.iloc[val_idx]
        print(f"  Train class balance: {np.bincount(y_train)}")
        print(f"  Val class balance: {np.bincount(y_val)}")
