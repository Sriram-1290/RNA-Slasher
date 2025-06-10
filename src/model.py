import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import os
from bio_features import (
    gc_content, at_content, melting_temp, length, base_frequencies,
    purine_content, pyrimidine_content, molecular_weight, dinucleotide_frequencies, shannon_entropy,
    longest_mononucleotide_run, au_gc_ratio, gc_skew, at_skew, unique_kmers, reverse_complement, is_palindromic, count_ambiguous
)

# --- Parameters ---
SEQ_LEN = 21  # siRNA length (adjust if needed)
MRNA_LEN = 80  # mRNA length (adjust if needed)
NUCLEOTIDES = 'AUCG'

# --- One-hot encoding ---
def one_hot_encode(seq, maxlen):
    seq = seq[:maxlen].ljust(maxlen, 'N')
    mapping = {n: i for i, n in enumerate(NUCLEOTIDES)}
    arr = np.zeros((maxlen, len(NUCLEOTIDES)), dtype=np.float32)
    for j, n in enumerate(seq):
        if n in mapping:
            arr[j, mapping[n]] = 1.0
    return arr.flatten()

# --- Custom Dataset ---
class SirnaDataset(Dataset):
    bio_feats_dim = None  # Class variable to store number of bio features
    def __init__(self, csv_path=None, df=None, scaler=None, fit_scaler=False):
        if df is not None:
            self.df = df.copy()
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError('Must provide csv_path or df')
        df = self.df
        # One-hot encoding
        self.X_siRNA = np.stack(df['siRNA'].apply(lambda x: one_hot_encode(x, SEQ_LEN)))
        self.X_mRNA = np.stack(df['mRNA'].apply(lambda x: one_hot_encode(x, MRNA_LEN)))
        # Biological features for siRNA
        siRNA_gc = df['siRNA'].apply(gc_content).values[:, None]
        siRNA_at = df['siRNA'].apply(at_content).values[:, None]
        siRNA_tm = df['siRNA'].apply(melting_temp).values[:, None]
        siRNA_len = df['siRNA'].apply(length).values[:, None]
        siRNA_basefreq = np.stack(df['siRNA'].apply(base_frequencies))
        siRNA_pur = df['siRNA'].apply(purine_content).values[:, None]
        siRNA_pyr = df['siRNA'].apply(pyrimidine_content).values[:, None]
        siRNA_mw = df['siRNA'].apply(molecular_weight).values[:, None]
        siRNA_dinuc = np.stack(df['siRNA'].apply(dinucleotide_frequencies))
        siRNA_entropy = df['siRNA'].apply(shannon_entropy).values[:, None]
        siRNA_run = df['siRNA'].apply(longest_mononucleotide_run).values[:, None]
        siRNA_au_gc = df['siRNA'].apply(au_gc_ratio).values[:, None]
        siRNA_gcskew = df['siRNA'].apply(gc_skew).values[:, None]
        siRNA_atskew = df['siRNA'].apply(at_skew).values[:, None]
        siRNA_kmers2 = df['siRNA'].apply(lambda x: unique_kmers(x, 2)).values[:, None]
        siRNA_kmers3 = df['siRNA'].apply(lambda x: unique_kmers(x, 3)).values[:, None]
        siRNA_ambig = df['siRNA'].apply(count_ambiguous).values[:, None]
        # Biological features for mRNA
        mRNA_gc = df['mRNA'].apply(gc_content).values[:, None]
        mRNA_at = df['mRNA'].apply(at_content).values[:, None]
        mRNA_tm = df['mRNA'].apply(melting_temp).values[:, None]
        mRNA_len = df['mRNA'].apply(length).values[:, None]
        mRNA_basefreq = np.stack(df['mRNA'].apply(base_frequencies))
        mRNA_pur = df['mRNA'].apply(purine_content).values[:, None]
        mRNA_pyr = df['mRNA'].apply(pyrimidine_content).values[:, None]
        mRNA_mw = df['mRNA'].apply(molecular_weight).values[:, None]
        mRNA_dinuc = np.stack(df['mRNA'].apply(dinucleotide_frequencies))
        mRNA_entropy = df['mRNA'].apply(shannon_entropy).values[:, None]
        mRNA_run = df['mRNA'].apply(longest_mononucleotide_run).values[:, None]
        mRNA_au_gc = df['mRNA'].apply(au_gc_ratio).values[:, None]
        mRNA_gcskew = df['mRNA'].apply(gc_skew).values[:, None]
        mRNA_atskew = df['mRNA'].apply(at_skew).values[:, None]
        mRNA_kmers2 = df['mRNA'].apply(lambda x: unique_kmers(x, 2)).values[:, None]
        mRNA_kmers3 = df['mRNA'].apply(lambda x: unique_kmers(x, 3)).values[:, None]
        mRNA_ambig = df['mRNA'].apply(count_ambiguous).values[:, None]
        # Concatenate all features
        bio_feats = np.concatenate([
            siRNA_gc, siRNA_at, siRNA_tm, siRNA_len, siRNA_basefreq,
            siRNA_pur, siRNA_pyr, siRNA_mw, siRNA_dinuc, siRNA_entropy, siRNA_run, siRNA_au_gc, siRNA_gcskew, siRNA_atskew, siRNA_kmers2, siRNA_kmers3, siRNA_ambig,
            mRNA_gc, mRNA_at, mRNA_tm, mRNA_len, mRNA_basefreq,
            mRNA_pur, mRNA_pyr, mRNA_mw, mRNA_dinuc, mRNA_entropy, mRNA_run, mRNA_au_gc, mRNA_gcskew, mRNA_atskew, mRNA_kmers2, mRNA_kmers3, mRNA_ambig
        ], axis=1)
        SirnaDataset.bio_feats_dim = bio_feats.shape[1]
        # Normalize biological features
        if scaler is not None:
            if fit_scaler:
                scaler.fit(bio_feats)
            bio_feats = scaler.transform(bio_feats)
        self.X = np.concatenate([
            self.X_siRNA, self.X_mRNA, bio_feats
        ], axis=1)
        self.y = df['label'].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# --- Neural Network ---
class ANN(nn.Module):
    def __init__(self, bio_feats_dim):
        super().__init__()
        # CNN block for siRNA and mRNA only
        self.cnn_siRNA = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.cnn_mRNA = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        # MLP block (CNN outputs + normalized bio features)
        self.mlp = nn.Sequential(
            nn.Linear(64+64+bio_feats_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        siRNA_flat_size = SEQ_LEN * 4
        mRNA_flat_size = MRNA_LEN * 4
        siRNA_flat = x[:, :siRNA_flat_size]
        mRNA_flat = x[:, siRNA_flat_size:siRNA_flat_size + mRNA_flat_size]
        extra_features = x[:, siRNA_flat_size + mRNA_flat_size:]
        siRNA_seq = siRNA_flat.view(batch_size, SEQ_LEN, 4)
        mRNA_seq = mRNA_flat.view(batch_size, MRNA_LEN, 4)
        siRNA_cnn = self.cnn_siRNA(siRNA_seq.permute(0,2,1)).view(batch_size, -1)
        mRNA_cnn = self.cnn_mRNA(mRNA_seq.permute(0,2,1)).view(batch_size, -1)
        features = torch.cat([siRNA_cnn, mRNA_cnn, extra_features], dim=1)
        out = self.mlp(features)
        return out.squeeze(-1)

# --- Training and Evaluation ---
def train_model(model, train_loader, val_loader, epochs=100, lr=5e-4, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for xb, yb in DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        val_losses = []
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(val_loader.dataset, batch_size=batch_size):
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())
                all_preds.append(pred.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        val_loss = np.mean(val_losses)
        # Compute ROC AUC and F1
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        y_true_bin = (y_true >= 0.5).astype(int)
        y_pred_bin = (y_pred >= 0.5).astype(int)
        try:
            val_roc = roc_auc_score(y_true_bin, y_pred)
        except ValueError:
            val_roc = float('nan')
        val_f1 = f1_score(y_true_bin, y_pred_bin)
        print(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.4f} - Val ROC AUC: {val_roc:.4f} - Val F1: {val_f1:.4f}")
    return model

def main():
    # K-Fold Cross-Validation on Hu.csv
    df = pd.read_csv('data/Hu.csv')
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()
    y_bin = (df['label'] >= 0.5).astype(int)
    print('Class balance in Hu.csv:', np.bincount(y_bin))
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f'Fold {fold+1}/{n_splits}')
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        # Fit scaler on train, transform both
        train_dataset = SirnaDataset(df=train_df, scaler=scaler, fit_scaler=True)
        val_dataset = SirnaDataset(df=val_df, scaler=scaler, fit_scaler=False)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        bio_feats_dim = SirnaDataset.bio_feats_dim
        model = ANN(bio_feats_dim)
        model = train_model(model, train_loader, val_loader, epochs=30, lr=5e-4, batch_size=16)
        # Evaluate ROC AUC on this fold
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(model(xb).numpy())
                targets.append(yb.numpy())
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        y_true_bin = (targets >= 0.5).astype(int)
        try:
            val_roc = roc_auc_score(y_true_bin, preds)
        except ValueError:
            val_roc = float('nan')
        val_f1 = f1_score(y_true_bin, (preds >= 0.5).astype(int))
        print(f'Fold {fold+1} ROC AUC: {val_roc:.4f} | F1: {val_f1:.4f}')
        fold_scores.append((val_roc, val_f1))
        # Save model weights for inference after each fold
        torch.save(model.state_dict(), f'model/ann_weights_fold{fold+1}.pth')
        # Optionally, save the last fold as the default
        if fold == n_splits - 1:
            torch.save(model.state_dict(), 'model/ann_weights.pth')
    # Print average scores
    roc_aucs = [score[0] for score in fold_scores]
    f1s = [score[1] for score in fold_scores]
    log_lines = []
    for i, (roc, f1) in enumerate(fold_scores):
        log_lines.append(f'Fold {i+1} ROC AUC: {roc:.4f} | F1: {f1:.4f}')
    log_lines.append(f'Average K-Fold ROC AUC: {np.nanmean(roc_aucs):.4f} | Average F1: {np.nanmean(f1s):.4f}')
    log_text = '\n'.join(log_lines)
    print(log_text)
    with open('model/training_log.txt', 'w') as f:
        f.write(log_text + '\n')

if __name__ == '__main__':
    main()
