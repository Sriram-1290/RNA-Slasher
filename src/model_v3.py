import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
import random
import os
from bio_features import (
    gc_content, at_content, melting_temp, length, base_frequencies,
    purine_content, pyrimidine_content, molecular_weight, dinucleotide_frequencies, shannon_entropy,
    longest_mononucleotide_run, au_gc_ratio, gc_skew, at_skew, unique_kmers, count_ambiguous,
    delta_g_ends, delta_g1, delta_h1, u1, g1, delta_h_all, u_all, uu1, g_all, gg1, gc1, gg_all, delta_g2, ua_all, u2, c1, cc_all, delta_g18, cc1, gc_all, cg1, delta_g13, uu_all, a19
)

SEQ_LEN = 21
MRNA_LEN = 80
NUCLEOTIDES = 'AUCGX'

def one_hot_encode(seq, maxlen):
    seq = seq[:maxlen].ljust(maxlen, 'X')
    mapping = {n: i for i, n in enumerate(NUCLEOTIDES)}
    arr = np.zeros((maxlen, len(NUCLEOTIDES)), dtype=np.float32)
    for j, n in enumerate(seq):
        arr[j, mapping.get(n, 4)] = 1.0
    return arr

def random_base_mutation(seq, p=0.05):
    bases = list("AUCG")
    out = []
    for c in seq:
        if c in bases and random.random() < p:
            new_bases = [b for b in bases if b != c]
            out.append(random.choice(new_bases))
        else:
            out.append(c)
    return ''.join(out)

def random_mask(seq, p=0.05):
    out = []
    for c in seq:
        if c in "AUCG" and random.random() < p:
            out.append('X')
        else:
            out.append(c)
    return ''.join(out)

def random_shuffle(seq, p=0.05):
    """Shuffle a random k-mer inside the sequence with probability p."""
    seq = list(seq)
    if random.random() < p:
        k = random.randint(2, 5)
        start = random.randint(0, max(0, len(seq) - k))
        kmer = seq[start:start+k]
        random.shuffle(kmer)
        seq[start:start+k] = kmer
    return ''.join(seq)

def random_reverse(seq, p=0.05):
    """Reverse a random subsequence with probability p."""
    seq = list(seq)
    if random.random() < p:
        start = random.randint(0, len(seq) - 2)
        end = random.randint(start+1, len(seq))
        seq[start:end] = seq[start:end][::-1]
    return ''.join(seq)

def augment_sequence(seq):
    # More diverse/stronger augmentations
    aug_types = ['none', 'mutate', 'mask', 'shuffle', 'reverse']
    aug_type = random.choice(aug_types)
    if aug_type == 'mutate':
        return random_base_mutation(seq, p=0.15)
    elif aug_type == 'mask':
        return random_mask(seq, p=0.15)
    elif aug_type == 'shuffle':
        return random_shuffle(seq, p=0.2)
    elif aug_type == 'reverse':
        return random_reverse(seq, p=0.2)
    else:
        return seq

class SirnaDataset(Dataset):
    bio_feats_dim = None
    def __init__(self, csv_path=None, df=None, scaler=None, fit_scaler=False, augment=False):
        if df is not None:
            self.df = df.copy()
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError('Must provide csv_path or df')
        df = self.df

        self.augment = augment

        self.siRNAs = df['siRNA'].tolist()
        self.mRNAs = df['mRNA'].tolist()
        self.labels = df['label'].values.astype(np.float32)

        def feats(label): return [
            gc_content(label), at_content(label), melting_temp(label), length(label),
            *base_frequencies(label), purine_content(label), pyrimidine_content(label),
            molecular_weight(label), *dinucleotide_frequencies(label), shannon_entropy(label),
            longest_mononucleotide_run(label), au_gc_ratio(label), gc_skew(label), at_skew(label),
            unique_kmers(label, 2), unique_kmers(label, 3), count_ambiguous(label),
            delta_g_ends(label), delta_g1(label), delta_h1(label), u1(label), g1(label),
            delta_h_all(label), u_all(label), uu1(label), g_all(label), gg1(label), gc1(label),
            gg_all(label), delta_g2(label), ua_all(label), u2(label), c1(label), cc_all(label),
            delta_g18(label), cc1(label), gc_all(label), cg1(label), delta_g13(label), uu_all(label), a19(label)
        ]

        siRNA_feats = np.stack(df['siRNA'].apply(feats))
        mRNA_feats = np.stack(df['mRNA'].apply(feats))
        bio_feats = np.concatenate([siRNA_feats, mRNA_feats], axis=1)
        SirnaDataset.bio_feats_dim = bio_feats.shape[1]

        if scaler is not None:
            if fit_scaler:
                scaler.fit(bio_feats)
            bio_feats = scaler.transform(bio_feats)
        self.bio_feats = bio_feats

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        siRNA = self.siRNAs[idx]
        mRNA = self.mRNAs[idx]
        if self.augment:
            siRNA = augment_sequence(siRNA)
            mRNA = augment_sequence(mRNA)
        x_siRNA = one_hot_encode(siRNA, SEQ_LEN).reshape(-1)
        x_mRNA = one_hot_encode(mRNA, MRNA_LEN).reshape(-1)
        feats = self.bio_feats[idx]
        x = np.concatenate([x_siRNA, x_mRNA, feats])
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    def forward(self, x):
        w = x.mean(dim=2)
        w = torch.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        w = w.unsqueeze(2)
        return x * w

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=(kernel_size-1)//2 * dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=(kernel_size-1)//2 * dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = self.downsample(x)
        return torch.relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_channels, num_channels, kernel_size=3, dropout=0.25):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers += [TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class FeatureEncoder(nn.Module):
    def __init__(self, bio_feats_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(bio_feats_dim, bio_feats_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(bio_feats_dim // 2),
            nn.Dropout(0.5),
            nn.Linear(bio_feats_dim // 2, bio_feats_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(bio_feats_dim // 4),
            nn.Dropout(0.5)
        )
    def forward(self, x):
        return self.layers(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
    def forward(self, siRNA_seq, mRNA_seq):
        siRNA2mRNA, _ = self.attn(query=siRNA_seq, key=mRNA_seq, value=mRNA_seq)
        mRNA2siRNA, _ = self.attn(query=mRNA_seq, key=siRNA_seq, value=siRNA_seq)
        return siRNA2mRNA, mRNA2siRNA

class CNN_SE_TCN_ANN(nn.Module):
    def __init__(self, bio_feats_dim):
        super().__init__()
        widen = 64
        self.siRNA_cnn = nn.Sequential(
            nn.Conv1d(5, widen, 3, padding=1),
            nn.BatchNorm1d(widen),
            nn.ReLU(),
            SEBlock(widen, reduction=8),
            nn.Conv1d(widen, widen, 3, padding=1),
            nn.BatchNorm1d(widen),
            nn.ReLU(),
            SEBlock(widen, reduction=8),
            nn.Dropout(0.5)
        )
        self.siRNA_tcn = TCN(widen, [widen, widen], kernel_size=3, dropout=0.4)
        self.siRNA_pool = nn.AdaptiveAvgPool1d(1)
        self.mRNA_cnn = nn.Sequential(
            nn.Conv1d(5, widen, 5, padding=2),
            nn.BatchNorm1d(widen),
            nn.ReLU(),
            SEBlock(widen, reduction=8),
            nn.Conv1d(widen, widen, 5, padding=2),
            nn.BatchNorm1d(widen),
            nn.ReLU(),
            SEBlock(widen, reduction=8),
            nn.Dropout(0.5)
        )
        self.mRNA_tcn = TCN(widen, [widen, widen], kernel_size=5, dropout=0.4)
        self.mRNA_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_encoder = FeatureEncoder(bio_feats_dim)
        self.cross_attention = CrossAttentionBlock(feature_dim=widen, num_heads=4)

        feature_dim = widen + widen + (bio_feats_dim // 4) + widen + widen
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        batch_size = x.size(0)
        siRNA_flat_size = SEQ_LEN * 5
        mRNA_flat_size = MRNA_LEN * 5
        siRNA_flat = x[:, :siRNA_flat_size]
        mRNA_flat = x[:, siRNA_flat_size:siRNA_flat_size + mRNA_flat_size]
        extra_features = x[:, siRNA_flat_size + mRNA_flat_size:]

        siRNA_seq = siRNA_flat.view(batch_size, SEQ_LEN, 5).permute(0, 2, 1)
        mRNA_seq = mRNA_flat.view(batch_size, MRNA_LEN, 5).permute(0, 2, 1)

        siRNA_cnn_out = self.siRNA_cnn(siRNA_seq)
        siRNA_tcn_out = self.siRNA_tcn(siRNA_cnn_out)
        siRNA_feat = self.siRNA_pool(siRNA_tcn_out).squeeze(-1)

        mRNA_cnn_out = self.mRNA_cnn(mRNA_seq)
        mRNA_tcn_out = self.mRNA_tcn(mRNA_cnn_out)
        mRNA_feat = self.mRNA_pool(mRNA_tcn_out).squeeze(-1)

        siRNA_tcn_out_t = siRNA_tcn_out.permute(0, 2, 1)
        mRNA_tcn_out_t = mRNA_tcn_out.permute(0, 2, 1)
        siRNA_att, mRNA_att = self.cross_attention(siRNA_tcn_out_t, mRNA_tcn_out_t)
        siRNA_att_feat = siRNA_att.mean(dim=1)
        mRNA_att_feat = mRNA_att.mean(dim=1)

        feat_out = self.feature_encoder(extra_features)

        features = torch.cat([siRNA_feat, mRNA_feat, feat_out, siRNA_att_feat, mRNA_att_feat], dim=1)
        out = self.mlp(features)
        return out.squeeze(-1)

def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, batch_size=16, device_str="cuda:2"):
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Add L2 regularization via weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    unique_labels, counts = np.unique((np.array(all_labels) >= 0.5).astype(int), return_counts=True)
    total_samples = len(all_labels)
    class_weights = total_samples / (len(unique_labels) * counts)
    pos_weight = torch.tensor(class_weights[1] / class_weights[0] if len(class_weights) > 1 else 1.0, device=device)
    weighted_bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse_loss = nn.MSELoss()
    best_roc = -float('inf')
    best_f1 = -float('inf')
    best_state = None
    best_epoch = -1
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            pred_logits = torch.logit(pred + 1e-8)
            loss = 0.7 * mse_loss(pred, yb) + 0.3 * weighted_bce_loss(pred_logits, (yb >= 0.5).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        model.eval()
        val_losses = []
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(val_loader.dataset, batch_size=batch_size):
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(mse_loss(pred, yb).item())
                all_preds.append(pred.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        val_loss = np.mean(val_losses)
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        y_true_bin = (y_true >= 0.5).astype(int)
        y_pred_bin = (y_pred >= 0.5).astype(int)
        try:
            val_roc = roc_auc_score(y_true_bin, y_pred)
        except ValueError:
            val_roc = float('nan')
        val_f1 = f1_score(y_true_bin, y_pred_bin)
        scheduler.step(val_roc if not np.isnan(val_roc) else 0)
        is_best = False
        if (val_roc > best_roc) and (val_f1 > 0.8):
            best_roc = val_roc
            best_f1 = val_f1
            best_state = model.state_dict()
            best_epoch = epoch + 1
            is_best = True
        log_str = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val ROC AUC: {val_roc:.4f} - Val F1: {val_f1:.4f}"
        if is_best:
            log_str += " ****"
        print(log_str)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_roc, best_f1, best_epoch

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    train_csv = os.path.join(base_dir, "data", "Hu.csv")
    val_csv = os.path.join(base_dir, "data", "Mix.csv")
    scaler = StandardScaler()
    train_dataset = SirnaDataset(csv_path=train_csv, scaler=scaler, fit_scaler=True, augment=True)
    val_dataset = SirnaDataset(csv_path=val_csv, scaler=scaler, fit_scaler=False, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    print(f"Bio features dimension: {SirnaDataset.bio_feats_dim}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    model = CNN_SE_TCN_ANN(SirnaDataset.bio_feats_dim)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("CNN + SE + TCN architecture + Cross-Attention block with data augmentation, L2 reg, high dropout, and LR scheduler")
    trained_model, best_roc, best_f1, best_epoch = train_model(
        model, train_loader, val_loader, epochs=100, lr=0.0001, batch_size=16, device_str="cuda:2"
    )
    output_dir = os.path.join(base_dir, "model")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(trained_model.state_dict(), os.path.join(output_dir, "cnn_se_tcn_ann_cross_attention_l2_dropout_aug_scheduler.pth"))
    print(f"Model saved from epoch {best_epoch} with ROC AUC: {best_roc:.4f}, F1: {best_f1:.4f}")
    print("Training completed!")