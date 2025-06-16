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
SEQ_LEN = 21  # siRNA length
MRNA_LEN = 80  # mRNA length
NUCLEOTIDES = 'AUCGX'  # Added X for padding/unknown nucleotides

# --- One-hot encoding ---
def one_hot_encode(seq, maxlen):
    seq = seq[:maxlen].ljust(maxlen, 'X')  # Pad with X instead of N
    mapping = {n: i for i, n in enumerate(NUCLEOTIDES)}
    arr = np.zeros((maxlen, len(NUCLEOTIDES)), dtype=np.float32)
    for j, n in enumerate(seq):
        if n in mapping:
            arr[j, mapping[n]] = 1.0
        # Handle any other unknown characters as X
        elif n not in 'AUCG':
            arr[j, 4] = 1.0  # X position (index 4)
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

# --- Enhanced Neural Network Components ---
class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels=5):  # Changed from 4 to 5
        super().__init__()
        # Different kernel sizes to capture different motif lengths
        self.conv3 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        
        self.bn = nn.BatchNorm1d(96)
        self.pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x):
        conv3_out = torch.relu(self.conv3(x))
        conv5_out = torch.relu(self.conv5(x))
        conv7_out = torch.relu(self.conv7(x))
        
        combined = torch.cat([conv3_out, conv5_out, conv7_out], dim=1)
        combined = self.bn(combined)
        return self.pool(combined).squeeze(-1)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=5, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)  # Residual connection

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=5, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key_value):
        attn_out, attn_weights = self.cross_attn(query, key_value, key_value)
        return self.norm(query + attn_out), attn_weights

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        return self.norm(x + self.block(x))

class FeatureEncoder(nn.Module):
    def __init__(self, bio_feats_dim):
        super().__init__()
        # Learn better representations of biological features
        self.bio_encoder = nn.Sequential(
            nn.Linear(bio_feats_dim, bio_feats_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(bio_feats_dim * 2, bio_feats_dim),
            nn.ReLU()
        )
        
    def forward(self, bio_feats):
        return self.bio_encoder(bio_feats)

# --- Enhanced Neural Network ---
class EnhancedANN(nn.Module):
    def __init__(self, bio_feats_dim):
        super().__init__()
        
        # Multi-scale CNNs for siRNA and mRNA
        self.cnn_siRNA = MultiScaleCNN(5)  # Changed from 4 to 5
        self.cnn_mRNA = MultiScaleCNN(5)   # Changed from 4 to 5
        
        # Self-attention for sequence understanding
        self.siRNA_attention = SelfAttention(5)  # Changed from 4 to 5
        self.mRNA_attention = SelfAttention(5)   # Changed from 4 to 5
        
        # Cross-attention for siRNA-mRNA interaction
        self.cross_attention = CrossAttention(5)  # Changed from 4 to 5
        
        # Enhanced BiLSTM with more layers and dropout
        self.bilstm_siRNA = nn.LSTM(
            input_size=5, hidden_size=64, num_layers=2,  # Changed from 4 to 5
            batch_first=True, bidirectional=True, dropout=0.2
        )
        self.bilstm_mRNA = nn.LSTM(
            input_size=5, hidden_size=64, num_layers=2,  # Changed from 4 to 5
            batch_first=True, bidirectional=True, dropout=0.2
        )
        
        # Feature encoder for biological features
        self.feature_encoder = FeatureEncoder(bio_feats_dim)
        
        # Enhanced MLP with residual connections
        # CNN: 96 + 96, BiLSTM: 128 + 128, Bio features: bio_feats_dim
        feature_dim = 96 + 96 + 128 + 128 + bio_feats_dim
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        siRNA_flat_size = SEQ_LEN * 5  # Changed from 4 to 5
        mRNA_flat_size = MRNA_LEN * 5  # Changed from 4 to 5
        
        # Extract sequences and features
        siRNA_flat = x[:, :siRNA_flat_size]
        mRNA_flat = x[:, siRNA_flat_size:siRNA_flat_size + mRNA_flat_size]
        extra_features = x[:, siRNA_flat_size + mRNA_flat_size:]
        
        # Reshape sequences
        siRNA_seq = siRNA_flat.view(batch_size, SEQ_LEN, 5)  # Changed from 4 to 5
        mRNA_seq = mRNA_flat.view(batch_size, MRNA_LEN, 5)   # Changed from 4 to 5
        
        # Apply self-attention
        siRNA_attended = self.siRNA_attention(siRNA_seq)
        mRNA_attended = self.mRNA_attention(mRNA_seq)
        
        # Apply cross-attention for siRNA-mRNA interaction
        siRNA_cross, _ = self.cross_attention(siRNA_attended, mRNA_attended)
        mRNA_cross, _ = self.cross_attention(mRNA_attended, siRNA_attended)
        
        # Combine original and cross-attended sequences
        siRNA_enhanced = siRNA_seq + siRNA_cross
        mRNA_enhanced = mRNA_seq + mRNA_cross
        
        # CNN outputs (multi-scale)
        siRNA_cnn = self.cnn_siRNA(siRNA_enhanced.permute(0, 2, 1))
        mRNA_cnn = self.cnn_mRNA(mRNA_enhanced.permute(0, 2, 1))
        
        # BiLSTM outputs
        _, (siRNA_hn, _) = self.bilstm_siRNA(siRNA_enhanced)
        _, (mRNA_hn, _) = self.bilstm_mRNA(mRNA_enhanced)
        
        # Concatenate final hidden states from both directions
        siRNA_bilstm = torch.cat([siRNA_hn[0], siRNA_hn[1]], dim=1)  # (batch, 128)
        mRNA_bilstm = torch.cat([mRNA_hn[0], mRNA_hn[1]], dim=1)    # (batch, 128)
        
        # Encode biological features
        encoded_bio_feats = self.feature_encoder(extra_features)
        
        # Concatenate all features
        features = torch.cat([
            siRNA_cnn, mRNA_cnn, siRNA_bilstm, mRNA_bilstm, encoded_bio_feats
        ], dim=1)
        
        out = self.mlp(features)
        return out.squeeze(-1)

# --- Enhanced Training and Evaluation ---
def train_model_enhanced(model, train_loader, val_loader, epochs=100, lr=1e-3, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Use AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Combined loss function
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
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
            
            # Combine regression and classification losses
            loss = 0.7 * mse_loss(pred, yb) + 0.3 * bce_loss(pred, (yb >= 0.5).float())
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            
        train_loss = np.mean(train_losses)
        
        # Validation
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
        
        # Update learning rate
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
    # Paths to training and validation CSVs
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    train_csv = os.path.join(base_dir, "data", "Hu.csv")
    val_csv = os.path.join(base_dir, "data", "Mix.csv")

    # Prepare scaler, datasets, and data loaders
    scaler = StandardScaler()
    train_dataset = SirnaDataset(csv_path=train_csv, scaler=scaler, fit_scaler=True)
    val_dataset = SirnaDataset(csv_path=val_csv, scaler=scaler, fit_scaler=False)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"Bio features dimension: {SirnaDataset.bio_feats_dim}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Initialize and train the enhanced model
    model = EnhancedANN(SirnaDataset.bio_feats_dim)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    trained_model, best_roc, best_f1, best_epoch = train_model_enhanced(
        model, train_loader, val_loader, epochs=50, lr=0.0001, batch_size=16
    )

    # Save model weights (only best)
    output_dir = os.path.join(base_dir, "model")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(trained_model.state_dict(), os.path.join(output_dir, "enhanced_ann_weights_v2.pth"))
    print(f"Best enhanced model saved from epoch {best_epoch} with ROC AUC: {best_roc:.4f}, F1: {best_f1:.4f}")
