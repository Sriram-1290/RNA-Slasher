import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import os
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    return arr

# --- Sequence Augmentation ---
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

def augment_sequence(seq):
    aug_types = ['none', 'mutate', 'mask']
    aug_type = random.choice(aug_types)
    if aug_type == 'mutate':
        return random_base_mutation(seq, p=0.1)
    elif aug_type == 'mask':
        return random_mask(seq, p=0.1)
    else:
        return seq

# --- Custom Dataset ---
class SirnaDataset(Dataset):
    bio_feats_dim = None  # Class variable to store number of bio features
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

        # Extract biological features
        def extract_features(seq):
            return [
                gc_content(seq), at_content(seq), melting_temp(seq), length(seq),
                *base_frequencies(seq), purine_content(seq), pyrimidine_content(seq),
                molecular_weight(seq), *dinucleotide_frequencies(seq), shannon_entropy(seq),
                longest_mononucleotide_run(seq), au_gc_ratio(seq), gc_skew(seq), at_skew(seq),
                unique_kmers(seq, 2), unique_kmers(seq, 3), count_ambiguous(seq)
            ]

        siRNA_feats = np.stack(df['siRNA'].apply(extract_features))
        mRNA_feats = np.stack(df['mRNA'].apply(extract_features))
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

# --- Enhanced Neural Network Components ---

class CrossAttentionBlock(nn.Module):
    """Enhanced cross-attention with residual connections and layer normalization"""
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = min(num_heads, feature_dim)  # Ensure num_heads <= feature_dim
        
        # Multi-head attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=self.num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
    def forward(self, query, key_value):
        # Cross-attention with residual connection
        attn_out, attn_weights = self.cross_attn(query, key_value, key_value)
        query = self.norm1(query + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(query)
        query = self.norm2(query + self.dropout(ffn_out))
        
        return query, attn_weights

class SelfAttentionBlock(nn.Module):
    """Self-attention with residual connections and layer normalization"""
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = min(num_heads, feature_dim)  # Ensure num_heads <= feature_dim
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x

class PositionalEncoding(nn.Module):
    """Learnable positional encoding for sequences"""
    def __init__(self, max_len, d_model):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, d_model))
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.encoding[:seq_len, :].unsqueeze(0)

class EnhancedFeatureEncoder(nn.Module):
    """Enhanced biological feature encoder with attention mechanism"""
    def __init__(self, bio_feats_dim):
        super().__init__()
        hidden_dim = max(bio_feats_dim // 2, 32)
        
        self.encoder = nn.Sequential(
            nn.Linear(bio_feats_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Feature attention mechanism
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        attention_weights = self.feature_attention(encoded)
        return encoded * attention_weights

# --- Main Model Class: Oligo ---
class Oligo(nn.Module):
    """Enhanced RNA-FM Sequence Model with Cross-Attention and Stable Training"""
    
    def __init__(self, bio_feats_dim):
        super().__init__()
        
        # Model dimensions
        self.embed_dim = 5  # One-hot encoding dimension
        self.hidden_dim = 64
        
        # Positional encoding
        self.siRNA_pos_enc = PositionalEncoding(SEQ_LEN, self.embed_dim)
        self.mRNA_pos_enc = PositionalEncoding(MRNA_LEN, self.embed_dim)
        
        # Self-attention layers
        self.siRNA_self_attn = SelfAttentionBlock(self.embed_dim, num_heads=1, dropout=0.1)
        self.mRNA_self_attn = SelfAttentionBlock(self.embed_dim, num_heads=1, dropout=0.1)
        
        # Cross-attention layers (bidirectional)
        self.siRNA_to_mRNA_attn = CrossAttentionBlock(self.embed_dim, num_heads=1, dropout=0.1)
        self.mRNA_to_siRNA_attn = CrossAttentionBlock(self.embed_dim, num_heads=1, dropout=0.1)
        
        # CNN layers for local pattern extraction
        self.siRNA_cnn = nn.Sequential(
            nn.Conv1d(self.embed_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.mRNA_cnn = nn.Sequential(
            nn.Conv1d(self.embed_dim, self.hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # BiLSTM layers
        self.siRNA_bilstm = nn.LSTM(
            input_size=self.embed_dim, 
            hidden_size=self.hidden_dim//2, 
            num_layers=2,
            batch_first=True, 
            bidirectional=True, 
            dropout=0.1
        )
        
        self.mRNA_bilstm = nn.LSTM(
            input_size=self.embed_dim, 
            hidden_size=self.hidden_dim//2, 
            num_layers=2,
            batch_first=True, 
            bidirectional=True, 
            dropout=0.1
        )
        
        # Enhanced feature encoder
        self.feature_encoder = EnhancedFeatureEncoder(bio_feats_dim)
        feature_out_dim = max(bio_feats_dim // 2, 32)
        
        # Final MLP with residual connections
        total_features = self.hidden_dim * 4 + feature_out_dim  # CNN + BiLSTM features + bio features
        
        self.mlp = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Better weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        batch_size = x.size(0)
        siRNA_flat_size = SEQ_LEN * 5
        mRNA_flat_size = MRNA_LEN * 5
        
        # Extract sequences and features
        siRNA_flat = x[:, :siRNA_flat_size]
        mRNA_flat = x[:, siRNA_flat_size:siRNA_flat_size + mRNA_flat_size]
        bio_features = x[:, siRNA_flat_size + mRNA_flat_size:]
        
        # Reshape sequences
        siRNA_seq = siRNA_flat.view(batch_size, SEQ_LEN, 5)
        mRNA_seq = mRNA_flat.view(batch_size, MRNA_LEN, 5)
        
        # Add positional encoding
        siRNA_seq = self.siRNA_pos_enc(siRNA_seq)
        mRNA_seq = self.mRNA_pos_enc(mRNA_seq)
        
        # Self-attention
        siRNA_attended = self.siRNA_self_attn(siRNA_seq)
        mRNA_attended = self.mRNA_self_attn(mRNA_seq)
        
        # Cross-attention (bidirectional)
        siRNA_cross, _ = self.siRNA_to_mRNA_attn(siRNA_attended, mRNA_attended)
        mRNA_cross, _ = self.mRNA_to_siRNA_attn(mRNA_attended, siRNA_attended)
        
        # Combine with residual connections
        siRNA_enhanced = siRNA_seq + siRNA_cross
        mRNA_enhanced = mRNA_seq + mRNA_cross
        
        # CNN features
        siRNA_cnn = self.siRNA_cnn(siRNA_enhanced.permute(0, 2, 1)).squeeze(-1)
        mRNA_cnn = self.mRNA_cnn(mRNA_enhanced.permute(0, 2, 1)).squeeze(-1)
        
        # BiLSTM features
        _, (siRNA_hn, _) = self.siRNA_bilstm(siRNA_enhanced)
        _, (mRNA_hn, _) = self.mRNA_bilstm(mRNA_enhanced)
        
        # Concatenate BiLSTM hidden states from both directions
        siRNA_lstm = torch.cat([siRNA_hn[0], siRNA_hn[1]], dim=1)
        mRNA_lstm = torch.cat([mRNA_hn[0], mRNA_hn[1]], dim=1)
        
        # Encode biological features
        bio_encoded = self.feature_encoder(bio_features)
        
        # Combine all features
        features = torch.cat([siRNA_cnn, mRNA_cnn, siRNA_lstm, mRNA_lstm, bio_encoded], dim=1)
        
        # Final prediction
        output = self.mlp(features)
        return output.squeeze(-1)

# --- Label Smoothing Loss ---
class LabelSmoothingBCELoss(nn.Module):
    """Binary Cross Entropy with Label Smoothing"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, predictions, targets):
        # Apply label smoothing
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy(predictions, targets_smooth)

# --- Early Stopping ---
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.best_weights = model.state_dict().copy()
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        return False

# --- Enhanced Training Function ---
def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, batch_size=16, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # CosineAnnealingWarmRestarts scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr/100
    )
    
    # Loss functions
    mse_loss = nn.MSELoss()
    label_smooth_bce = LabelSmoothingBCELoss(smoothing=0.1)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    
    best_roc = -float('inf')
    best_f1 = -float('inf')
    best_epoch = -1
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            pred = model(xb)
            
            # Combined loss: MSE + Label Smoothing BCE
            loss = 0.6 * mse_loss(pred, yb) + 0.4 * label_smooth_bce(pred, (yb >= 0.5).float())
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        
        # Validation phase
        model.eval()
        val_losses = []
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(mse_loss(pred, yb).item())
                all_preds.append(pred.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        
        val_loss = np.mean(val_losses)
        
        # Compute metrics
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
        scheduler.step()
        
        # Track best model
        is_best = False
        if not np.isnan(val_roc) and val_roc > best_roc and val_f1 > 0.8:
            best_roc = val_roc
            best_f1 = val_f1
            best_epoch = epoch + 1
            is_best = True
        
        # Logging
        log_str = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val ROC AUC: {val_roc:.4f} - Val F1: {val_f1:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}"
        if is_best:
            log_str += " ****"
        print(log_str)
        
        # Early stopping check
        if early_stopping(val_roc if not np.isnan(val_roc) else 0, model):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return model, best_roc, best_f1, best_epoch

if __name__ == "__main__":
    # Paths to training and validation CSVs
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    train_csv = os.path.join(base_dir, "data", "Hu.csv")
    val_csv = os.path.join(base_dir, "data", "Mix.csv")

    # Prepare scaler, datasets, and data loaders
    scaler = StandardScaler()
    train_dataset = SirnaDataset(csv_path=train_csv, scaler=scaler, fit_scaler=True, augment=True)
    val_dataset = SirnaDataset(csv_path=val_csv, scaler=scaler, fit_scaler=False, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"Bio features dimension: {SirnaDataset.bio_feats_dim}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Initialize the Oligo model
    model = Oligo(SirnaDataset.bio_feats_dim)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("Enhanced RNA-FM Sequence Model with Cross-Attention and Stable Training")
    print("Features: Bidirectional Cross-Attention, CosineAnnealingWarmRestarts, Label Smoothing, Early Stopping")
    
    # Train the model
    trained_model, best_roc, best_f1, best_epoch = train_model(
        model, train_loader, val_loader, epochs=100, lr=0.001, batch_size=16
    )

    # Save model weights
    output_dir = os.path.join(base_dir, "model")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(trained_model.state_dict(), os.path.join(output_dir, "rna_fm_seq_model_weights.pth"))
    print(f"Enhanced RNA-FM model saved from epoch {best_epoch} with ROC AUC: {best_roc:.4f}, F1: {best_f1:.4f}")