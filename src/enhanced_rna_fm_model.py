"""
Enhanced RNA-FM Sequence Model with improved cross-attention, stable training,
and interpretability features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import os

try:
    from .attention_utils import (
        InteractionEncoder, CrossAttention, SelfAttention, 
        AttentionVisualizer, create_position_encoding
    )
    from .config import EnhancedRNAConfig
except ImportError:
    from attention_utils import (
        InteractionEncoder, CrossAttention, SelfAttention, 
        AttentionVisualizer, create_position_encoding
    )
    from config import EnhancedRNAConfig


class SequenceEmbedding(nn.Module):
    """Enhanced sequence embedding with positional encoding"""
    
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Positional encoding
        self.register_buffer('position_encoding', 
                           create_position_encoding(max_seq_len, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, vocab_size]
        seq_len = x.size(1)
        
        # Linear transformation
        embedded = self.embedding(x) * np.sqrt(self.d_model)
        
        # Add positional encoding
        embedded = embedded + self.position_encoding[:, :seq_len, :]
        
        return self.dropout(embedded)


class MultiScaleCNN(nn.Module):
    """Multi-scale CNN for sequence feature extraction"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_sizes: List[int] = [3, 5, 7], dropout: float = 0.2):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels // len(kernel_sizes), kernel_size),
                nn.BatchNorm1d(out_channels // len(kernel_sizes)),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for kernel_size in kernel_sizes
        ])
        
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, in_channels, seq_len]
        conv_outputs = []
        
        for conv in self.convs:
            conv_out = conv(x)
            pooled = self.global_pool(conv_out)
            conv_outputs.append(pooled)
        
        # Concatenate multi-scale features
        output = torch.cat(conv_outputs, dim=1)  # [batch_size, out_channels, 1]
        return output.squeeze(-1)  # [batch_size, out_channels]


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, 
                 dropout: float = 0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_size]
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Concatenate forward and backward hidden states
        # hidden shape: [num_layers * 2, batch_size, hidden_size]
        forward_hidden = hidden[-2]  # Last layer forward
        backward_hidden = hidden[-1]  # Last layer backward
        
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        return self.dropout(combined_hidden)


class FeatureAttention(nn.Module):
    """Attention mechanism for biological features"""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # features shape: [batch_size, feature_dim]
        
        # Calculate attention weights
        attention_weights = F.softmax(self.attention(features), dim=1)
        
        # Apply attention
        attended_features = features * attention_weights
        
        return attended_features, attention_weights


class EnhancedRNAFM(nn.Module):
    """Enhanced RNA-FM sequence model with cross-attention and interpretability"""
    
    def __init__(self, config: EnhancedRNAConfig, bio_feats_dim: int):
        super().__init__()
        self.config = config
        self.bio_feats_dim = bio_feats_dim
        
        # Sequence embedding layers
        self.sirna_embedding = SequenceEmbedding(
            vocab_size=config.model.embed_dim,
            d_model=config.model.d_model,
            max_seq_len=config.model.seq_len,
            dropout=config.regularization.attention_dropout
        )
        
        self.mrna_embedding = SequenceEmbedding(
            vocab_size=config.model.embed_dim,
            d_model=config.model.d_model,
            max_seq_len=config.model.mrna_len,
            dropout=config.regularization.attention_dropout
        )
        
        # Multi-scale CNN layers
        self.sirna_cnn = MultiScaleCNN(
            in_channels=config.model.d_model,
            out_channels=sum(config.model.cnn_channels),
            kernel_sizes=config.model.cnn_kernel_sizes,
            dropout=config.regularization.cnn_dropout
        )
        
        self.mrna_cnn = MultiScaleCNN(
            in_channels=config.model.d_model,
            out_channels=sum(config.model.cnn_channels),
            kernel_sizes=config.model.cnn_kernel_sizes,
            dropout=config.regularization.cnn_dropout
        )
        
        # BiLSTM encoders
        self.sirna_lstm = BiLSTMEncoder(
            input_size=config.model.d_model,
            hidden_size=config.model.lstm_hidden_size,
            num_layers=config.model.lstm_num_layers,
            dropout=config.regularization.lstm_dropout
        )
        
        self.mrna_lstm = BiLSTMEncoder(
            input_size=config.model.d_model,
            hidden_size=config.model.lstm_hidden_size,
            num_layers=config.model.lstm_num_layers,
            dropout=config.regularization.lstm_dropout
        )
        
        # Cross-attention encoder
        self.interaction_encoder = InteractionEncoder(
            d_model=config.model.d_model,
            n_head=config.model.n_head,
            n_layers=config.model.n_layers,
            dropout=config.regularization.attention_dropout
        )
        
        # Biological feature processing
        self.feature_encoder = nn.Sequential(
            nn.Linear(bio_feats_dim, bio_feats_dim // 2),
            nn.BatchNorm1d(bio_feats_dim // 2) if config.regularization.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.regularization.mlp_dropout),
            nn.Linear(bio_feats_dim // 2, bio_feats_dim // 4),
            nn.BatchNorm1d(bio_feats_dim // 4) if config.regularization.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.regularization.mlp_dropout)
        )
        
        # Feature attention for interpretability
        self.feature_attention = FeatureAttention(bio_feats_dim // 4)
        
        # Calculate final feature dimension
        cnn_dim = sum(config.model.cnn_channels)
        lstm_dim = config.model.lstm_hidden_size * 2  # Bidirectional
        attention_dim = config.model.d_model
        feature_dim = bio_feats_dim // 4
        
        total_dim = cnn_dim * 2 + lstm_dim * 2 + attention_dim * 2 + feature_dim
        
        # Final MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.BatchNorm1d(total_dim // 2) if config.regularization.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.regularization.mlp_dropout),
            
            nn.Linear(total_dim // 2, total_dim // 4),
            nn.BatchNorm1d(total_dim // 4) if config.regularization.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.regularization.mlp_dropout),
            
            nn.Linear(total_dim // 4, total_dim // 8),
            nn.ReLU(),
            nn.Dropout(config.regularization.mlp_dropout // 2),
            
            nn.Linear(total_dim // 8, 1),
            nn.Sigmoid()
        )
        
        # Attention visualizer for interpretability
        self.attention_visualizer = AttentionVisualizer(config.logging.plot_dir)
        
        # Store attention weights and feature importance
        self.attention_weights = {}
        self.feature_importance = {}
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass through the enhanced RNA-FM model
        
        Args:
            x: Input tensor with concatenated sequences and features
            return_attention: Whether to return attention weights for visualization
        """
        batch_size = x.size(0)
        
        # Extract sequences and biological features
        sirna_flat_size = self.config.model.seq_len * self.config.model.embed_dim
        mrna_flat_size = self.config.model.mrna_len * self.config.model.embed_dim
        
        sirna_flat = x[:, :sirna_flat_size]
        mrna_flat = x[:, sirna_flat_size:sirna_flat_size + mrna_flat_size]
        bio_features = x[:, sirna_flat_size + mrna_flat_size:]
        
        # Reshape sequences from flat to 3D
        sirna_seq = sirna_flat.view(batch_size, self.config.model.seq_len, self.config.model.embed_dim)
        mrna_seq = mrna_flat.view(batch_size, self.config.model.mrna_len, self.config.model.embed_dim)
        
        # Sequence embeddings with positional encoding
        sirna_embedded = self.sirna_embedding(sirna_seq)
        mrna_embedded = self.mrna_embedding(mrna_seq)
        
        # CNN feature extraction
        sirna_cnn_features = self.sirna_cnn(sirna_embedded.transpose(1, 2))
        mrna_cnn_features = self.mrna_cnn(mrna_embedded.transpose(1, 2))
        
        # BiLSTM encoding
        sirna_lstm_features = self.sirna_lstm(sirna_embedded)
        mrna_lstm_features = self.mrna_lstm(mrna_embedded)
        
        # Cross-attention interaction modeling
        sirna_attended, mrna_attended, attention_weights = self.interaction_encoder(
            sirna_embedded, mrna_embedded
        )
        
        # Store attention weights for visualization
        if self.config.logging.track_attention_weights:
            self.attention_weights = attention_weights
        
        # Global pooling for attention features
        sirna_attention_features = sirna_attended.mean(dim=1)  # [batch_size, d_model]
        mrna_attention_features = mrna_attended.mean(dim=1)   # [batch_size, d_model]
        
        # Biological feature processing
        bio_features_encoded = self.feature_encoder(bio_features)
        bio_features_attended, feature_attn_weights = self.feature_attention(bio_features_encoded)
        
        # Store feature importance for interpretability
        if self.config.logging.track_attention_weights:
            self.feature_importance = {
                'attention_weights': feature_attn_weights.detach().cpu(),
                'feature_values': bio_features_encoded.detach().cpu()
            }
        
        # Combine all features
        combined_features = torch.cat([
            sirna_cnn_features,      # CNN features
            mrna_cnn_features,
            sirna_lstm_features,     # LSTM features
            mrna_lstm_features,
            sirna_attention_features, # Cross-attention features
            mrna_attention_features,
            bio_features_attended    # Biological features
        ], dim=1)
        
        # Final classification
        output = self.classifier(combined_features)
        
        if return_attention:
            return output, self.attention_weights, self.feature_importance
        
        return output.squeeze(-1)
    
    def get_attention_analysis(self, sirna_seq: str, mrna_seq: str) -> Dict:
        """Get attention analysis for interpretability"""
        if not self.attention_weights:
            raise ValueError("No attention weights available. Run forward pass first.")
        
        analysis = self.attention_visualizer.analyze_attention_patterns(
            self.attention_weights, sirna_seq, mrna_seq
        )
        
        return analysis
    
    def visualize_attention(self, sirna_seq: str, mrna_seq: str, save_prefix: str = 'attention'):
        """Visualize attention patterns"""
        if not self.attention_weights:
            raise ValueError("No attention weights available. Run forward pass first.")
        
        # Plot cross-attention analysis
        self.attention_visualizer.plot_cross_attention_analysis(
            self.attention_weights, sirna_seq, mrna_seq, save_prefix=save_prefix
        )
        
        # Plot attention evolution across layers
        evolution_path = os.path.join(self.attention_visualizer.save_dir, f'{save_prefix}_evolution.png')
        self.attention_visualizer.plot_attention_evolution(
            self.attention_weights, save_path=evolution_path
        )
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance scores for biological features"""
        if not self.feature_importance:
            raise ValueError("No feature importance available. Run forward pass first.")
        
        return self.feature_importance
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained model weights"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)
            
        print(f"Loaded pretrained model from {checkpoint_path}")
    
    def get_model_summary(self) -> Dict:
        """Get model architecture summary"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture_summary': {
                'embedding_dim': self.config.model.d_model,
                'attention_heads': self.config.model.n_head,
                'attention_layers': self.config.model.n_layers,
                'cnn_channels': self.config.model.cnn_channels,
                'lstm_hidden_size': self.config.model.lstm_hidden_size,
                'biological_features': self.bio_feats_dim
            }
        }


def create_enhanced_rna_fm_model(bio_feats_dim: int, 
                               config: Optional[EnhancedRNAConfig] = None) -> EnhancedRNAFM:
    """
    Factory function to create enhanced RNA-FM model
    
    Args:
        bio_feats_dim: Number of biological features
        config: Model configuration (uses default if None)
    
    Returns:
        EnhancedRNAFM model instance
    """
    if config is None:
        try:
            from .config import get_default_config
        except ImportError:
            from config import get_default_config
        config = get_default_config()
    
    model = EnhancedRNAFM(config, bio_feats_dim)
    
    # Print model summary
    summary = model.get_model_summary()
    print(f"Created Enhanced RNA-FM model:")
    print(f"  Total parameters: {summary['total_parameters']:,}")
    print(f"  Trainable parameters: {summary['trainable_parameters']:,}")
    print(f"  Model size: {summary['model_size_mb']:.2f} MB")
    
    return model