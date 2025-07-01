"""
Enhanced attention utilities for RNA-FM sequence model.
Includes cross-attention mechanisms and visualization capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Dict, List
import os


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with proper scaling and dropout"""
    
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # Output projection
        output = self.w_o(attended_values)
        
        return output, attention_weights


class CrossAttention(nn.Module):
    """Cross-attention mechanism for siRNA-mRNA interaction modeling"""
    
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cross-attention
        attended, attention_weights = self.attention(query, key_value, key_value, mask)
        
        # Residual connection and layer norm
        query = self.norm1(query + attended)
        
        # Feed-forward network
        ffn_output = self.ffn(query)
        
        # Residual connection and layer norm
        output = self.norm2(query + ffn_output)
        
        return output, attention_weights


class SelfAttention(nn.Module):
    """Self-attention mechanism for sequence modeling"""
    
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        attended, attention_weights = self.attention(x, x, x, mask)
        
        # Residual connection and layer norm
        x = self.norm1(x + attended)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        
        # Residual connection and layer norm
        output = self.norm2(x + ffn_output)
        
        return output, attention_weights


class InteractionEncoder(nn.Module):
    """Enhanced encoder with cross-attention between sequences"""
    
    def __init__(self, d_model: int, n_head: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Self-attention layers
        self.self_attention_layers = nn.ModuleList([
            SelfAttention(d_model, n_head, dropout) for _ in range(n_layers)
        ])
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(d_model, n_head, dropout) for _ in range(n_layers)
        ])
        
        # Store attention weights for visualization
        self.attention_weights = {}
        
    def forward(self, sirna_seq: torch.Tensor, mrna_seq: torch.Tensor,
                sirna_mask: Optional[torch.Tensor] = None,
                mrna_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        
        self.attention_weights = {'self': [], 'cross': []}
        
        # Apply self-attention and cross-attention layers
        for i in range(self.n_layers):
            # Self-attention on siRNA and mRNA
            sirna_seq, sirna_self_attn = self.self_attention_layers[i](sirna_seq, sirna_mask)
            mrna_seq, mrna_self_attn = self.self_attention_layers[i](mrna_seq, mrna_mask)
            
            # Cross-attention between siRNA and mRNA
            sirna_cross, sirna_cross_attn = self.cross_attention_layers[i](sirna_seq, mrna_seq, mrna_mask)
            mrna_cross, mrna_cross_attn = self.cross_attention_layers[i](mrna_seq, sirna_seq, sirna_mask)
            
            # Store attention weights
            self.attention_weights['self'].append({
                'sirna': sirna_self_attn.detach().cpu(),
                'mrna': mrna_self_attn.detach().cpu()
            })
            self.attention_weights['cross'].append({
                'sirna_to_mrna': sirna_cross_attn.detach().cpu(),
                'mrna_to_sirna': mrna_cross_attn.detach().cpu()
            })
            
            # Update sequences
            sirna_seq = sirna_cross
            mrna_seq = mrna_cross
        
        return sirna_seq, mrna_seq, self.attention_weights


class AttentionVisualizer:
    """Utility class for visualizing attention weights"""
    
    def __init__(self, save_dir: str = 'attention_plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_attention_heatmap(self, attention_weights: torch.Tensor, 
                             query_labels: List[str], key_labels: List[str],
                             title: str, save_path: Optional[str] = None):
        """Plot attention weights as heatmap"""
        # Average across heads and batch
        if attention_weights.dim() == 4:  # [batch, head, seq, seq]
            attention_weights = attention_weights.mean(dim=(0, 1))
        elif attention_weights.dim() == 3:  # [head, seq, seq]
            attention_weights = attention_weights.mean(dim=0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention_weights.numpy(), 
                   xticklabels=key_labels, yticklabels=query_labels,
                   cmap='Blues', cbar=True, square=True)
        plt.title(title)
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_cross_attention_analysis(self, attention_weights: Dict, 
                                    sirna_seq: str, mrna_seq: str,
                                    layer_idx: int = -1, save_prefix: str = 'cross_attention'):
        """Plot cross-attention analysis between siRNA and mRNA"""
        
        # Get attention weights for the specified layer
        cross_attn = attention_weights['cross'][layer_idx]
        
        # siRNA to mRNA attention
        sirna_to_mrna = cross_attn['sirna_to_mrna']
        self.plot_attention_heatmap(
            sirna_to_mrna,
            query_labels=list(sirna_seq),
            key_labels=list(mrna_seq),
            title=f'siRNA to mRNA Cross-Attention (Layer {layer_idx})',
            save_path=os.path.join(self.save_dir, f'{save_prefix}_sirna_to_mrna_layer_{layer_idx}.png')
        )
        
        # mRNA to siRNA attention
        mrna_to_sirna = cross_attn['mrna_to_sirna']
        self.plot_attention_heatmap(
            mrna_to_sirna,
            query_labels=list(mrna_seq),
            key_labels=list(sirna_seq),
            title=f'mRNA to siRNA Cross-Attention (Layer {layer_idx})',
            save_path=os.path.join(self.save_dir, f'{save_prefix}_mrna_to_sirna_layer_{layer_idx}.png')
        )
    
    def analyze_attention_patterns(self, attention_weights: Dict, 
                                 sirna_seq: str, mrna_seq: str) -> Dict:
        """Analyze attention patterns and return insights"""
        analysis = {}
        
        # Analyze cross-attention patterns
        for layer_idx, cross_attn in enumerate(attention_weights['cross']):
            sirna_to_mrna = cross_attn['sirna_to_mrna'].mean(dim=(0, 1))  # [sirna_len, mrna_len]
            mrna_to_sirna = cross_attn['mrna_to_sirna'].mean(dim=(0, 1))  # [mrna_len, sirna_len]
            
            # Find positions with highest attention
            sirna_max_attn_pos = torch.argmax(sirna_to_mrna, dim=1)  # For each siRNA position
            mrna_max_attn_pos = torch.argmax(mrna_to_sirna, dim=1)   # For each mRNA position
            
            # Find critical positions (top 20% attention)
            sirna_attention_scores = sirna_to_mrna.max(dim=1)[0]
            mrna_attention_scores = mrna_to_sirna.max(dim=1)[0]
            
            sirna_critical = sirna_attention_scores > torch.quantile(sirna_attention_scores, 0.8)
            mrna_critical = mrna_attention_scores > torch.quantile(mrna_attention_scores, 0.8)
            
            analysis[f'layer_{layer_idx}'] = {
                'sirna_critical_positions': torch.where(sirna_critical)[0].tolist(),
                'mrna_critical_positions': torch.where(mrna_critical)[0].tolist(),
                'sirna_critical_nucleotides': [sirna_seq[i] for i in torch.where(sirna_critical)[0]],
                'mrna_critical_nucleotides': [mrna_seq[i] for i in torch.where(mrna_critical)[0]],
                'average_attention_strength': {
                    'sirna_to_mrna': sirna_to_mrna.mean().item(),
                    'mrna_to_sirna': mrna_to_sirna.mean().item()
                }
            }
        
        return analysis
    
    def plot_attention_evolution(self, attention_weights: Dict, 
                               save_path: Optional[str] = None):
        """Plot how attention patterns evolve across layers"""
        n_layers = len(attention_weights['cross'])
        
        fig, axes = plt.subplots(2, n_layers, figsize=(4*n_layers, 8))
        if n_layers == 1:
            axes = axes.reshape(2, 1)
        
        for layer_idx in range(n_layers):
            cross_attn = attention_weights['cross'][layer_idx]
            
            # siRNA to mRNA
            sirna_to_mrna = cross_attn['sirna_to_mrna'].mean(dim=(0, 1))
            im1 = axes[0, layer_idx].imshow(sirna_to_mrna.numpy(), cmap='Blues')
            axes[0, layer_idx].set_title(f'siRNA→mRNA Layer {layer_idx}')
            axes[0, layer_idx].set_xlabel('mRNA Position')
            axes[0, layer_idx].set_ylabel('siRNA Position')
            plt.colorbar(im1, ax=axes[0, layer_idx])
            
            # mRNA to siRNA
            mrna_to_sirna = cross_attn['mrna_to_sirna'].mean(dim=(0, 1))
            im2 = axes[1, layer_idx].imshow(mrna_to_sirna.numpy(), cmap='Blues')
            axes[1, layer_idx].set_title(f'mRNA→siRNA Layer {layer_idx}')
            axes[1, layer_idx].set_xlabel('siRNA Position')
            axes[1, layer_idx].set_ylabel('mRNA Position')
            plt.colorbar(im2, ax=axes[1, layer_idx])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


def create_position_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Create sinusoidal position encoding"""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        (-np.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe.unsqueeze(0)  # Add batch dimension


def create_padding_mask(seq: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """Create padding mask for attention"""
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Create causal mask for self-attention"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0