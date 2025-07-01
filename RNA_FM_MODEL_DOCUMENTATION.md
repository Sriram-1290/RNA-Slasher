# RNA-FM Sequence Model Implementation

## Overview

This document describes the implementation of the enhanced RNA-FM sequence model for siRNA efficacy prediction, featuring improved cross-attention mechanisms and stable training procedures as specified in the requirements.

## Key Features Implemented

### 1. Enhanced Cross-Attention Mechanism

#### Bidirectional Cross-Attention
- **siRNA-to-mRNA Attention**: Models how siRNA features attend to mRNA features
- **mRNA-to-siRNA Attention**: Models how mRNA features attend to siRNA features
- **Residual Connections**: Added skip connections to improve gradient flow
- **Layer Normalization**: Stabilizes training and improves convergence

#### Self-Attention Integration
- Self-attention layers for both siRNA and mRNA sequences
- Captures intra-sequence dependencies before cross-sequence interactions
- Transformer-style architecture with feed-forward networks

#### Positional Encoding
- Learnable positional encodings for sequence order awareness
- Separate encodings for siRNA (21 bases) and mRNA (80 bases) sequences
- Maintains nucleotide position information throughout the model

### 2. Stable Training Improvements

#### CosineAnnealingWarmRestarts Scheduler
- Replaces aggressive ReduceLROnPlateau scheduler
- Smooth learning rate decay with periodic restarts
- Better convergence properties and reduced training plateaus
- Parameters: T_0=10, T_mult=2, eta_min=lr/100

#### Label Smoothing BCE Loss
- Prevents overconfident predictions
- Smoothing factor of 0.1 applied to binary targets
- Combined with MSE loss (60% MSE + 40% Label Smoothing BCE)
- Reduces overfitting and improves generalization

#### Early Stopping Mechanism
- Patience of 15 epochs with minimum delta of 0.001
- Automatically restores best weights when stopping
- Prevents overfitting and saves training time
- Monitors validation ROC AUC for stopping decisions

#### Enhanced Regularization
- Gradient clipping (max_norm=1.0) for training stability
- Dropout layers throughout the architecture (0.1-0.3 rates)
- Weight decay (1e-4) in AdamW optimizer
- Proper weight initialization (Xavier/Kaiming)

### 3. Architecture Improvements

#### Model Structure (Oligo Class)
```
Input → [siRNA Sequence | mRNA Sequence | Biological Features]
        ↓
Positional Encoding → Self-Attention → Cross-Attention
        ↓
[CNN Features | BiLSTM Features] → Enhanced Feature Encoder
        ↓
Feature Fusion → MLP → Sigmoid Output
```

#### Enhanced Components
- **CrossAttentionBlock**: Multi-head attention with residual connections and layer normalization
- **SelfAttentionBlock**: Self-attention with feed-forward networks
- **EnhancedFeatureEncoder**: Feature attention mechanism for biological features
- **PositionalEncoding**: Learnable position embeddings

#### Parameter Count
- Total parameters: ~221,297
- Efficient architecture with proper regularization
- Balanced between model capacity and overfitting prevention

### 4. Compatibility Features

#### Data Pipeline Compatibility
- Maintains compatibility with existing SirnaDataset class
- Supports all 70 biological features from bio_features.py
- Compatible with StandardScaler normalization
- Supports data augmentation (mutation, masking)

#### RNA-FM Integration Ready
- Architecture designed to work with RNA-FM embeddings
- Flexible embedding dimensions (currently 5 for one-hot, expandable)
- Modular design allows easy integration of pre-trained embeddings

#### GPU Optimization
- Batch-first operations for efficient GPU computation
- Memory-efficient attention mechanisms
- Optimized for both training and inference

## Usage Example

```python
import rna_fm_seq_model as rfm
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# Create datasets
scaler = StandardScaler()
train_dataset = rfm.SirnaDataset(csv_path='data/Hu.csv', scaler=scaler, fit_scaler=True, augment=True)
val_dataset = rfm.SirnaDataset(csv_path='data/Mix.csv', scaler=scaler, fit_scaler=False)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize model
model = rfm.Oligo(rfm.SirnaDataset.bio_feats_dim)

# Train with enhanced features
trained_model, best_roc, best_f1, best_epoch = rfm.train_model(
    model, train_loader, val_loader, epochs=100, lr=0.001, batch_size=16
)
```

## Technical Innovations

### 1. Bidirectional Cross-Attention
- **Innovation**: Simultaneous modeling of siRNA→mRNA and mRNA→siRNA interactions
- **Benefit**: Captures complex binding dynamics and sequence complementarity
- **Implementation**: Separate attention heads for each direction with residual fusion

### 2. Adaptive Learning Rate Scheduling
- **Innovation**: CosineAnnealingWarmRestarts instead of plateau-based reduction
- **Benefit**: Smoother convergence, reduced plateaus, better exploration
- **Implementation**: Periodic restarts with exponentially increasing cycle lengths

### 3. Label Smoothing for Regression-Classification
- **Innovation**: Label smoothing applied to binary classification component
- **Benefit**: Prevents overconfident predictions, improves calibration
- **Implementation**: Smooth targets toward 0.5 with 0.1 smoothing factor

### 4. Enhanced Feature Integration
- **Innovation**: Attention-based biological feature weighting
- **Benefit**: Dynamic importance assignment based on sequence context
- **Implementation**: Feature attention mechanism in EnhancedFeatureEncoder

## Performance Expectations

Based on the architecture and training improvements, the model is expected to achieve:

1. **Better Convergence**: Smoother training curves with reduced plateaus
2. **Improved Generalization**: Better validation performance through regularization
3. **Enhanced Interactions**: More accurate modeling of siRNA-mRNA binding
4. **Training Stability**: Reduced overfitting and more consistent results

## Future Enhancements

1. **Multi-Head Attention**: When sequence embeddings allow for more attention heads
2. **Transformer Encoder Stacks**: Multiple layers of self and cross-attention
3. **Pre-trained Embeddings**: Integration with RNA-FM or other pre-trained models
4. **Dynamic Attention Visualization**: Tools for interpreting attention patterns
5. **Multi-Task Learning**: Joint prediction of efficacy and off-target effects

## Files Structure

- `src/rna_fm_seq_model.py`: Main implementation with Oligo class and training functions
- `src/test_rna_fm_model.py`: Comprehensive test suite validating all features
- `src/bio_features.py`: Biological feature extraction functions (existing)

## Testing

The implementation includes comprehensive tests covering:
- Cross-attention mechanism functionality
- Label smoothing loss computation
- Early stopping behavior
- Model architecture forward pass
- Cosine annealing scheduler
- Enhanced feature components

All tests pass successfully, confirming the implementation meets the requirements.