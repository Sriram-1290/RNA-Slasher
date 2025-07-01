#!/usr/bin/env python3
"""
Test script for RNA-FM Sequence Model implementation
Tests the key features: cross-attention, training stability, and model architecture
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import rna_fm_seq_model as rfm
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

def test_cross_attention():
    """Test cross-attention mechanism"""
    print("Testing Cross-Attention Mechanism...")
    
    # Create small test data
    batch_size, seq_len, embed_dim = 2, 10, 5
    siRNA_seq = torch.randn(batch_size, seq_len, embed_dim)
    mRNA_seq = torch.randn(batch_size, seq_len, embed_dim)
    
    # Test cross-attention block
    cross_attn = rfm.CrossAttentionBlock(feature_dim=embed_dim, num_heads=1)
    output, attention_weights = cross_attn(siRNA_seq, mRNA_seq)
    
    assert output.shape == siRNA_seq.shape, f"Expected {siRNA_seq.shape}, got {output.shape}"
    assert attention_weights is not None, "Attention weights should not be None"
    print("✓ Cross-attention mechanism working correctly")

def test_label_smoothing():
    """Test label smoothing loss function"""
    print("Testing Label Smoothing Loss...")
    
    # Create test data
    predictions = torch.tensor([0.9, 0.1, 0.8, 0.2])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    
    # Test label smoothing
    loss_fn = rfm.LabelSmoothingBCELoss(smoothing=0.1)
    loss = loss_fn(predictions, targets)
    
    assert loss.item() >= 0, "Loss should be non-negative"
    print("✓ Label smoothing loss working correctly")

def test_early_stopping():
    """Test early stopping mechanism"""
    print("Testing Early Stopping...")
    
    # Create dummy model
    model = torch.nn.Linear(10, 1)
    early_stopping = rfm.EarlyStopping(patience=2, min_delta=0.01)
    
    # Simulate improving scores
    assert not early_stopping(0.5, model), "Should not stop on first call"
    assert not early_stopping(0.6, model), "Should not stop on improvement"
    
    # Simulate no improvement
    assert not early_stopping(0.55, model), "Should not stop yet (patience=2)"
    assert early_stopping(0.54, model), "Should stop after patience exhausted"
    
    print("✓ Early stopping mechanism working correctly")

def test_model_architecture():
    """Test complete model architecture"""
    print("Testing Model Architecture...")
    
    # Create dataset
    scaler = StandardScaler()
    train_dataset = rfm.SirnaDataset(csv_path='../data/Hu.csv', scaler=scaler, fit_scaler=True)
    
    # Create model
    model = rfm.Oligo(rfm.SirnaDataset.bio_feats_dim)
    model.eval()
    
    # Test forward pass
    sample_input, sample_target = train_dataset[0]
    output = model(sample_input.unsqueeze(0))
    
    assert output.shape == torch.Size([1]), f"Expected [1], got {output.shape}"
    assert 0 <= output.item() <= 1, f"Output should be between 0 and 1, got {output.item()}"
    
    print("✓ Model architecture working correctly")

def test_cosine_scheduler():
    """Test CosineAnnealingWarmRestarts scheduler"""
    print("Testing Cosine Annealing Scheduler...")
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=0.00001
    )
    
    # Test scheduler steps
    initial_lr = scheduler.get_last_lr()[0]
    for i in range(10):
        scheduler.step()
    
    # LR should have changed
    final_lr = scheduler.get_last_lr()[0]
    assert initial_lr != final_lr, "Learning rate should have changed"
    
    print("✓ Cosine annealing scheduler working correctly")

def test_enhanced_features():
    """Test enhanced features of the model"""
    print("Testing Enhanced Features...")
    
    # Test positional encoding
    pos_enc = rfm.PositionalEncoding(max_len=21, d_model=5)
    test_seq = torch.randn(2, 21, 5)
    encoded_seq = pos_enc(test_seq)
    assert encoded_seq.shape == test_seq.shape, "Positional encoding should preserve shape"
    
    # Test self-attention
    self_attn = rfm.SelfAttentionBlock(feature_dim=5, num_heads=1)
    attended_seq = self_attn(test_seq)
    assert attended_seq.shape == test_seq.shape, "Self-attention should preserve shape"
    
    print("✓ Enhanced features working correctly")

def main():
    """Run all tests"""
    print("Running RNA-FM Sequence Model Tests...")
    print("=" * 50)
    
    try:
        test_cross_attention()
        test_label_smoothing()
        test_early_stopping()
        test_model_architecture()
        test_cosine_scheduler()
        test_enhanced_features()
        
        print("=" * 50)
        print("✅ All tests passed successfully!")
        print("🎉 RNA-FM Sequence Model implementation is working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()