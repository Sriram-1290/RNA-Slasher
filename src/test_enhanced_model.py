"""
Test script for enhanced RNA-FM model.
Demonstrates model creation, training, and evaluation with improved stability.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np

from model import SirnaDataset
from enhanced_rna_fm_model import create_enhanced_rna_fm_model
from training_utils import StableTrainer
from config import get_default_config, get_fast_training_config
from sklearn.model_selection import train_test_split


def test_enhanced_model():
    """Test the enhanced RNA-FM model"""
    
    print("=" * 60)
    print("Testing Enhanced RNA-FM Model")
    print("=" * 60)
    
    # Load sample data
    print("Loading data...")
    df = pd.read_csv('../data/Mix.csv').head(100)  # Use small sample for testing
    print(f"Dataset size: {len(df)}")
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    # Create datasets with proper scaler handling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    train_dataset = SirnaDataset(df=train_df, scaler=scaler, fit_scaler=True)
    val_dataset = SirnaDataset(df=val_df, scaler=scaler, fit_scaler=False)
    
    bio_feats_dim = train_dataset.bio_feats_dim
    print(f"Biological features dimension: {bio_feats_dim}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Create model with fast training config
    print("\nCreating enhanced model...")
    config = get_fast_training_config()  # Use fast config for testing
    config.training.epochs = 5  # Even faster for testing
    config.logging.log_dir = '/tmp/test_logs'
    config.logging.checkpoint_dir = '/tmp/test_checkpoints'
    config.logging.plot_dir = '/tmp/test_plots'
    
    model = create_enhanced_rna_fm_model(bio_feats_dim, config)
    
    # Test single forward pass
    print("\nTesting forward pass...")
    sample_batch, sample_targets = next(iter(train_loader))
    
    model.eval()
    with torch.no_grad():
        outputs, attention_weights, feature_importance = model(sample_batch, return_attention=True)
    
    print(f"Sample outputs: {outputs[:3].numpy()}")
    print(f"Sample targets: {sample_targets[:3].numpy()}")
    print(f"Attention layers: {len(attention_weights['cross'])}")
    print(f"Feature importance shape: {feature_importance['attention_weights'].shape}")
    
    # Test stable training
    print("\nTesting stable training...")
    trainer = StableTrainer(model, config)
    
    # Train for a few epochs
    metrics_history = trainer.train(train_loader, val_loader, epochs=config.training.epochs)
    
    print(f"\nTraining completed!")
    print(f"Final metrics:")
    final_metrics = metrics_history[-1]
    print(f"  Final Val ROC-AUC: {final_metrics.val_roc_auc:.4f}")
    print(f"  Final Val F1: {final_metrics.val_f1:.4f}")
    print(f"  Final Val Loss: {final_metrics.val_loss:.4f}")
    
    # Test model interpretability
    print("\nTesting model interpretability...")
    
    # Get sample sequences for visualization
    sample_row = df.iloc[0]
    sirna_seq = sample_row['siRNA']
    mrna_seq = sample_row['mRNA']
    
    print(f"Sample siRNA: {sirna_seq}")
    print(f"Sample mRNA: {mrna_seq[:30]}...")  # Show first 30 chars
    
    # Run forward pass to get attention weights
    sample_input = train_dataset[0][0].unsqueeze(0)
    model.eval()
    
    with torch.no_grad():
        output, attention_weights, feature_importance = model(sample_input, return_attention=True)
    
    # Analyze attention patterns
    try:
        analysis = model.get_attention_analysis(sirna_seq, mrna_seq)
        print(f"Attention analysis for layer 0:")
        layer_0_analysis = analysis.get('layer_0', {})
        print(f"  siRNA critical positions: {layer_0_analysis.get('sirna_critical_positions', [])}")
        print(f"  mRNA critical positions: {layer_0_analysis.get('mrna_critical_positions', [])[:10]}...")  # Show first 10
    except Exception as e:
        print(f"Attention analysis failed: {e}")
    
    # Test feature importance
    feature_attn_weights = feature_importance['attention_weights']
    print(f"Feature attention weights shape: {feature_attn_weights.shape}")
    weights_numpy = feature_attn_weights.squeeze().numpy()
    if weights_numpy.ndim > 0:
        print(f"Top 5 feature importance scores: {weights_numpy[:5]}")
    else:
        print(f"Single feature importance score: {weights_numpy}")
    
    
    print("\n" + "=" * 60)
    print("Enhanced RNA-FM Model Test Completed Successfully!")
    print("=" * 60)
    
    return model, trainer, metrics_history


def compare_with_original():
    """Compare enhanced model with original model"""
    
    print("\n" + "=" * 60)
    print("Comparing Enhanced vs Original Model")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('../data/Mix.csv').head(50)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create datasets with proper scaler handling  
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    train_dataset = SirnaDataset(df=train_df, scaler=scaler, fit_scaler=True)
    val_dataset = SirnaDataset(df=val_df, scaler=scaler, fit_scaler=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    bio_feats_dim = train_dataset.bio_feats_dim
    
    # Test original model
    print("Testing original model...")
    from model import ANN, train_model
    
    original_model = ANN(bio_feats_dim)
    original_model = train_model(original_model, train_loader, val_loader, epochs=5, lr=1e-3)
    
    # Test enhanced model
    print("Testing enhanced model...")
    config = get_fast_training_config()
    config.training.epochs = 5
    config.logging.log_dir = '/tmp/comparison_logs'
    config.logging.checkpoint_dir = '/tmp/comparison_checkpoints'
    
    enhanced_model = create_enhanced_rna_fm_model(bio_feats_dim, config)
    trainer = StableTrainer(enhanced_model, config)
    metrics_history = trainer.train(train_loader, val_loader, epochs=5)
    
    # Compare final performance
    print("\nComparison Results:")
    print(f"Enhanced Model Final Val ROC-AUC: {metrics_history[-1].val_roc_auc:.4f}")
    print(f"Enhanced Model Parameters: {sum(p.numel() for p in enhanced_model.parameters()):,}")
    print(f"Original Model Parameters: {sum(p.numel() for p in original_model.parameters()):,}")
    
    print("Comparison completed!")


if __name__ == "__main__":
    # Run tests
    model, trainer, history = test_enhanced_model()
    
    # Optional: Run comparison
    # compare_with_original()