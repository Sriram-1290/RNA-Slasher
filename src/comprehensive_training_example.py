"""
Comprehensive training example for Enhanced RNA-FM model.
Demonstrates how to achieve ROC-AUC > 0.80 with proper hyperparameter tuning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

from model import SirnaDataset
from enhanced_rna_fm_model import create_enhanced_rna_fm_model
from training_utils import StableTrainer
from config import get_default_config, get_high_performance_config


def train_enhanced_model_full():
    """Train enhanced model on full dataset with optimized hyperparameters"""
    
    print("=" * 80)
    print("Enhanced RNA-FM Model - Full Training for ROC-AUC > 0.80")
    print("=" * 80)
    
    # Load full Mix dataset
    print("Loading Mix dataset...")
    df = pd.read_csv('../data/Mix.csv')
    print(f"Dataset size: {len(df)}")
    print(f"Label distribution: {df['label'].value_counts().sort_index()}")
    
    # Use K-fold cross-validation for robust evaluation
    print("\nSetting up K-fold cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n{'='*20} FOLD {fold + 1}/5 {'='*20}")
        
        # Split data
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
        
        # Create datasets with proper scaler
        scaler = StandardScaler()
        train_dataset = SirnaDataset(df=train_df, scaler=scaler, fit_scaler=True)
        val_dataset = SirnaDataset(df=val_df, scaler=scaler, fit_scaler=False)
        
        bio_feats_dim = train_dataset.bio_feats_dim
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Create model with high-performance configuration
        config = get_high_performance_config()
        
        # Optimize config for this dataset
        config.training.epochs = 100
        config.training.learning_rate = 5e-4
        config.training.early_stopping_patience = 20
        config.training.gradient_clip_val = 1.0
        config.training.label_smoothing = 0.05
        
        # Set up logging directories
        config.logging.log_dir = f'/tmp/enhanced_fold_{fold + 1}_logs'
        config.logging.checkpoint_dir = f'/tmp/enhanced_fold_{fold + 1}_checkpoints'
        config.logging.plot_dir = f'/tmp/enhanced_fold_{fold + 1}_plots'
        
        # Model architecture tuning
        config.model.d_model = 128
        config.model.n_head = 8
        config.model.n_layers = 3
        config.model.lstm_hidden_size = 64
        
        # Regularization tuning
        config.regularization.attention_dropout = 0.1
        config.regularization.cnn_dropout = 0.2
        config.regularization.lstm_dropout = 0.2
        config.regularization.mlp_dropout = 0.3
        
        print(f"Creating enhanced model for fold {fold + 1}...")
        model = create_enhanced_rna_fm_model(bio_feats_dim, config)
        
        # Train model
        print(f"Training model for fold {fold + 1}...")
        trainer = StableTrainer(model, config)
        
        try:
            metrics_history = trainer.train(train_loader, val_loader)
            
            # Get best validation metrics
            best_val_roc_auc = max(m.val_roc_auc for m in metrics_history)
            best_val_f1 = max(m.val_f1 for m in metrics_history)
            
            fold_results.append({
                'fold': fold + 1,
                'best_val_roc_auc': best_val_roc_auc,
                'best_val_f1': best_val_f1,
                'final_val_roc_auc': metrics_history[-1].val_roc_auc,
                'final_val_f1': metrics_history[-1].val_f1,
                'epochs_trained': len(metrics_history)
            })
            
            print(f"Fold {fold + 1} Results:")
            print(f"  Best Val ROC-AUC: {best_val_roc_auc:.4f}")
            print(f"  Best Val F1: {best_val_f1:.4f}")
            print(f"  Final Val ROC-AUC: {metrics_history[-1].val_roc_auc:.4f}")
            print(f"  Epochs trained: {len(metrics_history)}")
            
        except Exception as e:
            print(f"Error in fold {fold + 1}: {e}")
            fold_results.append({
                'fold': fold + 1,
                'best_val_roc_auc': 0.0,
                'best_val_f1': 0.0,
                'final_val_roc_auc': 0.0,
                'final_val_f1': 0.0,
                'epochs_trained': 0,
                'error': str(e)
            })
    
    # Calculate overall performance
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    if fold_results:
        avg_best_roc_auc = np.mean([r['best_val_roc_auc'] for r in fold_results])
        avg_best_f1 = np.mean([r['best_val_f1'] for r in fold_results])
        avg_final_roc_auc = np.mean([r['final_val_roc_auc'] for r in fold_results])
        avg_final_f1 = np.mean([r['final_val_f1'] for r in fold_results])
        
        std_best_roc_auc = np.std([r['best_val_roc_auc'] for r in fold_results])
        std_best_f1 = np.std([r['best_val_f1'] for r in fold_results])
        
        print(f"Average Best Val ROC-AUC: {avg_best_roc_auc:.4f} ± {std_best_roc_auc:.4f}")
        print(f"Average Best Val F1: {avg_best_f1:.4f} ± {std_best_f1:.4f}")
        print(f"Average Final Val ROC-AUC: {avg_final_roc_auc:.4f}")
        print(f"Average Final Val F1: {avg_final_f1:.4f}")
        
        print(f"\nPer-fold results:")
        for result in fold_results:
            if 'error' not in result:
                print(f"  Fold {result['fold']}: ROC-AUC {result['best_val_roc_auc']:.4f}, "
                      f"F1 {result['best_val_f1']:.4f}")
            else:
                print(f"  Fold {result['fold']}: Error - {result['error']}")
        
        # Check if we achieved target
        target_achieved = avg_best_roc_auc > 0.80
        print(f"\nTarget ROC-AUC > 0.80: {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
        
        if target_achieved:
            print("🎉 Enhanced RNA-FM model successfully achieved target performance!")
        else:
            print("💡 Suggestions for improvement:")
            print("  - Increase model capacity (d_model, n_layers)")
            print("  - Adjust learning rate and training schedule")
            print("  - Fine-tune regularization parameters")
            print("  - Ensure data quality and preprocessing")
    
    return fold_results


def compare_models_performance():
    """Compare enhanced model with original model"""
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    # Load subset for quick comparison
    df = pd.read_csv('../data/Mix.csv').head(200)
    train_df, val_df = df.iloc[:160], df.iloc[160:]
    
    # Create datasets
    scaler = StandardScaler()
    train_dataset = SirnaDataset(df=train_df, scaler=scaler, fit_scaler=True)
    val_dataset = SirnaDataset(df=val_df, scaler=scaler, fit_scaler=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    bio_feats_dim = train_dataset.bio_feats_dim
    
    # Test original model
    print("Training original model...")
    from model import ANN, train_model
    
    original_model = ANN(bio_feats_dim)
    original_model = train_model(original_model, train_loader, val_loader, epochs=20, lr=1e-3)
    
    # Evaluate original model
    original_model.eval()
    original_preds = []
    original_targets = []
    
    with torch.no_grad():
        for data, targets in val_loader:
            preds = original_model(data)
            original_preds.extend(preds.numpy())
            original_targets.extend(targets.numpy())
    
    original_roc_auc = roc_auc_score((np.array(original_targets) >= 0.5).astype(int), 
                                    original_preds)
    original_f1 = f1_score((np.array(original_targets) >= 0.5).astype(int), 
                          (np.array(original_preds) >= 0.5).astype(int))
    
    # Test enhanced model
    print("Training enhanced model...")
    config = get_default_config()
    config.training.epochs = 20
    config.logging.log_dir = '/tmp/comparison_enhanced_logs'
    config.logging.checkpoint_dir = '/tmp/comparison_enhanced_checkpoints'
    
    enhanced_model = create_enhanced_rna_fm_model(bio_feats_dim, config)
    trainer = StableTrainer(enhanced_model, config)
    metrics_history = trainer.train(train_loader, val_loader, epochs=20)
    
    enhanced_roc_auc = metrics_history[-1].val_roc_auc
    enhanced_f1 = metrics_history[-1].val_f1
    
    # Compare results
    print("\nComparison Results:")
    print("-" * 50)
    print(f"{'Metric':<20} {'Original':<12} {'Enhanced':<12} {'Improvement':<12}")
    print("-" * 50)
    print(f"{'ROC-AUC':<20} {original_roc_auc:<12.4f} {enhanced_roc_auc:<12.4f} "
          f"{((enhanced_roc_auc - original_roc_auc) / original_roc_auc * 100):+.1f}%")
    print(f"{'F1 Score':<20} {original_f1:<12.4f} {enhanced_f1:<12.4f} "
          f"{((enhanced_f1 - original_f1) / original_f1 * 100):+.1f}%")
    
    # Parameter comparison
    original_params = sum(p.numel() for p in original_model.parameters())
    enhanced_params = sum(p.numel() for p in enhanced_model.parameters())
    
    print(f"{'Parameters':<20} {original_params:<12,} {enhanced_params:<12,} "
          f"{((enhanced_params - original_params) / original_params * 100):+.1f}%")
    
    print("-" * 50)


def demonstrate_interpretability():
    """Demonstrate model interpretability features"""
    
    print("\n" + "=" * 80)
    print("MODEL INTERPRETABILITY DEMONSTRATION")
    print("=" * 80)
    
    # Load sample data
    df = pd.read_csv('../data/Mix.csv').head(10)
    scaler = StandardScaler()
    dataset = SirnaDataset(df=df, scaler=scaler, fit_scaler=True)
    
    # Create model
    config = get_default_config()
    config.logging.plot_dir = '/tmp/interpretability_plots'
    os.makedirs(config.logging.plot_dir, exist_ok=True)
    
    model = create_enhanced_rna_fm_model(dataset.bio_feats_dim, config)
    model.eval()
    
    # Get sample for analysis
    sample_input, target = dataset[0]
    sample_row = df.iloc[0]
    sirna_seq = sample_row['siRNA']
    mrna_seq = sample_row['mRNA']
    
    print(f"Analyzing sequence pair:")
    print(f"  siRNA: {sirna_seq}")
    print(f"  mRNA: {mrna_seq}")
    print(f"  Label: {target:.2f}")
    
    # Run forward pass with attention tracking
    with torch.no_grad():
        output, attention_weights, feature_importance = model(
            sample_input.unsqueeze(0), return_attention=True
        )
    
    print(f"  Prediction: {output.item():.4f}")
    
    # Analyze attention patterns
    try:
        analysis = model.get_attention_analysis(sirna_seq, mrna_seq)
        
        print(f"\nAttention Analysis:")
        for layer_name, layer_analysis in analysis.items():
            print(f"  {layer_name.upper()}:")
            print(f"    siRNA critical positions: {layer_analysis.get('sirna_critical_positions', [])}")
            print(f"    mRNA critical positions: {layer_analysis.get('mrna_critical_positions', [])[:5]}...")
            print(f"    Critical siRNA nucleotides: {layer_analysis.get('sirna_critical_nucleotides', [])}")
            print(f"    Average attention strength: {layer_analysis.get('average_attention_strength', {})}")
        
        # Generate visualizations
        print(f"\nGenerating attention visualizations...")
        model.visualize_attention(sirna_seq, mrna_seq, save_prefix='demo')
        print(f"Attention plots saved to: {config.logging.plot_dir}")
        
    except Exception as e:
        print(f"Attention analysis error: {e}")
    
    # Feature importance analysis
    feature_weights = feature_importance['attention_weights'].squeeze()
    print(f"\nFeature Importance Analysis:")
    print(f"  Feature attention shape: {feature_weights.shape}")
    if feature_weights.numel() > 1:
        print(f"  Top feature importance scores: {feature_weights.numpy()[:5]}")
    else:
        print(f"  Feature importance score: {feature_weights.item():.4f}")


if __name__ == "__main__":
    # Run comprehensive training
    results = train_enhanced_model_full()
    
    # Run comparison
    compare_models_performance()
    
    # Demonstrate interpretability
    demonstrate_interpretability()
    
    print("\n" + "=" * 80)
    print("ENHANCED RNA-FM MODEL EVALUATION COMPLETED")
    print("=" * 80)