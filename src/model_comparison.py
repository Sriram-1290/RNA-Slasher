import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import both model versions
from model_v1 import ANN as OriginalANN, SirnaDataset
from model_v2 import EnhancedANN, SirnaDataset as EnhancedSirnaDataset

def evaluate_model(model, data_loader, device):
    """Evaluate a model and return comprehensive metrics"""
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_true_bin = (y_true >= 0.5).astype(int)
    y_pred_bin = (y_pred >= 0.5).astype(int)
    
    metrics = {
        'roc_auc': roc_auc_score(y_true_bin, y_pred),
        'f1': f1_score(y_true_bin, y_pred_bin),
        'precision': precision_score(y_true_bin, y_pred_bin),
        'recall': recall_score(y_true_bin, y_pred_bin),
        'accuracy': accuracy_score(y_true_bin, y_pred_bin)
    }
    
    return metrics, y_true, y_pred

def compare_models():
    """Compare original and enhanced models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    
    # Load datasets
    scaler = StandardScaler()
    train_dataset = SirnaDataset(csv_path=os.path.join(base_dir, "data", "Hu.csv"), 
                                scaler=scaler, fit_scaler=True)
    val_dataset = SirnaDataset(csv_path=os.path.join(base_dir, "data", "Mix.csv"), 
                              scaler=scaler, fit_scaler=False)
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize models
    original_model = OriginalANN(SirnaDataset.bio_feats_dim)
    enhanced_model = EnhancedANN(SirnaDataset.bio_feats_dim)
    
    # Load weights if available
    original_weights_path = os.path.join(base_dir, "model", "ann_weights_v1.pth")
    enhanced_weights_path = os.path.join(base_dir, "model", "enhanced_ann_weights_v2.pth")
    
    results = {}
    
    # Evaluate original model if weights exist
    if os.path.exists(original_weights_path):
        original_model.load_state_dict(torch.load(original_weights_path, map_location=device))
        original_model.to(device)
        original_metrics, y_true_orig, y_pred_orig = evaluate_model(original_model, val_loader, device)
        results['original'] = {
            'metrics': original_metrics,
            'predictions': y_pred_orig,
            'targets': y_true_orig
        }
        print("Original Model Metrics:")
        for metric, value in original_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        print()
    
    # Evaluate enhanced model if weights exist
    if os.path.exists(enhanced_weights_path):
        enhanced_model.load_state_dict(torch.load(enhanced_weights_path, map_location=device))
        enhanced_model.to(device)
        enhanced_metrics, y_true_enh, y_pred_enh = evaluate_model(enhanced_model, val_loader, device)
        results['enhanced'] = {
            'metrics': enhanced_metrics,
            'predictions': y_pred_enh,
            'targets': y_true_enh
        }
        print("Enhanced Model Metrics:")
        for metric, value in enhanced_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        print()
    
    # Compare if both models are available
    if 'original' in results and 'enhanced' in results:
        print("Improvement Summary:")
        for metric in original_metrics.keys():
            original_val = results['original']['metrics'][metric]
            enhanced_val = results['enhanced']['metrics'][metric]
            improvement = ((enhanced_val - original_val) / original_val) * 100
            print(f"  {metric.upper()}: {improvement:+.2f}% ({original_val:.4f} → {enhanced_val:.4f})")
        
        # Create comparison plots
        create_comparison_plots(results, base_dir)
    
    return results

def create_comparison_plots(results, base_dir):
    """Create comparison plots between models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Metrics comparison
    metrics = list(results['original']['metrics'].keys())
    original_values = [results['original']['metrics'][m] for m in metrics]
    enhanced_values = [results['enhanced']['metrics'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, original_values, width, label='Original', alpha=0.8)
    axes[0, 0].bar(x + width/2, enhanced_values, width, label='Enhanced', alpha=0.8)
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Model Performance Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([m.upper() for m in metrics], rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Prediction distributions
    axes[0, 1].hist(results['original']['predictions'], bins=50, alpha=0.7, label='Original', density=True)
    axes[0, 1].hist(results['enhanced']['predictions'], bins=50, alpha=0.7, label='Enhanced', density=True)
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Prediction Distributions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter plot: Original vs Enhanced predictions
    axes[1, 0].scatter(results['original']['predictions'], results['enhanced']['predictions'], 
                       alpha=0.6, s=20)
    axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.8)
    axes[1, 0].set_xlabel('Original Model Predictions')
    axes[1, 0].set_ylabel('Enhanced Model Predictions')
    axes[1, 0].set_title('Prediction Correlation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Prediction errors
    original_errors = np.abs(results['original']['predictions'] - results['original']['targets'])
    enhanced_errors = np.abs(results['enhanced']['predictions'] - results['enhanced']['targets'])
    
    axes[1, 1].hist(original_errors, bins=50, alpha=0.7, label='Original', density=True)
    axes[1, 1].hist(enhanced_errors, bins=50, alpha=0.7, label='Enhanced', density=True)
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Prediction Error Distributions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'model', 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plots saved to: {os.path.join(base_dir, 'model', 'model_comparison.png')}")

def analyze_model_complexity():
    """Analyze and compare model complexity"""
    # Initialize models
    original_model = OriginalANN(50)  # Approximate bio_feats_dim
    enhanced_model = EnhancedANN(50)
    
    # Count parameters
    original_params = sum(p.numel() for p in original_model.parameters())
    enhanced_params = sum(p.numel() for p in enhanced_model.parameters())
    
    print("Model Complexity Analysis:")
    print(f"Original Model Parameters: {original_params:,}")
    print(f"Enhanced Model Parameters: {enhanced_params:,}")
    print(f"Parameter Increase: {((enhanced_params - original_params) / original_params) * 100:.1f}%")
    
    return original_params, enhanced_params

if __name__ == "__main__":
    print("=" * 60)
    print("RNA-Slasher Model Comparison")
    print("=" * 60)
    
    # Analyze model complexity
    analyze_model_complexity()
    print()
    
    # Compare model performance
    try:
        results = compare_models()
    except FileNotFoundError as e:
        print(f"Model weights not found: {e}")
        print("Please train the models first before running comparison.")
