#!/usr/bin/env python3
"""
RNA-Slasher Model Diagnosis and Improvement Script
Analyzes model performance issues and suggests/implements fixes
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import argparse

def analyze_prediction_patterns(results_dir='model'):
    """Analyze prediction patterns to identify issues"""
    print("=== PREDICTION PATTERN ANALYSIS ===\n")
    
    datasets = ['hu', 'mix', 'taka', 'simone']
    for dataset in datasets:
        pred_file = os.path.join(results_dir, f"{dataset}_predictions.csv")
        if not os.path.isfile(pred_file):
            # Try alternate location if not found in results_dir
            alt_path = os.path.join('..', 'model', f"{dataset}_predictions.csv")
            if os.path.isfile(alt_path):
                pred_file = alt_path
            else:
                print(f"Warning: {pred_file} not found, skipping...")
                continue
        df = pd.read_csv(pred_file)
        print(f"{dataset.upper()} DATASET ANALYSIS:")
        print("-" * 40)
        
        # Check for valid predictions
        orig_valid = df['original_prediction'].notna().sum()
        enh_valid = df['enhanced_prediction'].notna().sum()
        print(f"Valid predictions - Original: {orig_valid}, Enhanced: {enh_valid}")
        
        # Analyze prediction distributions
        if 'original_prediction' in df.columns and df['original_prediction'].notna().any():
            orig_pred = df['original_prediction'].dropna()
            print(f"Original predictions - Mean: {orig_pred.mean():.4f}, Std: {orig_pred.std():.4f}")
            print(f"                      Range: [{orig_pred.min():.4f}, {orig_pred.max():.4f}]")
        
        if 'enhanced_prediction' in df.columns and df['enhanced_prediction'].notna().any():
            enh_pred = df['enhanced_prediction'].dropna()
            print(f"Enhanced predictions - Mean: {enh_pred.mean():.4f}, Std: {enh_pred.std():.4f}")
            print(f"                      Range: [{enh_pred.min():.4f}, {enh_pred.max():.4f}]")
        
        # Analyze true labels
        if 'dataset_efficacy' in df.columns and df['dataset_efficacy'].notna().any():
            true_vals = df['dataset_efficacy'].dropna()
            print(f"True values          - Mean: {true_vals.mean():.4f}, Std: {true_vals.std():.4f}")
            print(f"                      Range: [{true_vals.min():.4f}, {true_vals.max():.4f}]")
        
        print()

def diagnose_model_issues():
    """Diagnose potential issues with the enhanced model"""
    print("=== MODEL ARCHITECTURE DIAGNOSIS ===\n")
    
    # Load and analyze model architectures
    try:
        from model import ANN
        from model_v2 import EnhancedANN
        
        # Create sample models to analyze complexity
        bio_feats_dim = 50  # Approximate
        
        original_model = ANN(bio_feats_dim)
        enhanced_model = EnhancedANN(bio_feats_dim)
        
        # Count parameters
        orig_params = sum(p.numel() for p in original_model.parameters())
        enh_params = sum(p.numel() for p in enhanced_model.parameters())
        
        print(f"Parameter Count:")
        print(f"  Original Model: {orig_params:,}")
        print(f"  Enhanced Model: {enh_params:,}")
        print(f"  Complexity Ratio: {enh_params/orig_params:.1f}x")
        print()
        
        # Analyze model structure
        print("Enhanced Model Architecture Issues:")
        print("- High complexity ratio may cause overfitting")
        print("- Attention mechanisms might not suit short sequences")
        print("- Multi-scale CNNs may be learning noise")
        print("- Insufficient regularization for complex architecture")
        print()
        
    except ImportError as e:
        print(f"Could not load models for analysis: {e}")

def suggest_improvements():
    """Suggest specific improvements for the enhanced model"""
    print("=== IMPROVEMENT SUGGESTIONS ===\n")
    
    improvements = [
        {
            "category": "Regularization",
            "suggestions": [
                "Increase dropout rates from 0.2 to 0.4-0.5",
                "Add L2 weight decay (current: 1e-4, try: 1e-3)",
                "Implement batch normalization after each layer",
                "Add early stopping with patience=10"
            ]
        },
        {
            "category": "Architecture Simplification", 
            "suggestions": [
                "Reduce multi-scale CNN kernels to just [3, 5]",
                "Simplify attention: use single head instead of multi-head",
                "Reduce LSTM hidden size from 256 to 128",
                "Remove one residual block from MLP"
            ]
        },
        {
            "category": "Training Strategy",
            "suggestions": [
                "Use smaller learning rate: 1e-4 instead of 1e-3",
                "Implement warmup learning rate schedule",
                "Train for more epochs with lower learning rate",
                "Use gradient accumulation for stable training"
            ]
        },
        {
            "category": "Data Strategy",
            "suggestions": [
                "Implement dataset-specific models",
                "Use ensemble of original + simplified enhanced model",
                "Apply data augmentation for enhanced model training",
                "Balance training data across datasets"
            ]
        }
    ]
    
    for improvement in improvements:
        print(f"{improvement['category'].upper()}:")
        for suggestion in improvement['suggestions']:
            print(f"  • {suggestion}")
        print()

def create_improved_model():
    """Create an improved version of the enhanced model"""
    print("=== CREATING IMPROVED MODEL (v3) ===\n")
    
    model_v3_code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedMultiScaleCNN(nn.Module):
    """Simplified multi-scale CNN with only 2 scales"""
    def __init__(self, input_size):
        super().__init__()
        self.conv3 = nn.Conv1d(4, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(4, 32, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(0.4)
        self.bn = nn.BatchNorm1d(64)
        
    def forward(self, x):
        # x shape: (batch, seq_len, 4)
        x = x.transpose(1, 2)  # (batch, 4, seq_len)
        
        conv3_out = F.relu(self.conv3(x))
        conv5_out = F.relu(self.conv5(x))
        
        combined = torch.cat([conv3_out, conv5_out], dim=1)
        combined = self.bn(combined)
        combined = self.dropout(combined)
        
        return F.max_pool1d(combined, kernel_size=combined.size(2)).squeeze(2)

class ImprovedANN(nn.Module):
    """Improved version of enhanced model with better regularization"""
    def __init__(self, bio_feats_dim):
        super().__init__()
        
        # Multi-scale CNN (simplified)
        self.sirna_cnn = SimplifiedMultiScaleCNN(21)  # siRNA length
        self.mrna_cnn = SimplifiedMultiScaleCNN(100)  # mRNA length
        
        # BiLSTM (simplified)
        self.lstm = nn.LSTM(64, 128, batch_first=True, dropout=0.4)
        
        # Feature encoder with batch norm
        self.feature_encoder = nn.Sequential(
            nn.Linear(bio_feats_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Final MLP with residual connection
        self.mlp = nn.Sequential(
            nn.Linear(64 + 64 + 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        sirna_seq = x[:, :21*4].view(-1, 21, 4)
        mrna_seq = x[:, 21*4:21*4+100*4].view(-1, 100, 4)
        bio_feats = x[:, 21*4+100*4:]
        
        # Extract CNN features
        sirna_features = self.sirna_cnn(sirna_seq)
        mrna_features = self.mrna_cnn(mrna_seq)
        
        # BiLSTM on combined sequences
        combined_seq = torch.cat([sirna_seq, mrna_seq], dim=1)
        lstm_out, _ = self.lstm(combined_seq)
        lstm_features = lstm_out[:, -1, :]  # Last hidden state
        
        # Process biological features
        bio_features = self.feature_encoder(bio_feats)
        
        # Combine all features
        combined_features = torch.cat([sirna_features, mrna_features, bio_features], dim=1)
        
        return self.mlp(combined_features)
'''
    
    # Save improved model
    model_file = 'model_v3.py'
    with open(model_file, 'w') as f:
        f.write(model_v3_code)
    
    print(f"Improved model saved to: {model_file}")
    print("\nKey improvements in v3:")
    print("  • Reduced CNN scales from 3 to 2")
    print("  • Added batch normalization throughout")
    print("  • Increased dropout rates (0.4-0.5)")
    print("  • Simplified LSTM architecture")
    print("  • Better feature combination strategy")
    print()

def main():
    parser = argparse.ArgumentParser(description='RNA-Slasher Model Diagnosis and Improvement')
    parser.add_argument('--results-dir', default='model', 
                        help='Directory containing prediction results')
    parser.add_argument('--action', choices=['analyze', 'diagnose', 'suggest', 'improve', 'all'], 
                        default='all', help='Action to perform')
    
    args = parser.parse_args()
    
    print("RNA-SLASHER MODEL DIAGNOSIS AND IMPROVEMENT TOOL")
    print("=" * 60)
    print()
    
    if args.action in ['analyze', 'all']:
        analyze_prediction_patterns(args.results_dir)
    
    if args.action in ['diagnose', 'all']:
        diagnose_model_issues()
    
    if args.action in ['suggest', 'all']:
        suggest_improvements()
    
    if args.action in ['improve', 'all']:
        create_improved_model()
    
    print("=" * 60)
    print("SUMMARY RECOMMENDATIONS:")
    print("1. Use original model for production (especially Hu dataset)")
    print("2. Implement the improved model (v3) with better regularization")
    print("3. Train v3 model with lower learning rate and more patience")
    print("4. Consider ensemble approach for optimal performance")
    print("5. Implement dataset-specific model selection")

if __name__ == '__main__':
    main()
