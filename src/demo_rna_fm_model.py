#!/usr/bin/env python3
"""
RNA-FM Sequence Model Demo
Demonstrates the key features and usage of the enhanced RNA-FM model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import rna_fm_seq_model as rfm
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import torch

def demonstrate_features():
    """Demonstrate key features of the RNA-FM model"""
    print("🧬 RNA-FM Sequence Model Demonstration")
    print("=" * 60)
    
    # 1. Dataset Creation
    print("1. Creating Enhanced Dataset...")
    scaler = StandardScaler()
    train_dataset = rfm.SirnaDataset(
        csv_path='../data/Hu.csv', 
        scaler=scaler, 
        fit_scaler=True, 
        augment=True  # Enhanced with sequence augmentation
    )
    val_dataset = rfm.SirnaDataset(
        csv_path='../data/Mix.csv', 
        scaler=scaler, 
        fit_scaler=False
    )
    
    print(f"   ✓ Training samples: {len(train_dataset):,}")
    print(f"   ✓ Validation samples: {len(val_dataset):,}")
    print(f"   ✓ Biological features: {rfm.SirnaDataset.bio_feats_dim}")
    print(f"   ✓ Sequence augmentation: Enabled")
    
    # 2. Model Architecture
    print("\n2. Enhanced Model Architecture (Oligo Class)...")
    model = rfm.Oligo(rfm.SirnaDataset.bio_feats_dim)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Cross-attention: Bidirectional (siRNA ↔ mRNA)")
    print(f"   ✓ Self-attention: Enabled for both sequences")
    print(f"   ✓ Positional encoding: Learnable embeddings")
    print(f"   ✓ Feature encoder: Enhanced with attention mechanism")
    
    # 3. Training Enhancements
    print("\n3. Stable Training Features...")
    print(f"   ✓ Scheduler: CosineAnnealingWarmRestarts")
    print(f"   ✓ Loss function: MSE + Label Smoothing BCE")
    print(f"   ✓ Early stopping: 15 epochs patience")
    print(f"   ✓ Regularization: Dropout, weight decay, gradient clipping")
    
    # 4. Sample Forward Pass
    print("\n4. Testing Forward Pass...")
    model.eval()
    sample_input, sample_target = train_dataset[0]
    with torch.no_grad():
        output = model(sample_input.unsqueeze(0))
    
    print(f"   ✓ Input shape: {sample_input.shape}")
    print(f"   ✓ Output shape: {output.shape}")
    print(f"   ✓ Output value: {output.item():.4f}")
    print(f"   ✓ Target value: {sample_target.item():.4f}")
    
    # 5. Feature Breakdown
    print("\n5. Input Feature Breakdown...")
    siRNA_size = 21 * 5  # SEQ_LEN * nucleotides
    mRNA_size = 80 * 5   # MRNA_LEN * nucleotides
    bio_size = rfm.SirnaDataset.bio_feats_dim
    
    print(f"   ✓ siRNA one-hot: {siRNA_size} features (21 bases × 5 nucleotides)")
    print(f"   ✓ mRNA one-hot: {mRNA_size} features (80 bases × 5 nucleotides)")
    print(f"   ✓ Biological features: {bio_size} features")
    print(f"   ✓ Total input size: {siRNA_size + mRNA_size + bio_size}")
    
    # 6. Training Configuration
    print("\n6. Recommended Training Configuration...")
    print(f"   ✓ Learning rate: 0.001 (with cosine annealing)")
    print(f"   ✓ Batch size: 16")
    print(f"   ✓ Epochs: 100 (with early stopping)")
    print(f"   ✓ Weight decay: 1e-4")
    print(f"   ✓ Gradient clipping: max_norm=1.0")
    
    print("\n🎉 RNA-FM Model ready for training!")
    print("   Use: rfm.train_model(model, train_loader, val_loader)")
    
def show_usage_example():
    """Show complete usage example"""
    print("\n" + "=" * 60)
    print("📝 USAGE EXAMPLE")
    print("=" * 60)
    
    example_code = '''
# Import the RNA-FM model
import rna_fm_seq_model as rfm
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# 1. Prepare datasets
scaler = StandardScaler()
train_dataset = rfm.SirnaDataset(
    csv_path='data/Hu.csv', 
    scaler=scaler, 
    fit_scaler=True, 
    augment=True
)
val_dataset = rfm.SirnaDataset(
    csv_path='data/Mix.csv', 
    scaler=scaler, 
    fit_scaler=False
)

# 2. Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 3. Initialize enhanced model
model = rfm.Oligo(rfm.SirnaDataset.bio_feats_dim)

# 4. Train with stable training features
trained_model, best_roc, best_f1, best_epoch = rfm.train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    lr=0.001,
    batch_size=16,
    device='cuda'  # or 'cpu'
)

# 5. Save the trained model
torch.save(trained_model.state_dict(), 'rna_fm_model_weights.pth')
print(f"Model saved! Best ROC: {best_roc:.4f}, F1: {best_f1:.4f}")
'''
    
    print(example_code)

def main():
    """Main demonstration function"""
    try:
        demonstrate_features()
        show_usage_example()
        
        print("\n" + "🌟" * 20)
        print("RNA-FM Sequence Model Implementation Complete!")
        print("Key Improvements Delivered:")
        print("  ✅ Enhanced Cross-Attention Mechanism")
        print("  ✅ Stable Training with Better Convergence")
        print("  ✅ Advanced Regularization Techniques")
        print("  ✅ RNA-FM Compatibility Ready")
        print("🌟" * 20)
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        raise

if __name__ == "__main__":
    main()