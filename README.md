# RNA-Slasher

A deep learning framework for predicting siRNA (small interfering RNA) efficacy using convolutional neural networks and biological feature engineering.

## Overview

RNA-Slasher is a machine learning tool that predicts the effectiveness of siRNA sequences in RNA interference (RNAi) experiments. The model combines convolutional neural networks for sequence analysis with engineered biological features to achieve high-accuracy predictions of siRNA knockdown efficacy.

## Features

- **Hybrid CNN-MLP Architecture**: Combines convolutional layers for sequence pattern recognition with multilayer perceptrons for feature integration
- **Comprehensive Biological Features**: Incorporates 34+ biological features including GC content, melting temperature, nucleotide frequencies, and k-mer analysis
- **Ensemble Learning**: Uses K-fold cross-validation with model averaging for robust predictions
- **Multiple Dataset Support**: Trained and validated on Hu, Mix, Taka, and Simone datasets
- **High Performance**: Achieves ROC AUC > 0.98 on test datasets

## Model Architecture

The RNA-Slasher model consists of:

1. **Input Processing**:
   - One-hot encoding of siRNA (21 nucleotides) and mRNA (80 nucleotides) sequences
   - Extraction and normalization of biological features

2. **CNN Layers**:
   - Separate convolutional blocks for siRNA and mRNA sequences
   - 32 and 64 filter convolutions with ReLU activation and max pooling

3. **Feature Engineering**:
   - GC/AT content, melting temperature, molecular weight
   - Base and dinucleotide frequencies
   - Shannon entropy, k-mer diversity, structural features

4. **MLP Classifier**:
   - Dense layers with dropout regularization
   - Sigmoid output for efficacy prediction (0-1 range)

## Performance Metrics

*K-fold cross-validation average: ROC AUC = 0.823, F1 = 0.765*

## Project Structure

```
RNA-Slasher/
├── src/                          # Source code
│   ├── model.py                  # Main model architecture and training
│   ├── model_v1.py              # Previous model version
│   ├── bio_features.py          # Biological feature extraction functions
│   ├── infer.py                 # Inference utilities and feature extraction
│   ├── model_performance.py     # Model evaluation and prediction generation
│   └── data_analysis.py         # Data analysis and cross-validation utilities
├── data/                        # Training datasets
│   ├── Hu.csv                   # Hu dataset (processed)
│   ├── Mix.csv                  # Mixed dataset (processed)
│   ├── Taka.csv                 # Taka dataset (processed)
│   ├── Hu_unprocessed.csv       # Original Hu dataset with TD features
│   ├── Mix_unprocessed.csv      # Original Mix dataset with TD features
│   └── Taka_unprocessed.csv     # Original Taka dataset with TD features
├── model/                       # Trained models and results
│   ├── ann_weights_fold*.pth    # Saved model weights (K-fold ensemble)
│   ├── ann_weights_v1.pth       # Previous version weights
│   ├── ann_weights.pth          # Main model weights
│   ├── *_predictions.csv        # Prediction results for each dataset
│   ├── *_predictions_metrics.txt # Evaluation metrics for each dataset
│   └── training_log*.txt        # Training logs and history
├── dataset_processor.py         # Dataset preprocessing utility
├── evaluate_predictions.py      # Model evaluation script
└── README.md                    # This file
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- scikit-learn
- pandas
- numpy

### Setup

```bash
git clone <repository-url>
cd RNA-Slasher
pip install torch pandas scikit-learn numpy
```

## Quick Start

To generate predictions on all datasets using pre-trained models:

```bash
python -c "from src.model_performance import main; main()"
```

To evaluate model performance on existing predictions:

```bash
python evaluate_predictions.py
```

## Usage

### Training a New Model

```python
from src.model import SirnaDataset, ANN, train_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

# Load data
df = pd.read_csv('data/Hu.csv')

# K-fold cross-validation training
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
scaler = StandardScaler()

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    # Prepare datasets with scaler
    train_dataset = SirnaDataset(df=train_df, scaler=scaler, fit_scaler=True)
    val_dataset = SirnaDataset(df=val_df, scaler=scaler, fit_scaler=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize and train model
    model = ANN(bio_feats_dim=train_dataset.bio_feats_dim)
    trained_model = train_model(model, train_loader, val_loader, epochs=50)
    
    # Save model weights for ensemble
    torch.save(model.state_dict(), f'model/ann_weights_fold{fold+1}.pth')
```

### Making Predictions

```python
from src.model_performance import extract_features, ensemble_predict
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Example siRNA and mRNA sequences
siRNA = "UUCUCUGGAAUGCCUGCAC"
mRNA = "CUCCUCCAGGUGACCGCGGGUGCAGGCAUUCCAGAGAAAGCGUUUAAUUUAACUUGG"

# Extract features
X_siRNA, X_mRNA, bio_feats = extract_features(siRNA, mRNA)

# Prepare scaler (fit on Mix dataset for consistency)
mix_df = pd.read_csv('data/Mix.csv')
scaler = StandardScaler()
# ... (fit scaler on Mix dataset biological features)

# Normalize biological features
bio_feats_norm = scaler.transform([bio_feats])[0]

# Combine all features
X = np.concatenate([X_siRNA, X_mRNA, bio_feats_norm])[None, :]

# Make prediction using ensemble
bio_feats_dim = len(bio_feats)
prediction = ensemble_predict(X, bio_feats_dim)

print(f"Predicted efficacy: {prediction:.4f}")
```

### Generating Predictions for Datasets

```python
# Run model performance script to generate predictions for all datasets
from src.model_performance import main
main()
```

### Evaluating Model Performance

```bash
python evaluate_predictions.py
```

This will generate metrics for all prediction files in the `model/` directory.

## Biological Features

The model incorporates the following biological features for both siRNA and mRNA sequences:

- **Composition**: GC content, AT content, purine/pyrimidine content
- **Thermodynamics**: Melting temperature, molecular weight
- **Sequence Properties**: Length, base frequencies, dinucleotide frequencies
- **Complexity**: Shannon entropy, longest mononucleotide runs
- **Structural**: GC skew, AT skew, AU/GC ratio
- **Diversity**: Unique k-mers (k=2,3), ambiguous base count

## Data Format

Input CSV files should contain the following columns:
- `siRNA`: 21-nucleotide siRNA sequence (A, U, C, G) - RNA sequences only
- `mRNA`: Target mRNA sequence (typically 80 nucleotides) - RNA sequences only
- `label`: Efficacy value (0.0 to 1.0)

**Note**: The model expects RNA sequences using Uracil (U) instead of Thymine (T).

## Data Preprocessing

The project includes a preprocessing utility to standardize datasets:

```bash
python dataset_processor.py
```

This script processes the raw unprocessed CSV files and creates clean versions by:
- Filtering to keep only required columns: siRNA, mRNA, label
- Ensuring consistent data format across datasets

## Model Versions

- **model.py**: Current production model with optimized architecture
- **model_v1.py**: Previous version for comparison
- **Ensemble**: K-fold cross-validation with 5 models for final predictions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

## License

[Specify license here]

## Citation

If you use RNA-Slasher in your research, please cite:
```
[Add citation information when available]
```

## Contact

[Add contact information]