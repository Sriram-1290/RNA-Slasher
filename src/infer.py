import torch
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.preprocessing import StandardScaler
from model import ANN, one_hot_encode, SEQ_LEN, MRNA_LEN, SirnaDataset
from model_v2 import EnhancedANN, SirnaDataset as EnhancedSirnaDataset
from bio_features import (
    gc_content, at_content, melting_temp, length, base_frequencies,
    purine_content, pyrimidine_content, molecular_weight, dinucleotide_frequencies, shannon_entropy,
    longest_mononucleotide_run, au_gc_ratio, gc_skew, at_skew, unique_kmers, count_ambiguous
)

# Helper to extract and normalize features for a single input

def extract_features(siRNA, mRNA):
    # One-hot encoding
    X_siRNA = one_hot_encode(siRNA, SEQ_LEN)
    X_mRNA = one_hot_encode(mRNA, MRNA_LEN)
    # Biological features (order must match model.py)
    siRNA_gc = gc_content(siRNA)
    siRNA_at = at_content(siRNA)
    siRNA_tm = melting_temp(siRNA)
    siRNA_len = length(siRNA)
    siRNA_basefreq = base_frequencies(siRNA)
    siRNA_pur = purine_content(siRNA)
    siRNA_pyr = pyrimidine_content(siRNA)
    siRNA_mw = molecular_weight(siRNA)
    siRNA_dinuc = dinucleotide_frequencies(siRNA)
    siRNA_entropy = shannon_entropy(siRNA)
    siRNA_run = longest_mononucleotide_run(siRNA)
    siRNA_au_gc = au_gc_ratio(siRNA)
    siRNA_gcskew = gc_skew(siRNA)
    siRNA_atskew = at_skew(siRNA)
    siRNA_kmers2 = unique_kmers(siRNA, 2)
    siRNA_kmers3 = unique_kmers(siRNA, 3)
    siRNA_ambig = count_ambiguous(siRNA)
    mRNA_gc = gc_content(mRNA)
    mRNA_at = at_content(mRNA)
    mRNA_tm = melting_temp(mRNA)
    mRNA_len = length(mRNA)
    mRNA_basefreq = base_frequencies(mRNA)
    mRNA_pur = purine_content(mRNA)
    mRNA_pyr = pyrimidine_content(mRNA)
    mRNA_mw = molecular_weight(mRNA)
    mRNA_dinuc = dinucleotide_frequencies(mRNA)
    mRNA_entropy = shannon_entropy(mRNA)
    mRNA_run = longest_mononucleotide_run(mRNA)
    mRNA_au_gc = au_gc_ratio(mRNA)
    mRNA_gcskew = gc_skew(mRNA)
    mRNA_atskew = at_skew(mRNA)
    mRNA_kmers2 = unique_kmers(mRNA, 2)
    mRNA_kmers3 = unique_kmers(mRNA, 3)
    mRNA_ambig = count_ambiguous(mRNA)
    bio_feats = np.concatenate([
        [siRNA_gc, siRNA_at, siRNA_tm, siRNA_len], siRNA_basefreq,
        [siRNA_pur, siRNA_pyr, siRNA_mw], siRNA_dinuc, [siRNA_entropy, siRNA_run, siRNA_au_gc, siRNA_gcskew, siRNA_atskew, siRNA_kmers2, siRNA_kmers3, siRNA_ambig],
        [mRNA_gc, mRNA_at, mRNA_tm, mRNA_len], mRNA_basefreq,
        [mRNA_pur, mRNA_pyr, mRNA_mw], mRNA_dinuc, [mRNA_entropy, mRNA_run, mRNA_au_gc, mRNA_gcskew, mRNA_atskew, mRNA_kmers2, mRNA_kmers3, mRNA_ambig]    ])
    return X_siRNA, X_mRNA, bio_feats

def get_bio_feats(df):
    """Extract biological features from a dataframe for scaler fitting"""
    siRNA_gc = df['siRNA'].apply(gc_content).values[:, None]
    siRNA_at = df['siRNA'].apply(at_content).values[:, None]
    siRNA_tm = df['siRNA'].apply(melting_temp).values[:, None]
    siRNA_len = df['siRNA'].apply(length).values[:, None]
    siRNA_basefreq = np.stack(df['siRNA'].apply(base_frequencies))
    siRNA_pur = df['siRNA'].apply(purine_content).values[:, None]
    siRNA_pyr = df['siRNA'].apply(pyrimidine_content).values[:, None]
    siRNA_mw = df['siRNA'].apply(molecular_weight).values[:, None]
    siRNA_dinuc = np.stack(df['siRNA'].apply(dinucleotide_frequencies))
    siRNA_entropy = df['siRNA'].apply(shannon_entropy).values[:, None]
    siRNA_run = df['siRNA'].apply(longest_mononucleotide_run).values[:, None]
    siRNA_au_gc = df['siRNA'].apply(au_gc_ratio).values[:, None]
    siRNA_gcskew = df['siRNA'].apply(gc_skew).values[:, None]
    siRNA_atskew = df['siRNA'].apply(at_skew).values[:, None]
    siRNA_kmers2 = df['siRNA'].apply(lambda x: unique_kmers(x, 2)).values[:, None]
    siRNA_kmers3 = df['siRNA'].apply(lambda x: unique_kmers(x, 3)).values[:, None]
    siRNA_ambig = df['siRNA'].apply(count_ambiguous).values[:, None]
    mRNA_gc = df['mRNA'].apply(gc_content).values[:, None]
    mRNA_at = df['mRNA'].apply(at_content).values[:, None]
    mRNA_tm = df['mRNA'].apply(melting_temp).values[:, None]
    mRNA_len = df['mRNA'].apply(length).values[:, None]
    mRNA_basefreq = np.stack(df['mRNA'].apply(base_frequencies))
    mRNA_pur = df['mRNA'].apply(purine_content).values[:, None]
    mRNA_pyr = df['mRNA'].apply(pyrimidine_content).values[:, None]
    mRNA_mw = df['mRNA'].apply(molecular_weight).values[:, None]
    mRNA_dinuc = np.stack(df['mRNA'].apply(dinucleotide_frequencies))
    mRNA_entropy = df['mRNA'].apply(shannon_entropy).values[:, None]
    mRNA_run = df['mRNA'].apply(longest_mononucleotide_run).values[:, None]
    mRNA_au_gc = df['mRNA'].apply(au_gc_ratio).values[:, None]
    mRNA_gcskew = df['mRNA'].apply(gc_skew).values[:, None]
    mRNA_atskew = df['mRNA'].apply(at_skew).values[:, None]
    mRNA_kmers2 = df['mRNA'].apply(lambda x: unique_kmers(x, 2)).values[:, None]
    mRNA_kmers3 = df['mRNA'].apply(lambda x: unique_kmers(x, 3)).values[:, None]
    mRNA_ambig = df['mRNA'].apply(count_ambiguous).values[:, None]
    return np.concatenate([
        siRNA_gc, siRNA_at, siRNA_tm, siRNA_len, siRNA_basefreq,
        siRNA_pur, siRNA_pyr, siRNA_mw, siRNA_dinuc, siRNA_entropy, siRNA_run, siRNA_au_gc, siRNA_gcskew, siRNA_atskew, siRNA_kmers2, siRNA_kmers3, siRNA_ambig,
        mRNA_gc, mRNA_at, mRNA_tm, mRNA_len, mRNA_basefreq,
        mRNA_pur, mRNA_pyr, mRNA_mw, mRNA_dinuc, mRNA_entropy, mRNA_run, mRNA_au_gc, mRNA_gcskew, mRNA_atskew, mRNA_kmers2, mRNA_kmers3, mRNA_ambig
    ], axis=1)

def ensemble_predict(X, bio_feats_dim, model_type='original'):
    """
    Predict using ensemble of models
    
    Args:
        X: Input features
        bio_feats_dim: Number of biological features
        model_type: 'original' for original model, 'enhanced' for enhanced model, 'both' for comparison
    
    Returns:
        Prediction(s) as float or dict
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type in ['original', 'both']:
        # Original model ensemble prediction
        original_preds = []
        for fold in range(1, 6):
            model_path = f'model/ann_weights_fold{fold}.pth'
            if os.path.exists(model_path):
                model = ANN(bio_feats_dim)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                with torch.no_grad():
                    x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                    pred = model(x_tensor).cpu().item()
                    original_preds.append(pred)
        
        original_pred = np.mean(original_preds) if original_preds else None
    
    if model_type in ['enhanced', 'both']:
        # Enhanced model prediction (single model for now)
        enhanced_pred = None
        enhanced_model_path = 'model/enhanced_ann_weights_v2.pth'
        if os.path.exists(enhanced_model_path):
            model = EnhancedANN(bio_feats_dim)
            model.load_state_dict(torch.load(enhanced_model_path, map_location=device))
            model.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                enhanced_pred = model(x_tensor).cpu().item()
    
    if model_type == 'original':
        return original_pred
    elif model_type == 'enhanced':
        return enhanced_pred
    elif model_type == 'both':
        return {
            'original': original_pred,
            'enhanced': enhanced_pred
        }
    else:
        raise ValueError("model_type must be 'original', 'enhanced', or 'both'")

def predict_from_file(csv_file, model_type='enhanced', output_file=None):
    """
    Make predictions on a CSV file containing siRNA and mRNA sequences
    
    Args:
        csv_file: Path to CSV file with 'siRNA' and 'mRNA' columns
        model_type: 'original', 'enhanced', or 'both'
        output_file: Optional output file path for predictions
    
    Returns:
        DataFrame with predictions
    """
    # Load data
    df = pd.read_csv(csv_file)
    
    if 'siRNA' not in df.columns or 'mRNA' not in df.columns:
        raise ValueError("CSV file must contain 'siRNA' and 'mRNA' columns")
    
    # Initialize scaler using reference dataset
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    ref_csv = os.path.join(base_dir, "data", "Mix.csv")
    
    scaler = StandardScaler()
    ref_dataset = SirnaDataset(csv_path=ref_csv, scaler=scaler, fit_scaler=True)
    bio_feats_dim = SirnaDataset.bio_feats_dim
    
    # Create dataset for predictions
    pred_dataset = SirnaDataset(df=df, scaler=scaler, fit_scaler=False)
    
    predictions = []
    
    print(f"Making predictions on {len(df)} sequences using {model_type} model(s)...")
    
    for i in range(len(pred_dataset)):
        x, _ = pred_dataset[i]
        x_input = x.numpy()[None, :]  # Add batch dimension
        
        pred = ensemble_predict(x_input, bio_feats_dim, model_type)
        predictions.append(pred)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(df)} sequences")
    
    # Add predictions to dataframe
    if model_type == 'both':
        df['original_prediction'] = [p['original'] if p['original'] is not None else np.nan for p in predictions]
        df['enhanced_prediction'] = [p['enhanced'] if p['enhanced'] is not None else np.nan for p in predictions]
    else:
        df[f'{model_type}_prediction'] = [p if p is not None else np.nan for p in predictions]
    
    # Save if output file specified
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='RNA-Slasher Inference Tool')
    parser.add_argument('--mode', choices=['interactive', 'file'], default='interactive',
                        help='Inference mode: interactive for single predictions, file for batch predictions')
    parser.add_argument('--model', choices=['original', 'enhanced', 'both'], default='enhanced',
                        help='Model type to use for predictions')
    parser.add_argument('--input', type=str, help='Input CSV file for batch predictions')
    parser.add_argument('--output', type=str, help='Output CSV file for batch predictions')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        # Interactive mode - single sequence prediction
        print("=== RNA-Slasher Interactive Inference ===")
        print(f"Using {args.model} model(s)")
        print()
        
        # Prompt user for siRNA and mRNA sequences
        siRNA = input('Enter siRNA sequence: ').strip().upper()
        mRNA = input('Enter mRNA sequence: ').strip().upper()
        
        # Validate sequences
        valid_nucleotides = set('AUCG')
        if not all(c in valid_nucleotides for c in siRNA):
            print("Warning: siRNA contains invalid nucleotides. Using only A, U, C, G.")
            siRNA = ''.join(c for c in siRNA if c in valid_nucleotides)
        
        if not all(c in valid_nucleotides for c in mRNA):
            print("Warning: mRNA contains invalid nucleotides. Using only A, U, C, G.")
            mRNA = ''.join(c for c in mRNA if c in valid_nucleotides)
        
        print(f"\nProcessing:")
        print(f"siRNA: {siRNA}")
        print(f"mRNA:  {mRNA}")
        print()

        # Extract features
        X_siRNA, X_mRNA, bio_feats = extract_features(siRNA, mRNA)
        
        # Load scaler from training set
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        ref_csv = os.path.join(base_dir, "data", "Mix.csv")
        
        df = pd.read_csv(ref_csv)
        scaler = StandardScaler()
        bio_feats_training = get_bio_feats(df)
        scaler.fit(bio_feats_training)
        
        bio_feats_norm = scaler.transform([bio_feats])[0]
        
        # Prepare input
        X = np.concatenate([X_siRNA, X_mRNA, bio_feats_norm])[None, :]
        
        # Determine number of bio features
        bio_feats_dim = bio_feats_norm.shape[0]
        
        # Make prediction
        pred = ensemble_predict(X, bio_feats_dim, args.model)
        
        # Display results
        print("=== Prediction Results ===")
        if args.model == 'both':
            if pred['original'] is not None:
                print(f"Original Model Efficacy: {pred['original']:.4f}")
            else:
                print("Original Model: Not available")
            
            if pred['enhanced'] is not None:
                print(f"Enhanced Model Efficacy: {pred['enhanced']:.4f}")
            else:
                print("Enhanced Model: Not available")
                
            if pred['original'] is not None and pred['enhanced'] is not None:
                diff = pred['enhanced'] - pred['original']
                print(f"Difference (Enhanced - Original): {diff:+.4f}")
        else:
            if pred is not None:
                print(f"{args.model.title()} Model Efficacy: {pred:.4f}")
            else:
                print(f"{args.model.title()} Model: Not available")
    
    elif args.mode == 'file':
        # File mode - batch predictions
        if not args.input:
            print("Error: --input file must be specified for file mode")
            return
        
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} not found")
            return
        
        try:
            results = predict_from_file(args.input, args.model, args.output)
            print(f"\nBatch prediction completed successfully!")
            print(f"Results shape: {results.shape}")
            
            # Show summary statistics
            if args.model == 'both':
                if 'original_prediction' in results.columns:
                    orig_valid = results['original_prediction'].notna()
                    print(f"Original model predictions: {orig_valid.sum()}/{len(results)} valid")
                    if orig_valid.any():
                        print(f"  Mean: {results.loc[orig_valid, 'original_prediction'].mean():.4f}")
                        print(f"  Std:  {results.loc[orig_valid, 'original_prediction'].std():.4f}")
                
                if 'enhanced_prediction' in results.columns:
                    enh_valid = results['enhanced_prediction'].notna()
                    print(f"Enhanced model predictions: {enh_valid.sum()}/{len(results)} valid")
                    if enh_valid.any():
                        print(f"  Mean: {results.loc[enh_valid, 'enhanced_prediction'].mean():.4f}")
                        print(f"  Std:  {results.loc[enh_valid, 'enhanced_prediction'].std():.4f}")
            else:
                pred_col = f'{args.model}_prediction'
                if pred_col in results.columns:
                    valid_preds = results[pred_col].notna()
                    print(f"{args.model.title()} model predictions: {valid_preds.sum()}/{len(results)} valid")
                    if valid_preds.any():
                        print(f"  Mean: {results.loc[valid_preds, pred_col].mean():.4f}")
                        print(f"  Std:  {results.loc[valid_preds, pred_col].std():.4f}")
                        
        except Exception as e:
            print(f"Error during batch prediction: {str(e)}")
            return

if __name__ == '__main__':
    main()
