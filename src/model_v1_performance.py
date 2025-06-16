import pandas as pd
import numpy as np
import torch
import os
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from model_v1 import ANN, one_hot_encode, SEQ_LEN, MRNA_LEN, SirnaDataset
from bio_features import (
    gc_content, at_content, melting_temp, length, base_frequencies,
    purine_content, pyrimidine_content, molecular_weight, dinucleotide_frequencies, shannon_entropy,
    longest_mononucleotide_run, au_gc_ratio, gc_skew, at_skew, unique_kmers, count_ambiguous
)

def extract_features(siRNA, mRNA):
    """Extract features for model_v1 (same format as in model_v1.py)"""
    X_siRNA = one_hot_encode(siRNA, SEQ_LEN)
    X_mRNA = one_hot_encode(mRNA, MRNA_LEN)
    
    # siRNA biological features
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
    
    # mRNA biological features
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
    
    # Concatenate biological features (same order as in SirnaDataset)
    bio_feats = np.concatenate([
        [siRNA_gc, siRNA_at, siRNA_tm, siRNA_len], siRNA_basefreq,
        [siRNA_pur, siRNA_pyr, siRNA_mw], siRNA_dinuc, 
        [siRNA_entropy, siRNA_run, siRNA_au_gc, siRNA_gcskew, siRNA_atskew, siRNA_kmers2, siRNA_kmers3, siRNA_ambig],
        [mRNA_gc, mRNA_at, mRNA_tm, mRNA_len], mRNA_basefreq,
        [mRNA_pur, mRNA_pyr, mRNA_mw], mRNA_dinuc, 
        [mRNA_entropy, mRNA_run, mRNA_au_gc, mRNA_gcskew, mRNA_atskew, mRNA_kmers2, mRNA_kmers3, mRNA_ambig]
    ])
    
    return X_siRNA, X_mRNA, bio_feats

def get_bio_feats(df):
    """Extract biological features from a dataframe for scaler fitting (matches model_v1 format)"""
    # siRNA features
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
    
    # mRNA features
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

def predict_v1(X, bio_feats_dim):
    """Predict using model_v1"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'model/ann_weights_v1.pth'
    
    if os.path.exists(model_path):
        try:
            model = ANN(bio_feats_dim)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                return model(x_tensor).cpu().item()
        except Exception as e:
            print(f"Warning: Could not load model_v1: {e}")
            return None
    else:
        print("Model_v1 weights not found at model/ann_weights_v1.pth")
        return None

def calculate_metrics(y_true, y_pred, dataset_name):
    """Calculate comprehensive metrics for predictions"""
    # Remove NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(y_true_clean) == 0:
        return None
    
    metrics = {}
    
    # Regression metrics
    metrics['mse'] = mean_squared_error(y_true_clean, y_pred_clean)
    metrics['mae'] = mean_absolute_error(y_true_clean, y_pred_clean)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    try:
        metrics['r2'] = r2_score(y_true_clean, y_pred_clean)
    except:
        metrics['r2'] = np.nan
    
    # Classification metrics (treating 0.5 as threshold)
    y_true_bin = (y_true_clean >= 0.5).astype(int)
    y_pred_bin = (y_pred_clean >= 0.5).astype(int)
    
    try:
        metrics['roc_auc'] = roc_auc_score(y_true_bin, y_pred_clean)
    except:
        metrics['roc_auc'] = np.nan
    
    try:
        metrics['f1'] = f1_score(y_true_bin, y_pred_bin)
    except:
        metrics['f1'] = np.nan
    
    # Additional stats
    metrics['correlation'] = np.corrcoef(y_true_clean, y_pred_clean)[0, 1] if len(y_true_clean) > 1 else np.nan
    metrics['mean_true'] = np.mean(y_true_clean)
    metrics['mean_pred'] = np.mean(y_pred_clean)
    metrics['std_true'] = np.std(y_true_clean)
    metrics['std_pred'] = np.std(y_pred_clean)
    metrics['n_samples'] = len(y_true_clean)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='RNA-Slasher Model_v1 Performance Evaluation')
    parser.add_argument('--datasets', nargs='+', 
                        choices=['Taka', 'Mix', 'Hu'], 
                        default=['Taka', 'Mix', 'Hu'],
                        help='Datasets to evaluate on')
    parser.add_argument('--output-dir', default='model',
                        help='Directory to save prediction results')
    
    args = parser.parse_args()
    
    # Dataset configurations
    dataset_configs = {
        'Taka': ('data/Taka.csv', f'{args.output_dir}/taka_predictions_v1.csv'),
        'Mix': ('data/Mix.csv', f'{args.output_dir}/mix_predictions_v1.csv'),
        'Hu': ('data/Hu.csv', f'{args.output_dir}/hu_predictions_v1.csv'),
    }
    
    # Load scaler from Mix.csv (reference dataset as used in model_v1 training)
    print("Loading reference dataset for scaler fitting...")
    mix_df = pd.read_csv('data/Mix.csv')
    scaler = StandardScaler()
    scaler.fit(get_bio_feats(mix_df))
    
    # Use a sample row from Mix to get feature dimensions
    sample_row = mix_df.iloc[0]
    _, _, sample_bio_feats = extract_features(sample_row['siRNA'], sample_row['mRNA'])
    bio_feats_dim = sample_bio_feats.shape[0]
    
    print(f"Biological features dimension: {bio_feats_dim}")
    print(f"Evaluating model_v1 on datasets: {args.datasets}")
    print()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_metrics = {}
    
    for dataset_name in args.datasets:
        if dataset_name not in dataset_configs:
            print(f"Warning: Unknown dataset {dataset_name}, skipping...")
            continue
            
        in_path, out_path = dataset_configs[dataset_name]
        
        if not os.path.exists(in_path):
            print(f"Warning: Dataset file {in_path} not found, skipping...")
            continue
        
        print(f"Processing {dataset_name} dataset...")
        df = pd.read_csv(in_path)
        results = []
        v1_preds = []
        true_labels = []
        
        for idx, (_, row) in enumerate(df.iterrows()):
            siRNA = row['siRNA']
            mRNA = row['mRNA']
            efficacy = row['label'] if 'label' in row else np.nan
            
            # Extract features
            X_siRNA, X_mRNA, bio_feats = extract_features(siRNA, mRNA)
            bio_feats_norm = scaler.transform([bio_feats])[0]
            X = np.concatenate([X_siRNA, X_mRNA, bio_feats_norm])[None, :]
            
            # Make prediction with model_v1
            v1_pred = predict_v1(X, bio_feats_dim)
            
            result_row = {
                'siRNA': siRNA,
                'mRNA': mRNA,
                'dataset_efficacy': efficacy,
                'v1_prediction': v1_pred
            }
            
            if v1_pred is not None:
                v1_preds.append(v1_pred)
                if not np.isnan(efficacy):
                    true_labels.append(efficacy)
            
            results.append(result_row)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)} sequences")
        
        # Save predictions
        out_df = pd.DataFrame(results)
        out_df.to_csv(out_path, index=False)
        print(f"  Predictions saved to {out_path}")
        
        # Calculate and save metrics
        if len(v1_preds) > 0:
            true_vals = out_df['dataset_efficacy'].values
            v1_vals = out_df['v1_prediction'].values
            v1_metrics = calculate_metrics(true_vals, v1_vals, dataset_name)
            
            if v1_metrics:
                all_metrics[dataset_name] = v1_metrics
                
                # Save detailed metrics
                metrics_file = out_path.replace('.csv', '_metrics.txt')
                with open(metrics_file, 'w') as f:
                    f.write(f"Metrics for {dataset_name} Dataset - Model_v1\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"MODEL_V1:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Samples: {v1_metrics['n_samples']}\n")
                    f.write(f"MSE: {v1_metrics['mse']:.6f}\n")
                    f.write(f"MAE: {v1_metrics['mae']:.6f}\n")
                    f.write(f"RMSE: {v1_metrics['rmse']:.6f}\n")
                    f.write(f"R²: {v1_metrics['r2']:.6f}\n")
                    f.write(f"ROC AUC: {v1_metrics['roc_auc']:.6f}\n")
                    f.write(f"F1 Score: {v1_metrics['f1']:.6f}\n")
                    f.write(f"Correlation: {v1_metrics['correlation']:.6f}\n")
                    f.write(f"Mean True: {v1_metrics['mean_true']:.6f}\n")
                    f.write(f"Mean Predicted: {v1_metrics['mean_pred']:.6f}\n")
                    f.write(f"Std True: {v1_metrics['std_true']:.6f}\n")
                    f.write(f"Std Predicted: {v1_metrics['std_pred']:.6f}\n")
                
                print(f"  Metrics saved to {metrics_file}")
        else:
            print(f"  Warning: No valid predictions generated for {dataset_name}")
        
        print()
    
    # Print summary
    print("=" * 60)
    print("MODEL_V1 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for dataset_name, metrics in all_metrics.items():
        print(f"\n{dataset_name.upper()} DATASET:")
        print("-" * 30)
        print(f"MODEL_V1: MSE={metrics['mse']:.4f}, "
              f"R²={metrics['r2']:.4f}, ROC AUC={metrics['roc_auc']:.4f}, "
              f"F1={metrics['f1']:.4f}")
    
    print(f"\nAll results saved to {args.output_dir}/ directory")
    print("Files generated:")
    for dataset_name in args.datasets:
        if dataset_name in dataset_configs:
            _, out_path = dataset_configs[dataset_name]
            print(f"  - {out_path}")
            print(f"  - {out_path.replace('.csv', '_metrics.txt')}")

if __name__ == '__main__':
    main()
