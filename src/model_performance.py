import pandas as pd
import numpy as np
import torch
import os
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from model import ANN, one_hot_encode, SEQ_LEN, MRNA_LEN, SirnaDataset
from model_v2 import EnhancedANN, SirnaDataset as EnhancedSirnaDataset
from bio_features import (
    gc_content, at_content, melting_temp, length, base_frequencies,
    purine_content, pyrimidine_content, molecular_weight, dinucleotide_frequencies, shannon_entropy,
    longest_mononucleotide_run, au_gc_ratio, gc_skew, at_skew, unique_kmers, count_ambiguous
)

def extract_features(siRNA, mRNA):
    X_siRNA = one_hot_encode(siRNA, SEQ_LEN)
    X_mRNA = one_hot_encode(mRNA, MRNA_LEN)
    siRNA_gcx = gc_content(siRNA)
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
        [mRNA_pur, mRNA_pyr, mRNA_mw], mRNA_dinuc, [mRNA_entropy, mRNA_run, mRNA_au_gc, mRNA_gcskew, mRNA_atskew, mRNA_kmers2, mRNA_kmers3, mRNA_ambig]
    ])
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

def ensemble_predict_original(X, bio_feats_dim):
    """Predict using ensemble of original models"""
    preds = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for fold in range(1, 6):
        model_path = f'model/ann_weights_fold{fold}.pth'
        if os.path.exists(model_path):
            try:
                model = ANN(bio_feats_dim)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                with torch.no_grad():
                    x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                    pred = model(x_tensor).cpu().item()
                    preds.append(pred)
            except Exception as e:
                print(f"Warning: Could not load original model fold {fold}: {e}")
                continue
    
    return np.mean(preds) if preds else None

def predict_enhanced(X, bio_feats_dim):
    """Predict using enhanced model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'model/enhanced_ann_weights_v2.pth'
    
    if os.path.exists(model_path):
        try:
            model = EnhancedANN(bio_feats_dim)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                return model(x_tensor).cpu().item()
        except Exception as e:
            print(f"Warning: Could not load enhanced model: {e}")
            return None
    else:
        print("Enhanced model weights not found")
        return None

def calculate_metrics(y_true, y_pred, dataset_name, model_name):
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
    parser = argparse.ArgumentParser(description='RNA-Slasher Model Performance Evaluation')
    parser.add_argument('--model', choices=['original', 'enhanced', 'both'], default='both',
                        help='Model type to evaluate: original, enhanced, or both')
    parser.add_argument('--datasets', nargs='+', 
                        choices=['Taka', 'Mix', 'Hu'], 
                        default=['Taka', 'Mix', 'Hu'],
                        help='Datasets to evaluate on')
    parser.add_argument('--output-dir', default='model',
                        help='Directory to save prediction results')
    
    args = parser.parse_args()
    
    # Dataset configurations
    dataset_configs = {
        'Taka': ('data/Taka.csv', f'{args.output_dir}/taka_predictions.csv'),
        'Mix': ('data/Mix.csv', f'{args.output_dir}/mix_predictions.csv'),
        'Hu': ('data/Hu.csv', f'{args.output_dir}/hu_predictions.csv'),
    }
    
    # Load scaler from Mix.csv (reference dataset)
    print("Loading reference dataset for scaler fitting...")
    mix_df = pd.read_csv('data/Mix.csv')
    scaler = StandardScaler()
    scaler.fit(get_bio_feats(mix_df))
    
    # Use a sample row from Mix to get feature dimensions
    sample_row = mix_df.iloc[0]
    _, _, sample_bio_feats = extract_features(sample_row['siRNA'], sample_row['mRNA'])
    bio_feats_dim = sample_bio_feats.shape[0]
    
    print(f"Biological features dimension: {bio_feats_dim}")
    print(f"Evaluating model(s): {args.model}")
    print(f"Datasets: {args.datasets}")
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
        original_preds = []
        enhanced_preds = []
        true_labels = []
        
        for idx, (_, row) in enumerate(df.iterrows()):
            siRNA = row['siRNA']
            mRNA = row['mRNA']
            efficacy = row['label'] if 'label' in row else np.nan
            
            # Extract features
            X_siRNA, X_mRNA, bio_feats = extract_features(siRNA, mRNA)
            bio_feats_norm = scaler.transform([bio_feats])[0]
            X = np.concatenate([X_siRNA, X_mRNA, bio_feats_norm])[None, :]
            
            # Make predictions
            result_row = {
                'siRNA': siRNA,
                'mRNA': mRNA,
                'dataset_efficacy': efficacy
            }
            
            if args.model in ['original', 'both']:
                original_pred = ensemble_predict_original(X, bio_feats_dim)
                result_row['original_prediction'] = original_pred
                if original_pred is not None:
                    original_preds.append(original_pred)
                    if not np.isnan(efficacy):
                        true_labels.append(efficacy)
            
            if args.model in ['enhanced', 'both']:
                enhanced_pred = predict_enhanced(X, bio_feats_dim)
                result_row['enhanced_prediction'] = enhanced_pred
                if enhanced_pred is not None:
                    enhanced_preds.append(enhanced_pred)
            
            results.append(result_row)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)} sequences")
        
        # Save predictions
        out_df = pd.DataFrame(results)
        out_df.to_csv(out_path, index=False)
        print(f"  Predictions saved to {out_path}")
        
        # Calculate and save metrics
        dataset_metrics = {}
        
        if args.model in ['original', 'both'] and len(original_preds) > 0:
            true_vals = out_df['dataset_efficacy'].values
            orig_vals = out_df['original_prediction'].values
            orig_metrics = calculate_metrics(true_vals, orig_vals, dataset_name, 'original')
            if orig_metrics:
                dataset_metrics['original'] = orig_metrics
        
        if args.model in ['enhanced', 'both'] and len(enhanced_preds) > 0:
            true_vals = out_df['dataset_efficacy'].values
            enh_vals = out_df['enhanced_prediction'].values
            enh_metrics = calculate_metrics(true_vals, enh_vals, dataset_name, 'enhanced')
            if enh_metrics:
                dataset_metrics['enhanced'] = enh_metrics
        
        all_metrics[dataset_name] = dataset_metrics
        
        # Save detailed metrics
        metrics_file = out_path.replace('.csv', '_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Metrics for {dataset_name} Dataset\n")
            f.write("=" * 50 + "\n\n")
            
            for model_type, metrics in dataset_metrics.items():
                f.write(f"{model_type.upper()} MODEL:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Samples: {metrics['n_samples']}\n")
                f.write(f"MSE: {metrics['mse']:.6f}\n")
                f.write(f"MAE: {metrics['mae']:.6f}\n")
                f.write(f"RMSE: {metrics['rmse']:.6f}\n")
                f.write(f"R²: {metrics['r2']:.6f}\n")
                f.write(f"ROC AUC: {metrics['roc_auc']:.6f}\n")
                f.write(f"F1 Score: {metrics['f1']:.6f}\n")
                f.write(f"Correlation: {metrics['correlation']:.6f}\n")
                f.write(f"Mean True: {metrics['mean_true']:.6f}\n")
                f.write(f"Mean Predicted: {metrics['mean_pred']:.6f}\n")
                f.write(f"Std True: {metrics['std_true']:.6f}\n")
                f.write(f"Std Predicted: {metrics['std_pred']:.6f}\n")
                f.write("\n")
        
        print(f"  Metrics saved to {metrics_file}")
        print()
    
    # Print summary
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for dataset_name, dataset_metrics in all_metrics.items():
        print(f"\n{dataset_name.upper()} DATASET:")
        print("-" * 30)
        
        for model_type, metrics in dataset_metrics.items():
            print(f"{model_type.upper()}: MSE={metrics['mse']:.4f}, "
                  f"R²={metrics['r2']:.4f}, ROC AUC={metrics['roc_auc']:.4f}, "
                  f"F1={metrics['f1']:.4f}")
    
    # Compare models if both were evaluated
    if args.model == 'both':
        print(f"\nMODEL COMPARISON:")
        print("-" * 30)
        
        for dataset_name, dataset_metrics in all_metrics.items():
            if 'original' in dataset_metrics and 'enhanced' in dataset_metrics:
                orig = dataset_metrics['original']
                enh = dataset_metrics['enhanced']
                
                mse_improvement = ((orig['mse'] - enh['mse']) / orig['mse']) * 100
                r2_improvement = ((enh['r2'] - orig['r2']) / abs(orig['r2'])) * 100 if orig['r2'] != 0 else 0
                auc_improvement = ((enh['roc_auc'] - orig['roc_auc']) / orig['roc_auc']) * 100 if not np.isnan(orig['roc_auc']) else 0
                
                print(f"{dataset_name}: MSE {mse_improvement:+.1f}%, "
                      f"R² {r2_improvement:+.1f}%, ROC AUC {auc_improvement:+.1f}%")
    
    print(f"\nAll results saved to {args.output_dir}/ directory")

if __name__ == '__main__':
    main()
