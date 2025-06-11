import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from model import ANN, one_hot_encode, SEQ_LEN, MRNA_LEN, SirnaDataset
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
        [mRNA_pur, mRNA_pyr, mRNA_mw], mRNA_dinuc, [mRNA_entropy, mRNA_run, mRNA_au_gc, mRNA_gcskew, mRNA_atskew, mRNA_kmers2, mRNA_kmers3, mRNA_ambig]
    ])
    return X_siRNA, X_mRNA, bio_feats

def ensemble_predict(X, bio_feats_dim):
    preds = []
    for fold in range(1, 6):
        model = ANN(bio_feats_dim)
        model.load_state_dict(torch.load(f'model/ann_weights_fold{fold}.pth', map_location='cpu'))
        model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32)
            pred = model(x_tensor).item()
            preds.append(pred)
    return np.mean(preds)

def main():
    # Prompt user for siRNA and mRNA sequences
    siRNA = input('Enter siRNA sequence: ').strip()
    mRNA = input('Enter mRNA sequence: ').strip()

    # Extract features
    X_siRNA, X_mRNA, bio_feats = extract_features(siRNA, mRNA)
    # Load scaler from training set
    df = pd.read_csv('data/Mix.csv')
    scaler = StandardScaler()
    def get_bio_feats(df):
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
    scaler.fit(get_bio_feats(df))
    bio_feats_norm = scaler.transform([bio_feats])[0]
    # Prepare input
    X = np.concatenate([X_siRNA, X_mRNA, bio_feats_norm])[None, :]
    # Determine number of bio features (matches model.py logic)
    bio_feats_dim = bio_feats_norm.shape[0]
    # Load model (ensemble of 5 folds)
    pred = ensemble_predict(X, bio_feats_dim)
    print(f'Ensemble predicted efficacy: {pred:.4f}')

if __name__ == '__main__':
    main()
