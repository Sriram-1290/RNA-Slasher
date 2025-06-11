import os
import glob
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

# Directory containing the prediction CSVs
dir_path = os.path.join('model')

# Find all *_predictions.csv files
csv_files = glob.glob(os.path.join(dir_path, '*_predictions.csv'))

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    y_true = df['dataset efficacy'].values
    y_pred = df['predicted efficacy'].values

    # Regression metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Binarize for classification metrics (threshold = 0.5)
    y_true_bin = (y_true >= 0.5).astype(int)
    y_pred_bin = (y_pred >= 0.5).astype(int)

    f1 = f1_score(y_true_bin, y_pred_bin)
    try:
        roc = roc_auc_score(y_true_bin, y_pred)
    except Exception:
        roc = 'N/A'
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin)
    rec = recall_score(y_true_bin, y_pred_bin)

    # Output file
    base = os.path.splitext(os.path.basename(csv_file))[0]
    out_file = os.path.join(dir_path, f'{base}_metrics.txt')
    with open(out_file, 'w') as f:
        f.write(f'Metrics for {csv_file}\n')
        f.write(f'MSE: {mse:.4f}\n')
        f.write(f'MAE: {mae:.4f}\n')
        f.write(f'R2: {r2:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')
        f.write(f'ROC AUC: {roc}\n')
        f.write(f'Accuracy: {acc:.4f}\n')
        f.write(f'Precision: {prec:.4f}\n')
        f.write(f'Recall: {rec:.4f}\n')
    print(f'Wrote metrics to {out_file}')
