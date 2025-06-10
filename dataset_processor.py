import pandas as pd
import os

# Define file mappings
unprocessed_files = [
    ('data/Hu_unprocessed.csv', 'data/Hu.csv'),
    ('data/Mix_unprocessed.csv', 'data/Mix.csv'),
    ('data/Taka_unprocessed.csv', 'data/Taka.csv'),
]

# Columns to retain
columns_to_keep = ['siRNA', 'mRNA', 'label']

def process_file(input_path, output_path):
    df = pd.read_csv(input_path)
    # Retain only the required columns if they exist
    cols = [col for col in columns_to_keep if col in df.columns]
    df = df[cols]
    df.to_csv(output_path, index=False)
    print(f"Processed {input_path} -> {output_path}")

if __name__ == "__main__":
    for in_file, out_file in unprocessed_files:
        process_file(in_file, out_file)
