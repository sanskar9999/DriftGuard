# src/preprocessing.py

import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

# Define the Universal Feature Set we identified from the census.
# These are the raw features we trust and will use as a base.
UNIVERSAL_FEATURES = [
    'BIS/BIS', 'BIS/EMG', 'BIS/SEF', 'BIS/SR', 'Orchestra/PPF20_VOL',
    'Solar8000/HR', 'Solar8000/ART_MBP', 'Solar8000/RR_CO2',
    'Tension_Index', 'Reversion_Pressure', 'patient_id'
]

# Define the final set of features the model will use after engineering.
FINAL_MODEL_FEATURES = [
    'BIS/BIS', 'BIS/EMG', 'BIS/SEF', 'BIS/SR', 'Solar8000/HR',
    'Solar8000/ART_MBP', 'Orchestra/PPF20_VOL', 'Tension_Index',
    'Reversion_Pressure', 'HR_volatility_300s', 'MBP_volatility_300s',
    'BIS_grad_1min', 'MBP_grad_1min', 'Shock_Index_Proxy',
    'BIS_per_PPF_rate', 'patient_id'
]

def engineer_features_for_batch(df):
    """
    Takes a DataFrame for a single batch and returns a new DataFrame
    with the final engineered features.
    """
    # --- Phase 1: Select only Universal Features ---
    features_to_load = [f for f in UNIVERSAL_FEATURES if f in df.columns]
    df_universal = df[features_to_load].copy()

    # --- Phase 2: Intelligent Imputation (per patient) ---
    df_universal = df_universal.groupby('patient_id').transform(lambda x: x.ffill().bfill())
    df_universal.fillna(0, inplace=True)

    # --- Phase 3: Create Advanced, Domain-Informed Features ---
    df_universal['HR_volatility_300s'] = df_universal['Solar8000/HR'].rolling(window=300, min_periods=1).std()
    df_universal['MBP_volatility_300s'] = df_universal['Solar8000/ART_MBP'].rolling(window=300, min_periods=1).std()
    df_universal['BIS_grad_1min'] = df_universal['BIS/BIS'].diff(periods=60)
    df_universal['MBP_grad_1min'] = df_universal['Solar8000/ART_MBP'].diff(periods=60)
    
    epsilon = 1e-6
    df_universal['Shock_Index_Proxy'] = df_universal['Solar8000/HR'] / (df_universal['Solar8000/ART_MBP'] + epsilon)
    
    ppf_rate = df_universal['Orchestra/PPF20_VOL'].diff()
    df_universal['BIS_per_PPF_rate'] = df_universal['BIS/BIS'].diff() / (ppf_rate + epsilon)
    
    df_universal.replace([np.inf, -np.inf], 0, inplace=True)
    df_universal.fillna(0, inplace=True)

    # Re-add the patient_id from the original dataframe
    df_universal['patient_id'] = df['patient_id']
    
    # --- Phase 4: Select Final Feature Set ---
    final_df = df_universal[FINAL_MODEL_FEATURES].copy()
    
    return final_df

def process_all_batches(source_dir, dest_dir):
    """
    Main function to run the entire feature engineering pipeline.
    """
    os.makedirs(dest_dir, exist_ok=True)
    source_files = sorted(glob.glob(os.path.join(source_dir, 'batch_*.parquet')))
    
    if not source_files:
        print(f"ERROR: No source files found in {source_dir}")
        return

    print(f"Starting feature engineering pipeline for {len(source_files)} files...")
    
    for file_path in tqdm(source_files, desc="Engineering features"):
        try:
            df = pd.read_parquet(file_path).reset_index(drop=True)
            
            engineered_df = engineer_features_for_batch(df)
            
            base_filename = os.path.basename(file_path)
            dest_path = os.path.join(dest_dir, base_filename)
            engineered_df.to_parquet(dest_path, index=False)

        except Exception as e:
            print(f"\nERROR processing file {file_path}: {e}")
            
    print(f"\n--- FEATURE ENGINEERING COMPLETE ---")
    print(f"--- All engineered feature files have been saved to: {dest_dir} ---")

if __name__ == '__main__':
    # This allows the script to be run from the command line.
    # Example: python src/preprocessing.py /path/to/preprocessed_batches /path/to/featured_batches
    import sys
    if len(sys.argv) != 3:
        print("Usage: python preprocessing.py <source_directory> <destination_directory>")
    else:
        source_directory = sys.argv[1]
        destination_directory = sys.argv[2]
        process_all_batches(source_directory, destination_directory)
