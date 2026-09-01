"""
Generate CSV files from raw NASA CMAPSS data with proper column names.
"""
import pandas as pd
import numpy as np

# Define column names based on CMAPSS dataset structure
COLUMN_NAMES = ['engine_id', 'cycle', 'op1', 'op2', 'op3'] + \
               [f's{i}' for i in range(1, 22)]

def load_raw_data(filepath):
    """Load raw space-separated data and assign column names."""
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=COLUMN_NAMES)
    return df

def generate_rul_target(df):
    """
    Generate RUL (Remaining Useful Life) target.
    RUL = max_cycle - current_cycle, clipped at 130.
    """
    # Calculate max cycle per engine
    max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    
    # Merge and calculate RUL
    df = df.merge(max_cycles, on='engine_id', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    
    # Clip RUL at 130 (piecewise linear target - standard in literature)
    df['RUL'] = df['RUL'].clip(upper=130)
    
    # Drop max_cycle helper column
    df = df.drop('max_cycle', axis=1)
    
    return df

def main():
    print("Loading raw training data...")
    train_df = load_raw_data('data/train_FD001.txt')
    
    print("Loading raw test data...")
    test_df = load_raw_data('data/test_FD001.txt')
    
    print("Generating RUL targets for training data...")
    train_df = generate_rul_target(train_df)
    
    # For test data, we need to load the RUL ground truth
    print("Loading test RUL ground truth...")
    test_rul = pd.read_csv('data/RUL_FD001.txt', header=None, names=['true_RUL'])
    test_rul['engine_id'] = test_rul.index + 1
    
    # For test data, calculate RUL based on the last cycle + true_RUL
    test_max_cycles = test_df.groupby('engine_id')['cycle'].max().reset_index()
    test_max_cycles.columns = ['engine_id', 'last_cycle']
    test_max_cycles = test_max_cycles.merge(test_rul, on='engine_id')
    test_max_cycles['max_cycle'] = test_max_cycles['last_cycle'] + test_max_cycles['true_RUL']
    
    test_df = test_df.merge(test_max_cycles[['engine_id', 'max_cycle']], on='engine_id')
    test_df['RUL'] = test_df['max_cycle'] - test_df['cycle']
    test_df['RUL'] = test_df['RUL'].clip(upper=130)
    test_df = test_df.drop('max_cycle', axis=1)
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"\nTrain RUL stats:\n{train_df['RUL'].describe()}")
    print(f"\nTest RUL stats:\n{test_df['RUL'].describe()}")
    
    # Save to CSV
    print("\nSaving processed data...")
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print("✓ Data generation complete!")
    print(f"  - data/train.csv: {len(train_df)} rows, {len(train_df['engine_id'].unique())} engines")
    print(f"  - data/test.csv: {len(test_df)} rows, {len(test_df['engine_id'].unique())} engines")

if __name__ == '__main__':
    main()
