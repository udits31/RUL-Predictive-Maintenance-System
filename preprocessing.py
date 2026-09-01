"""
Preprocessing pipeline for NASA CMAPSS data.
Includes feature engineering, scaling, and train/test split.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def drop_constant_sensors(df, threshold=0.01):
    """Drop sensors with near-zero variance (std < threshold)."""
    sensor_cols = [col for col in df.columns if col.startswith('s')]
    stds = df[sensor_cols].std()
    constant_sensors = stds[stds < threshold].index.tolist()
    
    print(f"Dropping {len(constant_sensors)} constant sensors: {constant_sensors}")
    df = df.drop(columns=constant_sensors)
    
    return df, constant_sensors

def engineer_features(df):
    """
    Feature engineering per engine group:
    - Rolling mean and std (windows 5, 10)
    - Lag differences (lags 1, 3)
    - Interaction features
    - Polynomial cycle features
    - Cumulative max degradation proxy
    """
    sensor_cols = [col for col in df.columns if col.startswith('s') and col != 's1']
    
    # Sort by engine and cycle to ensure proper time-series ordering
    df = df.sort_values(['engine_id', 'cycle']).reset_index(drop=True)
    
    new_features = []
    
    print("Engineering features...")
    
    # Rolling features (per engine group)
    for window in [5, 10]:
        print(f"  - Rolling window {window}...")
        for col in sensor_cols:
            # Rolling mean
            df[f'{col}_rolling_mean_{window}'] = df.groupby('engine_id')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            # Rolling std
            df[f'{col}_rolling_std_{window}'] = df.groupby('engine_id')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
            )
            new_features.extend([f'{col}_rolling_mean_{window}', f'{col}_rolling_std_{window}'])
    
    # Lag differences (per engine group)
    for lag in [1, 3]:
        print(f"  - Lag {lag} differences...")
        for col in sensor_cols:
            df[f'{col}_lag_{lag}'] = df.groupby('engine_id')[col].transform(
                lambda x: x.diff(lag).fillna(0)
            )
            new_features.append(f'{col}_lag_{lag}')
    
    # Interaction features
    print("  - Interaction features...")
    if 's3' in df.columns and 's4' in df.columns:
        df['T30_P30_ratio'] = df['s3'] / (df['s4'] + 1e-6)
        new_features.append('T30_P30_ratio')
    
    if 's7' in df.columns and 's2' in df.columns:
        df['T50_P15_ratio'] = df['s7'] / (df['s2'] + 1e-6)
        new_features.append('T50_P15_ratio')
    
    if 's8' in df.columns and 's9' in df.columns:
        df['Nf_Nc_ratio'] = df['s8'] / (df['s9'] + 1e-6)
        new_features.append('Nf_Nc_ratio')
    
    # Polynomial cycle features
    print("  - Polynomial cycle features...")
    df['cycle_squared'] = df['cycle'] ** 2
    df['cycle_log'] = np.log(df['cycle'] + 1)
    new_features.extend(['cycle_squared', 'cycle_log'])
    
    # Cumulative max of T30 (s3) per engine - degradation proxy
    if 's3' in df.columns:
        print("  - Cumulative degradation proxy...")
        df['s3_cummax'] = df.groupby('engine_id')['s3'].cummax()
        new_features.append('s3_cummax')
    
    print(f"✓ Created {len(new_features)} new features")
    
    return df, new_features

def prepare_data(train_path='data/train.csv', test_path='data/test.csv'):
    """
    Complete preprocessing pipeline:
    1. Load data
    2. Drop constant sensors
    3. Engineer features
    4. Scale features
    5. Save processed data and artifacts
    """
    print("="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"   Train: {train_df.shape}, {train_df['engine_id'].nunique()} engines")
    print(f"   Test: {test_df.shape}, {test_df['engine_id'].nunique()} engines")
    
    # Drop constant sensors
    print("\n2. Dropping constant sensors...")
    train_df, constant_sensors = drop_constant_sensors(train_df)
    test_df = test_df.drop(columns=constant_sensors)
    
    # Engineer features
    print("\n3. Engineering features...")
    train_df, new_features = engineer_features(train_df)
    test_df, _ = engineer_features(test_df)
    
    # Identify feature columns (exclude metadata and target)
    exclude_cols = ['engine_id', 'cycle', 'RUL']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    print(f"\n4. Total features: {len(feature_cols)}")
    
    # Scale features
    print("\n5. Scaling features...")
    scaler = StandardScaler()
    
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # Save processed data
    print("\n6. Saving processed data...")
    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/train_processed.csv', index=False)
    test_df.to_csv('data/test_processed.csv', index=False)
    
    # Save artifacts
    print("\n7. Saving preprocessing artifacts...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
    joblib.dump(constant_sensors, 'models/constant_sensors.pkl')
    
    print("\n" + "="*60)
    print("✓ PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Processed train: {train_df.shape}")
    print(f"Processed test: {test_df.shape}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Saved to: data/train_processed.csv, data/test_processed.csv")
    
    return train_df, test_df, feature_cols, scaler

if __name__ == '__main__':
    prepare_data()
