"""
Train and evaluate multiple models for RUL prediction.
Models: Ridge, Random Forest, XGBoost, LSTM
"""
import pandas as pd
import numpy as np
import joblib
import json
import time
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import shap
import os

def load_data():
    """Load preprocessed data."""
    train_df = pd.read_csv('data/train_processed.csv')
    test_df = pd.read_csv('data/test_processed.csv')
    feature_cols = joblib.load('models/feature_cols.pkl')
    
    X_train = train_df[feature_cols].values
    y_train = train_df['RUL'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['RUL'].values
    
    return X_train, y_train, X_test, y_test, train_df, test_df, feature_cols

def evaluate_model(y_true, y_pred, model_name):
    """Calculate evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R²:   {r2:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def train_ridge(X_train, y_train, X_test, y_test):
    """Train Ridge Regression."""
    print("\n" + "="*60)
    print("Training Ridge Regression")
    print("="*60)
    
    start_time = time.time()
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "Ridge Regression")
    metrics['train_time'] = train_time
    
    joblib.dump(model, 'models/ridge.pkl')
    print(f"✓ Model saved to models/ridge.pkl")
    
    return model, metrics, y_pred

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest."""
    print("\n" + "="*60)
    print("Training Random Forest")
    print("="*60)
    
    # Set numpy seed for reproducibility
    np.random.seed(42)
    
    start_time = time.time()
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "Random Forest")
    metrics['train_time'] = train_time
    
    joblib.dump(model, 'models/random_forest.pkl')
    print(f"✓ Model saved to models/random_forest.pkl")
    
    return model, metrics, y_pred

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with early stopping."""
    print("\n" + "="*60)
    print("Training XGBoost")
    print("="*60)
    
    # Split train into train/val for early stopping
    val_size = int(0.2 * len(X_train))
    X_train_fit, X_val = X_train[:-val_size], X_train[-val_size:]
    y_train_fit, y_val = y_train[:-val_size], y_train[-val_size:]
    
    start_time = time.time()
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "XGBoost")
    metrics['train_time'] = train_time
    
    joblib.dump(model, 'models/xgboost.pkl')
    print(f"✓ Model saved to models/xgboost.pkl")
    
    return model, metrics, y_pred

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting Regressor."""
    from sklearn.ensemble import GradientBoostingRegressor
    
    print("\n" + "="*60)
    print("Training Gradient Boosting")
    print("="*60)
    
    start_time = time.time()
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "Gradient Boosting")
    metrics['train_time'] = train_time
    
    joblib.dump(model, 'models/gradient_boosting.pkl')
    print(f"✓ Model saved to models/gradient_boosting.pkl")
    
    return model, metrics, y_pred

def create_sequences(X, y, engine_ids, seq_length=30):
    """Create sequences for LSTM training."""
    sequences = []
    targets = []
    
    unique_engines = np.unique(engine_ids)
    
    for engine_id in unique_engines:
        engine_mask = engine_ids == engine_id
        engine_X = X[engine_mask]
        engine_y = y[engine_mask]
        
        # Skip engines with insufficient data
        if len(engine_X) < seq_length:
            continue
        
        # Create sequences
        for i in range(len(engine_X) - seq_length + 1):
            sequences.append(engine_X[i:i+seq_length])
            targets.append(engine_y[i+seq_length-1])
    
    return np.array(sequences), np.array(targets)

def train_lstm(X_train, y_train, X_test, y_test, train_df, test_df, feature_cols):
    """Train LSTM model."""
    print("\n" + "="*60)
    print("Training LSTM")
    print("="*60)
    
    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    seq_length = 30
    
    # Create sequences
    print(f"Creating sequences (length={seq_length})...")
    train_engine_ids = train_df['engine_id'].values
    test_engine_ids = test_df['engine_id'].values
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, train_engine_ids, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, test_engine_ids, seq_length)
    
    print(f"  Train sequences: {X_train_seq.shape}")
    print(f"  Test sequences: {X_test_seq.shape}")
    
    # Build LSTM model
    print("\nBuilding LSTM architecture...")
    model = keras.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=(seq_length, len(feature_cols))),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.LSTM(64, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.LSTM(32),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    print(model.summary())
    
    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    
    # Train
    print("\nTraining LSTM...")
    start_time = time.time()
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = model.predict(X_test_seq, verbose=0).flatten()
    metrics = evaluate_model(y_test_seq, y_pred, "LSTM")
    metrics['train_time'] = train_time
    
    model.save('models/lstm_model.keras')
    print(f"✓ Model saved to models/lstm_model.keras")
    
    return model, metrics, y_pred

def compute_shap_values(model, X_test, feature_cols):
    """Compute SHAP values for XGBoost model."""
    print("\n" + "="*60)
    print("Computing SHAP Values")
    print("="*60)
    
    # Use subset for SHAP computation
    sample_size = min(200, len(X_test))
    X_sample = X_test[:sample_size]
    
    print(f"Computing SHAP values on {sample_size} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Calculate mean absolute SHAP values per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create feature importance dataframe
    shap_df = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    print("\nTop 15 Features by SHAP Importance:")
    print(shap_df.head(15).to_string(index=False))
    
    # Save SHAP results
    shap_results = {
        'top_features': shap_df.head(15).to_dict('records'),
        'all_features': shap_df.to_dict('records')
    }
    joblib.dump(shap_results, 'models/shap_results.pkl')
    print(f"\n✓ SHAP results saved to models/shap_results.pkl")
    
    return shap_results

def main():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    print("\nLoading preprocessed data...")
    X_train, y_train, X_test, y_test, train_df, test_df, feature_cols = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features: {len(feature_cols)}")
    
    results = {}
    predictions = {}
    
    # Train Ridge
    _, metrics, y_pred = train_ridge(X_train, y_train, X_test, y_test)
    results['ridge'] = metrics
    predictions['ridge'] = y_pred.tolist()
    
    # Train Random Forest
    rf_model, metrics, y_pred = train_random_forest(X_train, y_train, X_test, y_test)
    results['random_forest'] = metrics
    predictions['random_forest'] = y_pred.tolist()
    
    # Train XGBoost
    xgb_model, metrics, y_pred = train_xgboost(X_train, y_train, X_test, y_test)
    results['xgboost'] = metrics
    predictions['xgboost'] = y_pred.tolist()
    
    # Train Gradient Boosting
    gb_model, metrics, y_pred = train_gradient_boosting(X_train, y_train, X_test, y_test)
    results['gradient_boosting'] = metrics
    predictions['gradient_boosting'] = y_pred.tolist()
    
    # Create Ensemble (average of RF, XGB, GB)
    print("\n" + "="*60)
    print("Creating Ensemble Model")
    print("="*60)
    ensemble_pred = (
        np.array(predictions['random_forest']) + 
        np.array(predictions['xgboost']) + 
        np.array(predictions['gradient_boosting'])
    ) / 3.0
    ensemble_metrics = evaluate_model(y_test, ensemble_pred, "Ensemble")
    results['ensemble'] = ensemble_metrics
    predictions['ensemble'] = ensemble_pred.tolist()
    
    # Train LSTM
    _, metrics, y_pred_lstm = train_lstm(X_train, y_train, X_test, y_test, train_df, test_df, feature_cols)
    results['lstm'] = metrics
    # Note: LSTM predictions are on sequences, not full test set
    
    # Compute SHAP values using Random Forest model
    shap_results = compute_shap_values(rf_model, X_test, feature_cols)
    
    # Save results
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    with open('models/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Results saved to models/results.json")
    
    # Save test predictions
    test_predictions_df = test_df[['engine_id', 'cycle', 'RUL']].copy()
    test_predictions_df['ridge_pred'] = predictions['ridge']
    test_predictions_df['rf_pred'] = predictions['random_forest']
    test_predictions_df['xgb_pred'] = predictions['xgboost']
    test_predictions_df['gb_pred'] = predictions['gradient_boosting']
    test_predictions_df['ensemble_pred'] = predictions['ensemble']
    test_predictions_df.to_csv('data/test_predictions.csv', index=False)
    print("✓ Test predictions saved to data/test_predictions.csv")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'Time (s)':<10}")
    print("-"*60)
    for model_name, metrics in results.items():
        train_time = metrics.get('train_time', 0.0)  # Default to 0 if not present (ensemble)
        print(f"{model_name:<20} {metrics['rmse']:<10.2f} {metrics['mae']:<10.2f} "
              f"{metrics['r2']:<10.4f} {train_time:<10.2f}")
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    print(f"\n✓ Best model by RMSE: {best_model[0].upper()} ({best_model[1]['rmse']:.2f})")

if __name__ == '__main__':
    main()
