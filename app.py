"""
Flask REST API for predictive maintenance system.
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Load models and artifacts
print("Loading models and artifacts...")
models = {
    'ridge': joblib.load('models/ridge.pkl'),
    'random_forest': joblib.load('models/random_forest.pkl'),
    'xgboost': joblib.load('models/xgboost.pkl')
}
scaler = joblib.load('models/scaler.pkl')
feature_cols = joblib.load('models/feature_cols.pkl')
constant_sensors = joblib.load('models/constant_sensors.pkl')

with open('models/results.json', 'r') as f:
    model_metrics = json.load(f)

shap_results = joblib.load('models/shap_results.pkl')
test_df = pd.read_csv('data/test_processed.csv')
test_predictions = pd.read_csv('data/test_predictions.csv')

print("✓ All models and data loaded successfully")

# Nominal sensor values for defaults
NOMINAL_SENSORS = {
    's2': 642.0, 's3': 1590.0, 's4': 1400.0, 's6': 14.6, 's7': 21.6,
    's8': 554.0, 's9': 2388.0, 's11': 47.5, 's12': 522.0, 's13': 2388.0,
    's14': 8140.0, 's15': 8.4, 's17': 392.0, 's20': 39.0, 's21': 23.4
}

def get_risk_level(rul):
    """Determine risk level based on RUL."""
    if rul < 30:
        return 'CRITICAL'
    elif rul < 60:
        return 'WARNING'
    elif rul < 100:
        return 'CAUTION'
    else:
        return 'HEALTHY'

def get_recommendation(rul, risk_level):
    """Generate maintenance recommendation."""
    if risk_level == 'CRITICAL':
        return 'Immediate maintenance required. Ground aircraft.'
    elif risk_level == 'WARNING':
        return 'Schedule maintenance within next 30 cycles.'
    elif risk_level == 'CAUTION':
        return 'Plan maintenance within next 60 cycles.'
    else:
        return 'Continue normal operations. Monitor regularly.'

def preprocess_input(sensors, op_settings, cycle):
    """
    Preprocess single input for prediction.
    Apply same transformations as training data.
    """
    # Fill missing sensors with nominal values
    for sensor, default_val in NOMINAL_SENSORS.items():
        if sensor not in sensors:
            sensors[sensor] = default_val
    
    # Remove constant sensors
    for sensor in constant_sensors:
        sensors.pop(sensor, None)
    
    # Create base dataframe
    data = {
        'cycle': cycle,
        'op1': op_settings.get('op1', 0.0),
        'op2': op_settings.get('op2', 0.0),
        'op3': op_settings.get('op3', 100.0),
        **sensors
    }
    
    df = pd.DataFrame([data])
    
    # For single prediction, we can't compute rolling/lag features properly
    # Use simplified feature set or assume features are pre-computed
    # For now, fill missing features with zeros
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    # Ensure column order matches training
    df = df[feature_cols]
    
    return df.values

@app.route('/health', methods=['GET'])
def health():
    """Service health check."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': list(models.keys()),
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict RUL for given sensor readings.
    
    Request body:
    {
        "sensors": {"s3": 1610.0, "s9": 8950.0, ...},
        "op1": 0.0, "op2": 0.0, "op3": 100.0,
        "cycle": 150,
        "model": "random_forest"  # optional, defaults to random_forest
    }
    """
    try:
        data = request.json
        
        sensors = data.get('sensors', {})
        op_settings = {
            'op1': data.get('op1', 0.0),
            'op2': data.get('op2', 0.0),
            'op3': data.get('op3', 100.0)
        }
        cycle = data.get('cycle', 1)
        model_name = data.get('model', 'random_forest')
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 400
        
        # Preprocess input
        X = preprocess_input(sensors, op_settings, cycle)
        
        # Predict
        model = models[model_name]
        rul_pred = float(model.predict(X)[0])
        
        # Clamp to [0, 130]
        rul_pred = np.clip(rul_pred, 0, 130)
        
        # Determine risk and recommendation
        risk_level = get_risk_level(rul_pred)
        alert = risk_level == 'CRITICAL'
        recommendation = get_recommendation(rul_pred, risk_level)
        
        return jsonify({
            'predicted_rul': round(rul_pred, 2),
            'risk_level': risk_level,
            'alert': alert,
            'recommendation': recommendation,
            'model': model_name,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_models():
    """Get all trained model metrics."""
    return jsonify({
        'models': model_metrics,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/simulate/<int:engine_id>', methods=['GET'])
def simulate(engine_id):
    """
    Simulate full lifecycle for a test engine.
    Returns cycle-by-cycle predictions with risk levels.
    """
    try:
        # Get engine data
        engine_data = test_predictions[test_predictions['engine_id'] == engine_id]
        
        if engine_data.empty:
            return jsonify({'error': f'Engine {engine_id} not found'}), 404
        
        # Use random forest predictions (best model typically)
        model_col = 'rf_pred'
        
        # Build simulation results
        results = []
        alert_threshold = request.args.get('threshold', 30, type=int)
        
        for _, row in engine_data.iterrows():
            cycle = int(row['cycle'])
            actual_rul = float(row['RUL'])
            predicted_rul = float(row[model_col])
            
            risk_level = get_risk_level(predicted_rul)
            alert = predicted_rul < alert_threshold
            
            results.append({
                'cycle': cycle,
                'actual_rul': round(actual_rul, 2),
                'predicted_rul': round(predicted_rul, 2),
                'risk_level': risk_level,
                'alert': alert
            })
        
        return jsonify({
            'engine_id': engine_id,
            'total_cycles': len(results),
            'alert_threshold': alert_threshold,
            'simulation': results,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/shap', methods=['GET'])
def get_shap():
    """Get SHAP feature importances."""
    return jsonify({
        'top_features': shap_results['top_features'][:15],
        'model': 'xgboost',
        'samples_analyzed': 200,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/fleet', methods=['GET'])
def get_fleet():
    """
    Get fleet-wide health summary.
    Returns statistics across all test engines.
    """
    try:
        # Get latest cycle per engine
        latest_cycles = test_predictions.groupby('engine_id').last().reset_index()
        
        # Use random forest predictions
        latest_cycles['predicted_rul'] = latest_cycles['rf_pred']
        latest_cycles['risk_level'] = latest_cycles['predicted_rul'].apply(get_risk_level)
        
        # Count by risk level
        risk_counts = latest_cycles['risk_level'].value_counts().to_dict()
        
        # Engine summaries
        engines = []
        for _, row in latest_cycles.iterrows():
            engines.append({
                'engine_id': int(row['engine_id']),
                'predicted_rul': round(float(row['predicted_rul']), 2),
                'actual_rul': round(float(row['RUL']), 2),
                'risk_level': row['risk_level'],
                'last_cycle': int(row['cycle'])
            })
        
        return jsonify({
            'total_engines': len(engines),
            'risk_summary': {
                'critical': risk_counts.get('CRITICAL', 0),
                'warning': risk_counts.get('WARNING', 0),
                'caution': risk_counts.get('CAUTION', 0),
                'healthy': risk_counts.get('HEALTHY', 0)
            },
            'engines': engines,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Predictive Maintenance API")
    print("="*60)
    print("Available endpoints:")
    print("  GET  /health")
    print("  POST /predict")
    print("  GET  /models")
    print("  GET  /simulate/<engine_id>")
    print("  GET  /shap")
    print("  GET  /fleet")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
