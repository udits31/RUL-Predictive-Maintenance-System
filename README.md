# RUL Predictive Maintenance System

Full-stack predictive maintenance application using the NASA CMAPSS turbofan engine dataset to estimate Remaining Useful Life (RUL) of jet engines.

## Features

- **Complete ML Pipeline**: Ridge, Random Forest, XGBoost, and LSTM models
- **Feature Engineering**: Rolling statistics, lag features, interaction terms, and degradation proxies
- **SHAP Explainability**: Feature importance analysis using SHAP values
- **REST API**: Flask backend with prediction, simulation, and fleet monitoring endpoints
- **Interactive Dashboard**: Real-time visualization with 4 comprehensive tabs
- **Professional Design**: IBM Plex typography with industrial monitoring aesthetic

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
python run_pipeline.py
```

This executes all steps automatically:
- Data generation (converts raw NASA data to CSV)
- Preprocessing (feature engineering, scaling)
- Model training (Ridge, RF, XGBoost, LSTM)

Total time: ~10-15 minutes depending on hardware.

### 3. Start API Server

```bash
python app.py
```

The Flask API will start on `http://localhost:5000`

### 4. Open Dashboard

Open `frontend/index.html` in your browser. The dashboard will automatically connect to the API.

## Dashboard Features

### Tab 1: Model Comparison
- Performance metrics for all 4 models
- Radar chart showing model tradeoffs
- RMSE vs MAE comparison
- Detailed metrics table with best model highlighted

### Tab 2: Feature Importance
- SHAP values for top 10 features
- Sensor descriptions with physical meanings
- Explanation of rolling feature dominance
- Color-coded by sensor type

### Tab 3: Real-time Simulation
- Cycle-by-cycle RUL prediction
- Progressive line drawing animation
- Risk level monitoring with status banner
- Interactive playback controls (Play/Pause/Previous/Next)
- Manual scrubbing with slider

### Tab 4: Fleet Health
- Fleet-wide statistics (100 engines)
- RUL distribution histogram
- Predicted vs Actual scatter plot
- Individual engine tiles with color-coded risk levels
- Compact/Expanded view toggle
- Sort by ID, RUL, or risk level

## API Endpoints

- `GET /health` - Service health check
- `POST /predict` - Predict RUL for sensor readings
- `GET /models` - Get all model metrics
- `GET /simulate/<engine_id>` - Full lifecycle simulation
- `GET /shap` - SHAP feature importances
- `GET /fleet` - Fleet-wide health summary

## Project Structure

```
predictive-maintenance/
├── data/                      # Dataset files
├── models/                    # Trained models and artifacts
├── frontend/                  # Dashboard UI
│   ├── index.html
│   ├── dashboard.js
│   └── styles.css
├── generate_data.py           # Convert raw data to CSV
├── preprocessing.py           # Feature engineering pipeline
├── train_models.py            # Train all models
├── app.py                     # Flask API server
├── run_pipeline.py            # Automated pipeline runner
├── check_setup.py             # Environment validation
├── requirements.txt
└── README.md
```

## Dataset

The NASA CMAPSS FD001 dataset contains:
- 100 training engines (run to failure)
- 100 test engines (censored before failure)
- 21 sensor measurements per cycle
- 3 operational settings

## Key Implementation Notes

- **Time-series integrity**: Data split by engine_id, never shuffled
- **Feature engineering**: All rolling/lag features computed per engine group
- **LSTM sequences**: Fixed length of 30 cycles per engine
- **RUL clipping**: Target clipped at 130 cycles (piecewise linear)
- **Constant sensors**: s1, s5, s10, s16, s18, s19 removed (std < 0.01)

## Technologies

- **Backend**: Python, Flask, scikit-learn, XGBoost, TensorFlow
- **Frontend**: Vanilla JavaScript, Chart.js, HTML5, CSS3
- **ML**: Ridge, Random Forest, XGBoost, LSTM
- **Explainability**: SHAP
- **Design**: IBM Plex Mono/Sans fonts, dark industrial theme

## Troubleshooting

### API Connection Failed
- Ensure `python app.py` is running
- Check that port 5000 is not blocked
- Try accessing `http://localhost:5000/health` directly

### Charts Not Displaying
- Make sure the Flask server is running
- Open browser console (F12) to check for errors
- Hard refresh the page (Ctrl+Shift+R or Cmd+Shift+R)

### Module Not Found Errors
```bash
pip install -r requirements.txt --upgrade
```

### Models Not Found
Run the complete pipeline:
```bash
python run_pipeline.py
```

## License

MIT License - Dataset provided by NASA Ames Prognostics Data Repository
