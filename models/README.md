# Trained Models

The trained QSAR models are available in the **supplementary materials** of the publication.

## Files Included

- `xgboost_final_model.pkl` - Optimized XGBoost model (MCC = 0.583)
- `scaler.pkl` - Feature scaler
- `variance_selector.pkl` - Feature selector

## How to Access

Download **Data S3** from the publication's supplementary materials or contact:
- Sergio Montenegro: samontenegro@icesi.edu.co

## Usage

```python
import pickle

# Load models
model = pickle.load(open('xgboost_final_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
selector = pickle.load(open('variance_selector.pkl', 'rb'))

# Make predictions (see scripts/05_qsar_prediction_tool.py for complete workflow)
X_scaled = scaler.transform(X_descriptors)
X_selected = selector.transform(X_scaled)
predictions = model.predict_proba(X_selected)[:, 1]
```

## Citation

```
[Citation will be added after publication]
```
