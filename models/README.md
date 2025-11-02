# Trained Models

This directory should contain the trained QSAR models. Due to file size limitations on GitHub, the models are hosted externally.

## Available Models

### XGBoost Final Model
- **File**: `xgboost_final_model.pkl`
- **Performance**: MCC = 0.583
- **Size**: ~3.1 MB
- **Download**: [Zenodo link - to be added]

### Additional Files
- `scaler.pkl` - Feature scaler
- `variance_selector.pkl` - Feature selector
- `model_metadata.json` - Model configuration and metadata

## How to Download

### Option 1: Direct Download from Zenodo
```bash
# Download from Zenodo (link will be added after publication)
wget https://zenodo.org/record/[DOI]/files/xgboost_final_model.pkl
wget https://zenodo.org/record/[DOI]/files/scaler.pkl
wget https://zenodo.org/record/[DOI]/files/variance_selector.pkl
```

### Option 2: From Supplementary Materials
The models are also available in the supplementary materials of the published manuscript.

## Model Files Location

After publication, download the models from:
- **Primary**: Zenodo repository (DOI: [to be added])
- **Secondary**: Journal supplementary materials

Place the downloaded files in this `models/` directory.

## Usage

```python
import pickle
import pandas as pd

# Load model
with open('models/xgboost_final_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load variance selector
with open('models/variance_selector.pkl', 'rb') as f:
    variance_selector = pickle.load(f)

# Make predictions
# X = your_features  # Must be pre-processed molecular descriptors
# X_scaled = scaler.transform(X)
# X_selected = variance_selector.transform(X_scaled)
# predictions = model.predict_proba(X_selected)[:, 1]
```

## Model Details

- **Algorithm**: XGBoost
- **Hyperparameters**: Optimized via Bayesian optimization (50 trials)
- **Features**: Selected molecular descriptors (N = [number])
- **Training data**: [number] anti-dengue compounds from ChEMBL
- **Validation**: 5-fold cross-validation

## Citation

If you use these models, please cite:
```
[Citation will be added here]
```


