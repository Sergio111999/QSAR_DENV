# QSAR-DENV: Anti-Dengue Drug Discovery from Colombian Medicinal Flora

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the complete computational pipeline for QSAR (Quantitative Structure-Activity Relationship) modeling and virtual screening of anti-dengue compounds derived from Colombian medicinal flora. The study integrates ethnobotanical knowledge with machine learning to identify promising drug candidates.

## Key Features

- **Curated Dataset**: 358 Colombian medicinal plants with antiviral activity
- **Machine Learning**: XGBoost-based QSAR model (MCC = 0.583)
- **Virtual Screening**: 3,267 phytochemicals screened
- **Comprehensive Analysis**: Chemical diversity, applicability domain, and SAR analysis
- **Bayesian Optimization**: Automated hyperparameter tuning using Optuna

## Citation

If you use this code or data, please cite:

```
[Your citation will be added here once published]
```

## Repository Structure

```
QSAR-DENV/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment 
├── LICENSE                           # MIT License
├── scripts/                          # Main analysis pipeline
│   ├── 01_exploratory_data_analysis.py
│   ├── 02_chemical_diversity_analysis.py
│   ├── 03_data_preparation_baseline_models.py
│   ├── 04_advanced_model_optimization.py
│   └── 05_qsar_prediction_tool.py
├── data/                            # Example data
│   └── example_compounds.csv
├── models/                          # Trained models (download separately)
│   └── README.md
├── notebooks/                       # Jupyter notebooks
│   └── example_usage.ipynb
└── docs/                           # Additional documentation
    └── user_guide.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/[YOUR-USERNAME]/QSAR-DENV.git
cd QSAR-DENV

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
# Clone the repository
git clone https://github.com/[YOUR-USERNAME]/QSAR-DENV.git
cd QSAR-DENV

# Create conda environment
conda env create -f environment.yml
conda activate qsar-denv
```

## Quick Start

### 1. Exploratory Data Analysis

```python
python scripts/01_exploratory_data_analysis.py
```

Performs comprehensive EDA on the anti-dengue bioactivity dataset including:
- Activity distribution analysis
- Physicochemical property correlations
- Molecular weight and drug-likeness statistics

### 2. Chemical Diversity Analysis

```python
python scripts/02_chemical_diversity_analysis.py
```

Analyzes chemical diversity through:
- Molecular scaffolds identification
- Chemical clustering
- Fingerprint similarity analysis
- Pharmacophore feature extraction

### 3. Model Training and Optimization

```python
python scripts/04_advanced_model_optimization.py
```

Trains and optimizes QSAR models using:
- Multiple machine learning algorithms (XGBoost, LightGBM, ExtraTrees)
- Bayesian hyperparameter optimization (Optuna)
- 5-fold cross-validation
- Comprehensive performance metrics

### 4. Virtual Screening

```python
python scripts/05_qsar_prediction_tool.py --input your_compounds.csv
```

Screens new compounds with:
- Activity prediction
- Drug-likeness assessment (QED)
- Applicability domain evaluation

## Dependencies

Main dependencies (see `requirements.txt` for complete list):

- rdkit >= 2022.09.1
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.1.0
- xgboost >= 1.7.0
- lightgbm >= 3.3.0
- optuna >= 3.0.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0

## Data Availability

### Training Dataset
The curated anti-dengue bioactivity dataset from ChEMBL is available in the supplementary materials of the associated publication.

### Colombian Phytochemical Library
The complete library of 3,267 compounds from 358 Colombian medicinal plants is available in the supplementary materials.

### Trained Models
Pre-trained models are available for download:
- [Download from Zenodo](https://zenodo.org/record/[YOUR-DOI]) (will be added)

## Usage Example

```python
from rdkit import Chem
import pandas as pd
import pickle

# Load trained model
with open('models/xgboost_final_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load your compounds
df = pd.read_csv('your_compounds.csv')  # Must have 'SMILES' column

# Make predictions
predictions = model.predict_proba(X)[:, 1]

print(f"Predicted anti-dengue activity: {predictions[0]:.3f}")
```

## Model Performance

The final optimized XGBoost model achieved:

- **MCC**: 0.583
- **Accuracy**: [To be added]
- **Precision**: [To be added]
- **Recall**: [To be added]
- **AUC-ROC**: [To be added]

Performance was evaluated using 5-fold cross-validation on a balanced dataset.

## Results

### Virtual Screening Results

- **Total compounds screened**: 3,267
- **High-activity predictions** (>0.7): [Number]
- **High drug-likeness** (QED >0.5): [Number]
- **Top priority candidates**: 14 novel compounds

### Top Identified Compounds

The top priority compounds identified combine:
- High predicted anti-dengue activity
- Favorable drug-likeness properties (QED)
- Location within model's applicability domain
- Novel chemical scaffolds not present in training data

See supplementary materials for complete results.

## Documentation

- [User Guide](docs/user_guide.md) - Detailed usage instructions
- [Supplementary Materials](https://[journal-link]) - Complete data and results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaborations:

- **Sergio Montenegro**: samontenegro@icesi.edu.co
- **Alejandra Jerez** (Corresponding Author): amjerez@icesi.edu.co

**Institution**: Universidad Icesi, Cali, Colombia  
**Department**: Ciencias Básicas Médicas, Facultad de Ciencias de la Salud

## Acknowledgments

- ChEMBL database for bioactivity data
- Colombian ethnobotanical knowledge sources
- Universidad Icesi for computational resources

## Related Publications

- [Main manuscript citation will be added here]
- Supplementary materials: [Link to supplementary materials]

---

**Note**: This repository contains the computational pipeline. For complete datasets, trained models, and detailed results, please refer to the supplementary materials of the associated publication.
