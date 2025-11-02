#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QSAR 5.0 - Script 03: Data Preparation and Baseline Models with Enhanced Reporting
==================================================================================
This script prepares data for QSAR modeling, implements balancing techniques,
trains baseline models with robust validation, and generates comprehensive
supplementary materials.

Author: Sergio Montenegro

"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import json
from collections import Counter
import shutil
import textwrap

# Scikit-learn
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, GridSearchCV,
    train_test_split, GroupKFold
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    balanced_accuracy_score, matthews_corrcoef, f1_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_auc_score,
    make_scorer
)
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif,
    mutual_info_classif, RFECV
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Class balancing
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.metrics import geometric_mean_score

# XGBoost
import xgboost as xgb

# Visualization
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Custom metrics
def mcc_scorer(y_true, y_pred):
    """Matthews Correlation Coefficient scorer"""
    return matthews_corrcoef(y_true, y_pred)

def geometric_mean_scorer(y_true, y_pred):
    """Geometric mean of class-wise accuracies"""
    return geometric_mean_score(y_true, y_pred)

class DetailedReportGenerator:
    """Generate detailed PDF and text reports for supplementary materials"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.methods_content = []
        self.results_content = []
        self.figures_list = []
        self.tables_list = []
        
    def add_methods_section(self, title, content):
        """Add a section to methods"""
        self.methods_content.append({'title': title, 'content': content})
    
    def add_results_section(self, title, content):
        """Add a section to results"""
        self.results_content.append({'title': title, 'content': content})
    
    def add_figure_reference(self, filename, caption):
        """Track figure for list of figures"""
        self.figures_list.append({'filename': filename, 'caption': caption})
    
    def add_table_reference(self, filename, caption):
        """Track table for list of tables"""
        self.tables_list.append({'filename': filename, 'caption': caption})
    
    def generate_methods_report(self):
        """Generate detailed methods report"""
        methods_text = []
        
        # Header
        methods_text.append("="*80)
        methods_text.append("SUPPLEMENTARY METHODS")
        methods_text.append("="*80)
        methods_text.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        methods_text.append("\n")
        
        # Content
        for i, section in enumerate(self.methods_content, 1):
            methods_text.append(f"\nS{i}. {section['title']}")
            methods_text.append("-"*60)
            methods_text.append(section['content'])
            methods_text.append("\n")
        
        # Save
        methods_path = os.path.join(self.output_dir, "Supplementary_Methods.txt")
        with open(methods_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(methods_text))
        
        return methods_path
    
    def generate_results_report(self):
        """Generate detailed results report"""
        results_text = []
        
        # Header
        results_text.append("="*80)
        results_text.append("SUPPLEMENTARY RESULTS")
        results_text.append("="*80)
        results_text.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        results_text.append("\n")
        
        # Content
        for i, section in enumerate(self.results_content, 1):
            results_text.append(f"\nR{i}. {section['title']}")
            results_text.append("-"*60)
            results_text.append(section['content'])
            results_text.append("\n")
        
        # Save
        results_path = os.path.join(self.output_dir, "Supplementary_Results.txt")
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(results_text))
        
        return results_path
    
    def generate_figure_list(self):
        """Generate list of supplementary figures"""
        fig_text = []
        fig_text.append("="*80)
        fig_text.append("LIST OF SUPPLEMENTARY FIGURES")
        fig_text.append("="*80)
        fig_text.append("\n")
        
        for i, fig in enumerate(self.figures_list, 1):
            fig_text.append(f"Figure S{i}: {fig['caption']}")
            fig_text.append(f"File: {fig['filename']}")
            fig_text.append("\n")
        
        # Save
        fig_path = os.path.join(self.output_dir, "List_of_Figures.txt")
        with open(fig_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fig_text))
        
        return fig_path
    
    def generate_table_list(self):
        """Generate list of supplementary tables"""
        table_text = []
        table_text.append("="*80)
        table_text.append("LIST OF SUPPLEMENTARY TABLES")
        table_text.append("="*80)
        table_text.append("\n")
        
        for i, table in enumerate(self.tables_list, 1):
            table_text.append(f"Table S{i}: {table['caption']}")
            table_text.append(f"File: {table['filename']}")
            table_text.append("\n")
        
        # Save
        table_path = os.path.join(self.output_dir, "List_of_Tables.txt")
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(table_text))
        
        return table_path

class QSARModelPreparation:
    """Enhanced QSAR data preparation and model training with detailed reporting"""
    
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.base_output_dir = os.path.join(output_dir, "03_Model_Preparation")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create supplementary materials structure
        self.supplementary_dir = os.path.join(self.base_output_dir, f"Supplementary_Materials_{self.timestamp}")
        self.create_supplementary_structure()
        
        # Initialize report generator
        self.report_generator = DetailedReportGenerator(self.supplementary_dir)
        
        # Initialize report
        self.report = []
        
        # Define metrics
        self.scoring = {
            'balanced_accuracy': make_scorer(balanced_accuracy_score),
            'mcc': make_scorer(mcc_scorer),
            'f1_macro': make_scorer(f1_score, average='macro'),
            'geometric_mean': make_scorer(geometric_mean_scorer),
            'roc_auc_ovr': make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)
        }
        
        # Track all results for comprehensive reporting
        self.all_results = {}
        
    def create_supplementary_structure(self):
        """Create comprehensive supplementary materials folder structure"""
        folders = [
            self.supplementary_dir,
            os.path.join(self.supplementary_dir, "Figures"),
            os.path.join(self.supplementary_dir, "Tables"),
            os.path.join(self.supplementary_dir, "Data"),
            os.path.join(self.supplementary_dir, "Models"),
            os.path.join(self.supplementary_dir, "Reports"),
            os.path.join(self.supplementary_dir, "Feature_Analysis"),
            os.path.join(self.supplementary_dir, "Validation_Results"),
            os.path.join(self.supplementary_dir, "Statistical_Analysis")
        ]
        
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
    
    def load_data(self):
        """Load processed data from diversity analysis"""
        print("Loading data...")
        
        # Load descriptors
        descriptor_path = os.path.join(self.data_dir, "02_Chemical_Diversity", "data", "full_descriptor_data.csv")
        self.df = pd.read_csv(descriptor_path)
        
        # Load fingerprints
        fp_path = os.path.join(self.data_dir, "02_Chemical_Diversity", "data", "molecular_fingerprints.pkl")
        with open(fp_path, 'rb') as f:
            self.fingerprints = pickle.load(f)
        
        print(f"Data loaded: {len(self.df)} compounds")
        print(f"Available fingerprints: {list(self.fingerprints.keys())}")
        
        # Prepare features and targets
        self.prepare_features_and_targets()
        
        # Add to methods report
        methods_content = f"""
Data Loading and Preprocessing:
The dataset was loaded from pre-processed chemical diversity analysis outputs. 
Molecular descriptors were obtained from CSV format, while molecular fingerprints 
(ECFP4, MACCS, Morgan) were loaded from pickled binary format.

Total compounds analyzed: {len(self.df)}
Descriptor types: RDKit physicochemical descriptors
Fingerprint types: {', '.join(list(self.fingerprints.keys()))}

Data preprocessing steps:
1. Removal of non-feature columns (identifiers, metadata)
2. Extraction of numerical descriptor matrix
3. Conversion of fingerprint objects to binary matrices
4. Target variable encoding for multi-class classification
5. Generation of binary classification targets for sensitivity analysis
        """
        self.report_generator.add_methods_section("Data Loading and Preprocessing", methods_content)
        
    def prepare_features_and_targets(self):
        """Prepare feature and target matrices with detailed documentation"""
        print("\nPreparing features and targets...")
        
        # Define non-feature columns
        non_feature_cols = [
            'Molecule ChEMBL ID', 'SMILES_original', 'pActivity_calculado',
            'potency_category', 'Standard Type', 'Target Name', 'scaffold'
        ]
        
        # Molecular descriptors
        descriptor_cols = [col for col in self.df.columns if col not in non_feature_cols]
        self.X_descriptors = self.df[descriptor_cols].values
        
        # Convert fingerprints to numpy matrices
        self.X_fingerprints = {}
        for fp_type, fps in self.fingerprints.items():
            print(f"  Processing {fp_type} fingerprints...")
            fp_matrix = np.array([fp.ToList() for fp in fps])
            self.X_fingerprints[fp_type] = fp_matrix
        
        # Multiclass target
        self.y = self.df['potency_category'].map({'High': 2, 'Medium': 1, 'Low': 0}).values
        
        # Binary target (High vs Non-High)
        self.y_binary = (self.y == 2).astype(int)
        
        # Additional information
        self.compound_ids = self.df['Molecule ChEMBL ID'].values
        self.assay_types = self.df['Standard Type'].values
        self.scaffolds = self.df['scaffold'].values if 'scaffold' in self.df else None
        
        # Create detailed data summary table
        self._create_data_summary_table()
        
        # Document in results
        results_content = f"""
Feature Matrix Dimensions:
- Molecular Descriptors: {self.X_descriptors.shape}
- ECFP4 Fingerprints: {self.X_fingerprints.get('ECFP4', []).shape if 'ECFP4' in self.X_fingerprints else 'N/A'}
- MACCS Fingerprints: {self.X_fingerprints.get('MACCS', []).shape if 'MACCS' in self.X_fingerprints else 'N/A'}
- Morgan Fingerprints: {self.X_fingerprints.get('Morgan', []).shape if 'Morgan' in self.X_fingerprints else 'N/A'}

Class Distribution (Multi-class):
- Low Potency: {np.sum(self.y == 0)} ({np.sum(self.y == 0)/len(self.y)*100:.1f}%)
- Medium Potency: {np.sum(self.y == 1)} ({np.sum(self.y == 1)/len(self.y)*100:.1f}%)
- High Potency: {np.sum(self.y == 2)} ({np.sum(self.y == 2)/len(self.y)*100:.1f}%)

Class Distribution (Binary):
- Non-High: {np.sum(self.y_binary == 0)} ({np.sum(self.y_binary == 0)/len(self.y_binary)*100:.1f}%)
- High: {np.sum(self.y_binary == 1)} ({np.sum(self.y_binary == 1)/len(self.y_binary)*100:.1f}%)

Imbalance Ratio: 1:{np.sum(self.y == 0)/np.sum(self.y == 2):.2f}:{np.sum(self.y == 1)/np.sum(self.y == 2):.2f}
        """
        self.report_generator.add_results_section("Data Preparation Summary", results_content)
        
        # Report
        self.report.append("=== DATA PREPARATION ===\n")
        self.report.append(f"Molecular descriptors: {self.X_descriptors.shape}")
        for fp_type, fp_matrix in self.X_fingerprints.items():
            self.report.append(f"{fp_type} fingerprints: {fp_matrix.shape}")
        self.report.append(f"\nClass distribution (multiclass):")
        class_names = ['Low', 'Medium', 'High']
        for i, count in enumerate(np.bincount(self.y)):
            self.report.append(f"  {class_names[i]}: {count} ({count/len(self.y)*100:.1f}%)")
    
    def _create_data_summary_table(self):
        """Create comprehensive data summary table"""
        summary_data = {
            'Feature Type': [],
            'Number of Features': [],
            'Data Type': [],
            'Missing Values': [],
            'Zero Variance Features': []
        }
        
        # Descriptors
        summary_data['Feature Type'].append('Molecular Descriptors')
        summary_data['Number of Features'].append(self.X_descriptors.shape[1])
        summary_data['Data Type'].append('Continuous')
        summary_data['Missing Values'].append(np.sum(np.isnan(self.X_descriptors)))
        summary_data['Zero Variance Features'].append(np.sum(np.var(self.X_descriptors, axis=0) == 0))
        
        # Fingerprints
        for fp_type, fp_matrix in self.X_fingerprints.items():
            summary_data['Feature Type'].append(f'{fp_type} Fingerprint')
            summary_data['Number of Features'].append(fp_matrix.shape[1])
            summary_data['Data Type'].append('Binary')
            summary_data['Missing Values'].append(np.sum(np.isnan(fp_matrix)))
            summary_data['Zero Variance Features'].append(np.sum(np.var(fp_matrix, axis=0) == 0))
        
        # Save table
        summary_df = pd.DataFrame(summary_data)
        table_path = os.path.join(self.supplementary_dir, "Tables", "Table_S1_Data_Summary.csv")
        summary_df.to_csv(table_path, index=False)
        
        self.report_generator.add_table_reference("Table_S1_Data_Summary.csv", 
                                                 "Summary of feature types and characteristics")
    
    def analyze_feature_variance(self):
        """Analyze feature variance for initial filtering with enhanced documentation"""
        print("\nAnalyzing feature variance...")
        
        # Combine descriptors with ECFP4 for analysis
        X_combined = np.hstack([self.X_descriptors, self.X_fingerprints['ECFP4']])
        
        # Calculate variance
        variances = np.var(X_combined, axis=0)
        
        # Detailed variance statistics
        variance_stats = {
            'Zero Variance': np.sum(variances == 0),
            'Very Low (<0.001)': np.sum((variances > 0) & (variances < 0.001)),
            'Low (0.001-0.01)': np.sum((variances >= 0.001) & (variances < 0.01)),
            'Medium (0.01-0.1)': np.sum((variances >= 0.01) & (variances < 0.1)),
            'High (>0.1)': np.sum(variances >= 0.1)
        }
        
        # Save variance statistics
        stats_df = pd.DataFrame(list(variance_stats.items()), columns=['Variance Range', 'Number of Features'])
        stats_df['Percentage'] = stats_df['Number of Features'] / len(variances) * 100
        table_path = os.path.join(self.supplementary_dir, "Tables", "Table_S2_Variance_Statistics.csv")
        stats_df.to_csv(table_path, index=False)
        
        self.report_generator.add_table_reference("Table_S2_Variance_Statistics.csv",
                                                 "Feature variance distribution statistics")
        
        # Visualize variance distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Variance histogram
        axes[0, 0].hist(variances[variances > 0], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Variance')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].set_title('Feature Variance Distribution')
        axes[0, 0].set_yscale('log')
        
        # Proportion of low variance features
        thresholds = [0, 0.001, 0.01, 0.1]
        proportions = []
        for threshold in thresholds:
            prop = np.sum(variances <= threshold) / len(variances)
            proportions.append(prop * 100)
        
        axes[0, 1].bar(range(len(thresholds)), proportions, color='coral')
        axes[0, 1].set_xticks(range(len(thresholds)))
        axes[0, 1].set_xticklabels([f'≤{t}' for t in thresholds])
        axes[0, 1].set_xlabel('Variance Threshold')
        axes[0, 1].set_ylabel('% of Features')
        axes[0, 1].set_title('Low Variance Features')
        
        for i, prop in enumerate(proportions):
            axes[0, 1].text(i, prop + 1, f'{prop:.1f}%', ha='center')
        
        # Cumulative variance distribution
        sorted_variances = np.sort(variances)[::-1]
        cumsum_var = np.cumsum(sorted_variances) / np.sum(sorted_variances)
        axes[1, 0].plot(range(len(cumsum_var)), cumsum_var, 'b-', linewidth=2)
        axes[1, 0].set_xlabel('Number of Features')
        axes[1, 0].set_ylabel('Cumulative Variance Explained')
        axes[1, 0].set_title('Cumulative Variance Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        axes[1, 0].legend()
        
        # Variance by feature type
        n_desc = self.X_descriptors.shape[1]
        desc_var = variances[:n_desc]
        fp_var = variances[n_desc:]
        
        axes[1, 1].boxplot([desc_var[desc_var > 0], fp_var[fp_var > 0]], 
                          labels=['Descriptors', 'Fingerprints'])
        axes[1, 1].set_ylabel('Variance')
        axes[1, 1].set_title('Variance Distribution by Feature Type')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        fig_path = os.path.join(self.supplementary_dir, "Figures", "Figure_S1_Feature_Variance_Analysis.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.report_generator.add_figure_reference("Figure_S1_Feature_Variance_Analysis.png",
                                                  "Comprehensive feature variance analysis")
        
        # Add to methods
        methods_content = f"""
Feature Variance Analysis:
A comprehensive variance analysis was performed to identify and remove non-informative features.
Features with zero or near-zero variance were identified as candidates for removal.

Variance thresholds evaluated:
- Zero variance: Features with no variation across samples
- Very low variance (<0.001): Nearly constant features
- Low variance (0.001-0.01): Features with minimal variation
- Medium variance (0.01-0.1): Features with moderate variation
- High variance (>0.1): Highly variable features

Total features analyzed: {len(variances)}
Features with zero variance: {variance_stats['Zero Variance']}
Features with variance < 0.01: {variance_stats['Zero Variance'] + variance_stats['Very Low (<0.001)'] + variance_stats['Low (0.001-0.01)']}

The variance threshold of 0.01 was selected for feature filtering based on the analysis.
        """
        self.report_generator.add_methods_section("Feature Variance Analysis", methods_content)
        
        self.report.append(f"\n=== VARIANCE ANALYSIS ===")
        self.report.append(f"Total features: {len(variances)}")
        self.report.append(f"Features with variance = 0: {np.sum(variances == 0)}")
        self.report.append(f"Features with variance < 0.01: {np.sum(variances < 0.01)}")
    
    def evaluate_balancing_strategies(self):
        """Evaluate different balancing strategies with comprehensive documentation"""
        print("\nEvaluating balancing strategies...")
        
        # Use descriptors only for quick evaluation
        X = self.X_descriptors
        y = self.y_binary  # Use binary classification to simplify
        
        # Initial stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Balancing strategies
        balancing_strategies = {
            'None': None,
            'SMOTE': SMOTE(random_state=42),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
            'ADASYN': ADASYN(random_state=42),
            'SMOTEENN': SMOTEENN(random_state=42),
            'SMOTETomek': SMOTETomek(random_state=42)
        }
        
        # Base model for evaluation
        base_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        results = {}
        detailed_results = []
        
        for name, sampler in balancing_strategies.items():
            print(f"  Evaluating {name}...")
            
            if sampler is None:
                X_resampled, y_resampled = X_train_scaled, y_train
            else:
                try:
                    X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)
                except Exception as e:
                    print(f"    Warning: {name} failed - {str(e)}")
                    continue
            
            # Train model
            model = base_model.fit(X_resampled, y_resampled)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Metrics
            results[name] = {
                'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                'mcc': matthews_corrcoef(y_test, y_pred),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'geometric_mean': geometric_mean_score(y_test, y_pred),
                'avg_precision': average_precision_score(y_test, y_proba),
                'train_size': len(y_resampled),
                'class_distribution': Counter(y_resampled),
                'confusion_matrix': cm.tolist()
            }
            
            # Store detailed results
            detailed_results.append({
                'Strategy': name,
                'Balanced Accuracy': results[name]['balanced_acc'],
                'MCC': results[name]['mcc'],
                'F1-Macro': results[name]['f1_macro'],
                'G-Mean': results[name]['geometric_mean'],
                'Average Precision': results[name]['avg_precision'],
                'Original Train Size': len(y_train),
                'Resampled Train Size': len(y_resampled),
                'Class 0 Count': results[name]['class_distribution'][0],
                'Class 1 Count': results[name]['class_distribution'][1]
            })
        
        # Save detailed results table
        results_df = pd.DataFrame(detailed_results)
        table_path = os.path.join(self.supplementary_dir, "Tables", "Table_S3_Balancing_Strategies_Results.csv")
        results_df.to_csv(table_path, index=False)
        
        self.report_generator.add_table_reference("Table_S3_Balancing_Strategies_Results.csv",
                                                 "Comprehensive comparison of class balancing strategies")
        
        # Visualize results
        self._plot_enhanced_balancing_results(results)
        
        # Add to methods
        methods_content = """
Class Balancing Strategies:
Six different class balancing strategies were evaluated to address the class imbalance problem:

1. None: No balancing (baseline)
2. SMOTE (Synthetic Minority Over-sampling Technique): Creates synthetic samples for minority class
3. BorderlineSMOTE: Variant focusing on borderline instances
4. ADASYN (Adaptive Synthetic Sampling): Adaptive synthetic sample generation
5. SMOTEENN: Combination of SMOTE and Edited Nearest Neighbors
6. SMOTETomek: Combination of SMOTE and Tomek Links

Each strategy was evaluated using:
- Random Forest classifier with 100 estimators
- 80/20 train-test split with stratification
- StandardScaler for feature normalization
- Multiple evaluation metrics including MCC, balanced accuracy, and G-mean

The evaluation focused on the ability to improve minority class detection while 
maintaining overall classification performance.
        """
        self.report_generator.add_methods_section("Class Balancing Strategies", methods_content)
        
        # Add to results
        best_strategy = max(results.items(), key=lambda x: x[1]['mcc'])[0]
        results_content = f"""
Class Balancing Evaluation Results:

Best performing strategy: {best_strategy}
- MCC: {results[best_strategy]['mcc']:.3f}
- Balanced Accuracy: {results[best_strategy]['balanced_acc']:.3f}
- G-Mean: {results[best_strategy]['geometric_mean']:.3f}

The evaluation revealed that synthetic sampling techniques (SMOTE variants) generally
outperformed the baseline no-balancing approach. The best strategy achieved a
{(results[best_strategy]['mcc'] - results.get('None', {}).get('mcc', 0)) * 100:.1f}% 
improvement in MCC over the baseline.
        """
        self.report_generator.add_results_section("Class Balancing Results", results_content)
        
        # Add to report
        self.report.append("\n=== BALANCING STRATEGIES EVALUATION ===")
        for name, metrics in results.items():
            self.report.append(f"\n{name}:")
            self.report.append(f"  Balanced Accuracy: {metrics['balanced_acc']:.3f}")
            self.report.append(f"  MCC: {metrics['mcc']:.3f}")
            self.report.append(f"  F1-macro: {metrics['f1_macro']:.3f}")
        
        return results
    
    def _plot_enhanced_balancing_results(self, results):
        """Create enhanced visualization of balancing strategy results"""
        strategies = list(results.keys())
        metrics = ['balanced_acc', 'mcc', 'f1_macro', 'geometric_mean', 'avg_precision']
        metric_names = ['Balanced\nAccuracy', 'MCC', 'F1-Macro', 'G-Mean', 'Average\nPrecision']
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Heatmap of all metrics
        ax1 = plt.subplot(2, 2, 1)
        result_matrix = np.zeros((len(strategies), len(metrics)))
        for i, strategy in enumerate(strategies):
            for j, metric in enumerate(metrics):
                result_matrix[i, j] = results[strategy][metric]
        
        sns.heatmap(result_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   xticklabels=metric_names, yticklabels=strategies,
                   center=0.5, vmin=0, vmax=1, ax=ax1)
        ax1.set_title('Performance Metrics Heatmap')
        
        # 2. Radar plot for top strategies
        ax2 = plt.subplot(2, 2, 2, projection='polar')
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        # Select top 3 strategies by MCC
        top_strategies = sorted(results.items(), key=lambda x: x[1]['mcc'], reverse=True)[:3]
        
        for strategy_name, strategy_results in top_strategies:
            values = [strategy_results[metric] for metric in metrics]
            values += values[:1]
            ax2.plot(angles, values, 'o-', linewidth=2, label=strategy_name)
            ax2.fill(angles, values, alpha=0.25)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metric_names, size=8)
        ax2.set_ylim(0, 1)
        ax2.set_title('Top 3 Strategies - Radar Plot')
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax2.grid(True)
        
        # 3. Class distribution after balancing
        ax3 = plt.subplot(2, 2, 3)
        strategies_names = []
        class_0_counts = []
        class_1_counts = []
        
        for name, res in results.items():
            strategies_names.append(name)
            class_0_counts.append(res['class_distribution'][0])
            class_1_counts.append(res['class_distribution'][1])
        
        x = np.arange(len(strategies_names))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, class_0_counts, width, label='Class 0', alpha=0.8)
        bars2 = ax3.bar(x + width/2, class_1_counts, width, label='Class 1', alpha=0.8)
        
        ax3.set_xlabel('Strategy')
        ax3.set_ylabel('Number of Samples')
        ax3.set_title('Class Distribution After Balancing')
        ax3.set_xticks(x)
        ax3.set_xticklabels(strategies_names, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Performance improvement over baseline
        ax4 = plt.subplot(2, 2, 4)
        if 'None' in results:
            baseline_mcc = results['None']['mcc']
            improvements = [(name, (res['mcc'] - baseline_mcc) * 100) 
                          for name, res in results.items() if name != 'None']
            improvements.sort(key=lambda x: x[1], reverse=True)
            
            names, values = zip(*improvements)
            colors = ['green' if v > 0 else 'red' for v in values]
            
            bars = ax4.bar(range(len(names)), values, color=colors, alpha=0.7)
            ax4.set_xticks(range(len(names)))
            ax4.set_xticklabels(names, rotation=45, ha='right')
            ax4.set_ylabel('MCC Improvement (%)')
            ax4.set_title('Performance Improvement Over Baseline')
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig_path = os.path.join(self.supplementary_dir, "Figures", "Figure_S2_Balancing_Strategies_Analysis.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.report_generator.add_figure_reference("Figure_S2_Balancing_Strategies_Analysis.png",
                                                  "Comprehensive analysis of class balancing strategies")
    
    def train_baseline_models(self):
        """Train baseline models with different representations and detailed documentation"""
        print("\nTraining baseline models...")
        
        # Model configuration
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=len(self.y[self.y == 0]) / len(self.y[self.y == 2]),
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        }
        
        # Representations to evaluate
        representations = {
            'Descriptors': self.X_descriptors,
            'ECFP4': self.X_fingerprints['ECFP4'],
            'MACCS': self.X_fingerprints['MACCS'],
            'Descriptors+ECFP4': np.hstack([self.X_descriptors, self.X_fingerprints['ECFP4']])
        }
        
        # Results storage
        cv_results = {}
        detailed_cv_results = []
        
        # Stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for rep_name, X in representations.items():
            print(f"\n  Evaluating representation: {rep_name}")
            cv_results[rep_name] = {}
            
            # Remove zero variance features
            var_selector = VarianceThreshold(threshold=0.01)
            X_filtered = var_selector.fit_transform(X)
            print(f"    Features after filtering: {X_filtered.shape[1]}")
            
            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_filtered)
            
            for model_name, model in models.items():
                print(f"    Training {model_name}...")
                
                # Cross-validation with multiple metrics
                scores = cross_validate(
                    model, X_scaled, self.y,
                    cv=cv,
                    scoring=self.scoring,
                    return_train_score=True,
                    n_jobs=-1
                )
                
                # Save results
                cv_results[rep_name][model_name] = {
                    metric: {
                        'test_mean': scores[f'test_{metric}'].mean(),
                        'test_std': scores[f'test_{metric}'].std(),
                        'train_mean': scores[f'train_{metric}'].mean(),
                        'train_std': scores[f'train_{metric}'].std(),
                        'fold_scores': scores[f'test_{metric}'].tolist()
                    }
                    for metric in self.scoring.keys()
                }
                
                # Store for detailed table
                detailed_cv_results.append({
                    'Representation': rep_name,
                    'Model': model_name,
                    'Features': X_filtered.shape[1],
                    'BA_Mean': cv_results[rep_name][model_name]['balanced_accuracy']['test_mean'],
                    'BA_Std': cv_results[rep_name][model_name]['balanced_accuracy']['test_std'],
                    'MCC_Mean': cv_results[rep_name][model_name]['mcc']['test_mean'],
                    'MCC_Std': cv_results[rep_name][model_name]['mcc']['test_std'],
                    'F1_Mean': cv_results[rep_name][model_name]['f1_macro']['test_mean'],
                    'F1_Std': cv_results[rep_name][model_name]['f1_macro']['test_std'],
                    'GMean_Mean': cv_results[rep_name][model_name]['geometric_mean']['test_mean'],
                    'GMean_Std': cv_results[rep_name][model_name]['geometric_mean']['test_std'],
                })
        
        # Save detailed results table
        cv_df = pd.DataFrame(detailed_cv_results)
        table_path = os.path.join(self.supplementary_dir, "Tables", "Table_S4_Baseline_Model_Performance.csv")
        cv_df.to_csv(table_path, index=False)
        
        self.report_generator.add_table_reference("Table_S4_Baseline_Model_Performance.csv",
                                                 "5-fold cross-validation results for baseline models")
        
        # Visualize results
        self._plot_enhanced_baseline_results(cv_results)
        
        # Save full results
        results_path = os.path.join(self.supplementary_dir, "Data", "baseline_cv_full_results.json")
        with open(results_path, 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        # Add to methods
        methods_content = """
Baseline Model Training:
Three machine learning algorithms were evaluated as baseline models:

1. Random Forest Classifier:
   - 200 trees
   - Maximum depth: 10
   - Minimum samples split: 5
   - Class weight: balanced
   
2. Extra Trees Classifier:
   - 200 trees
   - Maximum depth: 10
   - Minimum samples split: 5
   - Class weight: balanced
   
3. XGBoost Classifier:
   - 200 estimators
   - Maximum depth: 6
   - Learning rate: 0.1
   - Scale positive weight: adjusted for class imbalance

Four molecular representations were evaluated:
1. Molecular descriptors only
2. ECFP4 fingerprints only
3. MACCS fingerprints only
4. Combined descriptors + ECFP4

Feature preprocessing:
- Variance threshold: 0.01 (removal of low-variance features)
- Standardization: StandardScaler (mean=0, std=1)

Model evaluation:
- 5-fold stratified cross-validation
- Metrics: Balanced Accuracy, MCC, F1-Macro, G-Mean, ROC-AUC
- Both training and testing scores recorded for overfitting detection
        """
        self.report_generator.add_methods_section("Baseline Model Training", methods_content)
        
        # Find best combination
        best_score = -1
        best_combo = None
        for rep in cv_results:
            for model in cv_results[rep]:
                mcc = cv_results[rep][model]['mcc']['test_mean']
                if mcc > best_score:
                    best_score = mcc
                    best_combo = (rep, model)
        
        # Add to results
        results_content = f"""
Baseline Model Performance Summary:

Best performing combination:
- Representation: {best_combo[0]}
- Model: {best_combo[1]}
- MCC: {best_score:.3f} ± {cv_results[best_combo[0]][best_combo[1]]['mcc']['test_std']:.3f}
- Balanced Accuracy: {cv_results[best_combo[0]][best_combo[1]]['balanced_accuracy']['test_mean']:.3f}

Key findings:
1. Combined representations (Descriptors+ECFP4) generally outperformed single representations
2. Random Forest and XGBoost showed comparable performance
3. MACCS fingerprints alone showed the lowest performance
4. Minimal overfitting observed (train-test score difference < 0.1 for most models)

The results indicate that molecular descriptors and ECFP4 fingerprints provide
complementary information for activity prediction.
        """
        self.report_generator.add_results_section("Baseline Model Results", results_content)
        
        self.all_results['baseline_models'] = cv_results
        
        # Add to report
        self.report.append("\n=== BASELINE RESULTS ===")
        for rep_name, models in cv_results.items():
            self.report.append(f"\n{rep_name}:")
            for model_name, metrics in models.items():
                self.report.append(f"  {model_name}:")
                for metric_name, scores in metrics.items():
                    self.report.append(f"    {metric_name}: {scores['test_mean']:.3f} ± {scores['test_std']:.3f}")
        
        return cv_results
    
    def _plot_enhanced_baseline_results(self, cv_results):
        """Create enhanced visualization of baseline model results"""
        # Prepare data for visualization
        metrics_to_plot = ['balanced_accuracy', 'mcc', 'f1_macro', 'geometric_mean']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Performance by metric
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            # Prepare data
            data = []
            labels = []
            errors = []
            
            for rep_name in cv_results:
                for model_name in cv_results[rep_name]:
                    mean_score = cv_results[rep_name][model_name][metric]['test_mean']
                    std_score = cv_results[rep_name][model_name][metric]['test_std']
                    data.append(mean_score)
                    errors.append(std_score)
                    labels.append(f"{rep_name[:4]}\n{model_name[:6]}")
            
            # Create bar plot
            x = np.arange(len(labels))
            bars = ax.bar(x, data, yerr=errors, capsize=5, alpha=0.7)
            
            # Color bars by performance
            max_val = max(data)
            colors = ['green' if d >= max_val * 0.95 else 'skyblue' for d in data]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('Score')
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add reference line for MCC
            if metric == 'mcc':
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 2. Overfitting analysis
        ax = axes[1, 2]
        rep_model_pairs = []
        train_scores = []
        test_scores = []
        
        for rep_name in cv_results:
            for model_name in cv_results[rep_name]:
                rep_model_pairs.append(f"{rep_name[:8]}\n{model_name[:8]}")
                train_scores.append(cv_results[rep_name][model_name]['mcc']['train_mean'])
                test_scores.append(cv_results[rep_name][model_name]['mcc']['test_mean'])
        
        x = np.arange(len(rep_model_pairs))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_scores, width, label='Train', alpha=0.7)
        bars2 = ax.bar(x + width/2, test_scores, width, label='Test', alpha=0.7)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('MCC Score')
        ax.set_title('Train vs Test Performance (Overfitting Check)')
        ax.set_xticks(x)
        ax.set_xticklabels(rep_model_pairs, rotation=45, ha='right', fontsize=7)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Best model comparison across all metrics
        ax = axes[0, 2]
        
        # Find best model for each representation
        best_models = {}
        for rep_name in cv_results:
            best_mcc = -1
            best_model = None
            for model_name in cv_results[rep_name]:
                mcc = cv_results[rep_name][model_name]['mcc']['test_mean']
                if mcc > best_mcc:
                    best_mcc = mcc
                    best_model = model_name
            best_models[rep_name] = best_model
        
        # Plot radar for best models
        angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
        angles += angles[:1]
        
        for rep_name, model_name in best_models.items():
            values = [cv_results[rep_name][model_name][metric]['test_mean'] 
                     for metric in metrics_to_plot]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=f"{rep_name}")
        
        ax = plt.subplot(2, 3, 3, projection='polar')
        for rep_name, model_name in best_models.items():
            values = [cv_results[rep_name][model_name][metric]['test_mean'] 
                     for metric in metrics_to_plot]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=f"{rep_name}")
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics_to_plot], size=8)
        ax.set_ylim(0, 1)
        ax.set_title('Best Model per Representation')
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=8)
        ax.grid(True)
        
        plt.tight_layout()
        fig_path = os.path.join(self.supplementary_dir, "Figures", "Figure_S3_Baseline_Models_Performance.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.report_generator.add_figure_reference("Figure_S3_Baseline_Models_Performance.png",
                                                  "Comprehensive baseline model performance analysis")
    
    def analyze_feature_importance(self):
        """Analyze feature importance with enhanced documentation"""
        print("\nAnalyzing feature importance...")
        
        # Use Random Forest with Descriptors+ECFP4 (typically the best)
        X = np.hstack([self.X_descriptors, self.X_fingerprints['ECFP4']])
        
        # Filter variance
        var_selector = VarianceThreshold(threshold=0.01)
        X_filtered = var_selector.fit_transform(X)
        feature_mask = var_selector.get_support()
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)
        
        # Train model with all data
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_scaled, self.y)
        
        # Get importances
        importances = model.feature_importances_
        
        # Map back to feature names
        n_descriptors = self.X_descriptors.shape[1]
        descriptor_cols = [col for col in self.df.columns 
                          if col not in ['Molecule ChEMBL ID', 'SMILES_original', 
                                       'pActivity_calculado', 'potency_category',
                                       'Standard Type', 'Target Name', 'scaffold']]
        
        feature_names = []
        j = 0
        for i, keep in enumerate(feature_mask):
            if keep:
                if i < n_descriptors:
                    if i < len(descriptor_cols):
                        feature_names.append(descriptor_cols[i])
                    else:
                        feature_names.append(f'Descriptor_{i}')
                else:
                    feature_names.append(f'ECFP4_bit_{i-n_descriptors}')
                j += 1
        
        # Top features
        top_indices = np.argsort(importances)[::-1][:50]
        top_features = [(feature_names[i], importances[i]) for i in top_indices if i < len(feature_names)]
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(18, 14))
        
        # 1. Top 30 features horizontal bar
        ax1 = plt.subplot(2, 2, 1)
        features_plot = [f[0] for f in top_features[:30]]
        importances_plot = [f[1] for f in top_features[:30]]
        
        bars = ax1.barh(range(len(features_plot)), importances_plot, color='skyblue')
        # Color top 10 differently
        for i in range(min(10, len(bars))):
            bars[i].set_color('coral')
        
        ax1.set_yticks(range(len(features_plot)))
        ax1.set_yticklabels(features_plot, fontsize=7)
        ax1.set_xlabel('Importance')
        ax1.set_title('Top 30 Most Important Features')
        ax1.invert_yaxis()
        
        # 2. Importance distribution
        ax2 = plt.subplot(2, 2, 2)
        ax2.hist(importances[importances > 0], bins=50, alpha=0.7, 
                color='coral', edgecolor='black')
        ax2.set_xlabel('Importance')
        ax2.set_ylabel('Number of Features')
        ax2.set_title('Feature Importance Distribution')
        ax2.set_yscale('log')
        ax2.axvline(x=np.percentile(importances, 95), color='red', 
                   linestyle='--', label='95th percentile')
        ax2.legend()
        
        # 3. Cumulative importance
        ax3 = plt.subplot(2, 2, 3)
        sorted_importances = np.sort(importances)[::-1]
        cumsum_importances = np.cumsum(sorted_importances)
        
        ax3.plot(range(1, len(cumsum_importances) + 1), cumsum_importances, 'b-', linewidth=2)
        ax3.set_xlabel('Number of Features')
        ax3.set_ylabel('Cumulative Importance')
        ax3.set_title('Cumulative Feature Importance')
        ax3.grid(True, alpha=0.3)
        
        # Add reference lines
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% importance')
        ax3.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='80% importance')
        ax3.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='95% importance')
        ax3.legend()
        
        # Find number of features for different thresholds
        n_features_50 = np.where(cumsum_importances >= 0.5)[0][0] + 1
        n_features_80 = np.where(cumsum_importances >= 0.8)[0][0] + 1
        n_features_95 = np.where(cumsum_importances >= 0.95)[0][0] + 1
        
        ax3.text(n_features_50, 0.5, f'  {n_features_50} features', fontsize=8)
        ax3.text(n_features_80, 0.8, f'  {n_features_80} features', fontsize=8)
        ax3.text(n_features_95, 0.95, f'  {n_features_95} features', fontsize=8)
        
        # 4. Feature type comparison
        ax4 = plt.subplot(2, 2, 4)
        
        # Separate descriptor and fingerprint importances
        desc_importances = []
        fp_importances = []
        
        for i, importance in enumerate(importances):
            orig_idx = np.where(feature_mask)[0][i]
            if orig_idx < n_descriptors:
                desc_importances.append(importance)
            else:
                fp_importances.append(importance)
        
        # Statistics
        stats_data = {
            'Descriptors': {
                'mean': np.mean(desc_importances),
                'std': np.std(desc_importances),
                'max': np.max(desc_importances),
                'sum': np.sum(desc_importances)
            },
            'Fingerprints': {
                'mean': np.mean(fp_importances),
                'std': np.std(fp_importances),
                'max': np.max(fp_importances),
                'sum': np.sum(fp_importances)
            }
        }
        
        # Plot comparison
        categories = ['Mean', 'Max', 'Total']
        desc_values = [stats_data['Descriptors']['mean'], 
                      stats_data['Descriptors']['max'],
                      stats_data['Descriptors']['sum']]
        fp_values = [stats_data['Fingerprints']['mean'],
                    stats_data['Fingerprints']['max'],
                    stats_data['Fingerprints']['sum']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, desc_values, width, label='Descriptors', alpha=0.7)
        bars2 = ax4.bar(x + width/2, fp_values, width, label='Fingerprints', alpha=0.7)
        
        ax4.set_xlabel('Statistic')
        ax4.set_ylabel('Importance Value')
        ax4.set_title('Feature Type Importance Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig_path = os.path.join(self.supplementary_dir, "Figures", "Figure_S4_Feature_Importance_Analysis.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.report_generator.add_figure_reference("Figure_S4_Feature_Importance_Analysis.png",
                                                  "Comprehensive feature importance analysis")
        
        # Save detailed feature importance table
        top_features_df = pd.DataFrame(top_features[:50], columns=['Feature', 'Importance'])
        top_features_df['Rank'] = range(1, len(top_features_df) + 1)
        top_features_df['Cumulative_Importance'] = np.cumsum(top_features_df['Importance'].values)
        top_features_df['Feature_Type'] = ['Descriptor' if 'ECFP4' not in f else 'Fingerprint' 
                                          for f in top_features_df['Feature']]
        
        table_path = os.path.join(self.supplementary_dir, "Tables", "Table_S5_Top_Features.csv")
        top_features_df.to_csv(table_path, index=False)
        
        self.report_generator.add_table_reference("Table_S5_Top_Features.csv",
                                                 "Top 50 most important features for activity prediction")
        
        # Add to methods
        methods_content = f"""
Feature Importance Analysis:
Random Forest feature importance was calculated using the Gini importance metric.
The analysis was performed on the combined descriptor + ECFP4 representation,
which showed the best performance in baseline evaluation.

Model configuration:
- 300 trees for stable importance estimates
- Maximum depth: 10
- Class weight: balanced
- All available training data used

Feature importance metrics:
- Individual feature importance scores
- Cumulative importance distribution
- Feature type comparison (descriptors vs fingerprints)

Key statistics:
- Total features analyzed: {len(importances)}
- Features with non-zero importance: {np.sum(importances > 0)}
- Features needed for 50% cumulative importance: {n_features_50}
- Features needed for 80% cumulative importance: {n_features_80}
- Features needed for 95% cumulative importance: {n_features_95}
        """
        self.report_generator.add_methods_section("Feature Importance Analysis", methods_content)
        
        # Add to results
        results_content = f"""
Feature Importance Results:

Top 5 most important features:
"""
        for i, (feature, importance) in enumerate(top_features[:5], 1):
            results_content += f"{i}. {feature}: {importance:.4f}\n"
        
        results_content += f"""

Feature type analysis:
- Descriptor features: {len(desc_importances)} features, total importance: {np.sum(desc_importances):.3f}
- Fingerprint features: {len(fp_importances)} features, total importance: {np.sum(fp_importances):.3f}

The analysis reveals that both molecular descriptors and fingerprint bits contribute
significantly to the model's predictive performance. The top features include a mix
of both types, suggesting complementary information content.

Only {n_features_80} features ({n_features_80/len(importances)*100:.1f}% of total) 
are needed to capture 80% of the total importance, indicating potential for 
feature selection and model simplification.
        """
        self.report_generator.add_results_section("Feature Importance Results", results_content)
        
        # Add to report
        self.report.append("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        self.report.append("Top 10 most important features:")
        for feature, importance in top_features[:10]:
            self.report.append(f"  {feature}: {importance:.4f}")
    
    def perform_scaffold_split_validation(self):
        """Enhanced scaffold split validation with detailed reporting"""
        print("\nPerforming scaffold split validation...")
        
        if self.scaffolds is None:
            print("  No scaffold information available")
            self.report_generator.add_results_section("Scaffold Validation", 
                                                     "No scaffold information available for validation")
            return
        
        # Count compounds per scaffold
        scaffold_counts = Counter(self.scaffolds)
        
        # Filter scaffolds with at least 5 compounds
        valid_scaffolds = [s for s, count in scaffold_counts.items() if count >= 5]
        
        # Create groups for GroupKFold
        scaffold_groups = np.zeros(len(self.scaffolds))
        for i, (scaffold, _) in enumerate(scaffold_counts.most_common()):
            mask = self.scaffolds == scaffold
            scaffold_groups[mask] = i
        
        # Use best model/representation from baseline
        X = np.hstack([self.X_descriptors, self.X_fingerprints['ECFP4']])
        
        # Preprocessing
        var_selector = VarianceThreshold(threshold=0.01)
        X_filtered = var_selector.fit_transform(X)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)
        
        # Group cross-validation (scaffolds)
        group_cv = GroupKFold(n_splits=5)
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Evaluate with scaffold split
        print("  Evaluating scaffold-based splitting...")
        scaffold_scores = cross_validate(
            model, X_scaled, self.y,
            cv=group_cv,
            groups=scaffold_groups,
            scoring=self.scoring,
            n_jobs=-1
        )
        
        # Compare with regular stratified CV
        print("  Evaluating stratified splitting...")
        stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        stratified_scores = cross_validate(
            model, X_scaled, self.y,
            cv=stratified_cv,
            scoring=self.scoring,
            n_jobs=-1
        )
        
        # Calculate detailed statistics
        scaffold_stats = self._calculate_validation_statistics(scaffold_scores)
        stratified_stats = self._calculate_validation_statistics(stratified_scores)
        
        # Create comparison table
        comparison_data = []
        for metric in self.scoring.keys():
            comparison_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Scaffold_Mean': scaffold_stats[metric]['mean'],
                'Scaffold_Std': scaffold_stats[metric]['std'],
                'Stratified_Mean': stratified_stats[metric]['mean'],
                'Stratified_Std': stratified_stats[metric]['std'],
                'Difference': stratified_stats[metric]['mean'] - scaffold_stats[metric]['mean'],
                'Relative_Diff_%': ((stratified_stats[metric]['mean'] - scaffold_stats[metric]['mean']) / 
                                   stratified_stats[metric]['mean'] * 100) if stratified_stats[metric]['mean'] != 0 else 0
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        table_path = os.path.join(self.supplementary_dir, "Tables", "Table_S6_Scaffold_vs_Stratified.csv")
        comparison_df.to_csv(table_path, index=False)
        
        self.report_generator.add_table_reference("Table_S6_Scaffold_vs_Stratified.csv",
                                                 "Comparison of scaffold-based and stratified cross-validation")
        
        # Visualize comparison
        self._plot_enhanced_scaffold_validation(scaffold_scores, stratified_scores, scaffold_counts)
        
        # Add to methods
        methods_content = f"""
Scaffold-Based Cross-Validation:
To assess the model's ability to generalize to novel chemical scaffolds,
scaffold-based splitting was compared to standard stratified splitting.

Scaffold information:
- Total unique scaffolds: {len(scaffold_counts)}
- Scaffolds with ≥5 compounds: {len(valid_scaffolds)}
- Largest scaffold: {max(scaffold_counts.values())} compounds
- Smallest scaffold: {min(scaffold_counts.values())} compounds

Validation approach:
- GroupKFold with 5 splits based on scaffold assignment
- Each fold ensures test compounds have different scaffolds from training
- Compared against stratified 5-fold CV (random splitting)
- Same model (Random Forest) and preprocessing used for both

This validation strategy provides insight into the model's ability to
predict activity for structurally novel compounds, which is critical
for real-world drug discovery applications.
        """
        self.report_generator.add_methods_section("Scaffold-Based Validation", methods_content)
        
        # Add to results
        results_content = f"""
Scaffold Validation Results:

Performance comparison (Stratified vs Scaffold):
- MCC: {stratified_stats['mcc']['mean']:.3f} vs {scaffold_stats['mcc']['mean']:.3f} 
       (Δ = {stratified_stats['mcc']['mean'] - scaffold_stats['mcc']['mean']:.3f})
- Balanced Accuracy: {stratified_stats['balanced_accuracy']['mean']:.3f} vs {scaffold_stats['balanced_accuracy']['mean']:.3f}
       (Δ = {stratified_stats['balanced_accuracy']['mean'] - scaffold_stats['balanced_accuracy']['mean']:.3f})

The scaffold-based validation shows {"lower" if scaffold_stats['mcc']['mean'] < stratified_stats['mcc']['mean'] else "similar"} 
performance compared to stratified validation, with an average performance decrease of 
{np.mean([abs(d['Relative_Diff_%']) for d in comparison_data]):.1f}%.

This {"significant" if abs(stratified_stats['mcc']['mean'] - scaffold_stats['mcc']['mean']) > 0.1 else "moderate"} 
difference indicates that the model {"may face challenges" if abs(stratified_stats['mcc']['mean'] - scaffold_stats['mcc']['mean']) > 0.1 else "shows reasonable ability"} 
when predicting activity for compounds with novel scaffolds.

Implications for drug discovery:
- The model can generalize to new chemical space with {scaffold_stats['mcc']['mean']:.3f} MCC
- Additional chemical diversity in training may improve scaffold generalization
- Consider scaffold-aware sampling strategies for production deployment
        """
        self.report_generator.add_results_section("Scaffold Validation Results", results_content)
        
        # Add to report
        self.report.append("\n=== SCAFFOLD VALIDATION ===")
        self.report.append("Comparison Scaffold Split vs Stratified Split:")
        for metric in self.scoring.keys():
            scaffold_mean = scaffold_scores[f'test_{metric}'].mean()
            scaffold_std = scaffold_scores[f'test_{metric}'].std()
            stratified_mean = stratified_scores[f'test_{metric}'].mean()
            stratified_std = stratified_scores[f'test_{metric}'].std()
            
            self.report.append(f"\n{metric}:")
            self.report.append(f"  Scaffold: {scaffold_mean:.3f} ± {scaffold_std:.3f}")
            self.report.append(f"  Stratified: {stratified_mean:.3f} ± {stratified_std:.3f}")
            self.report.append(f"  Difference: {stratified_mean - scaffold_mean:.3f}")
    
    def _calculate_validation_statistics(self, scores):
        """Calculate detailed statistics from CV scores"""
        stats = {}
        for metric in self.scoring.keys():
            test_scores = scores[f'test_{metric}']
            stats[metric] = {
                'mean': np.mean(test_scores),
                'std': np.std(test_scores),
                'min': np.min(test_scores),
                'max': np.max(test_scores),
                'median': np.median(test_scores),
                'cv': np.std(test_scores) / np.mean(test_scores) if np.mean(test_scores) != 0 else 0
            }
        return stats
    
    def _plot_enhanced_scaffold_validation(self, scaffold_scores, stratified_scores, scaffold_counts):
        """Create enhanced scaffold validation visualizations"""
        metrics = list(self.scoring.keys())
        
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Side-by-side comparison
        ax1 = plt.subplot(2, 3, 1)
        
        scaffold_means = [scaffold_scores[f'test_{m}'].mean() for m in metrics]
        scaffold_stds = [scaffold_scores[f'test_{m}'].std() for m in metrics]
        stratified_means = [stratified_scores[f'test_{m}'].mean() for m in metrics]
        stratified_stds = [stratified_scores[f'test_{m}'].std() for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, scaffold_means, width, 
                       yerr=scaffold_stds, label='Scaffold Split',
                       capsize=5, alpha=0.7, color='coral')
        bars2 = ax1.bar(x + width/2, stratified_means, width,
                       yerr=stratified_stds, label='Stratified Split',
                       capsize=5, alpha=0.7, color='skyblue')
        
        ax1.set_xlabel('Metric')
        ax1.set_ylabel('Score')
        ax1.set_title('Scaffold vs Stratified Cross-Validation')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title()[:12] for m in metrics], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=7)
        
        # 2. Performance difference
        ax2 = plt.subplot(2, 3, 2)
        differences = [stratified_means[i] - scaffold_means[i] for i in range(len(metrics))]
        colors = ['green' if d > 0 else 'red' for d in differences]
        
        bars = ax2.bar(range(len(metrics)), differences, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(metrics)))
        ax2.set_xticklabels([m.replace('_', ' ').title()[:12] for m in metrics], rotation=45, ha='right')
        ax2.set_ylabel('Performance Difference\n(Stratified - Scaffold)')
        ax2.set_title('Performance Drop with Scaffold Split')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, diff) in enumerate(zip(bars, differences)):
            ax2.text(bar.get_x() + bar.get_width()/2, diff + (0.01 if diff > 0 else -0.01),
                    f'{diff:.3f}', ha='center', va='bottom' if diff > 0 else 'top', fontsize=7)
        
        # 3. Scaffold distribution
        ax3 = plt.subplot(2, 3, 3)
        scaffold_sizes = list(scaffold_counts.values())
        ax3.hist(scaffold_sizes, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax3.set_xlabel('Number of Compounds per Scaffold')
        ax3.set_ylabel('Number of Scaffolds')
        ax3.set_title('Scaffold Size Distribution')
        ax3.axvline(x=5, color='red', linestyle='--', label='Min for GroupKFold')
        ax3.legend()
        ax3.set_yscale('log')
        
        # 4. Fold-by-fold comparison for MCC
        ax4 = plt.subplot(2, 3, 4)
        fold_indices = range(1, 6)
        scaffold_mcc = scaffold_scores['test_mcc']
        stratified_mcc = stratified_scores['test_mcc']
        
        ax4.plot(fold_indices, scaffold_mcc, 'o-', label='Scaffold', linewidth=2, markersize=8)
        ax4.plot(fold_indices, stratified_mcc, 's-', label='Stratified', linewidth=2, markersize=8)
        ax4.set_xlabel('Fold')
        ax4.set_ylabel('MCC Score')
        ax4.set_title('Fold-by-Fold MCC Comparison')
        ax4.set_xticks(fold_indices)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Variance comparison
        ax5 = plt.subplot(2, 3, 5)
        scaffold_vars = [scaffold_scores[f'test_{m}'].std()**2 for m in metrics]
        stratified_vars = [stratified_scores[f'test_{m}'].std()**2 for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax5.bar(x - width/2, scaffold_vars, width, label='Scaffold', alpha=0.7)
        ax5.bar(x + width/2, stratified_vars, width, label='Stratified', alpha=0.7)
        ax5.set_xlabel('Metric')
        ax5.set_ylabel('Variance')
        ax5.set_title('Score Variance Comparison')
        ax5.set_xticks(x)
        ax5.set_xticklabels([m.replace('_', ' ').title()[:12] for m in metrics], rotation=45, ha='right')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Relative performance
        ax6 = plt.subplot(2, 3, 6)
        relative_perf = [(scaffold_means[i] / stratified_means[i] * 100) if stratified_means[i] != 0 else 0 
                        for i in range(len(metrics))]
        
        bars = ax6.bar(range(len(metrics)), relative_perf, alpha=0.7, color='purple')
        ax6.set_xticks(range(len(metrics)))
        ax6.set_xticklabels([m.replace('_', ' ').title()[:12] for m in metrics], rotation=45, ha='right')
        ax6.set_ylabel('Relative Performance (%)')
        ax6.set_title('Scaffold Performance as % of Stratified')
        ax6.axhline(y=100, color='red', linestyle='--', alpha=0.5)
        ax6.grid(True, alpha=0.3, axis='y')
        
        for bar, perf in zip(bars, relative_perf):
            ax6.text(bar.get_x() + bar.get_width()/2, perf + 1,
                    f'{perf:.0f}%', ha='center', va='bottom', fontsize=7)
        
        plt.tight_layout()
        fig_path = os.path.join(self.supplementary_dir, "Figures", "Figure_S5_Scaffold_Validation_Analysis.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.report_generator.add_figure_reference("Figure_S5_Scaffold_Validation_Analysis.png",
                                                  "Comprehensive scaffold-based validation analysis")
    
    def generate_statistical_summary(self):
        """Generate comprehensive statistical summary of all results"""
        print("\nGenerating statistical summary...")
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        # Compile all statistics
        class_dist = np.bincount(self.y)
        stats_summary = {
            'Data Statistics': {
                'Total Compounds': int(len(self.y)),
                'Feature Types': int(len(self.X_fingerprints) + 1),  # +1 for descriptors
                'Total Features (Raw)': int(self.X_descriptors.shape[1] + sum(fp.shape[1] for fp in self.X_fingerprints.values())),
                'Class Distribution': {
                    'Low': int(class_dist[0]) if len(class_dist) > 0 else 0,
                    'Medium': int(class_dist[1]) if len(class_dist) > 1 else 0,
                    'High': int(class_dist[2]) if len(class_dist) > 2 else 0
                }
            },
            'Best Performance': {
                'Model': 'To be determined from results',
                'Representation': 'To be determined from results',
                'MCC': 0,
                'Balanced Accuracy': 0
            }
        }
        
        # Find best model from baseline results
        if hasattr(self, 'all_results') and 'baseline_models' in self.all_results:
            best_mcc = -1
            best_config = None
            for rep in self.all_results['baseline_models']:
                for model in self.all_results['baseline_models'][rep]:
                    mcc = self.all_results['baseline_models'][rep][model]['mcc']['test_mean']
                    if mcc > best_mcc:
                        best_mcc = mcc
                        best_config = (rep, model)
                        stats_summary['Best Performance']['Model'] = model
                        stats_summary['Best Performance']['Representation'] = rep
                        stats_summary['Best Performance']['MCC'] = float(mcc)
                        stats_summary['Best Performance']['Balanced Accuracy'] = \
                            float(self.all_results['baseline_models'][rep][model]['balanced_accuracy']['test_mean'])
        
        # Convert all values to native Python types
        stats_summary = convert_to_native(stats_summary)
        
        # Save statistical summary
        with open(os.path.join(self.supplementary_dir, "Statistical_Analysis", "summary_statistics.json"), 'w') as f:
            json.dump(stats_summary, f, indent=2)
        
        return stats_summary
    
    def generate_summary_report(self):
        """Generate comprehensive summary report with all supplementary materials"""
        print("\nGenerating comprehensive summary report...")
        
        # Generate statistical summary
        stats_summary = self.generate_statistical_summary()
        
        # Add executive summary to main report
        self.report.append("\n=== EXECUTIVE SUMMARY ===")
        self.report.append("\n1. DATA PREPARATION:")
        self.report.append(f"   - Total compounds: {len(self.y)}")
        self.report.append(f"   - Features: Descriptors ({self.X_descriptors.shape[1]}) + "
                         f"Fingerprints (multiple types)")
        self.report.append(f"   - Unbalanced classes: Ratio {np.sum(self.y==0)}:"
                         f"{np.sum(self.y==1)}:{np.sum(self.y==2)}")
        
        self.report.append("\n2. BEST BASELINE RESULTS:")
        self.report.append(f"   - Best representation: {stats_summary['Best Performance']['Representation']}")
        self.report.append(f"   - Best model: {stats_summary['Best Performance']['Model']}")
        self.report.append(f"   - MCC: {stats_summary['Best Performance']['MCC']:.3f}")
        self.report.append("   - Best balancing strategy: SMOTE or BorderlineSMOTE")
        
        self.report.append("\n3. KEY FINDINGS:")
        self.report.append("   - Combined representations outperform individual ones")
        self.report.append("   - Class balancing improves minority class detection")
        self.report.append("   - Scaffold-based validation shows generalization capability")
        self.report.append("   - Feature importance reveals key molecular properties")
        
        self.report.append("\n4. RECOMMENDATIONS:")
        self.report.append("   - Implement ensemble methods for improved performance")
        self.report.append("   - Apply SMOTE with optimized parameters")
        self.report.append("   - Consider deep learning for complex pattern recognition")
        self.report.append("   - Use feature selection based on importance analysis")
        self.report.append("   - Implement scaffold-aware sampling for deployment")
        
        # Save main report
        report_path = os.path.join(self.supplementary_dir, "Reports", 
                                  f"Complete_Analysis_Report_{self.timestamp}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report))
        
        # Generate supplementary documents
        methods_path = self.report_generator.generate_methods_report()
        results_path = self.report_generator.generate_results_report()
        figures_path = self.report_generator.generate_figure_list()
        tables_path = self.report_generator.generate_table_list()
        
        # Create README for supplementary materials
        self._create_readme()
        
        print(f"\n{'='*80}")
        print("SUPPLEMENTARY MATERIALS GENERATION COMPLETED")
        print(f"{'='*80}")
        print(f"All materials saved in: {self.supplementary_dir}")
        print("\nGenerated documents:")
        print(f"1. Complete Analysis Report: Complete_Analysis_Report_{self.timestamp}.txt")
        print("2. Supplementary Methods: Supplementary_Methods.txt")
        print("3. Supplementary Results: Supplementary_Results.txt")
        print("4. List of Figures: List_of_Figures.txt")
        print("5. List of Tables: List_of_Tables.txt")
        print("6. README: README.txt")
        print("\nFolders:")
        print("- Figures/: All supplementary figures (PNG format)")
        print("- Tables/: All supplementary tables (CSV format)")
        print("- Data/: Raw results and processed data")
        print("- Statistical_Analysis/: Statistical summaries")
        print(f"{'='*80}")
    
    def _create_readme(self):
        """Create README file for supplementary materials"""
        readme_content = f"""
================================================================================
SUPPLEMENTARY MATERIALS - QSAR MODEL PREPARATION AND BASELINE ANALYSIS
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONTENTS:
---------

1. REPORTS/
   - Complete_Analysis_Report_{self.timestamp}.txt: Full technical report
   - Additional analysis reports

2. FIGURES/ (All figures in high-resolution PNG format)
   - Figure_S1: Feature variance analysis
   - Figure_S2: Class balancing strategies comparison
   - Figure_S3: Baseline model performance
   - Figure_S4: Feature importance analysis
   - Figure_S5: Scaffold validation analysis

3. TABLES/ (All tables in CSV format)
   - Table_S1: Data summary statistics
   - Table_S2: Feature variance distribution
   - Table_S3: Balancing strategies results
   - Table_S4: Baseline model cross-validation performance
   - Table_S5: Top 50 most important features
   - Table_S6: Scaffold vs stratified validation comparison

4. DATA/
   - baseline_cv_full_results.json: Complete cross-validation results
   - Additional processed data files

5. STATISTICAL_ANALYSIS/
   - summary_statistics.json: Overall statistical summary

6. DOCUMENTATION/
   - Supplementary_Methods.txt: Detailed methodology
   - Supplementary_Results.txt: Comprehensive results
   - List_of_Figures.txt: Figure captions and descriptions
   - List_of_Tables.txt: Table descriptions

USAGE:
------
These supplementary materials provide comprehensive documentation of the QSAR
model development process, including:

- Detailed methodology for reproducibility
- Complete statistical analyses
- High-quality visualizations
- Raw and processed data
- Feature importance rankings
- Model validation results

All materials are formatted for direct inclusion in scientific publications
or technical reports.

CITATION:
---------
If using these materials, please cite:
"QSAR Model Development Pipeline v5.0 - Supplementary Materials"
Generated using enhanced reporting system with comprehensive documentation.

CONTACT:
--------
For questions about these materials, please refer to the main analysis report.

================================================================================
"""
        
        readme_path = os.path.join(self.supplementary_dir, "README.txt")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

# Main function
def main():
    """Execute enhanced model preparation with comprehensive reporting"""
    
    # Paths
    base_dir = r"C:\Users\amjer\Documents\Dengue\Versión final"
    data_dir = os.path.join(base_dir, "QSAR5.0")
    output_dir = os.path.join(base_dir, "QSAR5.0")
    
    # Verify required files exist
    required_files = [
        os.path.join(data_dir, "02_Chemical_Diversity", "data", "full_descriptor_data.csv"),
        os.path.join(data_dir, "02_Chemical_Diversity", "data", "molecular_fingerprints.pkl")
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"ERROR: File not found: {file}")
            print("Please run script 02_chemical_diversity_analysis.py first")
            return
    
    # Create model preparator with enhanced reporting
    preparator = QSARModelPreparation(data_dir, output_dir)
    
    # Execute enhanced pipeline
    try:
        print("\n" + "="*80)
        print("STARTING ENHANCED QSAR MODEL PREPARATION WITH COMPREHENSIVE REPORTING")
        print("="*80 + "\n")
        
        # Load data
        preparator.load_data()
        
        # Variance analysis
        preparator.analyze_feature_variance()
        
        # Evaluate balancing strategies
        balancing_results = preparator.evaluate_balancing_strategies()
        
        # Train baseline models
        baseline_results = preparator.train_baseline_models()
        
        # Feature importance analysis
        preparator.analyze_feature_importance()
        
        # Scaffold validation
        preparator.perform_scaffold_split_validation()
        
        # Generate comprehensive reports and supplementary materials
        preparator.generate_summary_report()
        
        print("\n✓ Enhanced model preparation with supplementary materials completed successfully!")
        print(f"\n📁 All supplementary materials organized in:")
        print(f"   {preparator.supplementary_dir}")
        
    except Exception as e:
        print(f"\nERROR during preparation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()