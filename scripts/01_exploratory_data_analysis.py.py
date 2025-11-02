#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QSAR 5.0 - Script 01: Exploratory Data Analysis (EDA) - Complete Version
=========================================================================
Autor: Sergio Montenegro
Analysis with proper IC₅₀/EC₅₀ rendering for updated dataset with 53 molecules having both assays
Save this file as: eda_analysis_complete.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Style configuration for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['font.family'] = 'sans-serif'

class DengueQSARAnalyzer:
    """Complete class for exploratory data analysis of anti-dengue QSAR data"""
    
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.base_dir = os.path.dirname(data_path)
        self.output_dir = os.path.join(self.base_dir, output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure for supplementary materials
        self.create_directory_structure()
        
        # Load data
        print("Loading data...")
        self.df = pd.read_csv(data_path, encoding='utf-8')
        print(f"Data loaded: {self.df.shape[0]} compounds, {self.df.shape[1]} columns")
        
        # Initialize report
        self.report = []
        
        # Create supplementary tables list
        self.supplementary_tables = {}
        
    def create_directory_structure(self):
        """Create folder structure for organizing results and supplementary materials"""
        folders = [
            self.output_dir,
            os.path.join(self.output_dir, "01_EDA"),
            os.path.join(self.output_dir, "01_EDA", "figures"),
            os.path.join(self.output_dir, "01_EDA", "data"),
            os.path.join(self.output_dir, "01_EDA", "reports"),
            os.path.join(self.output_dir, "01_EDA", "supplementary"),
            os.path.join(self.output_dir, "01_EDA", "supplementary", "figures"),
            os.path.join(self.output_dir, "01_EDA", "supplementary", "tables")
        ]
        
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            
    def analyze_activity_distribution(self):
        """Analyze biological activity distribution considering IC50 and EC50"""
        print("\n1. Analyzing biological activity distribution...")
        
        # Separate by assay type
        ic50_data = self.df[self.df['Standard Type'] == 'IC50']
        ec50_data = self.df[self.df['Standard Type'] == 'EC50']
        
        # Identify molecules with both assays
        molecule_counts = self.df.groupby('Molecule ChEMBL ID')['Standard Type'].nunique()
        molecules_with_both = molecule_counts[molecule_counts == 2].index.tolist()
        
        self.report.append("=== BIOLOGICAL ACTIVITY DISTRIBUTION ===\n")
        self.report.append(f"\nData by assay type:")
        self.report.append(f"- IC50: {len(ic50_data)} compounds ({len(ic50_data)/len(self.df)*100:.1f}%)")
        self.report.append(f"- EC50: {len(ec50_data)} compounds ({len(ec50_data)/len(self.df)*100:.1f}%)")
        self.report.append(f"- Unique molecules: {self.df['Molecule ChEMBL ID'].nunique()}")
        self.report.append(f"- Molecules with both assays: {len(molecules_with_both)}\n")
        
        # Basic statistics
        activity_stats = self.df['pActivity_calculado'].describe()
        self.report.append(f"Overall pActivity statistics:\n{activity_stats}\n")
        
        # Statistics by assay type
        ic50_stats = ic50_data['pActivity_calculado'].describe()
        ec50_stats = ec50_data['pActivity_calculado'].describe()
        
        # Save as supplementary table
        stats_comparison = pd.DataFrame({
            'Overall': activity_stats,
            'IC50': ic50_stats,
            'EC50': ec50_stats
        })
        self.supplementary_tables['Table_S1_Activity_Statistics'] = stats_comparison
        
        # Define potency categories
        def categorize_potency(pActivity):
            if pActivity > 6:
                return 'High'
            elif pActivity > 5:
                return 'Medium'
            else:
                return 'Low'
        
        self.df['potency_category'] = self.df['pActivity_calculado'].apply(categorize_potency)
        
        # Create main distribution figure (2x3 layout)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Overall pActivity histogram
        axes[0,0].hist(self.df['pActivity_calculado'], bins=50, alpha=0.7, 
                      color='skyblue', edgecolor='black')
        axes[0,0].axvline(5, color='orange', linestyle='--', linewidth=2,
                         label='Medium/Low threshold (5)')
        axes[0,0].axvline(6, color='red', linestyle='--', linewidth=2,
                         label='High/Medium threshold (6)')
        axes[0,0].set_xlabel(r'pActivity (-log$_{10}$ M)', fontsize=12)
        axes[0,0].set_ylabel('Frequency', fontsize=12)
        axes[0,0].set_title('Overall pActivity Distribution', fontsize=14)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. IC50 vs EC50 overlapped histograms
        axes[0,1].hist(ic50_data['pActivity_calculado'], bins=30, alpha=0.5, 
                      label='IC$_{50}$', color='blue', edgecolor='black')
        axes[0,1].hist(ec50_data['pActivity_calculado'], bins=30, alpha=0.5, 
                      label='EC$_{50}$', color='red', edgecolor='black')
        axes[0,1].axvline(5, color='orange', linestyle='--', linewidth=1)
        axes[0,1].axvline(6, color='darkred', linestyle='--', linewidth=1)
        axes[0,1].set_xlabel(r'pActivity (-log$_{10}$ M)', fontsize=12)
        axes[0,1].set_ylabel('Frequency', fontsize=12)
        axes[0,1].set_title('pActivity Distribution: IC$_{50}$ vs EC$_{50}$', fontsize=14)
        axes[0,1].legend(loc='upper right')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Cumulative distribution by assay type
        sorted_ic50 = np.sort(ic50_data['pActivity_calculado'])
        sorted_ec50 = np.sort(ec50_data['pActivity_calculado'])
        
        axes[0,2].plot(sorted_ic50, np.arange(len(sorted_ic50))/len(sorted_ic50), 
                      label='IC$_{50}$', color='blue', linewidth=2)
        axes[0,2].plot(sorted_ec50, np.arange(len(sorted_ec50))/len(sorted_ec50), 
                      label='EC$_{50}$', color='red', linewidth=2)
        axes[0,2].axhline(0.1, color='gray', linestyle='--', alpha=0.5)
        axes[0,2].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        axes[0,2].axhline(0.9, color='gray', linestyle='--', alpha=0.5)
        axes[0,2].set_xlabel(r'pActivity (-log$_{10}$ M)', fontsize=12)
        axes[0,2].set_ylabel('Cumulative Proportion', fontsize=12)
        axes[0,2].set_title('Cumulative Distribution Function', fontsize=14)
        axes[0,2].legend(loc='best')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Boxplot by potency category
        data_for_plot = []
        labels_for_plot = []
        colors_for_plot = []
        
        for category in ['High', 'Medium', 'Low']:
            cat_data = self.df[self.df['potency_category'] == category]['pActivity_calculado']
            if len(cat_data) > 0:
                data_for_plot.append(cat_data)
                labels_for_plot.append(category)
                if category == 'High':
                    colors_for_plot.append('#ff6b6b')
                elif category == 'Medium':
                    colors_for_plot.append('#4ecdc4')
                else:
                    colors_for_plot.append('#45b7d1')
        
        bp = axes[1,0].boxplot(data_for_plot, labels=labels_for_plot, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_for_plot):
            patch.set_facecolor(color)
        axes[1,0].set_ylabel(r'pActivity (-log$_{10}$ M)', fontsize=12)
        axes[1,0].set_xlabel('Potency Category', fontsize=12)
        axes[1,0].set_title('pActivity by Potency Category', fontsize=14)
        axes[1,0].grid(True, alpha=0.3, axis='y')
        
        # 5. Category counts by assay type
        category_counts = pd.crosstab(self.df['Standard Type'], self.df['potency_category'])
        
        x = np.arange(2)
        width = 0.25
        
        high_counts = [category_counts.loc['EC50', 'High'] if 'High' in category_counts.columns and 'EC50' in category_counts.index else 0,
                       category_counts.loc['IC50', 'High'] if 'High' in category_counts.columns and 'IC50' in category_counts.index else 0]
        low_counts = [category_counts.loc['EC50', 'Low'] if 'Low' in category_counts.columns and 'EC50' in category_counts.index else 0,
                      category_counts.loc['IC50', 'Low'] if 'Low' in category_counts.columns and 'IC50' in category_counts.index else 0]
        medium_counts = [category_counts.loc['EC50', 'Medium'] if 'Medium' in category_counts.columns and 'EC50' in category_counts.index else 0,
                         category_counts.loc['IC50', 'Medium'] if 'Medium' in category_counts.columns and 'IC50' in category_counts.index else 0]
        
        axes[1,1].bar(x - width, high_counts, width, label='High', color='#ff6b6b')
        axes[1,1].bar(x, low_counts, width, label='Low', color='#45b7d1')
        axes[1,1].bar(x + width, medium_counts, width, label='Medium', color='#4ecdc4')
        
        axes[1,1].set_xlabel('Assay Type', fontsize=12)
        axes[1,1].set_ylabel('Number of Compounds', fontsize=12)
        axes[1,1].set_title('Potency Distribution by Assay Type', fontsize=14)
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(['EC$_{50}$', 'IC$_{50}$'])
        axes[1,1].legend(title='Category')
        axes[1,1].grid(True, alpha=0.3, axis='y')
        
        # 6. IC50 vs EC50 correlation for molecules with both
        if len(molecules_with_both) > 0:
            ic50_values = []
            ec50_values = []
            for mol_id in molecules_with_both:
                mol_data = self.df[self.df['Molecule ChEMBL ID'] == mol_id]
                ic50_val = mol_data[mol_data['Standard Type'] == 'IC50']['pActivity_calculado'].values
                ec50_val = mol_data[mol_data['Standard Type'] == 'EC50']['pActivity_calculado'].values
                if len(ic50_val) > 0 and len(ec50_val) > 0:
                    ic50_values.append(ic50_val[0])
                    ec50_values.append(ec50_val[0])
            
            if len(ic50_values) > 0:
                axes[1,2].scatter(ic50_values, ec50_values, alpha=0.6, s=50, color='purple')
                
                # Add trend line
                z = np.polyfit(ic50_values, ec50_values, 1)
                p = np.poly1d(z)
                axes[1,2].plot(sorted(ic50_values), p(sorted(ic50_values)), 
                              "r--", alpha=0.8, linewidth=2)
                
                # Add diagonal line
                min_val = min(ic50_values + ec50_values)
                max_val = max(ic50_values + ec50_values)
                axes[1,2].plot([min_val, max_val], [min_val, max_val], 
                              'k--', alpha=0.5, linewidth=1, label='y=x')
                
                corr = np.corrcoef(ic50_values, ec50_values)[0,1]
                axes[1,2].set_xlabel(r'pActivity IC$_{50}$', fontsize=12)
                axes[1,2].set_ylabel(r'pActivity EC$_{50}$', fontsize=12)
                axes[1,2].set_title(f'IC$_{{50}}$ vs EC$_{{50}}$ Correlation\n({len(molecules_with_both)} molecules, r={corr:.3f})', 
                                   fontsize=14)
                axes[1,2].legend()
                axes[1,2].grid(True, alpha=0.3)
                
                self.report.append(f"\nIC50-EC50 Correlation: {corr:.3f}")
            else:
                axes[1,2].text(0.5, 0.5, 'Insufficient data for\ncorrelation analysis', 
                              ha='center', va='center', transform=axes[1,2].transAxes, fontsize=12)
                axes[1,2].set_title('IC$_{50}$ vs EC$_{50}$ Correlation', fontsize=14)
        else:
            axes[1,2].text(0.5, 0.5, 'No molecules with\nboth assay types', 
                          ha='center', va='center', transform=axes[1,2].transAxes, fontsize=12)
            axes[1,2].set_title('IC$_{50}$ vs EC$_{50}$ Correlation', fontsize=14)
        
        plt.tight_layout()
        
        # Save figures
        plt.savefig(os.path.join(self.output_dir, "01_EDA", "figures", "Figure_1_activity_distribution.png"), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, "01_EDA", "supplementary", "figures", 
                                "Figure_S1_activity_distribution.pdf"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Statistical test
        statistic, pvalue = stats.mannwhitneyu(ic50_data['pActivity_calculado'], 
                                               ec50_data['pActivity_calculado'], 
                                               alternative='two-sided')
        
        self.report.append(f"\nMann-Whitney U test (IC50 vs EC50):")
        self.report.append(f"Statistic: {statistic:.3f}, p-value: {pvalue:.4f}")

    def analyze_molecular_properties(self):
        """Analyze basic molecular properties"""
        print("\n2. Analyzing molecular properties...")
        
        # Calculate additional descriptors if not present
        if 'n_rings' not in self.df.columns:
            print("   Calculating additional molecular descriptors...")
            mols = [Chem.MolFromSmiles(smiles) if smiles else None 
                   for smiles in self.df['SMILES_original'] if pd.notna(smiles)]
            
            # Only calculate if we have valid molecules
            if mols:
                self.df['n_rings'] = [Chem.rdMolDescriptors.CalcNumRings(mol) if mol else None for mol in mols[:len(self.df)]]
                self.df['n_aromatic_rings'] = [Chem.rdMolDescriptors.CalcNumAromaticRings(mol) if mol else None for mol in mols[:len(self.df)]]
                self.df['n_heteroatoms'] = [Chem.rdMolDescriptors.CalcNumHeteroatoms(mol) if mol else None for mol in mols[:len(self.df)]]
                self.df['n_rotatable_bonds'] = [Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) if mol else None for mol in mols[:len(self.df)]]
        
        # Properties to analyze
        properties = ['MW_calculado', 'LogP_calculado', 'NumHBA', 'NumHBD', 'TPSA']
        
        # Add additional properties if they exist
        for prop in ['n_rings', 'n_aromatic_rings', 'n_heteroatoms', 'n_rotatable_bonds']:
            if prop in self.df.columns:
                properties.append(prop)
        
        # Create property-activity correlation plots
        n_props = len(properties)
        n_cols = 3
        n_rows = (n_props + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        correlations = {}
        for i, prop in enumerate(properties):
            if prop in self.df.columns:
                x = self.df[prop]
                y = self.df['pActivity_calculado']
                
                mask = ~(x.isna() | y.isna())
                if mask.sum() > 0:
                    x_clean = x[mask]
                    y_clean = y[mask]
                    
                    axes[i].scatter(x_clean, y_clean, alpha=0.5, s=20, color='steelblue')
                    
                    if len(x_clean) > 1:
                        z = np.polyfit(x_clean, y_clean, 1)
                        p = np.poly1d(z)
                        axes[i].plot(x_clean, p(x_clean), "r--", alpha=0.8, linewidth=2)
                        
                        corr = np.corrcoef(x_clean, y_clean)[0,1]
                        correlations[prop] = corr
                    else:
                        corr = 0
                    
                    # Format property names for display
                    display_name = prop.replace('_calculado', '').replace('_', ' ')
                    
                    axes[i].set_xlabel(display_name, fontsize=11)
                    axes[i].set_ylabel(r'pActivity (-log$_{10}$ M)', fontsize=11)
                    axes[i].set_title(f'{display_name}\n(r = {corr:.3f})', fontsize=12)
                    axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(properties), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "01_EDA", "figures", 
                                "Figure_2_property_correlations.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Lipinski's Rule of Five analysis
        print("   Analyzing Lipinski's Rule of Five...")
        lipinski_violations = []
        
        for idx, row in self.df.iterrows():
            violations = 0
            if row['MW_calculado'] > 500: violations += 1
            if row['LogP_calculado'] > 5: violations += 1
            if row['NumHBA'] > 10: violations += 1
            if row['NumHBD'] > 5: violations += 1
            lipinski_violations.append(violations)
        
        self.df['lipinski_violations'] = lipinski_violations
        
        # Save statistics
        self.supplementary_tables['Table_S3_Property_Stats'] = self.df[properties].describe()

    def analyze_chemical_space(self):
        """Analyze chemical space using PCA, t-SNE and UMAP"""
        print("\n3. Analyzing chemical space...")
        
        # Prepare data
        feature_cols = ['MW_calculado', 'LogP_calculado', 'NumHBA', 'NumHBD', 'TPSA']
        
        available_features = [col for col in feature_cols if col in self.df.columns]
        X = self.df[available_features].dropna()
        
        if len(X) < 10:
            print("   Insufficient data for chemical space analysis")
            return
            
        valid_indices = X.index
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        print("   Applying PCA...")
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components_90 = np.argmax(cumsum_var >= 0.90) + 1
        
        # t-SNE
        print("   Applying t-SNE...")
        perplexity = min(30, len(X)//4)
        tsne = TSNE(n_components=2, random_state=42, perplexity=max(5, perplexity))
        X_tsne = tsne.fit_transform(X_scaled)
        
        # UMAP
        print("   Applying UMAP...")
        n_neighbors = min(15, len(X)//10)
        reducer = umap.UMAP(n_neighbors=max(2, n_neighbors), min_dist=0.1, random_state=42)
        X_umap = reducer.fit_transform(X_scaled)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        
        colors = self.df.loc[valid_indices, 'potency_category'].map({
            'High': '#ff6b6b', 'Medium': '#4ecdc4', 'Low': '#45b7d1'
        })
        
        # PCA variance plots
        axes[0,0].plot(range(1, len(pca.explained_variance_ratio_)+1), 
                      pca.explained_variance_ratio_, 'bo-')
        axes[0,0].set_xlabel('Principal Component')
        axes[0,0].set_ylabel('Explained Variance')
        axes[0,0].set_title(f'Variance Explained\n({n_components_90} components for 90%)')
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].plot(range(1, len(cumsum_var)+1), cumsum_var, 'go-')
        axes[0,1].axhline(y=0.90, color='r', linestyle='--', alpha=0.5)
        axes[0,1].set_xlabel('Number of Components')
        axes[0,1].set_ylabel('Cumulative Explained Variance')
        axes[0,1].set_title('Cumulative Variance')
        axes[0,1].grid(True, alpha=0.3)
        
        # Scatter plots
        axes[0,2].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=30)
        axes[0,2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        axes[0,2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        axes[0,2].set_title('PCA Projection')
        
        axes[1,0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=0.6, s=30)
        axes[1,0].set_xlabel('t-SNE 1')
        axes[1,0].set_ylabel('t-SNE 2')
        axes[1,0].set_title('t-SNE Projection')
        
        axes[1,1].scatter(X_umap[:, 0], X_umap[:, 1], c=colors, alpha=0.6, s=30)
        axes[1,1].set_xlabel('UMAP 1')
        axes[1,1].set_ylabel('UMAP 2')
        axes[1,1].set_title('UMAP Projection')
        
        # Legend
        axes[1,2].axis('off')
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#ff6b6b', label='High Potency'),
                         Patch(facecolor='#4ecdc4', label='Medium Potency'),
                         Patch(facecolor='#45b7d1', label='Low Potency')]
        axes[1,2].legend(handles=legend_elements, loc='center', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "01_EDA", "figures", 
                                "Figure_5_chemical_space.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def save_supplementary_materials(self):
        """Save all supplementary tables in Excel format"""
        print("\n4. Saving supplementary materials...")
        
        excel_path = os.path.join(self.output_dir, "01_EDA", "supplementary", 
                                 "Supplementary_Tables_EDA.xlsx")
        
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for table_name, table_data in self.supplementary_tables.items():
                    sheet_name = table_name.replace('Table_', '')[:31]
                    table_data.to_excel(writer, sheet_name=sheet_name)
            print(f"   Supplementary tables saved to: {excel_path}")
        except Exception as e:
            print(f"   Warning: Could not save Excel file: {e}")
        
        # Save processed data
        processed_data_path = os.path.join(self.output_dir, "01_EDA", "supplementary", "tables",
                                          "Data_S1_processed_compounds.csv")
        self.df.to_csv(processed_data_path, index=False)
        print(f"   Processed data saved to: {processed_data_path}")

    def generate_summary_report(self):
        """Generate comprehensive and detailed summary report"""
        print("\n5. Generating detailed summary report...")
        
        # Clear report and start fresh
        self.report = []
        
        # Header
        self.report.append("="*80)
        self.report.append("QSAR ANTI-DENGUE EXPLORATORY DATA ANALYSIS - DETAILED REPORT")
        self.report.append("="*80)
        self.report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.report.append(f"Data file: {self.data_path}")
        self.report.append(f"Output directory: {self.output_dir}\n")
        
        # 1. DATASET OVERVIEW
        self.report.append("="*80)
        self.report.append("1. DATASET OVERVIEW")
        self.report.append("="*80)
        self.report.append(f"Total records: {len(self.df)}")
        self.report.append(f"Unique molecules: {self.df['Molecule ChEMBL ID'].nunique()}")
        self.report.append(f"Total features: {len(self.df.columns)}")
        self.report.append(f"Features: {', '.join(self.df.columns.tolist())}\n")
        
        # 2. BIOLOGICAL ACTIVITY DISTRIBUTION
        self.report.append("="*80)
        self.report.append("2. BIOLOGICAL ACTIVITY DISTRIBUTION")
        self.report.append("="*80)
        
        # Assay type distribution
        ic50_count = len(self.df[self.df['Standard Type'] == 'IC50'])
        ec50_count = len(self.df[self.df['Standard Type'] == 'EC50'])
        
        self.report.append("\n2.1 Assay Type Distribution:")
        self.report.append(f"    - IC50: {ic50_count} compounds ({ic50_count/len(self.df)*100:.2f}%)")
        self.report.append(f"    - EC50: {ec50_count} compounds ({ec50_count/len(self.df)*100:.2f}%)")
        
        # Molecules with both assays
        molecule_counts = self.df.groupby('Molecule ChEMBL ID')['Standard Type'].nunique()
        molecules_with_both = molecule_counts[molecule_counts == 2].index.tolist()
        molecules_ic50_only = self.df[(self.df['Standard Type'] == 'IC50') & 
                                      ~(self.df['Molecule ChEMBL ID'].isin(molecules_with_both))]['Molecule ChEMBL ID'].nunique()
        molecules_ec50_only = self.df[(self.df['Standard Type'] == 'EC50') & 
                                      ~(self.df['Molecule ChEMBL ID'].isin(molecules_with_both))]['Molecule ChEMBL ID'].nunique()
        
        self.report.append(f"\n2.2 Molecule-level Analysis:")
        self.report.append(f"    - Molecules with both IC50 and EC50: {len(molecules_with_both)}")
        self.report.append(f"    - Molecules with IC50 only: {molecules_ic50_only}")
        self.report.append(f"    - Molecules with EC50 only: {molecules_ec50_only}")
        
        # pActivity statistics
        self.report.append("\n2.3 pActivity Statistics:")
        self.report.append("    Overall:")
        for stat_name, stat_value in self.df['pActivity_calculado'].describe().items():
            self.report.append(f"        {stat_name}: {stat_value:.4f}")
        
        # IC50 specific stats
        ic50_data = self.df[self.df['Standard Type'] == 'IC50']['pActivity_calculado']
        self.report.append("\n    IC50 pActivity:")
        for stat_name, stat_value in ic50_data.describe().items():
            self.report.append(f"        {stat_name}: {stat_value:.4f}")
        
        # EC50 specific stats
        ec50_data = self.df[self.df['Standard Type'] == 'EC50']['pActivity_calculado']
        self.report.append("\n    EC50 pActivity:")
        for stat_name, stat_value in ec50_data.describe().items():
            self.report.append(f"        {stat_name}: {stat_value:.4f}")
        
        # IC50-EC50 correlation
        if len(molecules_with_both) > 0:
            ic50_values = []
            ec50_values = []
            for mol_id in molecules_with_both:
                mol_data = self.df[self.df['Molecule ChEMBL ID'] == mol_id]
                ic50_val = mol_data[mol_data['Standard Type'] == 'IC50']['pActivity_calculado'].values
                ec50_val = mol_data[mol_data['Standard Type'] == 'EC50']['pActivity_calculado'].values
                if len(ic50_val) > 0 and len(ec50_val) > 0:
                    ic50_values.append(ic50_val[0])
                    ec50_values.append(ec50_val[0])
            
            if len(ic50_values) > 1:
                corr = np.corrcoef(ic50_values, ec50_values)[0,1]
                self.report.append(f"\n2.4 IC50-EC50 Correlation Analysis:")
                self.report.append(f"    - Number of molecules: {len(ic50_values)}")
                self.report.append(f"    - Correlation coefficient: {corr:.4f}")
                self.report.append(f"    - Mean difference (IC50 - EC50): {np.mean(np.array(ic50_values) - np.array(ec50_values)):.4f}")
                self.report.append(f"    - Std difference: {np.std(np.array(ic50_values) - np.array(ec50_values)):.4f}")
        
        # 3. POTENCY CATEGORIES
        self.report.append("\n" + "="*80)
        self.report.append("3. POTENCY CATEGORY ANALYSIS")
        self.report.append("="*80)
        
        if 'potency_category' in self.df.columns:
            potency_dist = self.df['potency_category'].value_counts()
            
            self.report.append("\n3.1 Overall Distribution:")
            for category in ['High', 'Medium', 'Low']:
                if category in potency_dist.index:
                    count = potency_dist[category]
                    percentage = count/len(self.df)*100
                    self.report.append(f"    - {category} potency (pActivity {'> 6' if category=='High' else '5-6' if category=='Medium' else '< 5'}): {count} ({percentage:.2f}%)")
            
            # By assay type
            self.report.append("\n3.2 Distribution by Assay Type:")
            crosstab = pd.crosstab(self.df['Standard Type'], self.df['potency_category'])
            for assay in ['IC50', 'EC50']:
                if assay in crosstab.index:
                    self.report.append(f"\n    {assay}:")
                    for category in ['High', 'Medium', 'Low']:
                        if category in crosstab.columns:
                            count = crosstab.loc[assay, category]
                            percentage = count/crosstab.loc[assay].sum()*100
                            self.report.append(f"        - {category}: {count} ({percentage:.2f}%)")
            
            # Class imbalance analysis
            total_high = potency_dist.get('High', 0)
            total_medium = potency_dist.get('Medium', 0)
            total_low = potency_dist.get('Low', 0)
            
            self.report.append("\n3.3 Class Imbalance Analysis:")
            if total_high > 0:
                self.report.append(f"    - Ratio Low:Medium:High = {total_low/total_high:.2f}:{total_medium/total_high:.2f}:1")
                self.report.append(f"    - Imbalance factor: {total_low/total_high:.2f} (Low vs High)")
        
        # 4. MOLECULAR PROPERTIES
        self.report.append("\n" + "="*80)
        self.report.append("4. MOLECULAR PROPERTIES ANALYSIS")
        self.report.append("="*80)
        
        properties = ['MW_calculado', 'LogP_calculado', 'NumHBA', 'NumHBD', 'TPSA']
        
        self.report.append("\n4.1 Descriptor Statistics:")
        for prop in properties:
            if prop in self.df.columns:
                self.report.append(f"\n    {prop}:")
                for stat_name, stat_value in self.df[prop].describe().items():
                    self.report.append(f"        {stat_name}: {stat_value:.4f}")
        
        # Lipinski violations
        if 'lipinski_violations' in self.df.columns:
            self.report.append("\n4.2 Lipinski's Rule of Five Analysis:")
            violations_dist = self.df['lipinski_violations'].value_counts().sort_index()
            for n_violations, count in violations_dist.items():
                percentage = count/len(self.df)*100
                self.report.append(f"    - {n_violations} violations: {count} compounds ({percentage:.2f}%)")
            
            # Average violations by potency
            if 'potency_category' in self.df.columns:
                self.report.append("\n    Average violations by potency:")
                for category in ['High', 'Medium', 'Low']:
                    cat_data = self.df[self.df['potency_category'] == category]
                    if len(cat_data) > 0:
                        avg_violations = cat_data['lipinski_violations'].mean()
                        self.report.append(f"        - {category}: {avg_violations:.3f}")
        
        # Correlations with activity
        self.report.append("\n4.3 Property-Activity Correlations:")
        for prop in properties:
            if prop in self.df.columns:
                mask = ~(self.df[prop].isna() | self.df['pActivity_calculado'].isna())
                if mask.sum() > 1:
                    corr = np.corrcoef(self.df.loc[mask, prop], 
                                       self.df.loc[mask, 'pActivity_calculado'])[0,1]
                    self.report.append(f"    - {prop}: r = {corr:.4f}")
        
        # 5. STATISTICAL TESTS
        self.report.append("\n" + "="*80)
        self.report.append("5. STATISTICAL TESTS")
        self.report.append("="*80)
        
        # Mann-Whitney U test
        ic50_activity = self.df[self.df['Standard Type'] == 'IC50']['pActivity_calculado']
        ec50_activity = self.df[self.df['Standard Type'] == 'EC50']['pActivity_calculado']
        
        if len(ic50_activity) > 0 and len(ec50_activity) > 0:
            statistic, pvalue = stats.mannwhitneyu(ic50_activity, ec50_activity, alternative='two-sided')
            self.report.append(f"\n5.1 Mann-Whitney U Test (IC50 vs EC50):")
            self.report.append(f"    - U-statistic: {statistic:.2f}")
            self.report.append(f"    - p-value: {pvalue:.6f}")
            self.report.append(f"    - Significant difference: {'Yes (p < 0.05)' if pvalue < 0.05 else 'No (p >= 0.05)'}")
        
        # 6. DATA QUALITY
        self.report.append("\n" + "="*80)
        self.report.append("6. DATA QUALITY ASSESSMENT")
        self.report.append("="*80)
        
        # Missing values
        self.report.append("\n6.1 Missing Values:")
        missing_counts = self.df.isnull().sum()
        for col in missing_counts[missing_counts > 0].index:
            missing_pct = missing_counts[col]/len(self.df)*100
            self.report.append(f"    - {col}: {missing_counts[col]} ({missing_pct:.2f}%)")
        
        if missing_counts.sum() == 0:
            self.report.append("    No missing values in key columns")
        
        # 7. RECOMMENDATIONS
        self.report.append("\n" + "="*80)
        self.report.append("7. MODELING RECOMMENDATIONS")
        self.report.append("="*80)
        
        self.report.append("\n7.1 Data Preprocessing:")
        self.report.append("    - Include assay type (IC50/EC50) as a categorical feature")
        self.report.append(f"    - Leverage {len(molecules_with_both)} molecules with both assays for validation")
        self.report.append("    - Consider standardization of molecular descriptors")
        
        self.report.append("\n7.2 Class Imbalance Handling:")
        if 'potency_category' in self.df.columns and total_high > 0:
            self.report.append(f"    - Minority class (High potency): {total_high} samples")
            self.report.append("    - Recommended techniques:")
            self.report.append("        * SMOTE or ADASYN for oversampling")
            self.report.append("        * Class weight balancing")
            self.report.append("        * Stratified cross-validation")
        
        self.report.append("\n7.3 Model Selection:")
        self.report.append("    - Non-linear models recommended (Random Forest, XGBoost, ExtraTrees)")
        self.report.append("    - Consider ensemble methods for improved performance")
        self.report.append("    - Multi-task learning for IC50/EC50 prediction")
        
        self.report.append("\n7.4 Evaluation Metrics:")
        self.report.append("    - Matthews Correlation Coefficient (MCC)")
        self.report.append("    - Balanced Accuracy")
        self.report.append("    - F1-macro score")
        self.report.append("    - Area Under ROC Curve (AUC-ROC)")
        
        # 8. FILES GENERATED
        self.report.append("\n" + "="*80)
        self.report.append("8. OUTPUT FILES GENERATED")
        self.report.append("="*80)
        
        self.report.append("\nFigures:")
        self.report.append("    - Figure_1_activity_distribution.png")
        self.report.append("    - Figure_2_property_correlations.png")
        self.report.append("    - Figure_5_chemical_space.png")
        
        self.report.append("\nSupplementary Materials:")
        self.report.append("    - Supplementary_Tables_EDA.xlsx")
        self.report.append("    - Data_S1_processed_compounds.csv")
        
        # Footer
        self.report.append("\n" + "="*80)
        self.report.append("END OF REPORT")
        self.report.append("="*80)
        
        # Save report
        report_path = os.path.join(self.output_dir, "01_EDA", "reports", 
                                  f"EDA_detailed_report_{self.timestamp}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report))
        
        print(f"   Detailed report saved to: {report_path}")
        
        # Also print summary to console
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Total compounds analyzed: {len(self.df)}")
        print(f"Unique molecules: {self.df['Molecule ChEMBL ID'].nunique()}")
        print(f"Molecules with both assays: {len(molecules_with_both)}")
        print(f"Report saved to: {report_path}")
        print("="*60)

def main():
    """Execute complete exploratory analysis"""
    
    # Configuration - ADJUST THESE PATHS AS NEEDED
    data_path = r"C:\Users\amjer\Documents\Dengue\Versión final\chembl_antidengue_procesado.csv"
    output_dir = "QSAR5.0"
    
    # Check file exists
    if not os.path.exists(data_path):
        print(f"ERROR: File not found at {data_path}")
        return
    
    # Create analyzer and run analysis
    try:
        analyzer = DengueQSARAnalyzer(data_path, output_dir)
        
        # Run all analysis steps
        analyzer.analyze_activity_distribution()
        analyzer.analyze_molecular_properties()
        analyzer.analyze_chemical_space()
        analyzer.save_supplementary_materials()
        analyzer.generate_summary_report()
        
        print("\n✔ Analysis completed successfully!")
        print(f"\nAll results saved in: {analyzer.output_dir}")
        
    except Exception as e:
        print(f"\nERROR during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()