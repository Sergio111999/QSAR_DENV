#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QSAR 5.0 - Script 02: Chemical Diversity and Scaffolds Analysis
================================================================
This script analyzes the structural diversity of compounds,
identifies common scaffolds and prepares advanced molecular descriptors.
Author: Sergio Montenegro
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
import pickle

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdMolDescriptors, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

# Similarity analysis
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

# Other
import warnings
warnings.filterwarnings('ignore')

# Visualization configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class ChemicalDiversityAnalyzer:
    """Chemical diversity analyzer for QSAR data"""
    
    def __init__(self, data_path, output_base_dir):
        self.data_path = data_path
        self.output_base_dir = output_base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        self.output_dir = os.path.join(output_base_dir, "02_Chemical_Diversity")
        self.create_directory_structure()
        
        # Load data
        print("Loading processed data from EDA...")
        self.df = pd.read_csv(data_path)
        print(f"Data loaded: {len(self.df)} compounds")
        
        # Convert SMILES to RDKit molecules
        print("Converting SMILES to molecular objects...")
        self.mols = []
        self.valid_indices = []
        
        for idx, smiles in enumerate(self.df['SMILES_original']):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                self.mols.append(mol)
                self.valid_indices.append(idx)
            else:
                print(f"  Warning: Invalid SMILES at index {idx}")
        
        print(f"Valid molecules: {len(self.mols)} out of {len(self.df)}")
        
        # Initialize report
        self.report = []
        
    def create_directory_structure(self):
        """Create folder structure"""
        folders = [
            self.output_dir,
            os.path.join(self.output_dir, "figures"),
            os.path.join(self.output_dir, "data"),
            os.path.join(self.output_dir, "reports"),
            os.path.join(self.output_dir, "scaffolds")
        ]
        
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
    
    def analyze_scaffolds(self):
        """Analyze Murcko scaffolds and their distribution"""
        print("\n1. Analyzing molecular scaffolds...")
        
        # Extract scaffolds
        scaffolds = []
        scaffold_mols = []
        
        for mol in self.mols:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                scaffolds.append(scaffold_smiles)
                scaffold_mols.append(scaffold)
            except:
                scaffolds.append(None)
                scaffold_mols.append(None)
        
        # Add scaffolds to dataframe
        self.df.loc[self.valid_indices, 'scaffold'] = scaffolds
        
        # Count unique scaffolds
        scaffold_counts = Counter([s for s in scaffolds if s is not None])
        
        self.report.append("=== SCAFFOLD ANALYSIS ===\n")
        self.report.append(f"Total unique scaffolds: {len(scaffold_counts)}")
        self.report.append(f"Most common scaffolds (top 20):\n")
        
        # Top scaffolds
        top_scaffolds = scaffold_counts.most_common(20)
        scaffold_data = []
        
        for i, (scaffold_smiles, count) in enumerate(top_scaffolds):
            # Calculate statistics for each scaffold
            mask = self.df['scaffold'] == scaffold_smiles
            compounds_with_scaffold = self.df[mask]
            
            avg_activity = compounds_with_scaffold['pActivity_calculado'].mean()
            std_activity = compounds_with_scaffold['pActivity_calculado'].std()
            
            # Distribution by categories
            category_dist = compounds_with_scaffold['potency_category'].value_counts()
            
            scaffold_data.append({
                'rank': i+1,
                'scaffold': scaffold_smiles,
                'count': count,
                'percentage': count/len(self.mols)*100,
                'avg_pActivity': avg_activity,
                'std_pActivity': std_activity,
                'n_high': category_dist.get('High', 0),
                'n_medium': category_dist.get('Medium', 0),
                'n_low': category_dist.get('Low', 0)
            })
            
            self.report.append(f"{i+1}. {scaffold_smiles}: {count} compounds ({count/len(self.mols)*100:.1f}%), "
                             f"pActivity={avg_activity:.2f}±{std_activity:.2f}")
        
        # Create DataFrame with scaffold information
        scaffold_df = pd.DataFrame(scaffold_data)
        scaffold_df.to_csv(os.path.join(self.output_dir, "data", "top_scaffolds_analysis.csv"), index=False)
        
        # Visualization of top scaffolds
        self._visualize_top_scaffolds(top_scaffolds, scaffold_df)
        
        # Scaffold diversity analysis
        self._analyze_scaffold_diversity(scaffold_counts)
        
        return scaffold_df
    
    def _visualize_top_scaffolds(self, top_scaffolds, scaffold_df):
        """Visualize the most common scaffolds"""
        print("  Generating visualization of main scaffolds...")
        
        # 1. Draw structures of top 12 scaffolds
        fig = plt.figure(figsize=(16, 12))
        
        for i, (scaffold_smiles, count) in enumerate(top_scaffolds[:12]):
            ax = plt.subplot(3, 4, i+1)
            mol = Chem.MolFromSmiles(scaffold_smiles)
            
            if mol is not None:
                img = Draw.MolToImage(mol, size=(300, 300))
                ax.imshow(img)
                ax.set_title(f"Rank {i+1}: {count} compounds\n"
                           f"pActivity={scaffold_df.iloc[i]['avg_pActivity']:.2f}±"
                           f"{scaffold_df.iloc[i]['std_pActivity']:.2f}",
                           fontsize=10)
            
            ax.axis('off')
        
        plt.suptitle('Top 12 Most Frequent Scaffolds', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "figures", "top_scaffolds_structures.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Activity distribution by scaffold
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Frequency barplot
        axes[0,0].bar(range(len(scaffold_df)), scaffold_df['count'], color='skyblue')
        axes[0,0].set_xlabel('Scaffold Rank')
        axes[0,0].set_ylabel('Number of Compounds')
        axes[0,0].set_title('Frequency of Top 20 Scaffolds')
        axes[0,0].set_xticks(range(0, len(scaffold_df), 2))
        axes[0,0].set_xticklabels(range(1, len(scaffold_df)+1, 2))
        
        # Average activity by scaffold
        axes[0,1].bar(range(len(scaffold_df)), scaffold_df['avg_pActivity'], 
                     yerr=scaffold_df['std_pActivity'], capsize=5, color='coral')
        axes[0,1].axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Medium/Low Threshold')
        axes[0,1].axhline(y=6, color='red', linestyle='--', alpha=0.5, label='High/Medium Threshold')
        axes[0,1].set_xlabel('Scaffold Rank')
        axes[0,1].set_ylabel('Average pActivity')
        axes[0,1].set_title('Average Activity by Scaffold (±SD)')
        axes[0,1].legend()
        axes[0,1].set_xticks(range(0, len(scaffold_df), 2))
        axes[0,1].set_xticklabels(range(1, len(scaffold_df)+1, 2))
        
        # Category distribution by scaffold (stacked bar)
        categories = ['n_high', 'n_medium', 'n_low']
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        bottom = np.zeros(len(scaffold_df))
        
        for cat, color in zip(categories, colors):
            axes[1,0].bar(range(len(scaffold_df)), scaffold_df[cat], 
                         bottom=bottom, color=color, label=cat.replace('n_', '').capitalize())
            bottom += scaffold_df[cat]
        
        axes[1,0].set_xlabel('Scaffold Rank')
        axes[1,0].set_ylabel('Number of Compounds')
        axes[1,0].set_title('Category Distribution by Scaffold')
        axes[1,0].legend()
        axes[1,0].set_xticks(range(0, len(scaffold_df), 2))
        axes[1,0].set_xticklabels(range(1, len(scaffold_df)+1, 2))
        
        # High potency percentage by scaffold
        scaffold_df['pct_high'] = scaffold_df['n_high'] / scaffold_df['count'] * 100
        axes[1,1].bar(range(len(scaffold_df)), scaffold_df['pct_high'], color='darkred')
        axes[1,1].axhline(y=11.16, color='black', linestyle='--', alpha=0.5, 
                         label='Global average (11.16%)')
        axes[1,1].set_xlabel('Scaffold Rank')
        axes[1,1].set_ylabel('% High Potency Compounds')
        axes[1,1].set_title('High Potency Enrichment by Scaffold')
        axes[1,1].legend()
        axes[1,1].set_xticks(range(0, len(scaffold_df), 2))
        axes[1,1].set_xticklabels(range(1, len(scaffold_df)+1, 2))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "figures", "scaffold_activity_analysis.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_scaffold_diversity(self, scaffold_counts):
        """Analyze scaffold diversity"""
        print("  Analyzing scaffold diversity...")
        
        # Calculate diversity statistics
        counts = list(scaffold_counts.values())
        
        # Accumulation curve
        sorted_counts = sorted(counts, reverse=True)
        cumsum = np.cumsum(sorted_counts)
        cumsum_pct = cumsum / sum(sorted_counts) * 100
        
        # Find how many scaffolds represent 50%, 80%, 90% of compounds
        n_50 = np.argmax(cumsum_pct >= 50) + 1
        n_80 = np.argmax(cumsum_pct >= 80) + 1
        n_90 = np.argmax(cumsum_pct >= 90) + 1
        
        self.report.append(f"\n=== SCAFFOLD DIVERSITY ===")
        self.report.append(f"Unique scaffolds: {len(scaffold_counts)}")
        self.report.append(f"Singleton scaffolds (1 compound): {sum(1 for c in counts if c == 1)}")
        self.report.append(f"Scaffolds representing 50% of compounds: {n_50}")
        self.report.append(f"Scaffolds representing 80% of compounds: {n_80}")
        self.report.append(f"Scaffolds representing 90% of compounds: {n_90}\n")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Accumulation curve
        axes[0].plot(range(1, len(sorted_counts)+1), cumsum_pct, 'b-', linewidth=2)
        axes[0].axhline(y=50, color='orange', linestyle='--', alpha=0.5)
        axes[0].axhline(y=80, color='red', linestyle='--', alpha=0.5)
        axes[0].axhline(y=90, color='darkred', linestyle='--', alpha=0.5)
        axes[0].axvline(x=n_50, color='orange', linestyle=':', alpha=0.5)
        axes[0].axvline(x=n_80, color='red', linestyle=':', alpha=0.5)
        axes[0].axvline(x=n_90, color='darkred', linestyle=':', alpha=0.5)
        axes[0].set_xlabel('Number of Scaffolds')
        axes[0].set_ylabel('% Cumulative Compounds')
        axes[0].set_title('Scaffold Accumulation Curve')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')
        
        # Frequency distribution
        freq_dist = Counter(counts)
        axes[1].bar(freq_dist.keys(), freq_dist.values(), color='lightcoral')
        axes[1].set_xlabel('Number of Compounds per Scaffold')
        axes[1].set_ylabel('Number of Scaffolds')
        axes[1].set_title('Scaffold Frequency Distribution')
        axes[1].set_xlim(0, min(20, max(freq_dist.keys())))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "figures", "scaffold_diversity.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def calculate_molecular_fingerprints(self):
        """Calculate different types of molecular fingerprints"""
        print("\n2. Calculating molecular fingerprints...")
        
        fingerprints = {
            'ECFP4': [],
            'ECFP6': [],
            'FCFP4': [],
            'FCFP6': [],
            'MACCS': [],
            'Morgan2': [],
            'Morgan3': [],
            'AtomPair': [],
            'RDKit': []
        }
        
        for i, mol in enumerate(self.mols):
            if i % 200 == 0:
                print(f"  Processing molecule {i}/{len(self.mols)}...")
            
            # Extended Connectivity Fingerprints (Morgan)
            fingerprints['ECFP4'].append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
            fingerprints['ECFP6'].append(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048))
            
            # Feature-based Connectivity Fingerprints
            fingerprints['FCFP4'].append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=2048))
            fingerprints['FCFP6'].append(AllChem.GetMorganFingerprintAsBitVect(mol, 3, useFeatures=True, nBits=2048))
            
            # MACCS keys
            fingerprints['MACCS'].append(MACCSkeys.GenMACCSKeys(mol))
            
            # Morgan fingerprints with different radii
            fingerprints['Morgan2'].append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
            fingerprints['Morgan3'].append(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048))
            
            # Atom Pair fingerprints - use count method
            ap = Pairs.GetAtomPairFingerprint(mol)
            # Convert to fixed-size bit vector
            fp_bits = DataStructs.ExplicitBitVect(2048)
            for idx in ap.GetNonzeroElements():
                fp_bits.SetBit(idx % 2048)
            fingerprints['AtomPair'].append(fp_bits)
            
            # RDKit fingerprints
            fingerprints['RDKit'].append(Chem.RDKFingerprint(mol))
        
        # Save fingerprints
        print("  Saving fingerprints...")
        with open(os.path.join(self.output_dir, "data", "molecular_fingerprints.pkl"), 'wb') as f:
            pickle.dump(fingerprints, f)
        
        self.fingerprints = fingerprints
        
        # Analyze similarity with different fingerprints
        self._analyze_fingerprint_similarity()
        
        return fingerprints
    
    def _analyze_fingerprint_similarity(self):
        """Analyze molecular similarity using different fingerprints"""
        print("  Analyzing molecular similarity...")
        
        # Select key fingerprints for analysis
        fp_types = ['ECFP4', 'MACCS', 'Morgan3']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        similarity_stats = {}
        
        for idx, fp_type in enumerate(fp_types):
            fps = self.fingerprints[fp_type]
            
            # Calculate Tanimoto similarity matrix (random sample for efficiency)
            n_sample = min(500, len(fps))
            sample_indices = np.random.choice(len(fps), n_sample, replace=False)
            sample_fps = [fps[i] for i in sample_indices]
            
            similarity_matrix = np.zeros((n_sample, n_sample))
            
            for i in range(n_sample):
                for j in range(i+1, n_sample):
                    sim = DataStructs.TanimotoSimilarity(sample_fps[i], sample_fps[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
                similarity_matrix[i, i] = 1.0
            
            # Statistics
            upper_triangle = similarity_matrix[np.triu_indices(n_sample, k=1)]
            mean_sim = np.mean(upper_triangle)
            std_sim = np.std(upper_triangle)
            
            similarity_stats[fp_type] = {
                'mean': mean_sim,
                'std': std_sim,
                'min': np.min(upper_triangle),
                'max': np.max(upper_triangle)
            }
            
            # Visualizations
            # Similarity heatmap
            sns.heatmap(similarity_matrix, cmap='viridis', ax=axes[idx], 
                       cbar_kws={'label': 'Tanimoto Similarity'})
            axes[idx].set_title(f'Similarity Matrix - {fp_type}')
            axes[idx].set_xlabel('Compounds')
            axes[idx].set_ylabel('Compounds')
            
            # Similarity distribution
            axes[idx+3].hist(upper_triangle, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[idx+3].axvline(mean_sim, color='red', linestyle='--', 
                               label=f'Mean={mean_sim:.3f}')
            axes[idx+3].set_xlabel('Tanimoto Similarity')
            axes[idx+3].set_ylabel('Frequency')
            axes[idx+3].set_title(f'Similarity Distribution - {fp_type}')
            axes[idx+3].legend()
            axes[idx+3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "figures", "fingerprint_similarity_analysis.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add to report
        self.report.append("\n=== MOLECULAR SIMILARITY ANALYSIS ===")
        for fp_type, stats in similarity_stats.items():
            self.report.append(f"\n{fp_type}:")
            self.report.append(f"  Mean similarity: {stats['mean']:.3f} ± {stats['std']:.3f}")
            self.report.append(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    def analyze_chemical_clusters(self):
        """Identify chemical clusters using fingerprints"""
        print("\n3. Identifying chemical clusters...")
        
        # Use ECFP4 for clustering
        fps = self.fingerprints['ECFP4']
        
        # Calculate distance matrix (1 - Tanimoto similarity)
        print("  Calculating distance matrix...")
        n_compounds = len(fps)
        distance_matrix = np.zeros((n_compounds, n_compounds))
        
        for i in range(n_compounds):
            if i % 100 == 0:
                print(f"    Processing compound {i}/{n_compounds}...")
            for j in range(i+1, n_compounds):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                dist = 1 - sim
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # Hierarchical clustering
        print("  Performing hierarchical clustering...")
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method='ward')
        
        # Determine optimal number of clusters (using different thresholds)
        thresholds = [0.7, 0.8, 0.9]
        cluster_results = {}
        
        for threshold in thresholds:
            clusters = fcluster(linkage_matrix, threshold, criterion='distance')
            n_clusters = len(np.unique(clusters))
            cluster_results[threshold] = {
                'labels': clusters,
                'n_clusters': n_clusters
            }
            print(f"  Threshold {threshold}: {n_clusters} clusters")
        
        # Use threshold 0.8 for detailed analysis
        clusters = cluster_results[0.8]['labels']
        self.df.loc[self.valid_indices, 'chemical_cluster'] = clusters
        
        # Analyze clusters
        self._analyze_cluster_properties(clusters)
        
        # Visualize clusters in reduced chemical space
        self._visualize_chemical_clusters(fps, clusters)
        
        return cluster_results
    
    def _analyze_cluster_properties(self, clusters):
        """Analyze properties of chemical clusters"""
        print("  Analyzing cluster properties...")
        
        cluster_df = self.df.loc[self.valid_indices].copy()
        cluster_df['cluster'] = clusters
        
        # Statistics by cluster
        cluster_stats = []
        
        for cluster_id in np.unique(clusters):
            cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]
            
            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'pct_total': len(cluster_data) / len(cluster_df) * 100,
                'avg_pActivity': cluster_data['pActivity_calculado'].mean(),
                'std_pActivity': cluster_data['pActivity_calculado'].std(),
                'n_high': len(cluster_data[cluster_data['potency_category'] == 'High']),
                'n_medium': len(cluster_data[cluster_data['potency_category'] == 'Medium']),
                'n_low': len(cluster_data[cluster_data['potency_category'] == 'Low']),
                'pct_high': len(cluster_data[cluster_data['potency_category'] == 'High']) / len(cluster_data) * 100,
                'dominant_scaffold': cluster_data['scaffold'].mode()[0] if 'scaffold' in cluster_data else 'N/A'
            }
            
            cluster_stats.append(stats)
        
        cluster_stats_df = pd.DataFrame(cluster_stats)
        cluster_stats_df = cluster_stats_df.sort_values('size', ascending=False)
        
        # Save statistics
        cluster_stats_df.to_csv(os.path.join(self.output_dir, "data", "cluster_statistics.csv"), index=False)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top 20 clusters by size
        top_clusters = cluster_stats_df.head(20)
        
        # Cluster sizes
        axes[0,0].bar(range(len(top_clusters)), top_clusters['size'], color='skyblue')
        axes[0,0].set_xlabel('Cluster ID')
        axes[0,0].set_ylabel('Number of Compounds')
        axes[0,0].set_title('Size of Top 20 Clusters')
        axes[0,0].set_xticks(range(len(top_clusters)))
        axes[0,0].set_xticklabels(top_clusters['cluster_id'], rotation=45)
        
        # Average activity by cluster
        axes[0,1].bar(range(len(top_clusters)), top_clusters['avg_pActivity'],
                     yerr=top_clusters['std_pActivity'], capsize=5, color='coral')
        axes[0,1].axhline(y=5, color='orange', linestyle='--', alpha=0.5)
        axes[0,1].axhline(y=6, color='red', linestyle='--', alpha=0.5)
        axes[0,1].set_xlabel('Cluster ID')
        axes[0,1].set_ylabel('Average pActivity')
        axes[0,1].set_title('Average Activity by Cluster (Top 20)')
        axes[0,1].set_xticks(range(len(top_clusters)))
        axes[0,1].set_xticklabels(top_clusters['cluster_id'], rotation=45)
        
        # High potency percentage by cluster
        axes[1,0].bar(range(len(top_clusters)), top_clusters['pct_high'], color='darkred')
        axes[1,0].axhline(y=11.16, color='black', linestyle='--', alpha=0.5, 
                         label='Global average (11.16%)')
        axes[1,0].set_xlabel('Cluster ID')
        axes[1,0].set_ylabel('% High Potency')
        axes[1,0].set_title('High Potency Enrichment by Cluster')
        axes[1,0].legend()
        axes[1,0].set_xticks(range(len(top_clusters)))
        axes[1,0].set_xticklabels(top_clusters['cluster_id'], rotation=45)
        
        # Cluster size distribution
        cluster_sizes = cluster_stats_df['size'].values
        axes[1,1].hist(cluster_sizes, bins=50, color='lightgreen', edgecolor='black')
        axes[1,1].set_xlabel('Cluster Size')
        axes[1,1].set_ylabel('Number of Clusters')
        axes[1,1].set_title('Cluster Size Distribution')
        axes[1,1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "figures", "cluster_analysis.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add to report
        self.report.append("\n=== CHEMICAL CLUSTER ANALYSIS ===")
        self.report.append(f"Total number of clusters: {len(cluster_stats_df)}")
        self.report.append(f"Average cluster size: {cluster_stats_df['size'].mean():.1f}")
        self.report.append(f"Clusters with >10 compounds: {len(cluster_stats_df[cluster_stats_df['size'] > 10])}")
        self.report.append(f"\nTop 10 clusters by high potency enrichment:")
        
        top_enriched = cluster_stats_df.nlargest(10, 'pct_high')
        for _, row in top_enriched.iterrows():
            if row['size'] >= 5:  # Only clusters with at least 5 compounds
                self.report.append(f"  Cluster {row['cluster_id']}: {row['pct_high']:.1f}% high potency "
                                 f"({row['n_high']}/{row['size']} compounds)")
    
    def _visualize_chemical_clusters(self, fps, clusters):
        """Visualize clusters in reduced chemical space"""
        print("  Visualizing clusters in chemical space...")
        
        # Convert fingerprints to matrix
        fp_matrix = np.array([fp.ToList() for fp in fps])
        
        # Dimensionality reduction
        # PCA
        pca = PCA(n_components=50)
        X_pca = pca.fit_transform(fp_matrix)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_pca)  # Use PCA as preprocessing
        
        # UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        X_umap = reducer.fit_transform(X_pca)
        
        # Colors for clusters (top 10 + others)
        unique_clusters = np.unique(clusters)
        cluster_sizes = {c: np.sum(clusters == c) for c in unique_clusters}
        top_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        top_cluster_ids = [c[0] for c in top_clusters]
        
        # Assign colors
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        cluster_colors = []
        for c in clusters:
            if c in top_cluster_ids:
                idx = top_cluster_ids.index(c)
                cluster_colors.append(colors[idx])
            else:
                cluster_colors.append([0.7, 0.7, 0.7, 0.5])  # Grey for others
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # t-SNE
        scatter1 = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_colors, 
                                  s=30, alpha=0.7, edgecolors='none')
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        axes[0].set_title('Chemical Clusters - t-SNE Projection')
        
        # UMAP
        scatter2 = axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c=cluster_colors, 
                                  s=30, alpha=0.7, edgecolors='none')
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')
        axes[1].set_title('Chemical Clusters - UMAP Projection')
        
        # Add legend for top 10 clusters
        from matplotlib.patches import Patch
        legend_elements = []
        for i, (cluster_id, size) in enumerate(top_clusters):
            legend_elements.append(Patch(facecolor=colors[i], 
                                       label=f'Cluster {cluster_id} (n={size})'))
        legend_elements.append(Patch(facecolor=[0.7, 0.7, 0.7, 0.5], 
                                   label='Other clusters'))
        
        axes[1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "figures", "chemical_clusters_visualization.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def calculate_pharmacophore_features(self):
        """Calculate pharmacophoric features"""
        print("\n4. Calculating pharmacophoric features...")
        
        # Define types of pharmacophoric features
        pharma_features = {
            'n_aromatic': [],
            'n_hba_lipinski': [],
            'n_hbd_lipinski': [],
            'n_positive': [],
            'n_negative': [],
            'n_hydrophobic': [],
            'aromatic_proportion': [],
            'charged_proportion': []
        }
        
        for mol in self.mols:
            # Aromatic features
            aromatic_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsAromatic()]
            n_aromatic = len(aromatic_atoms)
            
            # Lipinski HBA/HBD (more strict)
            n_hba = rdMolDescriptors.CalcNumLipinskiHBA(mol)
            n_hbd = rdMolDescriptors.CalcNumLipinskiHBD(mol)
            
            # Formal charges
            positive_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetFormalCharge() > 0]
            negative_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetFormalCharge() < 0]
            
            # Hydrophobic atoms (C, S not in polar groups)
            hydrophobic_atoms = []
            for atom in mol.GetAtoms():
                if atom.GetSymbol() in ['C', 'S']:
                    # Check if connected to polar atoms
                    neighbors = [mol.GetAtomWithIdx(n.GetIdx()).GetSymbol() 
                               for n in atom.GetNeighbors()]
                    if not any(n in ['O', 'N', 'P'] for n in neighbors):
                        hydrophobic_atoms.append(atom.GetIdx())
            
            # Calculate proportions
            n_atoms = mol.GetNumAtoms()
            
            pharma_features['n_aromatic'].append(n_aromatic)
            pharma_features['n_hba_lipinski'].append(n_hba)
            pharma_features['n_hbd_lipinski'].append(n_hbd)
            pharma_features['n_positive'].append(len(positive_atoms))
            pharma_features['n_negative'].append(len(negative_atoms))
            pharma_features['n_hydrophobic'].append(len(hydrophobic_atoms))
            pharma_features['aromatic_proportion'].append(n_aromatic / n_atoms if n_atoms > 0 else 0)
            pharma_features['charged_proportion'].append((len(positive_atoms) + len(negative_atoms)) / n_atoms if n_atoms > 0 else 0)
        
        # Add to dataframe
        for feature, values in pharma_features.items():
            self.df.loc[self.valid_indices, feature] = values
        
        # Analyze correlation with activity
        self._analyze_pharmacophore_activity_relationship(pharma_features)
        
        return pharma_features
    
    def _analyze_pharmacophore_activity_relationship(self, pharma_features):
        """Analyze relationship between pharmacophoric features and activity"""
        print("  Analyzing pharmacophore-activity relationship...")
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        feature_names = list(pharma_features.keys())
        
        for i, feature in enumerate(feature_names):
            if i < len(axes):
                # Scatter plot with color by category
                valid_data = self.df.loc[self.valid_indices]
                
                colors = valid_data['potency_category'].map({
                    'High': '#ff6b6b',
                    'Medium': '#4ecdc4',
                    'Low': '#45b7d1'
                })
                
                axes[i].scatter(valid_data[feature], valid_data['pActivity_calculado'],
                              c=colors, alpha=0.5, s=20)
                
                # Trend line
                x = valid_data[feature]
                y = valid_data['pActivity_calculado']
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                axes[i].plot(sorted(x), p(sorted(x)), "r--", alpha=0.8)
                
                # Correlation
                corr = np.corrcoef(x, y)[0, 1]
                
                axes[i].set_xlabel(feature.replace('_', ' ').title())
                axes[i].set_ylabel('pActivity')
                axes[i].set_title(f'{feature.replace("_", " ").title()}\n(r={corr:.3f})')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "figures", "pharmacophore_activity_relationship.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_descriptor_matrix(self):
        """Generate complete descriptor matrix for modeling"""
        print("\n5. Generating complete descriptor matrix...")
        
        # Create complete descriptor matrix for modeling
        descriptor_list = []
        
        # Basic descriptors already calculated
        basic_descriptors = ['MW_calculado', 'LogP_calculado', 'NumHBA', 'NumHBD', 'TPSA',
                           'n_rings', 'n_aromatic_rings', 'n_heteroatoms', 'n_rotatable_bonds',
                           'lipinski_violations']
        
        # Pharmacophoric descriptors
        pharma_descriptors = ['n_aromatic', 'n_positive', 'n_negative', 'n_hydrophobic',
                            'aromatic_proportion', 'charged_proportion']
        
        # Additional descriptors to calculate
        additional_descriptors = {
            'BertzCT': Descriptors.BertzCT,
            'Chi0': Descriptors.Chi0,
            'Chi1': Descriptors.Chi1,
            'HallKierAlpha': Descriptors.HallKierAlpha,
            'Kappa1': Descriptors.Kappa1,
            'Kappa2': Descriptors.Kappa2,
            'Kappa3': Descriptors.Kappa3,
            'PEOE_VSA1': Descriptors.PEOE_VSA1,
            'PEOE_VSA2': Descriptors.PEOE_VSA2,
            'SMR_VSA1': Descriptors.SMR_VSA1,
            'SMR_VSA2': Descriptors.SMR_VSA2,
            'SlogP_VSA1': Descriptors.SlogP_VSA1,
            'SlogP_VSA2': Descriptors.SlogP_VSA2,
            'EState_VSA1': Descriptors.EState_VSA1,
            'EState_VSA2': Descriptors.EState_VSA2,
            'MolMR': Descriptors.MolMR,
            'BalabanJ': Descriptors.BalabanJ,
            'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings,
            'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings,
            'NumAromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles,
            'NumSaturatedHeterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles,
            'NumAliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles,
            'FractionCSP3': rdMolDescriptors.CalcFractionCSP3,
            'NumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms,
            'NumBridgeheadAtoms': rdMolDescriptors.CalcNumBridgeheadAtoms
        }
        
        print("  Calculating additional descriptors...")
        for desc_name, desc_func in additional_descriptors.items():
            values = []
            for mol in self.mols:
                try:
                    value = desc_func(mol)
                    values.append(value)
                except:
                    values.append(np.nan)
            
            self.df.loc[self.valid_indices, desc_name] = values
        
        # Combine all descriptors
        all_descriptors = basic_descriptors + pharma_descriptors + list(additional_descriptors.keys())
        
        # Create descriptor matrix
        descriptor_matrix = self.df.loc[self.valid_indices, all_descriptors].copy()
        
        # Add categorical information
        descriptor_matrix['assay_type_IC50'] = (self.df.loc[self.valid_indices, 'Standard Type'] == 'IC50').astype(int)
        descriptor_matrix['assay_type_EC50'] = (self.df.loc[self.valid_indices, 'Standard Type'] == 'EC50').astype(int)
        
        # Save descriptor matrix
        descriptor_matrix.to_csv(os.path.join(self.output_dir, "data", "descriptor_matrix.csv"), index=False)
        
        # Save also with target information
        full_data = pd.concat([
            self.df.loc[self.valid_indices, ['Molecule ChEMBL ID', 'SMILES_original', 
                                            'pActivity_calculado', 'potency_category',
                                            'Standard Type', 'Target Name']],
            descriptor_matrix
        ], axis=1)
        
        full_data.to_csv(os.path.join(self.output_dir, "data", "full_descriptor_data.csv"), index=False)
        
        self.report.append(f"\n=== DESCRIPTOR MATRIX ===")
        self.report.append(f"Total descriptors calculated: {len(all_descriptors) + 2}")
        self.report.append(f"Compounds with complete descriptors: {len(descriptor_matrix.dropna())}")
        self.report.append(f"Descriptors with missing values: {descriptor_matrix.isnull().sum().sum()}")
        
        return descriptor_matrix
    
    def generate_summary_report(self):
        """Generate summary report of the analysis"""
        print("\n6. Generating summary report...")
        
        # Executive summary
        self.report.append("\n=== EXECUTIVE SUMMARY ===")
        self.report.append(f"\n1. STRUCTURAL DIVERSITY:")
        self.report.append(f"   - Unique scaffolds: {self.df['scaffold'].nunique() if 'scaffold' in self.df else 'N/A'}")
        self.report.append(f"   - Chemical clusters identified: {self.df['chemical_cluster'].nunique() if 'chemical_cluster' in self.df else 'N/A'}")
        self.report.append(f"   - Average molecular similarity (ECFP4): See similarity analysis")
        
        self.report.append(f"\n2. PHARMACOPHORIC FEATURES:")
        self.report.append(f"   - Molecular descriptors calculated: {len([col for col in self.df.columns if col not in ['Molecule ChEMBL ID', 'SMILES_original', 'scaffold', 'chemical_cluster']])}")
        self.report.append(f"   - Fingerprints generated: ECFP4, ECFP6, FCFP4, FCFP6, MACCS, Morgan2/3, AtomPair, RDKit")
        
        self.report.append(f"\n3. MODELING RECOMMENDATIONS:")
        self.report.append(f"   - Use scaffold-based validation to evaluate generalization")
        self.report.append(f"   - Consider ensemble of fingerprints (ECFP4 + MACCS + descriptors)")
        self.report.append(f"   - Implement stratification by chemical clusters in cross-validation")
        self.report.append(f"   - Prioritize clusters with high enrichment of active compounds")
        
        # Save report
        report_path = os.path.join(self.output_dir, "reports", f"chemical_diversity_report_{self.timestamp}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report))
        
        print(f"\nReport saved at: {report_path}")
        
        # Console summary
        print("\n" + "="*60)
        print("CHEMICAL DIVERSITY ANALYSIS COMPLETED")
        print("="*60)
        print(f"Files generated in: {self.output_dir}")
        print("\nMain output files:")
        print("- molecular_fingerprints.pkl: All calculated fingerprints")
        print("- descriptor_matrix.csv: Molecular descriptor matrix")
        print("- full_descriptor_data.csv: Complete data for modeling")
        print("- top_scaffolds_analysis.csv: Analysis of main scaffolds")
        print("- cluster_statistics.csv: Chemical cluster statistics")
        print("="*60)
    
    def generate_supplementary_materials(self):
        """Generate organized supplementary materials for publication"""
        print("\n7. Generating supplementary materials for publication...")
        
        # Create supplementary materials directory structure
        supp_dir = os.path.join(self.output_dir, "Supplementary_Materials_Chemical_Diversity")
        supp_tables = os.path.join(supp_dir, "Tables")
        supp_figures = os.path.join(supp_dir, "Figures")
        supp_data = os.path.join(supp_dir, "Data")
        supp_reports = os.path.join(supp_dir, "Analysis_Reports")
        
        for folder in [supp_dir, supp_tables, supp_figures, supp_data, supp_reports]:
            os.makedirs(folder, exist_ok=True)
        
        print("  Creating supplementary tables...")
        
        # Table S8: Scaffold Statistics
        if 'scaffold' in self.df.columns:
            scaffold_counts = Counter([s for s in self.df['scaffold'].dropna()])
            scaffold_data = []
            
            for rank, (scaffold_smiles, count) in enumerate(scaffold_counts.most_common(50), 1):
                mask = self.df['scaffold'] == scaffold_smiles
                compounds = self.df[mask]
                
                scaffold_data.append({
                    'Rank': rank,
                    'Scaffold_SMILES': scaffold_smiles,
                    'Count': count,
                    'Percentage': count/len(self.df)*100,
                    'Avg_pActivity': compounds['pActivity_calculado'].mean(),
                    'Std_pActivity': compounds['pActivity_calculado'].std(),
                    'N_High': len(compounds[compounds['potency_category'] == 'High']),
                    'N_Medium': len(compounds[compounds['potency_category'] == 'Medium']),
                    'N_Low': len(compounds[compounds['potency_category'] == 'Low']),
                    'High_Potency_Enrichment': len(compounds[compounds['potency_category'] == 'High'])/count*100
                })
            
            # Save as Excel with multiple sheets
            with pd.ExcelWriter(os.path.join(supp_tables, "Table_S8_Scaffold_Statistics.xlsx")) as writer:
                # Sheet 1: Top 50 scaffolds
                pd.DataFrame(scaffold_data).to_excel(writer, sheet_name='Top_50_Scaffolds', index=False)
                
                # Sheet 2: Diversity metrics
                diversity_metrics = {
                    'Metric': ['Total unique scaffolds', 'Singleton scaffolds', 
                              'Scaffolds covering 50% compounds', 'Scaffolds covering 80% compounds',
                              'Scaffolds covering 90% compounds'],
                    'Value': [len(scaffold_counts), 
                             sum(1 for c in scaffold_counts.values() if c == 1),
                             self._get_coverage_count(scaffold_counts, 50),
                             self._get_coverage_count(scaffold_counts, 80),
                             self._get_coverage_count(scaffold_counts, 90)]
                }
                pd.DataFrame(diversity_metrics).to_excel(writer, sheet_name='Diversity_Metrics', index=False)
        
        # Table S9: Chemical Clusters
        if 'chemical_cluster' in self.df.columns:
            cluster_data = []
            for cluster_id in self.df['chemical_cluster'].unique():
                if pd.notna(cluster_id):
                    cluster_compounds = self.df[self.df['chemical_cluster'] == cluster_id]
                    
                    cluster_data.append({
                        'Cluster_ID': int(cluster_id),
                        'Size': len(cluster_compounds),
                        'Avg_pActivity': cluster_compounds['pActivity_calculado'].mean(),
                        'Std_pActivity': cluster_compounds['pActivity_calculado'].std(),
                        'Percent_High_Potency': len(cluster_compounds[cluster_compounds['potency_category'] == 'High'])/len(cluster_compounds)*100,
                        'Dominant_Scaffold': cluster_compounds['scaffold'].mode()[0] if len(cluster_compounds['scaffold'].mode()) > 0 else 'N/A'
                    })
            
            cluster_df = pd.DataFrame(cluster_data).sort_values('Size', ascending=False)
            
            with pd.ExcelWriter(os.path.join(supp_tables, "Table_S9_Chemical_Clusters.xlsx")) as writer:
                # Sheet 1: All clusters
                cluster_df.to_excel(writer, sheet_name='Cluster_Statistics', index=False)
                
                # Sheet 2: Top enriched clusters
                enriched = cluster_df[cluster_df['Percent_High_Potency'] > 20].sort_values('Percent_High_Potency', ascending=False)
                enriched.to_excel(writer, sheet_name='High_Potency_Enriched', index=False)
        
        # Table S10: Pharmacophore Features
        pharma_cols = ['n_aromatic', 'n_hba_lipinski', 'n_hbd_lipinski', 'n_positive', 
                      'n_negative', 'n_hydrophobic', 'aromatic_proportion', 'charged_proportion']
        
        if all(col in self.df.columns for col in pharma_cols):
            pharma_df = self.df[['Molecule ChEMBL ID', 'pActivity_calculado'] + pharma_cols].copy()
            
            # Add correlations
            correlations = {}
            for col in pharma_cols:
                valid_data = pharma_df.dropna(subset=[col, 'pActivity_calculado'])
                if len(valid_data) > 1:
                    corr = np.corrcoef(valid_data[col], valid_data['pActivity_calculado'])[0, 1]
                    correlations[col] = corr
            
            # Save pharmacophore analysis
            with pd.ExcelWriter(os.path.join(supp_tables, "Table_S10_Pharmacophore_Features.xlsx")) as writer:
                pharma_df.head(1000).to_excel(writer, sheet_name='Features_Sample', index=False)
                pd.DataFrame(correlations.items(), columns=['Feature', 'Correlation_with_Activity']).to_excel(
                    writer, sheet_name='Feature_Correlations', index=False)
        
        # Table S11: Fingerprint Summary
        if hasattr(self, 'fingerprints'):
            fp_summary = []
            for fp_type in self.fingerprints.keys():
                fp_summary.append({
                    'Fingerprint_Type': fp_type,
                    'Bit_Length': len(self.fingerprints[fp_type][0]) if self.fingerprints[fp_type] else 0,
                    'Number_Compounds': len(self.fingerprints[fp_type])
                })
            
            pd.DataFrame(fp_summary).to_excel(
                os.path.join(supp_tables, "Table_S11_Molecular_Fingerprints_Summary.xlsx"), index=False)
        
        print("  Copying figures to supplementary materials...")
        
        # Copy and rename figures
        import shutil
        figure_mapping = {
            'top_scaffolds_structures.png': 'Figure_S7_Top_Scaffolds_Structures.pdf',
            'scaffold_activity_analysis.png': 'Figure_S8_Scaffold_Activity_Analysis.pdf',
            'scaffold_diversity.png': 'Figure_S9_Scaffold_Diversity_Curves.pdf',
            'fingerprint_similarity_analysis.png': 'Figure_S10_Fingerprint_Similarity_Matrices.pdf',
            'chemical_clusters_visualization.png': 'Figure_S11_Chemical_Clusters_Visualization.pdf',
            'cluster_analysis.png': 'Figure_S12_Cluster_Properties.pdf',
            'pharmacophore_activity_relationship.png': 'Figure_S13_Pharmacophore_Activity_Correlations.pdf'
        }
        
        for src, dst in figure_mapping.items():
            src_path = os.path.join(self.output_dir, "figures", src)
            dst_path = os.path.join(supp_figures, dst.replace('.pdf', '.png'))
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
        
        print("  Organizing data files...")
        
        # Copy data files
        data_mapping = {
            'molecular_fingerprints.pkl': 'Data_S5_Molecular_Fingerprints.pkl',
            'top_scaffolds_analysis.csv': 'Data_S6_Scaffold_Assignments.csv',
            'cluster_statistics.csv': 'Data_S7_Cluster_Assignments.csv',
            'full_descriptor_data.csv': 'Data_S8_Full_Descriptor_Matrix.csv'
        }
        
        for src, dst in data_mapping.items():
            src_path = os.path.join(self.output_dir, "data", src)
            dst_path = os.path.join(supp_data, dst)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
        
        # Generate pharmacophore features file
        if all(col in self.df.columns for col in pharma_cols):
            pharma_df.to_csv(os.path.join(supp_data, "Data_S9_Pharmacophore_Features.csv"), index=False)
        
        print("  Creating README file...")
        
        # Create README
        readme_content = """# Supplementary Materials - Chemical Diversity Analysis

## Overview
Chemical diversity and structural analysis of 2,034 anti-dengue compounds from ChEMBL database.

## Key Statistics
- **Total Compounds**: 2,034 activity records (1,981 unique molecules)
- **Unique Scaffolds**: {:,} Murcko scaffolds
- **Chemical Clusters**: {} distinct clusters (ECFP4, threshold 0.8)
- **Molecular Descriptors**: 44 physicochemical + pharmacophoric features
- **Fingerprints**: 9 types (ECFP4/6, FCFP4/6, MACCS, Morgan2/3, AtomPair, RDKit)

## File Descriptions

### Tables (Excel format)
- **Table_S8**: Complete scaffold analysis with enrichment factors
- **Table_S9**: Chemical cluster statistics and high-potency enrichment  
- **Table_S10**: Pharmacophore features and activity correlations
- **Table_S11**: Molecular fingerprint specifications

### Figures
- **Figure_S7**: Top 12 scaffold structures with activity statistics
- **Figure_S8**: Scaffold frequency and activity distribution (4 panels)
- **Figure_S9**: Scaffold diversity and accumulation curves
- **Figure_S10**: Fingerprint similarity matrices and distributions
- **Figure_S11**: Chemical space visualization (t-SNE and UMAP)
- **Figure_S12**: Cluster size and enrichment analysis
- **Figure_S13**: Pharmacophore-activity relationships

### Data Files
- **Data_S5**: Binary fingerprints (Python pickle format)
- **Data_S6**: Scaffold assignments for all compounds
- **Data_S7**: Chemical cluster assignments
- **Data_S8**: Complete descriptor matrix for modeling
- **Data_S9**: Pharmacophoric features

## Usage Example
```python
import pickle
import pandas as pd

# Load fingerprints
with open('Data/Data_S5_Molecular_Fingerprints.pkl', 'rb') as f:
    fingerprints = pickle.load(f)

# Load descriptor matrix
descriptors = pd.read_csv('Data/Data_S8_Full_Descriptor_Matrix.csv')
```

## Software Requirements
- Python 3.9+
- RDKit 2023.03.1
- scikit-learn 1.3.0
- pandas 2.0.3
- numpy 1.24.3

## Citation
[Article citation to be added upon publication]

## Contact
For questions regarding these materials, please contact: [corresponding author email]
""".format(
            self.df['scaffold'].nunique() if 'scaffold' in self.df else 'N/A',
            self.df['chemical_cluster'].nunique() if 'chemical_cluster' in self.df else 'N/A'
        )
        
        with open(os.path.join(supp_dir, "README.md"), 'w') as f:
            f.write(readme_content)
        
        print(f"\n  Supplementary materials organized in: {supp_dir}")
        print("  Total files generated:")
        print(f"    - Tables: {len(os.listdir(supp_tables))} files")
        print(f"    - Figures: {len(os.listdir(supp_figures))} files")
        print(f"    - Data: {len(os.listdir(supp_data))} files")
        
        return supp_dir
    
    def _get_coverage_count(self, scaffold_counts, percentage):
        """Helper function to calculate scaffold coverage"""
        sorted_counts = sorted(scaffold_counts.values(), reverse=True)
        cumsum = np.cumsum(sorted_counts)
        total = sum(sorted_counts)
        cumsum_pct = cumsum / total * 100
        return np.argmax(cumsum_pct >= percentage) + 1

# Main function
def main():
    """Execute chemical diversity analysis"""
    
    # Paths
    base_dir = r"C:\Users\amjer\Documents\Dengue\Versión final"
    data_path = os.path.join(base_dir, "QSAR5.0", "01_EDA", "supplementary", "tables", "Data_S1_processed_compounds.csv")
    output_dir = os.path.join(base_dir, "QSAR5.0")
    
    # Verify file exists
    if not os.path.exists(data_path):
        print(f"ERROR: Processed EDA file not found at {data_path}")
        print("Please run script 01_exploratory_data_analysis.py first")
        return
    
    # Create analyzer
    analyzer = ChemicalDiversityAnalyzer(data_path, output_dir)
    
    # Execute analysis
    try:
        # Scaffold analysis
        scaffold_df = analyzer.analyze_scaffolds()
        
        # Fingerprint calculation
        fingerprints = analyzer.calculate_molecular_fingerprints()
        
        # Chemical clustering
        clusters = analyzer.analyze_chemical_clusters()
        
        # Pharmacophoric features
        pharma_features = analyzer.calculate_pharmacophore_features()
        
        # Generate descriptor matrix
        descriptor_matrix = analyzer.generate_descriptor_matrix()
        
        # Final report
        analyzer.generate_summary_report()
        
        # Generate supplementary materials
        supp_dir = analyzer.generate_supplementary_materials()
        
        print("\n✔ Chemical diversity analysis completed successfully!")
        print(f"✔ Supplementary materials organized in: {supp_dir}")
        
    except Exception as e:
        print(f"\nERROR during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()