#!/usr/bin/env python3
"""
SPRM Analysis: Comprehensive analysis of SPRM-processed CODEX data.
Creates figures, tables, and analysis results from the combined AnnData tissue files.
Compatible with both tissue-based and RIBCA cell type analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import json

import anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import scanpy as sc

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Optional GPU acceleration (RAPIDS cuML)
GPU_UMAP_AVAILABLE = False
try:
    import cupy as cp  # type: ignore
    from cuml.manifold import UMAP as cuUMAP  # type: ignore
    GPU_UMAP_AVAILABLE = True
except Exception:
    GPU_UMAP_AVAILABLE = False

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Constants
COMMON_CHANNELS = ['CD11c', 'CD21', 'CD4', 'CD8', 'Ki67']
TISSUE_NAMES = ['lymph_nodes', 'spleen', 'thymus', 'large_intestine', 'small_intestine']
TISSUE_COLORS = {
    'lymph_nodes': '#1f77b4',
    'spleen': '#ff7f0e', 
    'thymus': '#2ca02c',
    'large_intestine': '#d62728',
    'small_intestine': '#9467bd'
}


class UnifiedSPRMAnalyzer:
    """Main class for analyzing SPRM-processed CODEX data (tissue-based and RIBCA)."""
    
    def __init__(self, data_dir: Path, output_dir: Path, include_feature_types: Optional[List[str]] = None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for loaded data
        self.combined_adata = None
        self.feature_matrix = None
        self.feature_names = None
        self.analysis_type = None  # 'tissue' or 'ribca'

        # Shared preprocessing models
        self.scaler: Optional[StandardScaler] = None
        self.pca_model: Optional[PCA] = None

        # Which feature categories to include during extraction
        # Allowed values: 'mean', 'total', 'covariance', 'shape'
        default_features = ['mean', 'total', 'covariance', 'shape']
        if include_feature_types is None:
            self.include_feature_types = default_features
        else:
            allowed = set(default_features)
            requested = [f.lower() for f in include_feature_types]
            self.include_feature_types = [f for f in requested if f in allowed]
        
    def detect_analysis_type(self) -> str:
        """Detect whether this is tissue-based or RIBCA analysis."""
        # Check for RIBCA combined dataset
        ribca_file = self.data_dir / "ribca_combined_adata.h5ad"
        if ribca_file.exists():
            return 'ribca'
        
        # Check for tissue-based datasets
        tissue_files = [self.data_dir / f"{tissue}_combined_adata.h5ad" for tissue in TISSUE_NAMES]
        if any(f.exists() for f in tissue_files):
            return 'tissue'
        
        raise ValueError("No compatible datasets found in data directory!")
    
    def load_datasets(self) -> ad.AnnData:
        """Load datasets based on detected analysis type."""
        self.analysis_type = self.detect_analysis_type()
        print(f"Detected analysis type: {self.analysis_type}")
        
        if self.analysis_type == 'ribca':
            return self._load_ribca_datasets()
        else:
            return self._load_tissue_datasets()
    
    def _load_ribca_datasets(self) -> ad.AnnData:
        """Load RIBCA combined dataset."""
        print("Loading RIBCA combined dataset...")
        
        combined_file = self.data_dir / "ribca_combined_adata.h5ad"
        
        if combined_file.exists():
            print(f"  Loading RIBCA combined dataset...")
            self.combined_adata = ad.read_h5ad(combined_file)
            
            print(f"    RIBCA dataset: {self.combined_adata.n_obs:,} cells, {self.combined_adata.n_vars} features")
            
            # Check for cell type information
            if 'RIBCA_CellType' in self.combined_adata.obs.columns:
                cell_types = self.combined_adata.obs['RIBCA_CellType'].unique()
                print(f"    Cell types found: {len(cell_types)}")
                for ct in cell_types:
                    count = (self.combined_adata.obs['RIBCA_CellType'] == ct).sum()
                    print(f"      {ct}: {count:,} cells")
            else:
                print("    Warning: No RIBCA_CellType found in obs")
            
            return self.combined_adata
        else:
            raise ValueError(f"RIBCA combined dataset not found at {combined_file}")
    
    def _load_tissue_datasets(self) -> ad.AnnData:
        """Load all combined tissue datasets and concatenate them."""
        print("Loading combined tissue datasets...")
        
        adata_list = []
        total_cells = 0
        
        for tissue_name in TISSUE_NAMES:
            combined_file = self.data_dir / f"{tissue_name}_combined_adata.h5ad"
            
            if combined_file.exists():
                print(f"  Loading {tissue_name}...")
                tissue_adata = ad.read_h5ad(combined_file)
                
                # Add tissue information if not already present
                if 'tissue_type' not in tissue_adata.obs.columns:
                    tissue_adata.obs['tissue_type'] = tissue_name
                
                print(f"    {tissue_name}: {tissue_adata.n_obs:,} cells, {tissue_adata.n_vars} features")
                adata_list.append(tissue_adata)
                total_cells += tissue_adata.n_obs
            else:
                print(f"  Warning: {combined_file} not found")
        
        if not adata_list:
            raise ValueError("No combined datasets found!")
        
        # Concatenate all datasets
        print(f"\nConcatenating {len(adata_list)} datasets...")
        self.combined_adata = ad.concat(adata_list, join='outer', index_unique='_')
        
        print(f"Combined dataset: {self.combined_adata.n_obs:,} cells, {self.combined_adata.n_vars} features")
        print(f"Tissue distribution:")
        for tissue in TISSUE_NAMES:
            count = (self.combined_adata.obs['tissue_type'] == tissue).sum()
            print(f"  {tissue}: {count:,} cells")
        
        return self.combined_adata
    
    def extract_feature_set(self) -> Tuple[np.ndarray, List[str]]:
        """
        Extract comprehensive feature set for each cell based on analysis type.
        """
        if self.analysis_type == 'ribca':
            return self._extract_ribca_feature_set()
        else:
            return self._extract_tissue_feature_set()
    
    def _extract_ribca_feature_set(self) -> Tuple[np.ndarray, List[str]]:
        """
        Extract comprehensive feature set for each cell from RIBCA data:
        - Mean intensity features for common channels (from main X matrix)
        - Total intensity features for common channels (from layers)
        - Covariance between common channels (from obsm)
        - Shape features (from obsm)
        Note: Cluster features are excluded for UMAP analysis
        """
        print("\nExtracting comprehensive RIBCA feature set (excluding cluster features)...")
        
        features_list = []
        feature_names = []
        
        # 1. Mean intensity features (from main X matrix) - already filtered to COMMON_CHANNELS
        if self.combined_adata.n_vars > 0 and 'mean' in self.include_feature_types:
            mean_features = self.combined_adata.X
            features_list.append(mean_features)
            # Use variable names from AnnData (should be COMMON_CHANNELS)
            mean_names = [f"mean_{col}" for col in self.combined_adata.var_names]
            feature_names.extend(mean_names)
            print(f"  Mean intensity features: {mean_features.shape[1]} features (from main X matrix)")
            print(f"    Using channels: {list(self.combined_adata.var_names)}")
        
        # 2. Total intensity features (from layers) - already filtered to COMMON_CHANNELS
        if 'total_intensity_cell' in self.combined_adata.layers and 'total' in self.include_feature_types:
            total_features = self.combined_adata.layers['total_intensity_cell']
            features_list.append(total_features)
            # Use same variable names as mean features (should be COMMON_CHANNELS)
            total_names = [f"total_{col}" for col in self.combined_adata.var_names]
            feature_names.extend(total_names)
            print(f"  Total intensity features: {total_features.shape[1]} features (from layers)")
            print(f"    Using channels: {list(self.combined_adata.var_names)}")
        
        # 3. Covariance features (from obsm) - already filtered to COMMON_CHANNELS combinations
        if 'covariance_intensity_cell' in self.combined_adata.obsm and 'covariance' in self.include_feature_types:
            covar_features = self.combined_adata.obsm['covariance_intensity_cell']
            features_list.append(covar_features)
            # Generate covariance feature names based on COMMON_CHANNELS
            covar_names = []
            for i, ch1 in enumerate(COMMON_CHANNELS):
                for j, ch2 in enumerate(COMMON_CHANNELS):
                    covar_names.append(f"covar_{ch1}_{ch2}")
            # Limit to actual number of covariance features
            feature_names.extend(covar_names[:covar_features.shape[1]])
            print(f"  Covariance features: {covar_features.shape[1]} features")
            print(f"    Using channel combinations from: {COMMON_CHANNELS}")
        
        # 4. Shape features (from obsm) - exclude first column (scale factor)
        if 'shape_features' in self.combined_adata.obsm and 'shape' in self.include_feature_types:
            shape_features = self.combined_adata.obsm['shape_features']
            # Exclude first column (scale factor) to focus on shape characteristics
            # shape_features_no_scale = shape_features[:, 1:]
            # features_list.append(shape_features_no_scale)

            features_list.append(shape_features)
            # Generate shape feature names (excluding scale factor)
            # shape_names = [f"shape_{i}" for i in range(shape_features_no_scale.shape[1])]
            shape_names = [f"shape_{i}" for i in range(shape_features.shape[1])]
            feature_names.extend(shape_names)
            # print(f"  Shape features (excluding scale factor): {shape_features_no_scale.shape[1]} features")
            print(f"  Shape features: {shape_features.shape[1]} features")
        
        # Combine all features
        if features_list:
            self.feature_matrix = np.hstack(features_list)
            self.feature_names = feature_names
            
            print(f"\nComprehensive RIBCA feature set: {self.feature_matrix.shape[1]} total features")
            print(f"Feature breakdown:")
            print(f"  Mean intensity: {len([f for f in feature_names if f.startswith('mean_')])}")
            print(f"  Total intensity: {len([f for f in feature_names if f.startswith('total_')])}")
            print(f"  Covariance: {len([f for f in feature_names if f.startswith('covar_')])}")
            print(f"  Shape: {len([f for f in feature_names if f.startswith('shape_')])}")
            print(f"  Note: Cluster features excluded from UMAP analysis")
            print(f"  Using COMMON_CHANNELS: {COMMON_CHANNELS}")
            
            return self.feature_matrix, self.feature_names
        else:
            raise ValueError("No features found in the RIBCA dataset!")
    
    def _extract_tissue_feature_set(self) -> Tuple[np.ndarray, List[str]]:
        """
        Extract comprehensive feature set for each cell from tissue data:
        - Mean intensity for common channels
        - Total intensity for common channels  
        - Covariance between common channels
        - Shape features
        """
        print("\nExtracting comprehensive tissue feature set...")
        
        features_list = []
        feature_names = []
        
        # 1. Mean intensity features (main X matrix)
        if self.combined_adata.n_vars > 0 and 'mean' in self.include_feature_types:
            mean_features = self.combined_adata.X
            features_list.append(mean_features)
            feature_names.extend([f"mean_{col}" for col in self.combined_adata.var_names])
            print(f"  Mean intensity features: {mean_features.shape[1]} features")
        
        # 2. Total intensity features (from layers)
        if 'total_intensity_cell' in self.combined_adata.layers and 'total' in self.include_feature_types:
            total_features = self.combined_adata.layers['total_intensity_cell']
            features_list.append(total_features)
            feature_names.extend([f"total_{col}" for col in self.combined_adata.var_names])
            print(f"  Total intensity features: {total_features.shape[1]} features")
        
        # 3. Covariance features (from obsm)
        if 'covariance_intensity_cell' in self.combined_adata.obsm and 'covariance' in self.include_feature_types:
            covar_features = self.combined_adata.obsm['covariance_intensity_cell']
            features_list.append(covar_features)
            # Generate covariance feature names
            covar_names = []
            for i, ch1 in enumerate(COMMON_CHANNELS):
                for j, ch2 in enumerate(COMMON_CHANNELS):
                    covar_names.append(f"covar_{ch1}_{ch2}")
            feature_names.extend(covar_names[:covar_features.shape[1]])  # Limit to actual features
            print(f"  Covariance features: {covar_features.shape[1]} features")
        
        # 4. Shape features (from obsm) - exclude first column (scale factor)
        if 'shape_features' in self.combined_adata.obsm and 'shape' in self.include_feature_types:
            shape_features = self.combined_adata.obsm['shape_features']
            # Exclude first column (scale factor) to focus on shape characteristics
            # shape_features_no_scale = shape_features[:, 1:]
            # features_list.append(shape_features_no_scale)
            features_list.append(shape_features)

            # Generate shape feature names (excluding scale factor)
            # shape_names = [f"shape_{i}" for i in range(shape_features_no_scale.shape[1])]
            shape_names = [f"shape_{i}" for i in range(shape_features.shape[1])]
            feature_names.extend(shape_names)
            # print(f"  Shape features (excluding scale factor): {shape_features_no_scale.shape[1]} features")
            print(f"  Shape features: {shape_features.shape[1]} features")
        
        # Combine all features
        if features_list:
            self.feature_matrix = np.hstack(features_list)
            self.feature_names = feature_names
            
            print(f"\nComprehensive tissue feature set: {self.feature_matrix.shape[1]} total features")
            print(f"Feature breakdown:")
            print(f"  Mean intensity: {len([f for f in feature_names if f.startswith('mean_')])}")
            print(f"  Total intensity: {len([f for f in feature_names if f.startswith('total_')])}")
            print(f"  Covariance: {len([f for f in feature_names if f.startswith('covar_')])}")
            print(f"  Shape: {len([f for f in feature_names if f.startswith('shape_')])}")
            
            return self.feature_matrix, self.feature_names
        else:
            raise ValueError("No features found in the tissue dataset!")
    
    def preprocess_features(
        self,
        feature_matrix: np.ndarray,
        reduce_dimensions: bool = False,
        pca_n_components: int = 500,
        fit_pipeline: bool = True,
        existing_scaler: Optional[StandardScaler] = None,
        existing_pca: Optional[PCA] = None,
        force_float32: bool = True,
    ) -> np.ndarray:
        """Preprocess features for UMAP analysis with optional shared scaler/PCA.

        If fit_pipeline is True, this method fits a new StandardScaler (and PCA if requested)
        and stores them on the analyzer for later reuse. If fit_pipeline is False, it will use
        the provided existing models or previously fitted models on this analyzer.
        """
        print("\nPreprocessing features...")

        # Handle missing values
        feature_matrix_clean = np.nan_to_num(feature_matrix, nan=0.0)

        # Select scaler to use
        if fit_pipeline:
            self.scaler = StandardScaler()
            feature_matrix_scaled = self.scaler.fit_transform(feature_matrix_clean)
        else:
            scaler_to_use = existing_scaler or self.scaler
            if scaler_to_use is None:
                raise ValueError("No fitted scaler available for transform. Fit on tissue first or provide existing_scaler.")
            feature_matrix_scaled = scaler_to_use.transform(feature_matrix_clean)

        print(f"  Input shape: {feature_matrix.shape}")

        # Optional dimensionality reduction
        if reduce_dimensions:
            num_features = feature_matrix_scaled.shape[1]
            effective_components = min(pca_n_components, num_features - 1) if num_features > 1 else 1
            if effective_components < num_features and effective_components > 0:
                if fit_pipeline:
                    print(f"  Reducing dimensions from {num_features} to {effective_components} using PCA...")
                    self.pca_model = PCA(n_components=effective_components, random_state=42)
                    feature_matrix_scaled = self.pca_model.fit_transform(feature_matrix_scaled)
                    print(f"  PCA explained variance: {self.pca_model.explained_variance_ratio_.sum():.3f}")
                else:
                    pca_to_use = existing_pca or self.pca_model
                    if pca_to_use is None:
                        raise ValueError("No fitted PCA available for transform. Fit on tissue first or provide existing_pca.")
                    feature_matrix_scaled = pca_to_use.transform(feature_matrix_scaled)
            else:
                print(f"  Skipping PCA: requested components ({pca_n_components}) not less than feature count ({num_features})")

        # Enforce float32 to reduce memory footprint
        if force_float32 and feature_matrix_scaled.dtype != np.float32:
            feature_matrix_scaled = feature_matrix_scaled.astype(np.float32, copy=False)

        print(f"  Output shape: {feature_matrix_scaled.shape}")
        print(f"  Features standardized: mean={float(feature_matrix_scaled.mean()):.3f}, std={float(feature_matrix_scaled.std()):.3f}")

        return feature_matrix_scaled
    
    def create_umap_embedding(self, feature_matrix: np.ndarray, 
                            n_neighbors: int = 15, 
                            min_dist: float = 0.1,
                            n_components: int = 2,
                            random_state: int = 42,
                            fit_reducer: bool = True,
                            reducer: Optional[umap.UMAP] = None,
                            use_approximate: bool = True,
                            batch_size: Optional[int] = None) -> Tuple[np.ndarray, umap.UMAP]:
        """Create UMAP embedding of the feature matrix."""
        print(f"\nCreating UMAP embedding...")
        print(f"  Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}")
        print(f"  Fit reducer: {fit_reducer}")
        
        if fit_reducer or reducer is None:
            # Initialize and fit UMAP with optimizations for large datasets
            print(f"  Dataset size: {feature_matrix.shape[0]:,} cells × {feature_matrix.shape[1]:,} features")
            
            # Use approximate nearest neighbors for large datasets (CPU path)
            if use_approximate and feature_matrix.shape[0] > 100000:
                print(f"  Using approximate nearest neighbors for large dataset")
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    n_components=n_components,
                    random_state=random_state,
                    verbose=True,
                    n_jobs=-1,
                    low_memory=False,
                    metric='euclidean'
                )
            else:
                # Prefer GPU UMAP if available
                if GPU_UMAP_AVAILABLE:
                    print("  Using GPU-accelerated UMAP (RAPIDS cuML)")
                    reducer = cuUMAP(
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=n_components,
                        random_state=random_state,
                        metric='euclidean',
                        init='spectral',
                        output_type='numpy',
                        verbose=True,
                    )
                else:
                    reducer = umap.UMAP(
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=n_components,
                        random_state=random_state,
                        verbose=True
                    )
            
            # Fit and transform
            print(f"  Starting UMAP fitting (this may take a while for large datasets)...")
            if GPU_UMAP_AVAILABLE and isinstance(reducer, object) and reducer.__class__.__name__ == 'UMAP':
                # cuML path expects data on device; reducer outputs numpy because output_type='numpy'
                umap_embedding = reducer.fit_transform(cp.asarray(feature_matrix))  # type: ignore
            else:
                umap_embedding = reducer.fit_transform(feature_matrix)
            if umap_embedding.dtype != np.float32:
                umap_embedding = umap_embedding.astype(np.float32, copy=False)
            print(f"  Fitted new UMAP reducer")
        else:
            # Use existing reducer to transform
            total_cells = feature_matrix.shape[0]
            print(f"  Applying existing UMAP reducer to {total_cells:,} cells...")
            if batch_size is not None and total_cells > batch_size:
                print(f"  Transforming in batches of {batch_size}...")
                # Pre-allocate output
                umap_embedding = np.empty((total_cells, n_components), dtype=np.float32)
                start = 0
                while start < total_cells:
                    end = min(start + batch_size, total_cells)
                    batch = feature_matrix[start:end]
                    if GPU_UMAP_AVAILABLE and reducer.__class__.__module__.startswith('cuml'):
                        transformed = reducer.transform(cp.asarray(batch))  # type: ignore
                    else:
                        transformed = reducer.transform(batch)
                    if transformed.dtype != np.float32:
                        transformed = transformed.astype(np.float32, copy=False)
                    umap_embedding[start:end] = transformed
                    print(f"    Transformed rows {start:,}–{end-1:,}")
                    start = end
            else:
                if GPU_UMAP_AVAILABLE and reducer.__class__.__module__.startswith('cuml'):
                    umap_embedding = reducer.transform(cp.asarray(feature_matrix))  # type: ignore
                else:
                    umap_embedding = reducer.transform(feature_matrix)
                if umap_embedding.dtype != np.float32:
                    umap_embedding = umap_embedding.astype(np.float32, copy=False)
            print(f"  Applied existing UMAP reducer")
        
        print(f"  UMAP embedding shape: {umap_embedding.shape}")
        
        return umap_embedding, reducer
    
    def save_umap_reducer(self, reducer: umap.UMAP, save_path: Path):
        """Save UMAP reducer to file."""
        import joblib
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(reducer, save_path)
        print(f"  Saved UMAP reducer to: {save_path}")
    
    def load_umap_reducer(self, load_path: Path) -> umap.UMAP:
        """Load UMAP reducer from file."""
        import joblib
        if load_path.exists():
            reducer = joblib.load(load_path)
            print(f"  Loaded UMAP reducer from: {load_path}")
            return reducer
        else:
            raise FileNotFoundError(f"UMAP reducer not found at {load_path}")
    
    def create_umap_figure(self, umap_embedding: np.ndarray, 
                          save_path: Optional[Path] = None) -> plt.Figure:
        """Create UMAP visualization figure based on analysis type."""
        if self.analysis_type == 'ribca':
            return self._create_celltype_umap_figure(umap_embedding, save_path)
        else:
            return self._create_tissue_umap_figure(umap_embedding, save_path)
    
    def _create_celltype_umap_figure(self, umap_embedding: np.ndarray, 
                                    save_path: Optional[Path] = None) -> plt.Figure:
        """Create UMAP visualization figure colored by cell type with improved visibility."""
        print("\nCreating UMAP visualization colored by cell type...")
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Get cell types
        if 'RIBCA_CellType' in self.combined_adata.obs.columns:
            cell_types = self.combined_adata.obs['RIBCA_CellType'].values
            unique_cell_types = np.unique(cell_types)
            
            print(f"  Found {len(unique_cell_types)} unique cell types")
            
            # Create better color palette for cell types with more distinct colors
            # Use a combination of color maps for better distinction
            if len(unique_cell_types) <= 8:
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cell_types)))
            else:
                # Combine multiple color maps for more distinct colors
                colors1 = plt.cm.tab10(np.linspace(0, 1, min(10, len(unique_cell_types))))
                colors2 = plt.cm.Set2(np.linspace(0, 1, max(0, len(unique_cell_types) - 10)))
                colors = np.vstack([colors1, colors2])[:len(unique_cell_types)]
            
            # Calculate sampling parameters
            total_cells = len(umap_embedding)
            max_points_per_cell_type = 3000  # Reduced for better visibility
            min_points_per_cell_type = 500    # Reduced minimum for cleaner look
            
            print(f"  Implementing subsampling for better visibility...")
            
            # Plot each cell type with subsampling
            for i, cell_type in enumerate(unique_cell_types):
                mask = cell_types == cell_type
                if mask.sum() > 0:
                    cell_type_indices = np.where(mask)[0]
                    cell_count = len(cell_type_indices)
                    
                    # Determine number of points to sample
                    if cell_count > max_points_per_cell_type:
                        # Subsample for large cell types
                        sample_size = max_points_per_cell_type
                        sampled_indices = np.random.choice(cell_type_indices, size=sample_size, replace=False)
                        print(f"    {cell_type}: {cell_count:,} cells → {sample_size:,} sampled")
                    elif cell_count < min_points_per_cell_type:
                        # Use all cells for small cell types
                        sampled_indices = cell_type_indices
                        print(f"    {cell_type}: {cell_count:,} cells (all shown)")
                    else:
                        # Use all cells for medium-sized cell types
                        sampled_indices = cell_type_indices
                        print(f"    {cell_type}: {cell_count:,} cells (all shown)")
                    
                    # Calculate point size based on cell count (larger for fewer cells)
                    if cell_count < 500:
                        point_size = 12.0
                    elif cell_count < 2000:
                        point_size = 6.0
                    elif cell_count < 10000:
                        point_size = 3.0
                    else:
                        point_size = 1.5
                    
                    # Calculate transparency based on cell count (more transparent for more cells)
                    if cell_count < 500:
                        alpha = 0.9
                    elif cell_count < 2000:
                        alpha = 0.7
                    elif cell_count < 10000:
                        alpha = 0.5
                    else:
                        alpha = 0.3
                    
                    # Plot with optimized parameters
                    ax.scatter(umap_embedding[sampled_indices, 0], umap_embedding[sampled_indices, 1], 
                              c=[colors[i]], label=f"{cell_type} ({cell_count:,})", 
                              alpha=alpha, s=point_size, edgecolors='none')
            
            # Customize plot
            ax.set_xlabel('UMAP 1', fontsize=14)
            ax.set_ylabel('UMAP 2', fontsize=14)
            ax.set_title('UMAP Embedding of RIBCA Features Colored by Cell Type\n(Subsampled for visibility)', fontsize=16, fontweight='bold')
            
            # Add compact legend with better positioning
            ax.legend(title='Cell Type', title_fontsize=11, fontsize=9, 
                     bbox_to_anchor=(1.02, 1), loc='upper left', 
                     frameon=True, fancybox=True, shadow=True, ncol=1)
            
            # Add compact statistics text
            stats_text = f"Total: {total_cells:,} cells\n"
            stats_text += f"Types: {len(unique_cell_types)}\n"
            stats_text += f"Max sampled: {max_points_per_cell_type:,}\n"
            # Show only top 3 cell types by count
            cell_type_counts = [(ct, (cell_types == ct).sum()) for ct in unique_cell_types]
            cell_type_counts.sort(key=lambda x: x[1], reverse=True)
            for i, (cell_type, count) in enumerate(cell_type_counts[:3]):
                stats_text += f"{cell_type}: {count:,}\n"
            if len(cell_type_counts) > 3:
                stats_text += f"... +{len(cell_type_counts)-3} more"
            
            ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes, 
                    verticalalignment='top', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
        else:
            # Fallback if no cell type information
            ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], alpha=0.6, s=2.0)
            ax.set_xlabel('UMAP 1', fontsize=14)
            ax.set_ylabel('UMAP 2', fontsize=14)
            ax.set_title('UMAP Embedding of RIBCA Features', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved UMAP figure to: {save_path}")
        
        return fig
    
    def _create_tissue_umap_figure(self, umap_embedding: np.ndarray, 
                                  save_path: Optional[Path] = None) -> plt.Figure:
        """Create UMAP visualization figure with improved sampling and overlap handling."""
        print("\nCreating UMAP visualization colored by tissue type...")
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Get tissue types
        tissue_types = self.combined_adata.obs['tissue_type'].values
        
        # Fixed sampling parameters
        max_cells_per_tissue = 5000  # Hard-coded limit
        print(f"  Sampling max {max_cells_per_tissue:,} cells per tissue")
        
        # Calculate tissue counts and sampling
        tissue_counts = {}
        sampled_counts = {}
        
        for tissue in TISSUE_NAMES:
            tissue_counts[tissue] = (tissue_types == tissue).sum()
        
        # Create scatter plot with improved sampling and visualization
        for tissue in TISSUE_NAMES:
            mask = tissue_types == tissue
            if mask.sum() > 0:
                tissue_indices = np.where(mask)[0]
                total_cells = len(tissue_indices)
                
                # Sample cells for this tissue
                if total_cells > max_cells_per_tissue:
                    # Randomly sample max_cells_per_tissue from this tissue
                    sampled_indices = np.random.choice(tissue_indices, size=max_cells_per_tissue, replace=False)
                    sampled_counts[tissue] = max_cells_per_tissue
                    print(f"    {tissue}: {total_cells:,} cells → {max_cells_per_tissue:,} sampled")
                else:
                    # Use all cells if this tissue has fewer than max_cells_per_tissue
                    sampled_indices = tissue_indices
                    sampled_counts[tissue] = total_cells
                    print(f"    {tissue}: {total_cells:,} cells (all shown)")
                
                # Calculate point size based on cell count (smaller for more cells)
                if total_cells < 10000:
                    point_size = 8.0
                elif total_cells < 50000:
                    point_size = 4.0
                else:
                    point_size = 2.0
                
                # Calculate transparency based on cell count (more transparent for more cells)
                if total_cells < 10000:
                    alpha = 0.8
                elif total_cells < 50000:
                    alpha = 0.6
                else:
                    alpha = 0.4
                
                # Plot with optimized parameters for better overlap visibility
                ax.scatter(umap_embedding[sampled_indices, 0], umap_embedding[sampled_indices, 1], 
                          c=TISSUE_COLORS[tissue], label=f"{tissue} ({sampled_counts[tissue]:,})", 
                          alpha=alpha, s=point_size, edgecolors='none')
        
        # Customize plot
        ax.set_xlabel('UMAP 1', fontsize=14)
        ax.set_ylabel('UMAP 2', fontsize=14)
        ax.set_title('UMAP Embedding of SPRM Features Across All Tissues\n(Fixed sampling: 5000 cells per tissue)', fontsize=16, fontweight='bold')
        
        # Add legend with better positioning
        ax.legend(title='Tissue Type', title_fontsize=12, fontsize=10, 
                 bbox_to_anchor=(1.05, 1), loc='upper left', 
                 frameon=True, fancybox=True, shadow=True)
        
        # Add comprehensive statistics text
        total_cells = len(umap_embedding)
        total_sampled = sum(sampled_counts.values())
        stats_text = f"Total cells: {total_cells:,}\n"
        stats_text += f"Total sampled: {total_sampled:,}\n"
        stats_text += f"Max per tissue: {max_cells_per_tissue:,}\n"
        for tissue in TISSUE_NAMES:
            count = tissue_counts[tissue]
            sampled = sampled_counts.get(tissue, 0)
            stats_text += f"{tissue}: {count:,} → {sampled:,}\n"
        
        ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes, 
                verticalalignment='top', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved UMAP figure to: {save_path}")
        
        return fig
    
    def run_analysis(self):
        """Run the complete SPRM analysis pipeline."""
        print("=" * 60)
        print("SPRM Analysis: Comprehensive CODEX Data Analysis")
        print("=" * 60)
        
        # Load data
        self.load_datasets()
        
        # Extract features
        feature_matrix, feature_names = self.extract_feature_set()
        
        # Preprocess features
        feature_matrix_scaled = self.preprocess_features(feature_matrix)
        
        # Create UMAP embedding
        umap_embedding, reducer = self.create_umap_embedding(feature_matrix_scaled)
        
        # Create UMAP figure
        if self.analysis_type == 'ribca':
            umap_fig_path = self.output_dir / 'ribca_umap_celltypes_2.png'
        else:
            umap_fig_path = self.output_dir / 'umap_all_tissues_normalized.png'
        
        umap_fig = self.create_umap_figure(umap_embedding, umap_fig_path)
        
        # Save UMAP coordinates
        umap_coords_df = pd.DataFrame(
            umap_embedding, 
            columns=['UMAP_1', 'UMAP_2']
        )
        
        if self.analysis_type == 'ribca':
            umap_coords_df['dataset_id'] = self.combined_adata.obs['dataset_id'].values
            if 'RIBCA_CellType' in self.combined_adata.obs.columns:
                umap_coords_df['cell_type'] = self.combined_adata.obs['RIBCA_CellType'].values
            umap_coords_path = self.output_dir / 'ribca_umap_coordinates.csv'
        else:
            umap_coords_df['tissue_type'] = self.combined_adata.obs['tissue_type'].values
            umap_coords_df['dataset_id'] = self.combined_adata.obs['dataset_id'].values
            umap_coords_path = self.output_dir / 'umap_coordinates.csv'
        
        umap_coords_df.to_csv(umap_coords_path, index=False)
        print(f"\nSaved UMAP coordinates to: {umap_coords_path}")
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Analysis type: {self.analysis_type}")
        print(f"Total cells analyzed: {self.combined_adata.n_obs:,}")
        print(f"Total features: {len(feature_names)}")
        
        if self.analysis_type == 'ribca':
            if 'RIBCA_CellType' in self.combined_adata.obs.columns:
                cell_types = self.combined_adata.obs['RIBCA_CellType'].unique()
                print(f"Cell types: {len(cell_types)}")
                
                print(f"\nCell type distribution:")
                for cell_type in cell_types:
                    count = (self.combined_adata.obs['RIBCA_CellType'] == cell_type).sum()
                    print(f"  {cell_type}: {count:,} cells")
        else:
            print(f"Tissue types: {len(TISSUE_NAMES)}")
            print(f"\nTissue distribution:")
            for tissue in TISSUE_NAMES:
                count = (self.combined_adata.obs['tissue_type'] == tissue).sum()
                print(f"  {tissue}: {count:,} cells")
        
        print(f"\nOutput files saved to: {self.output_dir}")
        if self.analysis_type == 'ribca':
            print("  - ribca_umap_celltypes_2.png: UMAP visualization")
            print("  - ribca_umap_coordinates.csv: UMAP coordinates")
        else:
            print("  - umap_all_tissues_normalized.png: UMAP visualization")
            print("  - umap_coordinates.csv: UMAP coordinates")
        
        return {
            'combined_adata': self.combined_adata,
            'umap_embedding': umap_embedding,
            'feature_matrix': feature_matrix_scaled,
            'feature_names': feature_names,
            'analysis_type': self.analysis_type
        }
    
    def run_multi_dataset_analysis(self, ribca_data_dir=None, tissue_data_dir=None, use_shared_umap: bool = True,
                                   include_feature_types: Optional[List[str]] = None,
                                   ribca_use_pca: bool = False,
                                   pca_components: int = 500):
        """Run analysis on both RIBCA and tissue datasets from different directories."""
        print("=" * 80)
        print("MULTI-DATASET SPRM ANALYSIS")
        print("=" * 80)
        
        results = {}
        shared_reducer = None
        shared_reducer_path = self.output_dir / 'shared_umap_reducer.pkl'
        # Resolve feature selection for this run
        feature_types = [f.lower() for f in (include_feature_types if include_feature_types is not None else self.include_feature_types)]
        
        # Check if shared UMAP reducer already exists
        if use_shared_umap and shared_reducer_path.exists():
            print(f"\n{'='*40}")
            print("FOUND EXISTING SHARED UMAP REDUCER")
            print(f"{'='*40}")
            print(f"Loading existing UMAP reducer from: {shared_reducer_path}")
            
            # Load existing reducer
            try:
                shared_reducer = self.load_umap_reducer(shared_reducer_path)
                print(f"  ✅ Successfully loaded existing UMAP reducer")
                print(f"  Skipping tissue dataset processing (already completed)")
            except Exception as e:
                print(f"  ❌ Failed to load existing reducer: {e}")
                print(f"  Will process tissue dataset to create new reducer")
                shared_reducer = None
        else:
            shared_reducer = None
        
        # Process tissue datasets FIRST (to create shared UMAP transformation) - only if needed
        if tissue_data_dir and (shared_reducer is None or not use_shared_umap):
            tissue_data_path = Path(tissue_data_dir)
            tissue_files = [tissue_data_path / f"{tissue}_combined_adata.h5ad" for tissue in TISSUE_NAMES]
            available_tissue_files = [f for f in tissue_files if f.exists()]
            
            if available_tissue_files:
                print(f"\n{'='*40}")
                print("PROCESSING TISSUE DATASETS (FITTING UMAP)")
                print(f"{'='*40}")
                print(f"Tissue data directory: {tissue_data_path}")
                print(f"Available tissue files: {len(available_tissue_files)}")
                
                # Create temporary analyzer for tissue
                tissue_analyzer = UnifiedSPRMAnalyzer(tissue_data_path, self.output_dir, include_feature_types=feature_types)
                tissue_analyzer.analysis_type = 'tissue'
                
                # Create tissue-specific output directory
                tissue_output_dir = self.output_dir / 'tissue_analysis'
                tissue_output_dir.mkdir(parents=True, exist_ok=True)
                tissue_analyzer.output_dir = tissue_output_dir
                
                # Run tissue analysis
                try:
                    # Extract features and preprocess
                    tissue_analyzer.load_datasets()
                    feature_matrix, feature_names = tissue_analyzer.extract_feature_set()
                    # Fit scaler and optional PCA on tissue
                    feature_matrix_scaled = tissue_analyzer.preprocess_features(
                        feature_matrix,
                        reduce_dimensions=False,
                        pca_n_components=pca_components,
                        fit_pipeline=True,
                        force_float32=True,
                    )
                    
                    # Create UMAP embedding (FIT NEW REDUCER on tissue data)
                    print(f"  🔄 Fitting UMAP reducer on tissue dataset...")
                    umap_embedding, reducer = tissue_analyzer.create_umap_embedding(
                        feature_matrix_scaled, fit_reducer=True
                    )
                    
                    # Save shared reducer for RIBCA dataset
                    if use_shared_umap:
                        tissue_analyzer.save_umap_reducer(reducer, shared_reducer_path)
                        shared_reducer = reducer
                        print(f"  ✅ Saved shared UMAP reducer (fitted on tissue) for use with RIBCA dataset")

                        # Persist scaler and PCA for reuse
                        try:
                            import joblib
                            joblib.dump(tissue_analyzer.scaler, self.output_dir / 'shared_scaler.pkl')
                            if tissue_analyzer.pca_model is not None:
                                joblib.dump(tissue_analyzer.pca_model, self.output_dir / 'shared_pca.pkl')
                            print("  ✅ Saved shared scaler/PCA")
                        except Exception as e:
                            print(f"  ⚠️  Failed to save scaler/PCA: {e}")

                        # Persist the exact feature names used for tissue to enforce alignment
                        try:
                            shared_features_path = self.output_dir / 'shared_feature_names.json'
                            with open(shared_features_path, 'w') as f:
                                json.dump(feature_names, f)
                            print(f"  ✅ Saved shared feature names: {shared_features_path}")
                        except Exception as e:
                            print(f"  ⚠️  Failed to save shared feature names: {e}")
                    
                    # Create UMAP figure
                    umap_fig_path = tissue_output_dir / 'umap_all_tissues_normalized.png'
                    umap_fig = tissue_analyzer.create_umap_figure(umap_embedding, umap_fig_path)
                    
                    # Save UMAP coordinates
                    umap_coords_df = pd.DataFrame(
                        umap_embedding, 
                        columns=['UMAP_1', 'UMAP_2']
                    )
                    umap_coords_df['tissue_type'] = tissue_analyzer.combined_adata.obs['tissue_type'].values
                    umap_coords_df['dataset_id'] = tissue_analyzer.combined_adata.obs['dataset_id'].values
                    umap_coords_path = tissue_output_dir / 'umap_coordinates.csv'
                    umap_coords_df.to_csv(umap_coords_path, index=False)
                    
                    # Store results
                    results['tissue'] = {
                        'combined_adata': tissue_analyzer.combined_adata,
                        'umap_embedding': umap_embedding,
                        'feature_matrix': feature_matrix_scaled,
                        'feature_names': feature_names,
                        'analysis_type': 'tissue',
                        'reducer': reducer
                    }
                    
                    print(f"\n✅ Tissue analysis completed successfully!")
                except Exception as e:
                    print(f"\n❌ Tissue analysis failed: {e}")
            else:
                print(f"\n⚠️  No tissue datasets found in {tissue_data_path}")
        elif shared_reducer is not None:
            print(f"\n✅ Using existing shared UMAP reducer (skipping tissue processing)")
        else:
            print(f"\n⚠️  No tissue data directory specified")
        
        # Process RIBCA dataset using shared UMAP transformation from tissue
        if ribca_data_dir:
            ribca_data_path = Path(ribca_data_dir)
            ribca_file = ribca_data_path / "ribca_combined_adata.h5ad"
            
            if ribca_file.exists():
                print(f"\n{'='*40}")
                print("PROCESSING RIBCA DATASET (USING TISSUE UMAP TRANSFORMATION)")
                print(f"{'='*40}")
                print(f"RIBCA data directory: {ribca_data_path}")
                
                # Create temporary analyzer for RIBCA
                ribca_analyzer = UnifiedSPRMAnalyzer(ribca_data_path, self.output_dir, include_feature_types=feature_types)
                ribca_analyzer.analysis_type = 'ribca'
                
                # Create RIBCA-specific output directory
                ribca_output_dir = self.output_dir / 'ribca_analysis'
                ribca_output_dir.mkdir(parents=True, exist_ok=True)
                ribca_analyzer.output_dir = ribca_output_dir
                
                # Run RIBCA analysis
                try:
                    # Extract features
                    ribca_analyzer.load_datasets()
                    feature_matrix, feature_names = ribca_analyzer.extract_feature_set()

                    print(f"  RIBCA feature matrix shape: {feature_matrix.shape}")
                    print(f"  RIBCA feature names: {feature_names}")

                    # Enforce shared feature set and ordering
                    if use_shared_umap:
                        shared_features_path = self.output_dir / 'shared_feature_names.json'
                        if not shared_features_path.exists():
                            raise ValueError("Shared feature names not found. Run tissue processing first to create them (shared_feature_names.json).")
                        try:
                            with open(shared_features_path, 'r') as f:
                                shared_feature_names = json.load(f)
                            print(f"  Aligning RIBCA features to {len(shared_feature_names)} shared feature columns")
                        except Exception as e:
                            raise ValueError(f"Failed to load shared feature names: {e}")

                        # Map current RIBCA feature names to indices
                        ribca_name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
                        missing_features = [name for name in shared_feature_names if name not in ribca_name_to_idx]
                        if missing_features:
                            raise ValueError(
                                "RIBCA dataset is missing required shared features: " + ", ".join(missing_features[:10]) +
                                (" ..." if len(missing_features) > 10 else "")
                            )
                        aligned_indices = [ribca_name_to_idx[name] for name in shared_feature_names]
                        feature_matrix = feature_matrix[:, aligned_indices]
                        feature_names = shared_feature_names
                        print("  ✅ RIBCA features aligned to shared columns")

                    # Load shared scaler/PCA from tissue if available
                    loaded_scaler = None
                    loaded_pca = None
                    try:
                        import joblib
                        scaler_path = self.output_dir / 'shared_scaler.pkl'
                        pca_path = self.output_dir / 'shared_pca.pkl'
                        if scaler_path.exists():
                            loaded_scaler = joblib.load(scaler_path)
                        if pca_path.exists():
                            loaded_pca = joblib.load(pca_path)
                    except Exception as e:
                        print(f"  ⚠️  Failed to load shared scaler/PCA: {e}")

                    if use_shared_umap:
                        # Require shared assets; fail fast with clear message if missing
                        if shared_reducer is None:
                            raise ValueError("Shared UMAP reducer not found. Run tissue processing first to create it.")
                        if loaded_scaler is None:
                            raise ValueError("Shared StandardScaler not found. Run tissue processing first to create it (shared_scaler.pkl).")
                        if ribca_use_pca and loaded_pca is None:
                            raise ValueError("Shared PCA model not found but ribca_use_pca=True. Run tissue processing with PCA or disable ribca PCA.")

                        # Preprocess using shared scaler/PCA
                        feature_matrix_scaled = ribca_analyzer.preprocess_features(
                            feature_matrix,
                            reduce_dimensions=ribca_use_pca,
                            pca_n_components=pca_components,
                            fit_pipeline=False,
                            existing_scaler=loaded_scaler,
                            existing_pca=loaded_pca if ribca_use_pca else None,
                            force_float32=True,
                        )

                        print(f"  🔄 Using shared UMAP transformation from tissue dataset")
                        umap_embedding, _ = ribca_analyzer.create_umap_embedding(
                            feature_matrix_scaled,
                            fit_reducer=False,
                            reducer=shared_reducer,
                            batch_size=100000,
                        )
                    else:
                        # Local fit path (only when explicitly not using shared UMAP)
                        feature_matrix_scaled = ribca_analyzer.preprocess_features(
                            feature_matrix,
                            reduce_dimensions=ribca_use_pca,
                            pca_n_components=pca_components,
                            fit_pipeline=True,
                            force_float32=True,
                        )
                        print(f"  🔄 Creating new UMAP transformation for RIBCA dataset (shared disabled)")
                        umap_embedding, reducer = ribca_analyzer.create_umap_embedding(
                            feature_matrix_scaled, fit_reducer=True
                        )
                    
                    # Create UMAP figure
                    umap_fig_path = ribca_output_dir / 'ribca_umap_celltypes_2.png'
                    umap_fig = ribca_analyzer.create_umap_figure(umap_embedding, umap_fig_path)
                    
                    # Save UMAP coordinates
                    umap_coords_df = pd.DataFrame(
                        umap_embedding, 
                        columns=['UMAP_1', 'UMAP_2']
                    )
                    umap_coords_df['dataset_id'] = ribca_analyzer.combined_adata.obs['dataset_id'].values
                    if 'RIBCA_CellType' in ribca_analyzer.combined_adata.obs.columns:
                        umap_coords_df['cell_type'] = ribca_analyzer.combined_adata.obs['RIBCA_CellType'].values
                    umap_coords_path = ribca_output_dir / 'ribca_umap_coordinates.csv'
                    umap_coords_df.to_csv(umap_coords_path, index=False)
                    
                    # Store results
                    results['ribca'] = {
                        'combined_adata': ribca_analyzer.combined_adata,
                        'umap_embedding': umap_embedding,
                        'feature_matrix': feature_matrix_scaled,
                        'feature_names': feature_names,
                        'analysis_type': 'ribca',
                        'reducer': shared_reducer if use_shared_umap else reducer
                    }
                    
                    print(f"\n✅ RIBCA analysis completed successfully!")
                except Exception as e:
                    print(f"\n❌ RIBCA analysis failed: {e}")
            else:
                print(f"\n⚠️  RIBCA dataset not found at {ribca_file}")
        else:
            print(f"\n⚠️  No RIBCA data directory specified")
        
        # Print final summary
        print(f"\n{'='*80}")
        print("MULTI-DATASET ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"Datasets processed: {len(results)}")
        print(f"Shared UMAP transformation: {'Yes (fitted on tissue)' if use_shared_umap else 'No'}")
        for dataset_type, result in results.items():
            print(f"  {dataset_type.upper()}: {result['combined_adata'].n_obs:,} cells")
        
        if not results:
            print("❌ No datasets were successfully processed!")
        else:
            print(f"\n✅ Multi-dataset analysis completed!")
            print(f"Results saved to:")
            for dataset_type in results.keys():
                if dataset_type == 'ribca':
                    print(f"  - {self.output_dir}/ribca_analysis/")
                else:
                    print(f"  - {self.output_dir}/tissue_analysis/")
            if use_shared_umap and shared_reducer_path.exists():
                print(f"  - Shared UMAP reducer (fitted on tissue): {shared_reducer_path}")
        
        # Compose side-by-side UMAP figure if both images are present
        tissue_png = self.output_dir / 'tissue_analysis' / 'umap_all_tissues_normalized.png'
        ribca_png = self.output_dir / 'ribca_analysis' / 'ribca_umap_celltypes_2.png'
        side_by_side_png = self.output_dir / 'umap_tissue_ribca_side_by_side.png'
        self.compose_side_by_side_umap(tissue_png, ribca_png, side_by_side_png)
        
        return results


    def compose_side_by_side_umap(self, tissue_png: Path, ribca_png: Path, out_png: Path) -> Optional[Path]:
        """Create a side-by-side figure from tissue and RIBCA UMAP PNGs."""
        try:
            if not tissue_png.exists() or not ribca_png.exists():
                print(f"⚠️  Skipping side-by-side: missing files. Tissue: {tissue_png.exists()}, RIBCA: {ribca_png.exists()}")
                return None

            tissue_img = plt.imread(str(tissue_png))
            ribca_img = plt.imread(str(ribca_png))

            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            axes[0].imshow(tissue_img)
            axes[0].axis('off')
            axes[0].set_title('Tissue UMAP')

            axes[1].imshow(ribca_img)
            axes[1].axis('off')
            axes[1].set_title('RIBCA UMAP')

            plt.tight_layout()
            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out_png), dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Saved side-by-side UMAP figure to: {out_png}")
            return out_png
        except Exception as e:
            print(f"❌ Failed to compose side-by-side UMAP: {e}")
            return None
    
    def extract_quality_metrics(self, adata: ad.AnnData) -> Tuple[pd.DataFrame, List[str]]:
        """Extract quality metrics from AnnData object(s).
        
        Returns:
            Tuple of (quality_metrics_df, dataset_ids) where:
            - quality_metrics_df: DataFrame with quality metrics as rows, features as columns
            - dataset_ids: List of dataset IDs corresponding to each row
        """
        print("\nExtracting quality metrics from AnnData...")
        
        qm_data = []
        dataset_ids = []
        
        # Check if this is a combined AnnData with multiple datasets
        if 'dataset_info' in adata.uns:
            # Single dataset case
            qm_dict = adata.uns['dataset_info'].get('quality_metrics', {})
            if qm_dict:
                # Exclude dataset_id from features
                qm_features = {k: v for k, v in qm_dict.items() if k != 'dataset_id'}
                if qm_features:
                    qm_data.append(qm_features)
                    dataset_ids.append(adata.uns['dataset_info'].get('dataset_id', 'unknown'))
        else:
            # Multiple datasets case - need to check individual AnnData objects
            # This would require loading individual files or having them in memory
            print("  Warning: Combined AnnData detected, but quality metrics extraction from combined objects not yet implemented")
            print("  Quality metrics should be extracted from individual dataset AnnData objects")
        
        if not qm_data:
            # Try alternative: extract from unique dataset_ids in obs
            unique_datasets = adata.obs['dataset_id'].unique() if 'dataset_id' in adata.obs.columns else []
            print(f"  Found {len(unique_datasets)} unique datasets in obs")
            print("  Note: Quality metrics are stored per-dataset in uns['dataset_info']['quality_metrics']")
            print("  For combined AnnData, quality metrics should be extracted from individual dataset files")
            
            return pd.DataFrame(), []
        
        # Create DataFrame
        qm_df = pd.DataFrame(qm_data)
        
        # Convert numeric columns
        for col in qm_df.columns:
            try:
                qm_df[col] = pd.to_numeric(qm_df[col], errors='coerce')
            except:
                pass
        
        # Fill NaN with 0
        qm_df = qm_df.fillna(0)
        
        print(f"  Extracted quality metrics for {len(qm_df)} dataset(s)")
        print(f"  Quality metric features: {list(qm_df.columns)}")
        
        return qm_df, dataset_ids
    
    def extract_quality_metrics_from_directory(self, data_dir: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Extract specific quality metrics from all individual dataset AnnData files in a directory.
        
        Extracts only the metrics specified in Table 4:
        - Fraction of pixels in Image Background
        - Fraction of Image occupied by cells
        - Otsu:CD11c, Otsu:CD21, Otsu:CD4, Otsu:CD8, Otsu:Ki67
        - M/SD:CD11c, M/SD:CD21, M/SD:CD4, M/SD:CD8, M/SD:Ki67 (stored as Z-Score)
        - meanInt:CD11c, meanInt:CD21, meanInt:CD4, meanInt:CD8, meanInt:Ki67
        
        Returns:
            Tuple of (quality_metrics_df, dataset_ids, tissue_types)
        """
        print(f"\nExtracting quality metrics from directory: {data_dir}")
        
        # Define the specific metrics we want (matching Table 4)
        required_metrics = [
            'FracPixInImgBG',  # Fraction of pixels in Image Background
            'FracImgOfCells',  # Fraction of Image occupied by cells
        ]
        
        # Add Otsu metrics for each channel
        for channel in COMMON_CHANNELS:
            required_metrics.append(f'Otsu: {channel}')  # Original format with space
        
        # Add M/SD metrics (stored as Z-Score in the data)
        for channel in COMMON_CHANNELS:
            required_metrics.append(f'Z-Score: {channel}')  # Original format
        
        # Add meanInt metrics for each channel
        for channel in COMMON_CHANNELS:
            required_metrics.append(f'meanInt: {channel}')  # Original format with space
        
        qm_data = []
        dataset_ids = []
        tissue_types = []
        
        # First, try to find individual dataset AnnData files
        pattern = "*_adata.h5ad"
        adata_files = list(data_dir.glob(pattern))
        
        # Also check for combined datasets
        combined_files = list(data_dir.glob("*_combined_adata.h5ad"))
        
        if adata_files:
            print(f"  Found {len(adata_files)} individual dataset file(s)")
            
            # Load each file and extract quality metrics
            for adata_file in adata_files:
                try:
                    adata = ad.read_h5ad(adata_file)
                    dataset_info = adata.uns.get('dataset_info', {})
                    qm_dict = dataset_info.get('quality_metrics', {})
                    
                    if qm_dict:
                        # Extract only the required metrics
                        qm_row = {}
                        for metric_key in required_metrics:
                            if metric_key in qm_dict:
                                qm_row[metric_key] = qm_dict[metric_key]
                            else:
                                qm_row[metric_key] = np.nan
                        
                        # Only add if we have at least some data
                        if any(not pd.isna(v) for v in qm_row.values()):
                            qm_data.append(qm_row)
                            dataset_id = qm_dict.get('dataset_id') or dataset_info.get('dataset_id', adata_file.stem)
                            dataset_ids.append(dataset_id)
                            # Get tissue type from dataset_info
                            tissue_type = dataset_info.get('tissue_type', 'unknown')
                            tissue_types.append(tissue_type)
                except Exception as e:
                    print(f"  Warning: Could not extract quality metrics from {adata_file.name}: {e}")
                    continue
        
        # If no individual files found, try combined datasets
        if not qm_data and combined_files:
            print(f"  No individual files found, trying {len(combined_files)} combined dataset file(s)")
            print(f"  Note: For combined datasets, quality metrics should be in individual dataset files")
            
            # For combined datasets, we'd need to load individual files
            # This is a limitation - quality metrics are per-dataset
            pass
        
        if not qm_data:
            print("  No quality metrics found in any dataset files")
            print("  Make sure AnnData files contain quality metrics in uns['dataset_info']['quality_metrics']")
            return pd.DataFrame(), [], []
        
        # Create DataFrame with only the required metrics
        qm_df = pd.DataFrame(qm_data)
        
        # Rename columns to match Table 4 format (no spaces, M/SD instead of Z-Score)
        column_mapping = {}
        for channel in COMMON_CHANNELS:
            # Otsu: CD11c -> Otsu:CD11c
            column_mapping[f'Otsu: {channel}'] = f'Otsu:{channel}'
            # Z-Score: CD11c -> M/SD:CD11c
            column_mapping[f'Z-Score: {channel}'] = f'M/SD:{channel}'
            # meanInt: CD11c -> meanInt:CD11c
            column_mapping[f'meanInt: {channel}'] = f'meanInt:{channel}'
        
        qm_df = qm_df.rename(columns=column_mapping)
        
        # Convert numeric columns
        for col in qm_df.columns:
            try:
                qm_df[col] = pd.to_numeric(qm_df[col], errors='coerce')
            except:
                pass
        
        # Fill NaN with 0
        qm_df = qm_df.fillna(0)
        
        print(f"  Extracted quality metrics for {len(qm_df)} dataset(s)")
        print(f"  Quality metric features: {list(qm_df.columns)}")
        print(f"  Tissue types found: {set(tissue_types)}")
        
        return qm_df, dataset_ids, tissue_types
    
    def run_quality_metrics_umap(self, data_dir: Optional[Path] = None, 
                                 n_neighbors: int = 15,
                                 min_dist: float = 0.1,
                                 random_state: int = 42) -> Tuple[np.ndarray, pd.DataFrame, List[str], List[str]]:
        """Run UMAP analysis on image quality metrics.
        
        Args:
            data_dir: Directory containing AnnData files. If None, uses self.data_dir
            n_neighbors: UMAP n_neighbors parameter (smaller for fewer samples)
            min_dist: UMAP min_dist parameter
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (umap_embedding, quality_metrics_df, dataset_ids, tissue_types)
        """
        print("\n" + "=" * 60)
        print("QUALITY METRICS UMAP ANALYSIS")
        print("=" * 60)
        
        data_dir = data_dir or self.data_dir
        
        # Extract quality metrics (only the specific metrics from Table 4)
        qm_df, dataset_ids, tissue_types = self.extract_quality_metrics_from_directory(data_dir)
        
        if len(qm_df) == 0:
            raise ValueError("No quality metrics found! Make sure AnnData files contain quality metrics in uns['dataset_info']['quality_metrics']")
        
        if len(qm_df) < 2:
            raise ValueError(f"Need at least 2 datasets for UMAP, found {len(qm_df)}")
        
        print(f"\nQuality metrics matrix shape: {qm_df.shape}")
        print(f"Number of datasets: {len(dataset_ids)}")
        # print(f"Using only the metrics specified in Table 4")
        
        # Preprocess quality metrics
        qm_matrix = qm_df.values.astype(np.float32)
        
        # Standardize
        scaler = StandardScaler()
        qm_matrix_scaled = scaler.fit_transform(qm_matrix)
        
        print(f"  Standardized quality metrics: mean={float(qm_matrix_scaled.mean()):.3f}, std={float(qm_matrix_scaled.std()):.3f}")
        
        # Create UMAP embedding
        print(f"\nCreating UMAP embedding for quality metrics...")
        print(f"  Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}")
        print(f"  Dataset size: {qm_matrix_scaled.shape[0]} datasets × {qm_matrix_scaled.shape[1]} features")
        
        reducer = umap.UMAP(
            n_neighbors=min(n_neighbors, len(qm_df) - 1),  # Ensure n_neighbors < n_samples
            min_dist=min_dist,
            n_components=2,
            random_state=random_state,
            verbose=True
        )
        
        umap_embedding = reducer.fit_transform(qm_matrix_scaled)
        
        print(f"  UMAP embedding shape: {umap_embedding.shape}")
        
        return umap_embedding, qm_df, dataset_ids, tissue_types
    
    def create_quality_metrics_umap_figure(self, umap_embedding: np.ndarray, 
                                           dataset_ids: List[str],
                                           tissue_types: List[str],
                                           save_path: Optional[Path] = None) -> plt.Figure:
        """Create UMAP visualization figure for quality metrics colored by tissue type.
        
        Args:
            umap_embedding: UMAP embedding (n_datasets, 2)
            dataset_ids: List of dataset IDs (for tracking)
            tissue_types: List of tissue types (for coloring)
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        print("\nCreating quality metrics UMAP visualization...")
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Get unique tissue types and assign colors
        unique_tissues = list(set(tissue_types))
        n_tissues = len(unique_tissues)
        
        print(f"  Found {n_tissues} unique tissue types: {unique_tissues}")
        print(f"  Total datasets: {len(dataset_ids)}")
        
        # Use predefined tissue colors if available, otherwise generate
        tissue_to_color = {}
        for i, tissue in enumerate(unique_tissues):
            if tissue in TISSUE_COLORS:
                tissue_to_color[tissue] = TISSUE_COLORS[tissue]
            else:
                # Use a color palette for unknown tissues
                colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_tissues)))
                tissue_to_color[tissue] = colors[i % len(colors)]
        
        # Plot each dataset colored by tissue type
        for i, (dataset_id, tissue_type) in enumerate(zip(dataset_ids, tissue_types)):
            color = tissue_to_color.get(tissue_type, 'gray')
            # Only label first occurrence of each tissue type
            label = tissue_type if tissue_type not in [tissue_types[j] for j in range(i)] else None
            ax.scatter(umap_embedding[i, 0], umap_embedding[i, 1], 
                      c=[color], label=label,
                      alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
        
        # Customize plot
        ax.set_xlabel('Component 1', fontsize=14)
        ax.set_ylabel('Component 2', fontsize=14)
        ax.set_title('Image Quality Metrics UMAP', 
                    fontsize=16, fontweight='bold')
        
        # Add legend for tissue types
        ax.legend(title='Tissue Type', title_fontsize=12, fontsize=10,
                 bbox_to_anchor=(1.02, 1), loc='upper left',
                 frameon=True, fancybox=True, shadow=True, ncol=1)
        
        # Add statistics text
        stats_text = f"Total datasets: {len(dataset_ids)}\n"
        stats_text += f"Tissue types: {n_tissues}\n"
        stats_text += f"UMAP components: 2"
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
               verticalalignment='bottom', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=500, bbox_inches='tight')
            print(f"  Saved quality metrics UMAP figure to: {save_path}")
        
        return fig
    
    def compute_quality_metrics_correlations(self, umap_embedding: np.ndarray,
                                            qm_df: pd.DataFrame,
                                            save_path: Optional[Path] = None) -> pd.DataFrame:
        """Compute correlations between quality metrics and UMAP components.
        
        Args:
            umap_embedding: UMAP embedding (n_datasets, 2)
            qm_df: DataFrame with quality metrics (n_datasets, n_features)
            save_path: Optional path to save correlation table
            
        Returns:
            DataFrame with correlations (features × components)
        """
        print("\nComputing correlations between quality metrics and UMAP components...")
        
        # Create DataFrame for correlations
        correlation_data = []
        
        # Compute correlation for each quality metric with each UMAP component
        for feature_name in qm_df.columns:
            feature_values = qm_df[feature_name].values
            
            # Compute correlation with Component 1
            corr_comp1 = np.corrcoef(feature_values, umap_embedding[:, 0])[0, 1]
            
            # Compute correlation with Component 2
            corr_comp2 = np.corrcoef(feature_values, umap_embedding[:, 1])[0, 1]
            
            correlation_data.append({
                'Feature': feature_name,
                'Component 1': corr_comp1,
                'Component 2': corr_comp2
            })
        
        # Create DataFrame
        corr_df = pd.DataFrame(correlation_data)
        
        # Sort by feature name for consistent ordering
        # Order: FracPixInImgBG, FracImgOfCells, then Otsu, M/SD, meanInt for each channel
        feature_order = ['FracPixInImgBG', 'FracImgOfCells']
        for channel in COMMON_CHANNELS:
            feature_order.append(f'Otsu:{channel}')
        for channel in COMMON_CHANNELS:
            feature_order.append(f'M/SD:{channel}')
        for channel in COMMON_CHANNELS:
            feature_order.append(f'meanInt:{channel}')
        
        # Create a mapping for custom sorting
        def sort_key(row):
            feature = row['Feature']
            if feature in feature_order:
                return feature_order.index(feature)
            return len(feature_order)
        
        corr_df['_sort_key'] = corr_df.apply(sort_key, axis=1)
        corr_df = corr_df.sort_values('_sort_key').drop(columns=['_sort_key'])
        
        # Format feature names for display (matching Table 4 format)
        display_names = {
            'FracPixInImgBG': 'Fraction of pixels in Image Background',
            'FracImgOfCells': 'Fraction of Image occupied by cells',
            # Channel-specific metrics keep their names (already in correct format)
        }
        
        # Create display DataFrame with formatted names
        display_df = corr_df.copy()
        display_df['Feature'] = display_df['Feature'].replace(display_names)
        
        # Round to 6 decimal places (matching Table 4 format)
        display_df['Component 1'] = display_df['Component 1'].round(6)
        display_df['Component 2'] = display_df['Component 2'].round(6)
        
        # Save to CSV
        if save_path:
            display_df.to_csv(save_path, index=False)
            print(f"  Saved correlation table to: {save_path}")
        
        # Print table
        print("\n" + "=" * 80)
        print("CORRELATIONS BETWEEN QUALITY METRICS AND UMAP COMPONENTS")
        print("=" * 80)
        print(display_df.to_string(index=False))
        print("=" * 80)
        
        return corr_df
    
    def run_quality_metrics_analysis(self, data_dir: Optional[Path] = None,
                                     n_neighbors: int = 15,
                                     min_dist: float = 0.1,
                                     random_state: int = 42) -> Dict:
        """Run complete quality metrics UMAP analysis pipeline.
        
        Uses only the specific quality metrics from Table 4:
        - Fraction of pixels in Image Background
        - Fraction of Image occupied by cells
        - Otsu:CD11c, Otsu:CD21, Otsu:CD4, Otsu:CD8, Otsu:Ki67
        - M/SD:CD11c, M/SD:CD21, M/SD:CD4, M/SD:CD8, M/SD:Ki67
        - meanInt:CD11c, meanInt:CD21, meanInt:CD4, meanInt:CD8, meanInt:Ki67
        
        Args:
            data_dir: Directory containing AnnData files. If None, uses self.data_dir
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with analysis results
        """
        # Run UMAP analysis
        umap_embedding, qm_df, dataset_ids, tissue_types = self.run_quality_metrics_umap(
            data_dir=data_dir,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )
        
        # Create visualization (colored by tissue type)
        save_path = self.output_dir / 'quality_metrics_umap.png'
        fig = self.create_quality_metrics_umap_figure(umap_embedding, dataset_ids, tissue_types, save_path)
        
        # Save UMAP coordinates with dataset_id and tissue_type
        umap_coords_df = pd.DataFrame(
            umap_embedding,
            columns=['UMAP_1', 'UMAP_2']
        )
        umap_coords_df['dataset_id'] = dataset_ids
        umap_coords_df['tissue_type'] = tissue_types
        
        # Add quality metrics columns
        for col in qm_df.columns:
            umap_coords_df[col] = qm_df[col].values
        
        umap_coords_path = self.output_dir / 'quality_metrics_umap_coordinates.csv'
        umap_coords_df.to_csv(umap_coords_path, index=False)
        print(f"\nSaved quality metrics UMAP coordinates to: {umap_coords_path}")
        
        # Compute and save correlations
        corr_path = self.output_dir / 'quality_metrics_umap_correlations.csv'
        corr_df = self.compute_quality_metrics_correlations(umap_embedding, qm_df, corr_path)
        
        # Print summary
        print("\n" + "=" * 60)
        print("QUALITY METRICS UMAP ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total datasets analyzed: {len(dataset_ids)}")
        print(f"Quality metric features: {len(qm_df.columns)}")
        print(f"Tissue types: {set(tissue_types)}")
        print(f"Output files:")
        print(f"  - quality_metrics_umap.png: UMAP visualization (colored by tissue type)")
        print(f"  - quality_metrics_umap_coordinates.csv: UMAP coordinates with quality metrics, dataset_id, and tissue_type")
        print(f"  - quality_metrics_umap_correlations.csv: Correlations between quality metrics and UMAP components")
        
        return {
            'umap_embedding': umap_embedding,
            'quality_metrics_df': qm_df,
            'dataset_ids': dataset_ids,
            'tissue_types': tissue_types,
            'correlations_df': corr_df,
            'figure': fig
        }

def main():
    """Main entry point."""
    # Set up paths for different datasets
    ribca_data_dir = Path.home() / 'workspace' / 'FIGURES' / 'ribca_int_dataset'
    tissue_data_dir = Path.home() / 'workspace' / 'FIGURES' / 'new_int_dataset'
    output_dir = Path.home() / 'workspace' / 'FIGURES' / 'new_output_120925'
    
    # print(f"RIBCA data directory: {ribca_data_dir}")
    # print(f"Tissue data directory: {tissue_data_dir}")
    # print(f"Output directory: {output_dir}")
    
    # # Create analyzer and run multi-dataset analysis with shared UMAP transformation
    analyzer = UnifiedSPRMAnalyzer(ribca_data_dir, output_dir)  # Use any data_dir as base
    # results = analyzer.run_multi_dataset_analysis(
    #     ribca_data_dir=ribca_data_dir,
    #     tissue_data_dir=tissue_data_dir,
    #     use_shared_umap=True,
    #     include_feature_types=['mean', 'total'], #'mean', 'total', 'covariance', 'shape'
    #     ribca_use_pca=False,
    #     pca_components=500
    # )
    
    # print(f"\nMulti-dataset analysis complete!")
    # print(f"Processed {len(results)} dataset(s)")
    
    # Run quality metrics UMAP analysis
    print("\n" + "=" * 60)
    print("Running Quality Metrics UMAP Analysis")
    print("=" * 60)
    try:
        qm_results = analyzer.run_quality_metrics_analysis(
            data_dir=tissue_data_dir,  # Use tissue data directory for quality metrics
            n_neighbors=15,
            min_dist=0.1,
            random_state=42
        )
        print("\n✅ Quality metrics UMAP analysis completed successfully!")
    except Exception as e:
        print(f"\n⚠️  Quality metrics UMAP analysis failed: {e}")
        print("  This may be expected if quality metrics are not available in the AnnData files")


if __name__ == "__main__":
    main() 