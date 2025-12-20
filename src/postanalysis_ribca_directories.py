#!/usr/bin/env python3
"""
Modified version of postanalysis_clean.py to process specific directories with RIBCA_CellType data.
Processes CODEX data from directories listed in directories_with_ribca_celltype.txt
and creates AnnData objects with 20,000 cell subsamples per dataset.
"""

import os
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from itertools import product

import numpy as np
import pandas as pd
import tifffile
import scipy.io
from scipy import sparse
import anndata as ad

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Constants
COMMON_CHANNELS = ['CD11c', 'CD21', 'CD4', 'CD8', 'Ki67']
# Generate all possible channel combinations for covariance
COVAR_COMMON_CHANNELS = [f"{ch1}:{ch2}" for ch1, ch2 in product(COMMON_CHANNELS, COMMON_CHANNELS)]
SUBSAMPLE_SIZE = 20000
RANDOM_SEED = 42


class RIBCACODEXDataProcessor:
    """Main class for processing CODEX data from specific RIBCA directories and creating AnnData objects."""
    
    def __init__(self, output_dir: Path, directories_file: Path):
        self.output_dir = output_dir
        self.directories_file = directories_file
        self.rng = np.random.default_rng(RANDOM_SEED)
        
        # Data paths
        self.stanford = Path('/hive/hubmap/data/consortium/Stanford TMC/')
        
        # Initialize storage
        self.adj_matrix_list = []
        self.cell_center_list = []
        
        # Load directories from file
        self.directories = self._load_directories()
        
    def _load_directories(self) -> List[str]:
        """Load directory names from the directories file."""
        directories = []
        with open(self.directories_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Remove the './' prefix if present
                    if line.startswith('./'):
                        line = line[2:]
                    directories.append(line)
        return directories
    
    def find_feature_files(self, dataset_path: Path) -> Dict[str, List[Path]]:
        """Find all feature files for a specific dataset."""
        sprm_outputs_path = dataset_path / 'sprm_outputs'
        
        if not sprm_outputs_path.exists():
            print(f"Warning: sprm_outputs directory not found in {dataset_path}")
            return {}
        
        patterns = {
            'cell_cluster': r'*-cell_cluster.csv',
            'cell_shape': r'*-cell_shape.csv',
            'cell_polygons': r'*-cell_polygons_spatial.csv',
            'cell_centers': r'*-cell_centers.csv',
            'adj_matrix': r'*_AdjacencyMatrix.mtx',
            'quality_metrics': r'*.json',
            'mean': r'*_channel_mean.csv',
            'covar': r'*_channel_covar.csv',
            'total': r'*_channel_total.csv',
            'meanall': r'*_channel_meanAll.csv'
        }
        
        return {key: list(sprm_outputs_path.glob(pattern)) 
                for key, pattern in patterns.items()}
    
    def get_dataset_pixels(self, dataset_path: Path, dataset_id: str) -> int:
        """Calculate pixel count for a specific dataset."""
        try:
            # Look for mask files in the dataset
            mask_files = list(dataset_path.glob(r'**/reg001_*'))
            if not mask_files:
                mask_files = list(dataset_path.glob(r'**/reg1_stitched_mask.*'))
            
            if mask_files:
                mask = tifffile.imread(mask_files[0])
                return mask.shape[-1] * mask.shape[-2]
            else:
                print(f"Warning: No mask file found for {dataset_id}")
                return 0
        except Exception as e:
            print(f"Warning: Could not get pixels for {dataset_id}: {e}")
            return 0
    
    def load_and_process_features(self, feature_files: Dict[str, List[Path]], dataset_id: str) -> Dict[str, pd.DataFrame]:
        """Load and process feature files for a specific dataset."""
        processed_features = {}
        
        # Load cell cluster data (this contains RIBCA_CellType)
        if 'cell_cluster' in feature_files and feature_files['cell_cluster']:
            cell_cluster_file = feature_files['cell_cluster'][0]
            try:
                cell_cluster_data = pd.read_csv(cell_cluster_file)
                processed_features['cell_cluster'] = cell_cluster_data
                print(f"  Loaded cell cluster data: {len(cell_cluster_data)} cells")
            except Exception as e:
                print(f"Warning: Could not load cell cluster data for {dataset_id}: {e}")
                processed_features['cell_cluster'] = pd.DataFrame()
        else:
            processed_features['cell_cluster'] = pd.DataFrame()
        
        # Load shape data
        if 'cell_shape' in feature_files and feature_files['cell_shape']:
            shape_file = feature_files['cell_shape'][0]
            try:
                shape_data = pd.read_csv(shape_file)
                processed_features['shape'] = shape_data
                print(f"  Loaded shape data: {len(shape_data)} cells")
            except Exception as e:
                print(f"Warning: Could not load shape data for {dataset_id}: {e}")
                processed_features['shape'] = pd.DataFrame()
        else:
            processed_features['shape'] = pd.DataFrame()
        
        # Load cell centers for spatial coordinates
        if 'cell_centers' in feature_files and feature_files['cell_centers']:
            centers_file = feature_files['cell_centers'][0]
            try:
                centers_data = pd.read_csv(centers_file)
                processed_features['cell_centers'] = centers_data
                print(f"  Loaded cell centers: {len(centers_data)} cells")
            except Exception as e:
                print(f"Warning: Could not load cell centers for {dataset_id}: {e}")
                processed_features['cell_centers'] = pd.DataFrame()
        else:
            processed_features['cell_centers'] = pd.DataFrame()
        
        # Load adjacency matrix
        if 'adj_matrix' in feature_files and feature_files['adj_matrix']:
            adj_file = feature_files['adj_matrix'][0]
            try:
                adj_matrix = scipy.io.mmread(adj_file)
                adj_matrix = adj_matrix.tocsr()  # Convert to CSR for subscriptability
                processed_features['adjacency'] = adj_matrix
                print(f"  Loaded adjacency matrix: {adj_matrix.shape}")
            except Exception as e:
                print(f"Warning: Could not load adjacency matrix for {dataset_id}: {e}")
                processed_features['adjacency'] = sparse.csr_matrix((0, 0))
        else:
            processed_features['adjacency'] = sparse.csr_matrix((0, 0))
        
        # Load quality metrics
        if 'quality_metrics' in feature_files and feature_files['quality_metrics']:
            qm_file = feature_files['quality_metrics'][0]
            try:
                with open(qm_file) as f:
                    qm = json.load(f)
                processed_features['quality_metrics'] = qm
                print(f"  Loaded quality metrics for {dataset_id}")
            except Exception as e:
                print(f"Warning: Could not load quality metrics for {dataset_id}: {e}")
                processed_features['quality_metrics'] = {}
        else:
            processed_features['quality_metrics'] = {}
        
        # Load intensity features (mean, covar, total, meanall) with COMMON_CHANNELS filtering
        feature_types = ['mean', 'covar', 'total', 'meanall']
        for feature_type in feature_types:
            if feature_type in feature_files and feature_files[feature_type]:
                feature_file = feature_files[feature_type][0]
                try:
                    feature_data = pd.read_csv(feature_file)
                    
                    # Filter to COMMON_CHANNELS (matching postanalysis_clean.py behavior)
                    if feature_type == 'covar':
                        # For covariance data, filter by channel combinations
                        available_columns = feature_data.columns.tolist()
                        covar_columns = [col for col in COVAR_COMMON_CHANNELS if col in available_columns]
                        if covar_columns:
                            filtered_data = feature_data[covar_columns]
                            print(f"  Loaded {feature_type} data: {len(filtered_data)} cells, {len(covar_columns)} covariance features (filtered to COMMON_CHANNELS)")
                        else:
                            print(f"Warning: No covariance columns found in available columns: {available_columns[:10]}...")
                            filtered_data = pd.DataFrame()
                    else:
                        # For mean, total, and meanall data, filter by individual channels
                        available_columns = feature_data.columns.tolist()
                        common_columns = [col for col in COMMON_CHANNELS if col in available_columns]
                        if common_columns:
                            filtered_data = feature_data[common_columns]
                            print(f"  Loaded {feature_type} data: {len(filtered_data)} cells, {len(common_columns)} channels (filtered to COMMON_CHANNELS)")
                        else:
                            print(f"Warning: No COMMON_CHANNELS found in available columns: {available_columns[:10]}...")
                            filtered_data = pd.DataFrame()
                    
                    processed_features[feature_type] = filtered_data
                except Exception as e:
                    print(f"Warning: Could not load {feature_type} data for {dataset_id}: {e}")
                    processed_features[feature_type] = pd.DataFrame()
            else:
                processed_features[feature_type] = pd.DataFrame()
        
        return processed_features
    
    def create_ann_data_object(self, 
                              features: Dict[str, pd.DataFrame],
                              dataset_id: str,
                              pixel_count: int,
                              subsampled_indices: np.ndarray) -> ad.AnnData:
        """Create an AnnData object from processed features (matching postanalysis_clean.py format)."""
        
        # Get main feature matrices (matching postanalysis_clean.py structure)
        mean_data = features.get('mean', pd.DataFrame())
        total_data = features.get('total', pd.DataFrame())
        covar_data = features.get('covar', pd.DataFrame())
        shape_data = features.get('shape', pd.DataFrame())
        cell_cluster_data = features.get('cell_cluster', pd.DataFrame())
        quality_metrics = features.get('quality_metrics', {})
        
        # Check if we have enough data for subsampling
        if len(mean_data) == 0:
            print(f"Warning: No mean intensity data found for dataset {dataset_id}")
            return self._create_empty_ann_data(dataset_id, pixel_count)
        
        # Ensure subsampled indices don't exceed available data
        max_index = len(mean_data) - 1
        valid_indices = subsampled_indices[subsampled_indices <= max_index]
        
        if len(valid_indices) == 0:
            print(f"Warning: No valid indices for dataset {dataset_id}")
            return self._create_empty_ann_data(dataset_id, pixel_count)
        
        # Subsample data
        subsampled_data = {
            'mean': mean_data.iloc[valid_indices] if len(mean_data) > 0 else pd.DataFrame(),
            'total': total_data.iloc[valid_indices] if len(total_data) > 0 else pd.DataFrame(),
            'covar': covar_data.iloc[valid_indices] if len(covar_data) > 0 else pd.DataFrame(),
            'shape': shape_data.iloc[valid_indices] if len(shape_data) > 0 else pd.DataFrame(),
            'cell_cluster': cell_cluster_data.iloc[valid_indices] if len(cell_cluster_data) > 0 else pd.DataFrame()
        }
        
        # Create observation metadata (matching postanalysis_clean.py format)
        obs = pd.DataFrame({
            'dataset_id': dataset_id,
            'cell_id': valid_indices,
            'total_pixels_in_dataset': pixel_count
        }, index=range(len(valid_indices)))
        
        # Add cell type annotations from cell cluster data if available
        if len(subsampled_data['cell_cluster']) > 0:
            cell_cluster = subsampled_data['cell_cluster']
            
            # Add RIBCA_CellType if available
            if 'RIBCA_CellType' in cell_cluster.columns:
                obs['RIBCA_CellType'] = cell_cluster['RIBCA_CellType'].values
            
            # Add DeepCellTypes_CellType if available
            if 'DeepCellTypes_CellType' in cell_cluster.columns:
                obs['DeepCellTypes_CellType'] = cell_cluster['DeepCellTypes_CellType'].values

            # Add ID if available
            if 'ID' in cell_cluster.columns:
                obs['ID'] = cell_cluster['ID'].values
        
        # Create variable metadata (matching postanalysis_clean.py format)
        channel_names = list(subsampled_data['mean'].columns) if len(subsampled_data['mean']) > 0 else []
        var = pd.DataFrame({
            'channel_name': channel_names,
            'feature_type': ['mean_intensity'] * len(channel_names)
        }, index=channel_names)
        
        # Create spatial coordinates (matching postanalysis_clean.py format)
        spatial_coords = pd.DataFrame({
            'x': np.random.rand(len(valid_indices)) * 1000,
            'y': np.random.rand(len(valid_indices)) * 1000
        })
        
        # Create empty adjacency matrix (matching postanalysis_clean.py format)
        adj_matrix = sparse.csr_matrix((len(valid_indices), len(valid_indices)))
        
        # Create AnnData object (matching postanalysis_clean.py format)
        adata = ad.AnnData(
            X=subsampled_data['mean'].values if len(subsampled_data['mean']) > 0 else np.zeros((len(valid_indices), 0)),
            obs=obs,
            var=var,
            obsm={'spatial': spatial_coords.values},
            obsp={'adjacency': adj_matrix}
        )
        
        # Add feature layers (matching postanalysis_clean.py format)
        if len(subsampled_data['total']) > 0:
            adata.layers['total_intensity_cell'] = subsampled_data['total'].values
        
        # Add data with different dimensions to obsm (matching postanalysis_clean.py format)
        if len(subsampled_data['covar']) > 0:
            adata.obsm['covariance_intensity_cell'] = subsampled_data['covar'].values
        if len(subsampled_data['shape']) > 0:
            adata.obsm['shape_features'] = subsampled_data['shape'].values
        
        # Add cell cluster features to obsm (preserving the extra cell cluster data)
        if len(subsampled_data['cell_cluster']) > 0:
            cell_cluster = subsampled_data['cell_cluster']
            # Exclude non-feature columns
            exclude_cols = ['ID', 'RIBCA_CellType', 'DeepCellTypes_CellType', 'RIBCA_CellType Factorized', 'DeepCellTypes_CellType Factorized']
            feature_cols = [col for col in cell_cluster.columns if col not in exclude_cols]
            
            if len(feature_cols) > 0:
                # Convert to numeric and ensure it's serializable
                cluster_features = cell_cluster[feature_cols].copy()
                for col in feature_cols:
                    cluster_features[col] = pd.to_numeric(cluster_features[col], errors='coerce')
                cluster_features = cluster_features.fillna(0)
                adata.obsm['cell_cluster_features'] = cluster_features.values.astype(np.float64)
        
        # Store dataset-level metadata in uns (matching postanalysis_clean.py format)
        dataset_info = {
            'dataset_id': dataset_id,
            'total_pixels': pixel_count,
            'subsampled_size': len(valid_indices),
            'original_data_size': len(mean_data),
            'data_type': 'cell_only',
            'common_channels': COMMON_CHANNELS,
            'covariance_channels': COVAR_COMMON_CHANNELS
        }
        
        # Add quality metrics if available
        if quality_metrics:
            dataset_info['quality_metrics'] = quality_metrics
        
        adata.uns['dataset_info'] = dataset_info
        
        # Add cell type mappings if available
        if len(subsampled_data['cell_cluster']) > 0:
            cell_cluster = subsampled_data['cell_cluster']
            
            # Create mappings from factorized codes to labels
            if 'RIBCA_CellType' in cell_cluster.columns and 'RIBCA_CellType Factorized' in cell_cluster.columns:
                ribca_mapping = dict(pd.Series(
                    cell_cluster['RIBCA_CellType'].values,
                    index=cell_cluster['RIBCA_CellType Factorized'].values.astype(str)
                ).drop_duplicates())
                adata.uns['RIBCA_CellType_mapping'] = ribca_mapping
                
            if 'DeepCellTypes_CellType' in cell_cluster.columns and 'DeepCellTypes_CellType Factorized' in cell_cluster.columns:
                deepcell_mapping = dict(pd.Series(
                    cell_cluster['DeepCellTypes_CellType'].values,
                    index=cell_cluster['DeepCellTypes_CellType Factorized'].values.astype(str)
                ).drop_duplicates())
                adata.uns['DeepCellTypes_CellType_mapping'] = deepcell_mapping
        
        return adata
    
    def _create_empty_ann_data(self, dataset_id: str, pixel_count: int) -> ad.AnnData:
        """Create an empty AnnData object when no data is available (matching postanalysis_clean.py format)."""
        obs = pd.DataFrame({
            'dataset_id': dataset_id,
            'cell_id': [],
            'total_pixels_in_dataset': pixel_count
        })
        
        var = pd.DataFrame({
            'channel_name': [],
            'feature_type': []
        })
        
        adata = ad.AnnData(
            X=np.zeros((0, 0)),
            obs=obs,
            var=var,
            obsm={'spatial': np.zeros((0, 2))},
            obsp={'adjacency': sparse.csr_matrix((0, 0))}
        )
        
        adata.uns['dataset_info'] = {
            'dataset_id': dataset_id,
            'total_pixels': pixel_count,
            'subsampled_size': 0,
            'original_data_size': 0,
            'data_type': 'cell_only',
            'common_channels': COMMON_CHANNELS,
            'covariance_channels': COVAR_COMMON_CHANNELS,
            'quality_metrics': {}
        }
        
        return adata
    
    def process_dataset(self, dataset_id: str) -> Optional[ad.AnnData]:
        """Process a single dataset and create AnnData object."""
        print(f"\nProcessing dataset: {dataset_id}")
        
        # Check if output already exists
        output_filename = f"{dataset_id}_adata.h5ad"
        output_path = self.output_dir / output_filename
        if output_path.exists():
            # print(f"  Output already exists: {output_filename}")
            # print(f"  Skipping processing for {dataset_id}")
            
            print(f"  Loading existing AnnData object: {output_filename}")
            # Load the existing AnnData object
            adata = ad.read_h5ad(output_path)
            return adata
        
        # Construct dataset path
        dataset_path = self.stanford / dataset_id
        
        if not dataset_path.exists():
            print(f"  Warning: Dataset path {dataset_path} does not exist")
            return None
        
        # Find feature files
        feature_files = self.find_feature_files(dataset_path)
        
        if not feature_files:
            print(f"  Warning: No feature files found for {dataset_id}")
            return None
        
        # Get dataset pixel count
        pixel_count = self.get_dataset_pixels(dataset_path, dataset_id)
        
        # Load and process features
        print("  Loading and processing features...")
        features = self.load_and_process_features(feature_files, dataset_id)
        
        # Get the actual number of cells (using mean data as primary source)
        mean_data = features.get('mean', pd.DataFrame())
        actual_cells = len(mean_data)
        
        if actual_cells == 0:
            print(f"    Warning: No mean intensity data found for dataset {dataset_id}")
            adata = self._create_empty_ann_data(dataset_id, pixel_count)
        else:
            # Determine subsample size
            if actual_cells < SUBSAMPLE_SIZE:
                print(f"    Found {actual_cells} cells, using all")
                subsample_size = actual_cells
            else:
                print(f"    Found {actual_cells} cells, subsampling {SUBSAMPLE_SIZE}")
                subsample_size = SUBSAMPLE_SIZE
            
            # Generate subsampled indices
            subsampled_indices = self.rng.choice(actual_cells, 
                                               size=subsample_size, 
                                               replace=False)
            
            # Create AnnData object
            adata = self.create_ann_data_object(
                features, dataset_id, pixel_count, subsampled_indices
            )
        
        # Save AnnData
        adata.write_h5ad(output_path)
        print(f"    Saved: {output_filename}")
        
        return adata
    
    def run(self):
        """Main processing pipeline."""
        print("Starting RIBCA CODEX data processing...")
        print(f"Processing {len(self.directories)} directories from {self.directories_file}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each dataset
        all_adata = []
        successful_datasets = []
        failed_datasets = []
        
        for dataset_id in self.directories:
            try:
                adata = self.process_dataset(dataset_id)
                if adata is not None:
                    all_adata.append(adata)
                    successful_datasets.append(dataset_id)
            except Exception as e:
                print(f"Error processing dataset {dataset_id}: {e}")
                failed_datasets.append(dataset_id)
        
        # Create combined AnnData for all successful datasets
        if all_adata:
            print(f"\nCreating combined AnnData for {len(all_adata)} datasets")
            combined_adata = ad.concat(all_adata, join='outer', index_unique='_')
            
            # Store uns metadata from all datasets for later reference
            if all_adata and hasattr(all_adata[0], 'uns') and all_adata[0].uns:
                print(f"  Storing uns metadata from all datasets")
                combined_adata.uns = {}
                for i, adata in enumerate(all_adata):
                    if hasattr(adata, 'uns') and adata.uns:
                        dataset_id = adata.obs['dataset_id'].iloc[0] if 'dataset_id' in adata.obs.columns else f"dataset_{i}"
                        combined_adata.uns[f'uns_dataset_{i}'] = {
                            'dataset_id': dataset_id,
                            'mean_intensity_columns': adata.uns.get('mean_intensity_columns', []),
                            'total_intensity_columns': adata.uns.get('total_intensity_columns', [])
                        }
            
            combined_filename = "ribca_combined_adata.h5ad"
            combined_adata.write_h5ad(self.output_dir / combined_filename)
            print(f"  Saved: {combined_filename}")
        
        # Print summary
        print(f"\nProcessing complete!")
        print(f"  Successful datasets: {len(successful_datasets)}")
        print(f"  Failed datasets: {len(failed_datasets)}")
        print(f"  Created {len(all_adata)} AnnData objects.")
        
        if failed_datasets:
            print(f"\nFailed datasets:")
            for dataset_id in failed_datasets:
                print(f"  - {dataset_id}")


def main():
    """Main entry point."""
    output_dir = Path.home() / 'workspace' / 'FIGURES' / 'ribca_int_dataset'
    directories_file = Path.home() / 'workspace' / 'directories_with_ribca_celltype.txt'
    
    processor = RIBCACODEXDataProcessor(output_dir, directories_file)
    processor.run()


if __name__ == "__main__":
    main() 