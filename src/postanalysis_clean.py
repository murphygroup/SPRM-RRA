#!/usr/bin/env python3
"""
Clean, refactored version of postanalysis for creating AnnData intermediate datasets.
Processes CODEX data and creates AnnData objects with 20,000 cell subsamples per dataset.
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


class CODEXDataProcessor:
    """Main class for processing CODEX data and creating AnnData objects."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.rng = np.random.default_rng(RANDOM_SEED)
        
        # Data paths
        self.root = Path('/hive/users/tedz/workspace/REPROCESS_CODEX/')
        self.stanford = Path('/hive/hubmap/data/consortium/Stanford TMC/')
        self.bfconvert = Path('/hive/users/tedz/workspace/BFCONVERTED/reprocess-v2.4.2/')
        
        # Tissue configuration
        self.tissue_config = {
            'ln': 'lymph_nodes',
            'spleen': 'spleen', 
            'thymus': 'thymus',
            'li': 'large_intestine',
            'si': 'small_intestine'
        }
        
        # Initialize storage
        self.adj_matrix_list = []
        self.cell_center_list = []
        
    def get_tissue_paths(self) -> List[Tuple[Path, str]]:
        """Get list of tissue paths and their names."""
        return [(self.root / tissue_key, tissue_name) 
                for tissue_key, tissue_name in self.tissue_config.items()]
    
    def find_feature_files(self, tissue_path: Path) -> Dict[str, List[Path]]:
        """Find all feature files for a tissue."""
        patterns = {
            'mean': r'**/*_channel_mean.csv',
            'covar': r'**/*_channel_covar.csv', 
            'total': r'**/*_channel_total.csv',
            'shape': r'**/*-cell_shape.csv',
            'meanall': r'**/*_channel_meanAll.csv',
            'qm': r'**/*.json',
            'cell_polygons': r'**/*-cell_polygons_spatial.csv',
            'adj_matrix': r'**/*_AdjacencyMatrix.mtx',
            'cell_centers': r'**/*-cell_centers.csv'
        }
        
        return {key: list(tissue_path.glob(pattern)) 
                for key, pattern in patterns.items()}
    
    def get_dataset_pixels(self, tissue_path: Path, tissue_name: str) -> Dict[str, int]:
        """Calculate pixel counts for each dataset in a tissue."""
        dataset_ids = [d for d in os.listdir(tissue_path) 
                      if os.path.isdir(tissue_path / d)]
        
        pixel_counts = {}
        failed_datasets = []
        
        # First pass: try to get pixel counts for all datasets
        for dataset_id in dataset_ids:
            try:
                mask_path = self._find_mask_path(dataset_id, tissue_name)
                mask = tifffile.imread(mask_path)
                pixel_counts[dataset_id] = mask.shape[-1] * mask.shape[-2]
            except Exception as e:
                print(f"Warning: Could not get pixels for {dataset_id}: {e}")
                failed_datasets.append(dataset_id)
        
        # Second pass: calculate average for failed datasets
        if failed_datasets and pixel_counts:
            successful_pixels = list(pixel_counts.values())
            avg_pixels = int(np.mean(successful_pixels))
            print(f"  Using average pixel count ({avg_pixels:,}) for {len(failed_datasets)} failed datasets")
            
            for dataset_id in failed_datasets:
                pixel_counts[dataset_id] = avg_pixels
        elif failed_datasets:
            # If all datasets failed, use default
            print(f"  All datasets failed for {tissue_name}, using default pixel count")
            for dataset_id in failed_datasets:
                pixel_counts[dataset_id] = 0  # Default fallback
                
        return pixel_counts
    
    def _find_mask_path(self, dataset_id: str, tissue_name: str) -> Path:
        """Find the mask file path for a dataset."""
        # Stanford datasets (tissue indices 3, 4)
        if tissue_name in ['large_intestine', 'small_intestine']:
            mask_path_root = self.stanford / dataset_id
            try:
                return list(mask_path_root.glob(r'**/reg001_*'))[0]
            except:
                try:
                    return list(mask_path_root.glob(r'**/reg1_stitched_mask.*'))[0]
                except:
                    new_root = Path('/hive/hubmap/data/public/') / dataset_id
                    return list(new_root.glob(r'**/reg001_*'))[0]
        else:
            # BFConvert datasets
            mask_path_root = self.bfconvert / dataset_id / 'pipeline_output'
            return list(mask_path_root.glob(r'**/reg001_mask.*'))[0]
    
    def load_and_process_features(self, feature_files: Dict[str, List[Path]]) -> Dict[str, pd.DataFrame]:
        """Load and process feature files with dataset information preserved."""
        # Load shape data with dataset information
        shape_data = self._load_shape_data_with_dataset_info(feature_files['shape'])
        
        # Process intensity features with dataset information
        intensity_features = self._process_intensity_features_with_dataset_info(feature_files)
        
        # Process quality metrics
        quality_metrics = self._process_quality_metrics(feature_files['qm'])
        
        return {
            'shape': shape_data,
            'quality_metrics': quality_metrics,
            **intensity_features
        }
    
    def _load_shape_data_with_dataset_info(self, shape_files: List[Path]) -> pd.DataFrame:
        """Load shape data while preserving dataset information."""
        shape_dataframes = []
        
        for shape_file in shape_files:
            # Extract dataset ID from file path
            dataset_id = self._extract_dataset_id_from_path(shape_file)
            
            # Load the shape data
            df = pd.read_csv(shape_file)
            
            # Add dataset information
            df['dataset_id'] = dataset_id
            
            shape_dataframes.append(df)
        
        # Concatenate with dataset information preserved
        if shape_dataframes:
            return pd.concat(shape_dataframes, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _process_intensity_features_with_dataset_info(self, feature_files: Dict[str, List[Path]]) -> Dict[str, pd.DataFrame]:
        """Process intensity features while preserving dataset information - only cell data."""
        feature_types = ['mean', 'covar', 'total']
        processed_features = {}
        
        for feature_type in feature_types:
            file_key = feature_type
            if file_key in feature_files:
                paths = feature_files[file_key]
                
                # Only process cell-specific files
                cell_paths = [p for p in paths if 'cell_channel' in p.stem]
                
                if cell_paths:
                    # Load and find common channels with dataset info
                    dataframes_with_info = []
                    for path in cell_paths:
                        df = pd.read_csv(path)
                        dataset_id = self._extract_dataset_id_from_path(path)
                        df['dataset_id'] = dataset_id
                        dataframes_with_info.append(df)
                    
                    # Filter to common channels and concatenate
                    if feature_type == 'covar':
                        # For covariance data, filter by channel combinations
                        available_columns = dataframes_with_info[0].columns.tolist() if dataframes_with_info else []
                        covar_columns = [col for col in COVAR_COMMON_CHANNELS if col in available_columns]
                        if covar_columns:
                            filtered_data = [df[covar_columns + ['dataset_id']] for df in dataframes_with_info]
                        else:
                            print(f"Warning: No covariance columns found in available columns: {available_columns[:10]}...")
                            filtered_data = [df[['dataset_id']] for df in dataframes_with_info]
                    else:
                        # For mean and total data, filter by individual channels
                        filtered_data = [df[COMMON_CHANNELS + ['dataset_id']] for df in dataframes_with_info]
                    
                    concatenated = pd.concat(filtered_data, ignore_index=True)
                    
                    # Store only cell data
                    processed_features[f'{feature_type}_cells'] = concatenated
                else:
                    print(f"Warning: No cell-specific files found for {feature_type}")
                    processed_features[f'{feature_type}_cells'] = pd.DataFrame()
        
        return processed_features
    
    def _process_quality_metrics(self, qm_files: List[Path]) -> pd.DataFrame:
        """Process quality metrics files and extract dataset-level information."""
        if not qm_files:
            print("Warning: No quality metrics files found")
            return pd.DataFrame()
        
        qm_data = []
        
        for qm_file in qm_files:
            try:
                # Extract dataset ID from file path
                dataset_id = self._extract_dataset_id_from_path(qm_file)
                
                # Load JSON quality metrics
                with open(qm_file) as f:
                    qm = json.load(f)
                
                # Extract relevant metrics (similar to original qm_process)
                frac_img_cells = qm['Image Quality Metrics that require cell segmentation']['Fraction of Image Occupied by Cells']
                frac_pix_bg = qm['Image Quality Metrics requiring background segmentation']['Fraction of Pixels in Image Background']
                seg_qs = qm['Segmentation Evaluation Metrics']['QualityScore']
                
                # Extract signal-to-noise ratios for common channels
                s2n_otsu = qm['Image Quality Metrics not requiring image segmentation']['Signal To Noise Otsu']
                s2n_z = qm['Image Quality Metrics not requiring image segmentation']['Signal To Noise Z-Score']
                
                # Extract total intensity for common channels
                total_intensity = qm['Image Quality Metrics not requiring image segmentation']['Total Intensity']
                
                # Extract cell/background ratios
                avg_cell_ratios = qm['Image Quality Metrics that require cell segmentation']['Channel Statistics']['Average per Cell Ratios']
                
                # Build row data
                row_data = [dataset_id, frac_img_cells, frac_pix_bg, seg_qs]
                
                # Add Otsu SNR for common channels
                for channel in COMMON_CHANNELS:
                    if channel in s2n_otsu:
                        row_data.append(s2n_otsu[channel])
                    else:
                        row_data.append(np.nan)
                
                # Add Z-Score SNR for common channels
                for channel in COMMON_CHANNELS:
                    if channel in s2n_z:
                        row_data.append(s2n_z[channel])
                    else:
                        row_data.append(np.nan)
                
                # Add mean intensity for common channels
                for channel in COMMON_CHANNELS:
                    if channel in total_intensity:
                        row_data.append(total_intensity[channel])
                    else:
                        row_data.append(np.nan)
                
                # Add cell/background ratios for common channels
                for channel in COMMON_CHANNELS:
                    if channel in avg_cell_ratios:
                        row_data.append(avg_cell_ratios[channel]['Cell / Background'])
                    else:
                        row_data.append(np.nan)
                
                qm_data.append(row_data)
                
            except Exception as e:
                print(f"Warning: Could not process quality metrics file {qm_file}: {e}")
                continue
        
        if not qm_data:
            return pd.DataFrame()
        
        # Create DataFrame with same structure as original
        columns = ['dataset_id', 'FracImgOfCells', 'FracPixInImgBG', 'SegQS']
        columns.extend([f'Otsu: {channel}' for channel in COMMON_CHANNELS])
        columns.extend([f'Z-Score: {channel}' for channel in COMMON_CHANNELS])
        columns.extend([f'meanInt: {channel}' for channel in COMMON_CHANNELS])
        columns.extend([f'Cell/BG: {channel}' for channel in COMMON_CHANNELS])
        
        qm_df = pd.DataFrame(qm_data, columns=columns)
        return qm_df
    
    def _extract_dataset_id_from_path(self, file_path: Path) -> str:
        """Extract dataset ID from file path."""
        # Navigate up the path to find the dataset directory
        path_parts = file_path.parts
        
        # Look for the dataset ID in the path
        # Typically it's a directory name that looks like a hash
        for part in path_parts:
            # Dataset IDs are typically long alphanumeric strings
            if len(part) > 20 and all(c.isalnum() for c in part):
                return part
        
        # Fallback: use the parent directory name
        return file_path.parent.name
    
    def _segment_paths_by_region(self, paths: List[Path]) -> List[List[Path]]:
        """Segment paths by region type (cell, nuclei, boundaries, etc.)."""
        regions = ['cell_channel', 'nuclei', 'cell_boundaries', 'nucleus']
        segmented = [[] for _ in regions]
        
        for path in paths:
            for i, region in enumerate(regions):
                if region in path.stem:
                    segmented[i].append(path)
                    break
        
        return segmented
    
    def create_ann_data_object(self, 
                              features: Dict[str, pd.DataFrame],
                              dataset_id: str,
                              tissue_name: str,
                              pixel_count: int,
                              subsampled_indices: np.ndarray) -> ad.AnnData:
        """Create an AnnData object from processed features (cell data only)."""
        
        # Filter features by dataset_id
        dataset_features = {}
        for feature_name, feature_df in features.items():
            if feature_name == 'quality_metrics':
                # Quality metrics are dataset-level, not cell-level
                dataset_features[feature_name] = feature_df
            elif 'dataset_id' in feature_df.columns:
                # Filter to only this dataset
                dataset_data = feature_df[feature_df['dataset_id'] == dataset_id].copy()
                # Remove dataset_id column for the actual data
                if len(dataset_data) > 0:
                    dataset_features[feature_name] = dataset_data.drop(columns=['dataset_id'])
                else:
                    dataset_features[feature_name] = pd.DataFrame()
            else:
                # If no dataset_id column, use all data (fallback)
                dataset_features[feature_name] = feature_df
        
        # Get main feature matrices (only cell data)
        mean_cells = dataset_features.get('mean_cells', pd.DataFrame())
        total_cells = dataset_features.get('total_cells', pd.DataFrame())
        covar_cells = dataset_features.get('covar_cells', pd.DataFrame())
        shape_data = dataset_features.get('shape', pd.DataFrame())
        quality_metrics = dataset_features.get('quality_metrics', pd.DataFrame())
        
        # Check if we have enough data for subsampling
        if len(mean_cells) == 0:
            print(f"Warning: No cell data found for dataset {dataset_id}")
            # Create empty AnnData
            return self._create_empty_ann_data(dataset_id, tissue_name, pixel_count)
        
        # Ensure subsampled indices don't exceed available data
        max_index = len(mean_cells) - 1
        valid_indices = subsampled_indices[subsampled_indices <= max_index]
        
        if len(valid_indices) == 0:
            print(f"Warning: No valid indices for dataset {dataset_id}")
            return self._create_empty_ann_data(dataset_id, tissue_name, pixel_count)
        
        # Subsample data
        subsampled_data = {
            'mean_cells': mean_cells.iloc[valid_indices] if len(mean_cells) > 0 else pd.DataFrame(),
            'total_cells': total_cells.iloc[valid_indices] if len(total_cells) > 0 else pd.DataFrame(),
            'covar_cells': covar_cells.iloc[valid_indices] if len(covar_cells) > 0 else pd.DataFrame(),
            'shape': shape_data.iloc[valid_indices] if len(shape_data) > 0 else pd.DataFrame()
        }
        
        # Create observation metadata (cell-level data only)
        obs = pd.DataFrame({
            'dataset_id': dataset_id,
            'tissue_type': tissue_name,
            'cell_id': valid_indices,
            'total_pixels_in_dataset': pixel_count
        }, index=range(len(valid_indices)))
        
        # Create variable metadata
        channel_names = list(subsampled_data['mean_cells'].columns) if len(subsampled_data['mean_cells']) > 0 else []
        var = pd.DataFrame({
            'channel_name': channel_names,
            'feature_type': ['mean_intensity'] * len(channel_names)
        }, index=channel_names)
        
        # Create placeholder spatial coordinates
        spatial_coords = pd.DataFrame({
            'x': np.random.rand(len(valid_indices)) * 1000,
            'y': np.random.rand(len(valid_indices)) * 1000
        })
        
        # Create empty adjacency matrix
        adj_matrix = sparse.csr_matrix((len(valid_indices), len(valid_indices)))
        
        # Create AnnData object
        adata = ad.AnnData(
            X=subsampled_data['mean_cells'].values if len(subsampled_data['mean_cells']) > 0 else np.zeros((len(valid_indices), 0)),
            obs=obs,
            var=var,
            obsm={'spatial': spatial_coords.values},
            obsp={'adjacency': adj_matrix}
        )
        
        # Add feature layers (only cell data)
        if len(subsampled_data['total_cells']) > 0:
            adata.layers['total_intensity_cell'] = subsampled_data['total_cells'].values
        
        # Add data with different dimensions to obsm (observations × matrices)
        if len(subsampled_data['covar_cells']) > 0:
            adata.obsm['covariance_intensity_cell'] = subsampled_data['covar_cells'].values
        if len(subsampled_data['shape']) > 0:
            adata.obsm['shape_features'] = subsampled_data['shape'].values
        
        # Store dataset-level metadata in uns
        dataset_info = {
            'dataset_id': dataset_id,
            'tissue_type': tissue_name,
            'total_pixels': pixel_count,
            'subsampled_size': len(valid_indices),
            'original_data_size': len(mean_cells),
            'data_type': 'cell_only',
            'common_channels': COMMON_CHANNELS,
            'covariance_channels': COVAR_COMMON_CHANNELS
        }
        
        # Add quality metrics if available
        if len(quality_metrics) > 0:
            dataset_qm = quality_metrics[quality_metrics['dataset_id'] == dataset_id]
            if len(dataset_qm) > 0:
                # Convert quality metrics to dictionary for storage
                qm_dict = dataset_qm.iloc[0].to_dict()
                dataset_info['quality_metrics'] = qm_dict
        
        adata.uns['dataset_info'] = dataset_info
        
        return adata
    
    def _create_empty_ann_data(self, dataset_id: str, tissue_name: str, pixel_count: int) -> ad.AnnData:
        """Create an empty AnnData object when no data is available."""
        obs = pd.DataFrame({
            'dataset_id': dataset_id,
            'tissue_type': tissue_name,
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
            'tissue_type': tissue_name,
            'total_pixels': pixel_count,
            'subsampled_size': 0,
            'original_data_size': 0,
            'data_type': 'cell_only',
            'common_channels': COMMON_CHANNELS,
            'covariance_channels': COVAR_COMMON_CHANNELS,
            'quality_metrics': {}  # Empty quality metrics
        }
        
        return adata
    
    def process_tissue(self, tissue_path: Path, tissue_name: str) -> List[ad.AnnData]:
        """Process a single tissue and create AnnData objects for all datasets."""
        print(f"\nProcessing tissue: {tissue_name}")
        
        # Check if combined dataset already exists
        combined_filename = f"{tissue_name}_combined_adata.h5ad"
        combined_path = self.output_dir / combined_filename
        if combined_path.exists():
            print(f"  Combined dataset already exists: {combined_filename}")
            print(f"  Skipping processing for {tissue_name}")
            return []
        
        # Find feature files
        feature_files = self.find_feature_files(tissue_path)
        
        # Get dataset pixel counts
        pixel_counts = self.get_dataset_pixels(tissue_path, tissue_name)
        
        # Save pixel counts
        # with open(self.output_dir / f"{tissue_name}-pixels.pkl", "wb") as f:
        #     pickle.dump(pixel_counts, f)
        
        # Load and process features (now with dataset information preserved)
        print("Loading and processing features...")
        features = self.load_and_process_features(feature_files)
        
        # Create AnnData objects for each dataset
        adata_list = []
        dataset_ids = list(pixel_counts.keys())
        
        for dataset_id in dataset_ids:
            print(f"  Processing dataset: {dataset_id}")
            
            # Get the actual number of cells for this dataset from the features
            actual_cells = 0
            if 'mean_cells' in features and 'dataset_id' in features['mean_cells'].columns:
                dataset_data = features['mean_cells'][features['mean_cells']['dataset_id'] == dataset_id]
                actual_cells = len(dataset_data)
            
            if actual_cells == 0:
                print(f"    Warning: No data found for dataset {dataset_id}")
                # Create empty AnnData
                adata = self._create_empty_ann_data(dataset_id, tissue_name, pixel_counts[dataset_id])
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
                    features, dataset_id, tissue_name, 
                    pixel_counts[dataset_id], subsampled_indices
                )
            
            # Save individual AnnData
            filename = f"{tissue_name}_{dataset_id}_adata.h5ad"
            adata.write_h5ad(self.output_dir / filename)
            print(f"    Saved: {filename}")
            
            adata_list.append(adata)
        
        # Create combined AnnData for tissue
        if adata_list:
            print(f"  Creating combined AnnData for {tissue_name}")
            combined_adata = ad.concat(adata_list, join='outer', index_unique='_')
            combined_filename = f"{tissue_name}_combined_adata.h5ad"
            combined_adata.write_h5ad(self.output_dir / combined_filename)
            print(f"    Saved: {combined_filename}")
        
        return adata_list
    
    def run(self):
        """Main processing pipeline."""
        print("Starting CODEX data processing...")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each tissue
        all_adata = []
        for tissue_path, tissue_name in self.get_tissue_paths():
            if tissue_path.exists():
                tissue_adata = self.process_tissue(tissue_path, tissue_name)
                all_adata.extend(tissue_adata)
            else:
                print(f"Warning: Tissue path {tissue_path} does not exist")
        
        print(f"\nProcessing complete! Created {len(all_adata)} AnnData objects.")


def main():
    """Main entry point."""
    output_dir = Path.home() / 'workspace' / 'FIGURES' / 'new_int_dataset'
    
    processor = CODEXDataProcessor(output_dir)
    processor.run()


if __name__ == "__main__":
    main() 