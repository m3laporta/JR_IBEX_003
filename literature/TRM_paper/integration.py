#!/usr/bin/env python3
"""
TRM Paper Methods Integration for JR_IBEX_003
Tissue Resident Memory T cell analysis methods adapted for IBEX data format
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "scripts"))

from spatial_utils import SpatialAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, spatial
from sklearn.cluster import DBSCAN

class TRMAnalyzer:
    """
    Tissue Resident Memory T cell analysis following TRM paper methods
    Adapted for IBEX immunofluorescence data
    """
    
    def __init__(self, ibex_data, pixel_size_um=0.325):
        """
        Initialize TRM analyzer
        
        Args:
            ibex_data: IBEX cell data DataFrame
            pixel_size_um: Pixel size in micrometers
        """
        self.ibex_data = ibex_data
        self.pixel_size = pixel_size_um
        self.spatial_analyzer = SpatialAnalyzer(pixel_size_um)
        
        # TRM-specific parameters
        self.trm_markers = ['CD8', 'CD103', 'CD69']  # Classic TRM markers
        self.memory_markers = ['CD45RO', 'CD127']     # Memory markers (if available)
        
    def identify_trm_cells(self, cd8_threshold=0.5, cd103_threshold=0.3, 
                          cd69_threshold=0.3, require_all_markers=True):
        """
        Identify tissue resident memory T cells based on marker expression
        
        Args:
            cd8_threshold: Minimum CD8 expression (normalized)
            cd103_threshold: Minimum CD103 expression (normalized) 
            cd69_threshold: Minimum CD69 expression (normalized)
            require_all_markers: Whether all markers must be positive
            
        Returns:
            DataFrame: Cells with TRM classification
        """
        trm_data = self.ibex_data.copy()
        
        # Normalize marker intensities (0-1 scale)
        for marker in self.trm_markers:
            if marker in trm_data.columns:
                marker_col = f"{marker}_intensity"
                if marker_col in trm_data.columns:
                    intensity_col = marker_col
                else:
                    intensity_col = marker
                    
                # Min-max normalization
                min_val = trm_data[intensity_col].min()
                max_val = trm_data[intensity_col].max()
                trm_data[f"{marker}_normalized"] = (
                    (trm_data[intensity_col] - min_val) / (max_val - min_val)
                )
        
        # Define TRM criteria
        cd8_positive = trm_data.get('CD8_normalized', 0) >= cd8_threshold
        cd103_positive = trm_data.get('CD103_normalized', 0) >= cd103_threshold  
        cd69_positive = trm_data.get('CD69_normalized', 0) >= cd69_threshold
        
        if require_all_markers:
            trm_mask = cd8_positive & cd103_positive & cd69_positive
        else:
            # At least CD8+ and one other marker
            trm_mask = cd8_positive & (cd103_positive | cd69_positive)
            
        trm_data['is_TRM'] = trm_mask
        trm_data['TRM_score'] = (
            trm_data.get('CD8_normalized', 0) * 0.4 +
            trm_data.get('CD103_normalized', 0) * 0.3 +
            trm_data.get('CD69_normalized', 0) * 0.3
        )
        
        n_trm = np.sum(trm_mask)
        print(f"🎯 Identified {n_trm} TRM cells ({n_trm/len(trm_data)*100:.1f}%)")
        
        return trm_data
    
    def trm_spatial_neighborhoods(self, trm_data, radius_um=50):
        """
        Analyze TRM spatial neighborhoods following paper methods
        
        Args:
            trm_data: DataFrame with TRM classifications
            radius_um: Neighborhood radius
            
        Returns:
            DataFrame: Neighborhood analysis results
        """
        # Load data into spatial analyzer
        self.spatial_analyzer.load_cell_data(
            trm_data, coord_columns=['X', 'Y'], id_column='CellID'
        )
        
        # Standard neighborhood analysis
        neighborhoods = self.spatial_analyzer.spatial_neighborhoods(
            radius_um=radius_um, cell_type_column='is_TRM'
        )
        
        # TRM-specific neighborhood features
        trm_neighborhoods = []
        
        coordinates = trm_data[['X', 'Y']].values * self.pixel_size
        tree = spatial.cKDTree(coordinates)
        
        for i, cell in trm_data.iterrows():
            cell_coord = np.array([cell['X'], cell['Y']]) * self.pixel_size
            
            # Find neighbors within radius
            neighbor_indices = tree.query_ball_point(cell_coord, radius_um)
            neighbor_indices.remove(i) if i in neighbor_indices else None
            
            if neighbor_indices:
                neighbors = trm_data.iloc[neighbor_indices]
                
                neighborhood_features = {
                    'cell_id': i,
                    'is_TRM': cell['is_TRM'],
                    'n_total_neighbors': len(neighbors),
                    'n_TRM_neighbors': np.sum(neighbors['is_TRM']),
                    'TRM_neighborhood_fraction': np.mean(neighbors['is_TRM']),
                    'neighborhood_TRM_density': np.sum(neighbors['is_TRM']) / (np.pi * radius_um**2),
                    'mean_neighbor_TRM_score': neighbors['TRM_score'].mean() if 'TRM_score' in neighbors.columns else 0
                }
                
                # Calculate TRM clustering coefficient
                if len(neighbors) > 1:
                    trm_neighbors = neighbors[neighbors['is_TRM']]
                    if len(trm_neighbors) >= 3:
                        neighborhood_features['TRM_clustering'] = self._calculate_local_clustering(
                            trm_neighbors[['X', 'Y']].values * self.pixel_size, radius_um
                        )
                    else:
                        neighborhood_features['TRM_clustering'] = 0
                else:
                    neighborhood_features['TRM_clustering'] = 0
                    
            else:
                neighborhood_features = {
                    'cell_id': i,
                    'is_TRM': cell['is_TRM'],
                    'n_total_neighbors': 0,
                    'n_TRM_neighbors': 0,
                    'TRM_neighborhood_fraction': 0,
                    'neighborhood_TRM_density': 0,
                    'mean_neighbor_TRM_score': 0,
                    'TRM_clustering': 0
                }
            
            trm_neighborhoods.append(neighborhood_features)
            
        return pd.DataFrame(trm_neighborhoods)
    
    def _calculate_local_clustering(self, coordinates, radius_um):
        """Calculate local clustering coefficient for TRM cells"""
        if len(coordinates) < 3:
            return 0
            
        # Build distance matrix
        distances = spatial.distance_matrix(coordinates, coordinates)
        
        # Count triangles (3-way connections within radius)
        n_cells = len(coordinates)
        triangles = 0
        possible_triangles = 0
        
        for i in range(n_cells):
            neighbors_i = np.where(distances[i] <= radius_um)[0]
            neighbors_i = neighbors_i[neighbors_i != i]
            
            if len(neighbors_i) >= 2:
                for j in range(len(neighbors_i)):
                    for k in range(j + 1, len(neighbors_i)):
                        possible_triangles += 1
                        if distances[neighbors_i[j], neighbors_i[k]] <= radius_um:
                            triangles += 1
                            
        return triangles / possible_triangles if possible_triangles > 0 else 0
    
    def trm_tissue_distribution(self, trm_data, tissue_regions=None):
        """
        Analyze TRM distribution across tissue regions
        
        Args:
            trm_data: DataFrame with TRM classifications
            tissue_regions: Dict mapping region names to coordinate bounds
            
        Returns:
            DataFrame: TRM distribution by tissue region
        """
        if tissue_regions is None:
            # Default: divide tissue into quadrants
            x_coords = trm_data['X'] * self.pixel_size
            y_coords = trm_data['Y'] * self.pixel_size
            
            x_mid = (x_coords.max() + x_coords.min()) / 2
            y_mid = (y_coords.max() + y_coords.min()) / 2
            
            tissue_regions = {
                'upper_left': {'x_min': x_coords.min(), 'x_max': x_mid,
                              'y_min': y_mid, 'y_max': y_coords.max()},
                'upper_right': {'x_min': x_mid, 'x_max': x_coords.max(),
                               'y_min': y_mid, 'y_max': y_coords.max()},
                'lower_left': {'x_min': x_coords.min(), 'x_max': x_mid,
                              'y_min': y_coords.min(), 'y_max': y_mid},
                'lower_right': {'x_min': x_mid, 'x_max': x_coords.max(),
                               'y_min': y_coords.min(), 'y_max': y_mid}
            }
        
        region_analysis = []
        
        for region_name, bounds in tissue_regions.items():
            # Select cells in region
            x_coords = trm_data['X'] * self.pixel_size
            y_coords = trm_data['Y'] * self.pixel_size
            
            in_region = (
                (x_coords >= bounds['x_min']) & (x_coords <= bounds['x_max']) &
                (y_coords >= bounds['y_min']) & (y_coords <= bounds['y_max'])
            )
            
            region_cells = trm_data[in_region]
            
            if len(region_cells) > 0:
                region_stats = {
                    'region': region_name,
                    'total_cells': len(region_cells),
                    'trm_cells': np.sum(region_cells['is_TRM']),
                    'trm_fraction': np.mean(region_cells['is_TRM']),
                    'trm_density': np.sum(region_cells['is_TRM']) / (
                        (bounds['x_max'] - bounds['x_min']) * 
                        (bounds['y_max'] - bounds['y_min'])
                    ),
                    'mean_trm_score': region_cells[region_cells['is_TRM']]['TRM_score'].mean() if np.any(region_cells['is_TRM']) else 0
                }
            else:
                region_stats = {
                    'region': region_name,
                    'total_cells': 0,
                    'trm_cells': 0,
                    'trm_fraction': 0,
                    'trm_density': 0,
                    'mean_trm_score': 0
                }
                
            region_analysis.append(region_stats)
            
        return pd.DataFrame(region_analysis)
    
    def trm_interaction_analysis(self, trm_data, interaction_radius_um=20):
        """
        Analyze TRM interactions with other cell types
        
        Args:
            trm_data: DataFrame with TRM classifications
            interaction_radius_um: Interaction distance threshold
            
        Returns:
            DataFrame: Interaction analysis results
        """
        # Use spatial analyzer for interaction analysis
        self.spatial_analyzer.load_cell_data(
            trm_data, coord_columns=['X', 'Y']
        )
        
        interactions = self.spatial_analyzer.cell_interaction_analysis(
            distance_threshold_um=interaction_radius_um,
            cell_type_column='is_TRM'
        )
        
        # Add TRM-specific interaction classifications
        if not interactions.empty:
            interactions['involves_TRM'] = (
                (interactions['cell1_type'] == True) | 
                (interactions['cell2_type'] == True)
            )
            
            interactions['TRM_TRM_interaction'] = (
                (interactions['cell1_type'] == True) & 
                (interactions['cell2_type'] == True)
            )
            
        return interactions
    
    def generate_trm_report(self, trm_data):
        """
        Generate comprehensive TRM analysis report
        
        Args:
            trm_data: DataFrame with TRM analysis results
            
        Returns:
            dict: Comprehensive analysis report
        """
        report = {
            'summary': {
                'total_cells': len(trm_data),
                'trm_cells': np.sum(trm_data['is_TRM']),
                'trm_percentage': np.mean(trm_data['is_TRM']) * 100,
                'mean_trm_score': trm_data[trm_data['is_TRM']]['TRM_score'].mean() if np.any(trm_data['is_TRM']) else 0
            }
        }
        
        # Spatial analysis
        neighborhoods = self.trm_spatial_neighborhoods(trm_data)
        report['spatial'] = {
            'mean_trm_clustering': neighborhoods['TRM_clustering'].mean(),
            'mean_trm_neighborhood_density': neighborhoods['neighborhood_TRM_density'].mean(),
            'trm_spatial_enrichment': neighborhoods[neighborhoods['is_TRM']]['TRM_neighborhood_fraction'].mean()
        }
        
        # Tissue distribution
        tissue_dist = self.trm_tissue_distribution(trm_data)
        report['tissue_distribution'] = tissue_dist.to_dict('records')
        
        print("📋 TRM Analysis Report Generated")
        return report

def plot_trm_spatial_distribution(trm_data, pixel_size_um=0.325, 
                                 save_path=None, figsize=(12, 8)):
    """
    Plot TRM spatial distribution
    
    Args:
        trm_data: DataFrame with TRM classifications
        pixel_size_um: Pixel size for coordinate conversion
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Convert coordinates to microns
    x_coords = trm_data['X'] * pixel_size_um
    y_coords = trm_data['Y'] * pixel_size_um
    
    # Plot 1: All cells with TRM highlighted
    axes[0].scatter(x_coords[~trm_data['is_TRM']], y_coords[~trm_data['is_TRM']], 
                   c='lightgray', s=1, alpha=0.5, label='Other cells')
    axes[0].scatter(x_coords[trm_data['is_TRM']], y_coords[trm_data['is_TRM']], 
                   c='red', s=3, alpha=0.8, label='TRM cells')
    axes[0].set_xlabel('X (μm)')
    axes[0].set_ylabel('Y (μm)')
    axes[0].set_title('TRM Cell Distribution')
    axes[0].legend()
    axes[0].axis('equal')
    
    # Plot 2: TRM score heatmap
    if 'TRM_score' in trm_data.columns:
        scatter = axes[1].scatter(x_coords, y_coords, 
                                 c=trm_data['TRM_score'], 
                                 s=2, alpha=0.7, cmap='Reds')
        axes[1].set_xlabel('X (μm)')
        axes[1].set_ylabel('Y (μm)')
        axes[1].set_title('TRM Score Distribution')
        axes[1].axis('equal')
        plt.colorbar(scatter, ax=axes[1], label='TRM Score')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig

# Integration functions for IBEX pipeline
def adapt_ibex_for_trm_analysis(ibex_cycles, channel_mapping):
    """
    Adapt IBEX multi-cycle data for TRM analysis
    
    Args:
        ibex_cycles: Dict of cycle data
        channel_mapping: Mapping of channels to markers
        
    Returns:
        DataFrame: Processed data ready for TRM analysis
    """
    # Placeholder for IBEX adaptation logic
    # This would integrate with existing IBEX processing pipeline
    print("🔄 Adapting IBEX data for TRM analysis...")
    print("   Integrating with existing alignment and segmentation...")
    
    # Return template structure
    return pd.DataFrame({
        'CellID': range(100),  # Placeholder
        'X': np.random.randn(100) * 100,
        'Y': np.random.randn(100) * 100,
        'CD8_intensity': np.random.rand(100),
        'CD103_intensity': np.random.rand(100),
        'CD69_intensity': np.random.rand(100)
    })

if __name__ == "__main__":
    print("🧬 TRM Paper Methods Integration")
    print("Available classes: TRMAnalyzer")
    print("Available functions: plot_trm_spatial_distribution, adapt_ibex_for_trm_analysis")