#!/usr/bin/env python3
"""
JR_IBEX_003 Spatial Analysis Utilities
Comprehensive spatial analysis functions for immunofluorescence data
"""

import numpy as np
import pandas as pd
from scipy import spatial, stats
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

class SpatialAnalyzer:
    """Main spatial analysis class for IBEX data"""
    
    def __init__(self, pixel_size_um=0.325):
        """
        Initialize spatial analyzer
        
        Args:
            pixel_size_um: Pixel size in micrometers (default IBEX value)
        """
        self.pixel_size = pixel_size_um
        self.cell_data = None
        self.coordinates = None
        
    def load_cell_data(self, cell_data, coord_columns=['X', 'Y'], id_column='CellID'):
        """
        Load cell data for spatial analysis
        
        Args:
            cell_data: DataFrame with cell measurements and coordinates
            coord_columns: Column names for X, Y coordinates
            id_column: Column name for cell IDs
        """
        self.cell_data = cell_data.copy()
        
        # Extract coordinates and convert to microns
        if isinstance(coord_columns, list) and len(coord_columns) == 2:
            x_col, y_col = coord_columns
            self.coordinates = self.cell_data[[x_col, y_col]].values * self.pixel_size
        else:
            raise ValueError("coord_columns must be list of 2 column names [X, Y]")
            
        print(f"📊 Loaded {len(self.cell_data)} cells for spatial analysis")
        
    def calculate_nearest_neighbors(self, n_neighbors=6):
        """
        Calculate nearest neighbor distances and indices
        
        Args:
            n_neighbors: Number of nearest neighbors to calculate
            
        Returns:
            dict: Contains distances and indices arrays
        """
        if self.coordinates is None:
            raise ValueError("Must load cell data first")
            
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 to exclude self
        nn.fit(self.coordinates)
        distances, indices = nn.kneighbors(self.coordinates)
        
        # Remove self (first neighbor)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        return {
            'distances': distances,
            'indices': indices,
            'mean_distance': np.mean(distances[:, 0]),  # Nearest neighbor
            'median_distance': np.median(distances[:, 0])
        }
        
    def spatial_neighborhoods(self, radius_um=50, cell_type_column=None):
        """
        Analyze spatial neighborhoods around each cell
        
        Args:
            radius_um: Neighborhood radius in micrometers
            cell_type_column: Column name for cell type classification
            
        Returns:
            DataFrame: Neighborhood analysis results
        """
        if self.coordinates is None:
            raise ValueError("Must load cell data first")
            
        # Find neighbors within radius
        tree = spatial.cKDTree(self.coordinates)
        neighbor_lists = tree.query_ball_tree(tree, radius_um)
        
        results = []
        
        for i, neighbors in enumerate(neighbor_lists):
            # Remove self from neighbors
            neighbors = [n for n in neighbors if n != i]
            
            neighborhood_data = {
                'cell_id': i,
                'n_neighbors': len(neighbors),
                'neighborhood_density': len(neighbors) / (np.pi * radius_um**2)
            }
            
            if cell_type_column and cell_type_column in self.cell_data.columns:
                # Count neighbors by cell type
                neighbor_types = self.cell_data.iloc[neighbors][cell_type_column].value_counts()
                for cell_type, count in neighbor_types.items():
                    neighborhood_data[f'{cell_type}_neighbors'] = count
                    
            results.append(neighborhood_data)
            
        return pd.DataFrame(results)
        
    def ripley_k_function(self, radii_um=None, cell_type_column=None, 
                         boundary_correction=True):
        """
        Calculate Ripley's K function for spatial clustering analysis
        
        Args:
            radii_um: Array of radii to test (default: 10-200 um)
            cell_type_column: Column for cell type-specific analysis
            boundary_correction: Apply boundary correction
            
        Returns:
            DataFrame: K function values for each radius
        """
        if radii_um is None:
            radii_um = np.arange(10, 201, 10)
            
        if self.coordinates is None:
            raise ValueError("Must load cell data first")
            
        # Image boundaries for correction
        x_min, y_min = np.min(self.coordinates, axis=0)
        x_max, y_max = np.max(self.coordinates, axis=0)
        area = (x_max - x_min) * (y_max - y_min)
        
        n_cells = len(self.coordinates)
        density = n_cells / area
        
        tree = spatial.cKDTree(self.coordinates)
        
        k_values = []
        
        for r in radii_um:
            # Count neighbors within radius r for each cell
            neighbor_counts = []
            
            for i, coord in enumerate(self.coordinates):
                neighbors = tree.query_ball_point(coord, r)
                # Remove self
                neighbors = [n for n in neighbors if n != i]
                
                count = len(neighbors)
                
                # Boundary correction (simplified edge correction)
                if boundary_correction:
                    x, y = coord
                    edge_factor = 1.0
                    
                    # Reduce weight for cells near edges
                    if x - r < x_min or x + r > x_max or y - r < y_min or y + r > y_max:
                        # Simplified: reduce by fraction of circle outside boundary
                        edge_factor = 0.8  # Conservative correction
                    
                    count = count / edge_factor
                    
                neighbor_counts.append(count)
            
            # Calculate K(r)
            k_r = np.sum(neighbor_counts) / (n_cells * density)
            k_values.append(k_r)
            
        # Expected K under random distribution
        k_expected = np.pi * radii_um**2
        
        # L function (normalized K)
        l_values = np.sqrt(np.array(k_values) / np.pi)
        l_expected = radii_um
        
        return pd.DataFrame({
            'radius_um': radii_um,
            'K_observed': k_values,
            'K_expected': k_expected,
            'L_observed': l_values,
            'L_expected': l_expected,
            'L_deviation': l_values - l_expected
        })
        
    def cell_interaction_analysis(self, distance_threshold_um=20, 
                                cell_type_column=None):
        """
        Analyze cell-cell interactions based on proximity
        
        Args:
            distance_threshold_um: Maximum distance for interaction
            cell_type_column: Column for cell type analysis
            
        Returns:
            DataFrame: Interaction analysis results
        """
        if self.coordinates is None:
            raise ValueError("Must load cell data first")
            
        # Build distance matrix
        distances = spatial.distance_matrix(self.coordinates, self.coordinates)
        
        # Find interactions (excluding self-interactions)
        interactions = []
        
        for i in range(len(self.coordinates)):
            for j in range(i + 1, len(self.coordinates)):
                if distances[i, j] <= distance_threshold_um:
                    interaction = {
                        'cell1_id': i,
                        'cell2_id': j,
                        'distance_um': distances[i, j]
                    }
                    
                    if cell_type_column and cell_type_column in self.cell_data.columns:
                        interaction['cell1_type'] = self.cell_data.iloc[i][cell_type_column]
                        interaction['cell2_type'] = self.cell_data.iloc[j][cell_type_column]
                        interaction['interaction_type'] = f"{interaction['cell1_type']}-{interaction['cell2_type']}"
                        
                    interactions.append(interaction)
                    
        return pd.DataFrame(interactions)
        
    def spatial_clustering(self, method='dbscan', eps_um=25, min_samples=5, 
                          cell_type_column=None):
        """
        Perform spatial clustering analysis
        
        Args:
            method: Clustering method ('dbscan')
            eps_um: Clustering radius in micrometers
            min_samples: Minimum samples for DBSCAN
            cell_type_column: Column for cell type-specific clustering
            
        Returns:
            DataFrame: Clustering results
        """
        if self.coordinates is None:
            raise ValueError("Must load cell data first")
            
        results = self.cell_data.copy()
        
        if method == 'dbscan':
            # Overall clustering
            clustering = DBSCAN(eps=eps_um, min_samples=min_samples)
            cluster_labels = clustering.fit_predict(self.coordinates)
            results['cluster_id'] = cluster_labels
            
            # Cell type-specific clustering
            if cell_type_column and cell_type_column in self.cell_data.columns:
                for cell_type in self.cell_data[cell_type_column].unique():
                    mask = self.cell_data[cell_type_column] == cell_type
                    if np.sum(mask) >= min_samples:
                        type_coords = self.coordinates[mask]
                        type_clustering = DBSCAN(eps=eps_um, min_samples=min_samples)
                        type_labels = type_clustering.fit_predict(type_coords)
                        results.loc[mask, f'{cell_type}_cluster'] = type_labels
                        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"Found {n_clusters} spatial clusters")
        
        return results
        
    def territorial_analysis(self, territory_radius_um=75, cell_type_column=None):
        """
        Analyze cellular territories (e.g., microglial territories)
        
        Args:
            territory_radius_um: Territory radius in micrometers
            cell_type_column: Column for cell type analysis
            
        Returns:
            dict: Territory analysis results
        """
        if self.coordinates is None:
            raise ValueError("Must load cell data first")
            
        # Calculate Voronoi diagram for territories
        vor = spatial.Voronoi(self.coordinates)
        
        territories = []
        
        for i, point in enumerate(self.coordinates):
            # Find Voronoi cell vertices
            region_idx = vor.point_region[i]
            vertex_indices = vor.regions[region_idx]
            
            if len(vertex_indices) > 0 and -1 not in vertex_indices:
                vertices = vor.vertices[vertex_indices]
                
                # Calculate territory area (simplified as convex hull)
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(vertices)
                    territory_area = hull.volume  # 2D area
                except:
                    territory_area = np.nan
            else:
                territory_area = np.nan
                
            territory_data = {
                'cell_id': i,
                'territory_area_um2': territory_area
            }
            
            if cell_type_column and cell_type_column in self.cell_data.columns:
                territory_data['cell_type'] = self.cell_data.iloc[i][cell_type_column]
                
            territories.append(territory_data)
            
        return pd.DataFrame(territories)
        
    def multi_scale_analysis(self, scale_levels_um=[10, 25, 50, 100, 200], 
                           cell_type_column=None):
        """
        Perform multi-scale spatial analysis
        
        Args:
            scale_levels_um: List of spatial scales to analyze
            cell_type_column: Column for cell type analysis
            
        Returns:
            dict: Multi-scale analysis results
        """
        results = {}
        
        for scale in scale_levels_um:
            print(f"Analyzing scale: {scale} μm")
            
            # Neighborhood analysis at this scale
            neighborhood_results = self.spatial_neighborhoods(
                radius_um=scale, 
                cell_type_column=cell_type_column
            )
            
            results[f'scale_{scale}um'] = {
                'neighborhoods': neighborhood_results,
                'mean_density': neighborhood_results['neighborhood_density'].mean(),
                'std_density': neighborhood_results['neighborhood_density'].std()
            }
            
        return results

def plot_spatial_distribution(coordinates, cell_types=None, title="Spatial Distribution"):
    """
    Plot spatial distribution of cells
    
    Args:
        coordinates: Array of (x, y) coordinates
        cell_types: Array of cell type labels
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    if cell_types is not None:
        unique_types = np.unique(cell_types)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_types)))
        
        for i, cell_type in enumerate(unique_types):
            mask = cell_types == cell_type
            plt.scatter(coordinates[mask, 0], coordinates[mask, 1], 
                       c=[colors[i]], label=cell_type, s=1, alpha=0.7)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(coordinates[:, 0], coordinates[:, 1], s=1, alpha=0.7)
    
    plt.xlabel('X (μm)')
    plt.ylabel('Y (μm)')
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    
def plot_ripley_k(ripley_results, title="Ripley's K Function"):
    """
    Plot Ripley's K function results
    
    Args:
        ripley_results: DataFrame from ripley_k_function
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # K function
    ax1.plot(ripley_results['radius_um'], ripley_results['K_observed'], 
             label='Observed', linewidth=2)
    ax1.plot(ripley_results['radius_um'], ripley_results['K_expected'], 
             label='Expected (Random)', linestyle='--', linewidth=2)
    ax1.set_xlabel('Radius (μm)')
    ax1.set_ylabel('K(r)')
    ax1.set_title('Ripley\'s K Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # L function deviation
    ax2.plot(ripley_results['radius_um'], ripley_results['L_deviation'], 
             linewidth=2, color='red')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Radius (μm)')
    ax2.set_ylabel('L(r) - r')
    ax2.set_title('L Function Deviation')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()

# Convenience functions for common analyses
def quick_spatial_analysis(cell_data, coord_columns=['X', 'Y'], 
                          cell_type_column=None, pixel_size_um=0.325):
    """
    Perform quick spatial analysis with standard parameters
    
    Args:
        cell_data: DataFrame with cell data
        coord_columns: Column names for coordinates
        cell_type_column: Column name for cell types
        pixel_size_um: Pixel size in micrometers
        
    Returns:
        dict: Analysis results
    """
    analyzer = SpatialAnalyzer(pixel_size_um=pixel_size_um)
    analyzer.load_cell_data(cell_data, coord_columns=coord_columns)
    
    results = {}
    
    # Basic neighborhood analysis
    results['neighborhoods'] = analyzer.spatial_neighborhoods(
        radius_um=50, cell_type_column=cell_type_column
    )
    
    # Nearest neighbors
    results['nearest_neighbors'] = analyzer.calculate_nearest_neighbors()
    
    # Spatial clustering
    results['clustering'] = analyzer.spatial_clustering(
        cell_type_column=cell_type_column
    )
    
    # Ripley's K
    results['ripley_k'] = analyzer.ripley_k_function(
        cell_type_column=cell_type_column
    )
    
    print("✅ Quick spatial analysis complete")
    return results

if __name__ == "__main__":
    print("🔬 JR_IBEX_003 Spatial Analysis Utilities")
    print("Available classes: SpatialAnalyzer")
    print("Available functions: plot_spatial_distribution, plot_ripley_k, quick_spatial_analysis")