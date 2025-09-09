#!/usr/bin/env python3
"""
Kipnis Lab Methods for JR_IBEX_003
Brain immune interactions and microglial territory analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from scipy import spatial, ndimage
from sklearn.cluster import DBSCAN
from skimage import measure, morphology, segmentation
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "scripts"))

from spatial_utils import SpatialAnalyzer

class KipnisAnalyzer:
    """
    Kipnis lab-style analysis for brain immune interactions
    Focus on microglial territories and brain-immune cell interactions
    """
    
    def __init__(self, ibex_data, pixel_size_um=0.325):
        """
        Initialize Kipnis analyzer
        
        Args:
            ibex_data: IBEX cell data DataFrame
            pixel_size_um: Pixel size in micrometers
        """
        self.ibex_data = ibex_data
        self.pixel_size = pixel_size_um
        self.spatial_analyzer = SpatialAnalyzer(pixel_size_um)
        
        # Define Kipnis lab-relevant cell types
        self.immune_markers = ['CD45', 'CD3', 'CD68', 'IBA1']
        self.neuronal_markers = ['NeuN']
        self.glial_markers = ['GFAP', 'IBA1']
        
    def identify_cell_types(self):
        """
        Identify brain cell types based on marker expression
        
        Returns:
            DataFrame: IBEX data with cell type classifications
        """
        cell_data = self.ibex_data.copy()
        
        # Initialize cell type column
        cell_data['kipnis_cell_type'] = 'unknown'
        
        # Define thresholds (would need optimization for specific data)
        threshold_percentile = 75
        
        # Microglia (IBA1+)
        if 'IBA1' in cell_data.columns:
            iba1_threshold = np.percentile(cell_data['IBA1'], threshold_percentile)
            microglia_mask = cell_data['IBA1'] > iba1_threshold
            cell_data.loc[microglia_mask, 'kipnis_cell_type'] = 'microglia'
        
        # Neurons (NeuN+, non-immune)
        if 'NeuN' in cell_data.columns:
            neun_threshold = np.percentile(cell_data['NeuN'], threshold_percentile)
            cd45_low = cell_data.get('CD45', 0) < np.percentile(cell_data.get('CD45', [0]), 25)
            neuron_mask = (cell_data['NeuN'] > neun_threshold) & cd45_low
            cell_data.loc[neuron_mask, 'kipnis_cell_type'] = 'neuron'
        
        # Astrocytes (GFAP+, non-immune)
        if 'GFAP' in cell_data.columns:
            gfap_threshold = np.percentile(cell_data['GFAP'], threshold_percentile)
            cd45_low = cell_data.get('CD45', 0) < np.percentile(cell_data.get('CD45', [0]), 25)
            astrocyte_mask = (cell_data['GFAP'] > gfap_threshold) & cd45_low
            cell_data.loc[astrocyte_mask, 'kipnis_cell_type'] = 'astrocyte'
        
        # T cells (CD3+)
        if 'CD3' in cell_data.columns:
            cd3_threshold = np.percentile(cell_data['CD3'], threshold_percentile)
            t_cell_mask = cell_data['CD3'] > cd3_threshold
            cell_data.loc[t_cell_mask, 'kipnis_cell_type'] = 'T_cell'
        
        # Macrophages (CD68+, not microglia)
        if 'CD68' in cell_data.columns and 'IBA1' in cell_data.columns:
            cd68_threshold = np.percentile(cell_data['CD68'], threshold_percentile)
            iba1_threshold = np.percentile(cell_data['IBA1'], threshold_percentile)
            macrophage_mask = (cell_data['CD68'] > cd68_threshold) & (cell_data['IBA1'] <= iba1_threshold)
            cell_data.loc[macrophage_mask, 'kipnis_cell_type'] = 'macrophage'
        
        # General immune cells (CD45+, not otherwise classified)
        if 'CD45' in cell_data.columns:
            cd45_threshold = np.percentile(cell_data['CD45'], threshold_percentile)
            immune_mask = (cell_data['CD45'] > cd45_threshold) & (cell_data['kipnis_cell_type'] == 'unknown')
            cell_data.loc[immune_mask, 'kipnis_cell_type'] = 'immune_other'
        
        # Count cell types
        type_counts = cell_data['kipnis_cell_type'].value_counts()
        print("🧠 Kipnis cell type identification:")
        for cell_type, count in type_counts.items():
            print(f"   {cell_type}: {count} cells ({count/len(cell_data)*100:.1f}%)")
        
        return cell_data
    
    def analyze_microglial_territories(self, cell_data=None, territory_radius_um=75):
        """
        Analyze microglial territorial organization
        
        Args:
            cell_data: DataFrame with cell type classifications
            territory_radius_um: Radius for territory analysis
            
        Returns:
            DataFrame: Microglial territory analysis
        """
        if cell_data is None:
            cell_data = self.identify_cell_types()
        
        # Get microglia
        microglia = cell_data[cell_data['kipnis_cell_type'] == 'microglia'].copy()
        
        if len(microglia) == 0:
            print("⚠ No microglia identified")
            return pd.DataFrame()
        
        print(f"🔬 Analyzing {len(microglia)} microglial territories")
        
        # Calculate microglial territories using Voronoi diagrams
        microglia_coords = microglia[['X', 'Y']].values * self.pixel_size
        
        # Voronoi territories
        if len(microglia_coords) >= 3:
            vor = spatial.Voronoi(microglia_coords)
            
            territories = []
            
            for i, (idx, microglia_cell) in enumerate(microglia.iterrows()):
                territory_data = {
                    'cell_id': idx,
                    'microglia_id': i,
                    'x_um': microglia_cell['X'] * self.pixel_size,
                    'y_um': microglia_cell['Y'] * self.pixel_size
                }
                
                # Calculate territory area (simplified)
                region_idx = vor.point_region[i]
                if region_idx >= 0 and region_idx < len(vor.regions):
                    vertex_indices = vor.regions[region_idx]
                    
                    if len(vertex_indices) > 0 and -1 not in vertex_indices:
                        vertices = vor.vertices[vertex_indices]
                        
                        # Calculate convex hull area
                        try:
                            from scipy.spatial import ConvexHull
                            hull = ConvexHull(vertices)
                            territory_area = hull.volume  # 2D area
                        except:
                            territory_area = np.pi * territory_radius_um**2  # Default circular area
                    else:
                        territory_area = np.pi * territory_radius_um**2
                else:
                    territory_area = np.pi * territory_radius_um**2
                
                territory_data['territory_area_um2'] = territory_area
                territory_data['territory_radius_equivalent'] = np.sqrt(territory_area / np.pi)
                
                # Find cells within territory
                cell_coord = np.array([microglia_cell['X'], microglia_cell['Y']]) * self.pixel_size
                all_coords = cell_data[['X', 'Y']].values * self.pixel_size
                
                distances = np.linalg.norm(all_coords - cell_coord, axis=1)
                territory_cells = cell_data[distances <= territory_radius_um]
                
                # Count different cell types in territory
                territory_data['total_cells_in_territory'] = len(territory_cells)
                
                if 'kipnis_cell_type' in territory_cells.columns:
                    type_counts = territory_cells['kipnis_cell_type'].value_counts()
                    for cell_type in ['neuron', 'astrocyte', 'T_cell', 'macrophage', 'immune_other']:
                        territory_data[f'{cell_type}_count'] = type_counts.get(cell_type, 0)
                        territory_data[f'{cell_type}_density'] = type_counts.get(cell_type, 0) / territory_area
                
                # Calculate microglial morphology features (if intensity data available)
                if 'IBA1' in microglia_cell:
                    territory_data['iba1_intensity'] = microglia_cell['IBA1']
                
                territories.append(territory_data)
            
            return pd.DataFrame(territories)
        
        else:
            print("⚠ Insufficient microglia for territory analysis")
            return pd.DataFrame()
    
    def brain_immune_interactions(self, cell_data=None, interaction_radius_um=30):
        """
        Analyze interactions between brain cells and immune cells
        
        Args:
            cell_data: DataFrame with cell type classifications
            interaction_radius_um: Radius for interaction detection
            
        Returns:
            DataFrame: Brain-immune interaction analysis
        """
        if cell_data is None:
            cell_data = self.identify_cell_types()
        
        # Load data for spatial analysis
        self.spatial_analyzer.load_cell_data(
            cell_data, coord_columns=['X', 'Y']
        )
        
        # Perform interaction analysis
        interactions = self.spatial_analyzer.cell_interaction_analysis(
            distance_threshold_um=interaction_radius_um,
            cell_type_column='kipnis_cell_type'
        )
        
        if interactions.empty:
            print("⚠ No interactions found")
            return pd.DataFrame()
        
        # Focus on brain-immune interactions
        brain_types = ['neuron', 'astrocyte', 'microglia']
        immune_types = ['T_cell', 'macrophage', 'immune_other', 'microglia']
        
        brain_immune_interactions = interactions[
            ((interactions['cell1_type'].isin(brain_types)) & (interactions['cell2_type'].isin(immune_types))) |
            ((interactions['cell1_type'].isin(immune_types)) & (interactions['cell2_type'].isin(brain_types)))
        ].copy()
        
        # Classify interaction types
        def classify_interaction(row):
            types = sorted([row['cell1_type'], row['cell2_type']])
            return f"{types[0]}-{types[1]}"
        
        brain_immune_interactions['interaction_class'] = brain_immune_interactions.apply(classify_interaction, axis=1)
        
        print(f"🧠 Found {len(brain_immune_interactions)} brain-immune interactions")
        
        # Summary statistics
        interaction_summary = brain_immune_interactions.groupby('interaction_class').agg({
            'distance_um': ['count', 'mean', 'std'],
            'cell1_id': 'count'
        }).round(3)
        
        interaction_summary.columns = ['count', 'mean_distance', 'std_distance', 'total']
        interaction_summary = interaction_summary.reset_index()
        
        return brain_immune_interactions, interaction_summary
    
    def microglial_activation_analysis(self, cell_data=None):
        """
        Analyze microglial activation states based on morphology and markers
        
        Args:
            cell_data: DataFrame with cell classifications
            
        Returns:
            DataFrame: Microglial activation analysis
        """
        if cell_data is None:
            cell_data = self.identify_cell_types()
        
        microglia = cell_data[cell_data['kipnis_cell_type'] == 'microglia'].copy()
        
        if len(microglia) == 0:
            print("⚠ No microglia identified")
            return pd.DataFrame()
        
        # Calculate activation features
        activation_features = []
        
        for idx, cell in microglia.iterrows():
            features = {'cell_id': idx}
            
            # Marker-based activation (if available)
            if 'IBA1' in cell:
                features['iba1_intensity'] = cell['IBA1']
                
            if 'CD68' in cell:
                features['cd68_intensity'] = cell['CD68'] 
                features['activation_score'] = cell.get('CD68', 0) / (cell.get('IBA1', 1) + 1e-6)
            
            # Spatial activation features
            cell_coord = np.array([cell['X'], cell['Y']]) * self.pixel_size
            
            # Find nearby cells within 50um
            all_coords = cell_data[['X', 'Y']].values * self.pixel_size
            distances = np.linalg.norm(all_coords - cell_coord, axis=1)
            nearby_cells = cell_data[distances <= 50]
            
            # Count immune cells nearby (activation context)
            if 'kipnis_cell_type' in nearby_cells.columns:
                immune_nearby = np.sum(nearby_cells['kipnis_cell_type'].isin(['T_cell', 'macrophage', 'immune_other']))
                features['immune_cells_nearby'] = immune_nearby
                features['activation_environment'] = 'high' if immune_nearby >= 3 else 'low'
            
            activation_features.append(features)
        
        activation_df = pd.DataFrame(activation_features)
        
        # Classify activation state
        if 'activation_score' in activation_df.columns:
            score_threshold = np.percentile(activation_df['activation_score'], 75)
            activation_df['activation_state'] = activation_df['activation_score'] > score_threshold
            activation_df['activation_state'] = activation_df['activation_state'].map({True: 'activated', False: 'resting'})
            
            state_counts = activation_df['activation_state'].value_counts()
            print(f"🔥 Microglial activation states:")
            for state, count in state_counts.items():
                print(f"   {state}: {count} cells ({count/len(activation_df)*100:.1f}%)")
        
        return activation_df
    
    def meningeal_border_analysis(self, cell_data=None, border_width_um=200):
        """
        Analyze immune cell distribution at meningeal borders
        
        Args:
            cell_data: DataFrame with cell classifications
            border_width_um: Width of border region to analyze
            
        Returns:
            dict: Border analysis results
        """
        if cell_data is None:
            cell_data = self.identify_cell_types()
        
        # Identify tissue borders (simplified - use image edges)
        coords = cell_data[['X', 'Y']].values * self.pixel_size
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Define border regions
        border_cells = cell_data[
            ((cell_data['X'] * self.pixel_size - x_min) <= border_width_um) |
            ((x_max - cell_data['X'] * self.pixel_size) <= border_width_um) |
            ((cell_data['Y'] * self.pixel_size - y_min) <= border_width_um) |
            ((y_max - cell_data['Y'] * self.pixel_size) <= border_width_um)
        ]
        
        central_cells = cell_data[
            ((cell_data['X'] * self.pixel_size - x_min) > border_width_um) &
            ((x_max - cell_data['X'] * self.pixel_size) > border_width_um) &
            ((cell_data['Y'] * self.pixel_size - y_min) > border_width_um) &
            ((y_max - cell_data['Y'] * self.pixel_size) > border_width_um)
        ]
        
        # Analyze immune cell enrichment at borders
        border_analysis = {}
        
        if 'kipnis_cell_type' in cell_data.columns:
            border_types = border_cells['kipnis_cell_type'].value_counts()
            central_types = central_cells['kipnis_cell_type'].value_counts()
            
            for cell_type in ['microglia', 'T_cell', 'macrophage', 'immune_other']:
                border_count = border_types.get(cell_type, 0)
                central_count = central_types.get(cell_type, 0)
                
                border_density = border_count / len(border_cells) if len(border_cells) > 0 else 0
                central_density = central_count / len(central_cells) if len(central_cells) > 0 else 0
                
                enrichment = border_density / central_density if central_density > 0 else np.inf
                
                border_analysis[cell_type] = {
                    'border_count': border_count,
                    'central_count': central_count,
                    'border_density': border_density,
                    'central_density': central_density,
                    'border_enrichment': enrichment
                }
        
        print(f"🏥 Meningeal border analysis ({len(border_cells)} border cells, {len(central_cells)} central cells)")
        
        return border_analysis
    
    def generate_kipnis_report(self, cell_data=None):
        """
        Generate comprehensive Kipnis lab-style analysis report
        
        Args:
            cell_data: DataFrame with analysis results
            
        Returns:
            dict: Comprehensive analysis report
        """
        if cell_data is None:
            cell_data = self.identify_cell_types()
        
        report = {}
        
        # Cell type summary
        type_counts = cell_data['kipnis_cell_type'].value_counts()
        report['cell_types'] = type_counts.to_dict()
        
        # Microglial territories
        territories = self.analyze_microglial_territories(cell_data)
        if not territories.empty:
            report['microglial_territories'] = {
                'n_territories': len(territories),
                'mean_territory_area': territories['territory_area_um2'].mean(),
                'mean_cells_per_territory': territories['total_cells_in_territory'].mean()
            }
        
        # Brain-immune interactions
        interactions, interaction_summary = self.brain_immune_interactions(cell_data)
        if not interactions.empty:
            report['brain_immune_interactions'] = {
                'total_interactions': len(interactions),
                'interaction_types': interaction_summary.to_dict('records')
            }
        
        # Microglial activation
        activation = self.microglial_activation_analysis(cell_data)
        if not activation.empty and 'activation_state' in activation.columns:
            activation_counts = activation['activation_state'].value_counts()
            report['microglial_activation'] = activation_counts.to_dict()
        
        # Border analysis
        border_analysis = self.meningeal_border_analysis(cell_data)
        report['meningeal_border'] = border_analysis
        
        print("📋 Kipnis Lab Analysis Report Generated")
        return report

def plot_microglial_territories(territories_df, cell_data, pixel_size_um=0.325, save_path=None):
    """
    Plot microglial territorial organization
    
    Args:
        territories_df: Microglial territories DataFrame
        cell_data: All cell data
        pixel_size_um: Pixel size for coordinate conversion
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Territory map with Voronoi cells
    all_coords = cell_data[['X', 'Y']].values * pixel_size_um
    microglia_coords = territories_df[['x_um', 'y_um']].values
    
    # Plot all cells
    axes[0].scatter(all_coords[:, 0], all_coords[:, 1], c='lightgray', s=1, alpha=0.5, label='All cells')
    
    # Plot microglia
    axes[0].scatter(microglia_coords[:, 0], microglia_coords[:, 1], 
                   c='red', s=20, alpha=0.8, label='Microglia', edgecolors='black')
    
    # Draw territory boundaries (simplified circles)
    for _, territory in territories_df.iterrows():
        circle = plt.Circle((territory['x_um'], territory['y_um']), 
                          territory['territory_radius_equivalent'], 
                          fill=False, color='red', alpha=0.3, linestyle='--')
        axes[0].add_patch(circle)
    
    axes[0].set_xlabel('X (μm)')
    axes[0].set_ylabel('Y (μm)')
    axes[0].set_title('Microglial Territories')
    axes[0].legend()
    axes[0].axis('equal')
    
    # Plot 2: Territory size distribution
    if 'territory_area_um2' in territories_df.columns:
        axes[1].hist(territories_df['territory_area_um2'], bins=20, alpha=0.7, color='red')
        axes[1].axvline(territories_df['territory_area_um2'].mean(), color='black', 
                       linestyle='--', label=f"Mean: {territories_df['territory_area_um2'].mean():.0f} μm²")
        axes[1].set_xlabel('Territory Area (μm²)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Territory Size Distribution')
        axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig

def plot_brain_immune_interactions(interactions_df, interaction_summary, save_path=None):
    """
    Plot brain-immune interaction analysis
    
    Args:
        interactions_df: Interactions DataFrame
        interaction_summary: Summary statistics DataFrame
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Interaction count by type
    if not interaction_summary.empty:
        axes[0].bar(interaction_summary['interaction_class'], interaction_summary['count'])
        axes[0].set_xlabel('Interaction Type')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Brain-Immune Interactions')
        axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Distance distribution
    if not interactions_df.empty:
        axes[1].hist(interactions_df['distance_um'], bins=20, alpha=0.7, color='blue')
        axes[1].axvline(interactions_df['distance_um'].mean(), color='red', 
                       linestyle='--', label=f"Mean: {interactions_df['distance_um'].mean():.1f} μm")
        axes[1].set_xlabel('Interaction Distance (μm)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Interaction Distance Distribution')
        axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig

if __name__ == "__main__":
    print("🧠 Kipnis Lab Methods")
    print("Available classes: KipnisAnalyzer")
    print("Available functions: plot_microglial_territories, plot_brain_immune_interactions")