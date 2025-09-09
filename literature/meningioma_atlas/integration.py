#!/usr/bin/env python3
"""
Meningioma Atlas Integration for JR_IBEX_003
Tissue segmentation and spatial profiling methods adapted for IBEX data format
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from scipy import ndimage, spatial
from skimage import measure, morphology, segmentation, filters
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "scripts"))

from spatial_utils import SpatialAnalyzer

class MeningiomaAtlasAnalyzer:
    """
    Meningioma atlas-based tissue analysis for IBEX immunofluorescence data
    Implements tissue segmentation and spatial profiling methods
    """
    
    def __init__(self, ibex_data, tissue_image=None, pixel_size_um=0.325):
        """
        Initialize meningioma atlas analyzer
        
        Args:
            ibex_data: IBEX cell data DataFrame
            tissue_image: Optional tissue/DAPI image for segmentation
            pixel_size_um: Pixel size in micrometers
        """
        self.ibex_data = ibex_data
        self.tissue_image = tissue_image
        self.pixel_size = pixel_size_um
        self.spatial_analyzer = SpatialAnalyzer(pixel_size_um)
        
        # Define tissue region types
        self.region_types = ['tumor_core', 'tumor_edge', 'interface', 'normal']
        
    def segment_tissue_regions(self, dapi_image=None, method='intensity_based'):
        """
        Segment tissue into different regions (tumor, interface, normal)
        
        Args:
            dapi_image: DAPI channel image for segmentation
            method: Segmentation method ('intensity_based', 'gradient_based')
            
        Returns:
            numpy.ndarray: Segmentation mask with region labels
        """
        if dapi_image is None and self.tissue_image is None:
            print("⚠ No tissue image provided, using coordinate-based regions")
            return self._coordinate_based_segmentation()
            
        image = dapi_image if dapi_image is not None else self.tissue_image
        
        if method == 'intensity_based':
            return self._intensity_based_segmentation(image)
        elif method == 'gradient_based':
            return self._gradient_based_segmentation(image)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
    
    def _intensity_based_segmentation(self, image):
        """Segment based on intensity thresholds"""
        # Smooth image
        smoothed = filters.gaussian(image, sigma=2)
        
        # Calculate intensity percentiles
        p25, p50, p75, p90 = np.percentile(smoothed[smoothed > 0], [25, 50, 75, 90])
        
        # Create segmentation mask
        segmentation_mask = np.zeros_like(image, dtype=int)
        
        # High intensity = tumor core
        segmentation_mask[smoothed >= p90] = 3
        
        # Medium-high intensity = tumor edge
        segmentation_mask[(smoothed >= p75) & (smoothed < p90)] = 2
        
        # Medium intensity = interface
        segmentation_mask[(smoothed >= p50) & (smoothed < p75)] = 1
        
        # Low intensity = normal tissue
        segmentation_mask[(smoothed > 0) & (smoothed < p50)] = 0
        
        # Apply morphological operations to clean up
        for label in [0, 1, 2, 3]:
            mask = segmentation_mask == label
            mask = morphology.binary_closing(mask, morphology.disk(3))
            mask = morphology.remove_small_objects(mask, min_size=100)
            segmentation_mask[mask] = label
            
        return segmentation_mask
    
    def _gradient_based_segmentation(self, image):
        """Segment based on intensity gradients"""
        # Calculate gradients
        grad_x = filters.sobel_h(image)
        grad_y = filters.sobel_v(image)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Smooth gradient
        gradient_smooth = filters.gaussian(gradient_magnitude, sigma=1)
        
        # Threshold gradients to find boundaries
        grad_thresh = filters.threshold_otsu(gradient_smooth)
        boundaries = gradient_smooth > grad_thresh
        
        # Use watershed to segment regions
        distance = ndimage.distance_transform_edt(~boundaries)
        coords = measure.peak_local_maxima(distance, min_distance=20, threshold_abs=10)
        
        if len(coords) > 0:
            markers = np.zeros_like(image, dtype=int)
            markers[tuple(zip(*coords))] = np.arange(1, len(coords) + 1)
            
            segmentation_mask = segmentation.watershed(-distance, markers, mask=image > 0)
        else:
            # Fallback to simple thresholding
            segmentation_mask = self._intensity_based_segmentation(image)
            
        return segmentation_mask
    
    def _coordinate_based_segmentation(self):
        """Create regions based on cell coordinates when no image is available"""
        x_coords = self.ibex_data['X'] * self.pixel_size
        y_coords = self.ibex_data['Y'] * self.pixel_size
        
        # Create spatial regions based on cell density
        from sklearn.cluster import KMeans
        
        coords = np.column_stack([x_coords, y_coords])
        
        # Use KMeans to define regions
        kmeans = KMeans(n_clusters=4, random_state=42)
        region_labels = kmeans.fit_predict(coords)
        
        # Map region labels to standard names
        region_mapping = {
            0: 'normal',
            1: 'interface', 
            2: 'tumor_edge',
            3: 'tumor_core'
        }
        
        self.ibex_data['region'] = [region_mapping[label] for label in region_labels]
        
        print("📊 Created coordinate-based tissue regions")
        return region_labels
    
    def assign_cells_to_regions(self, segmentation_mask=None):
        """
        Assign cells to tissue regions based on their coordinates
        
        Args:
            segmentation_mask: Tissue segmentation mask
            
        Returns:
            DataFrame: IBEX data with region assignments
        """
        if segmentation_mask is not None:
            # Convert cell coordinates to image coordinates
            cell_coords_img = np.column_stack([
                (self.ibex_data['Y'] / self.pixel_size).astype(int),  # Note: Y->row, X->col
                (self.ibex_data['X'] / self.pixel_size).astype(int)
            ])
            
            # Clip coordinates to image bounds
            cell_coords_img[:, 0] = np.clip(cell_coords_img[:, 0], 0, segmentation_mask.shape[0] - 1)
            cell_coords_img[:, 1] = np.clip(cell_coords_img[:, 1], 0, segmentation_mask.shape[1] - 1)
            
            # Get region labels for each cell
            region_labels = segmentation_mask[cell_coords_img[:, 0], cell_coords_img[:, 1]]
            
            # Map numeric labels to region names
            region_names = ['normal', 'interface', 'tumor_edge', 'tumor_core']
            self.ibex_data['region'] = [region_names[min(label, 3)] for label in region_labels]
            
        else:
            # Use coordinate-based assignment
            self._coordinate_based_segmentation()
            
        # Count cells per region
        region_counts = self.ibex_data['region'].value_counts()
        print("📊 Cell distribution by region:")
        for region, count in region_counts.items():
            print(f"   {region}: {count} cells ({count/len(self.ibex_data)*100:.1f}%)")
            
        return self.ibex_data
    
    def calculate_regional_profiles(self, marker_columns=None):
        """
        Calculate marker expression profiles for each tissue region
        
        Args:
            marker_columns: List of marker column names
            
        Returns:
            DataFrame: Regional expression profiles
        """
        if marker_columns is None:
            # Auto-detect marker columns (look for intensity columns)
            marker_columns = [col for col in self.ibex_data.columns 
                            if 'intensity' in col.lower() or 
                            col in ['CD45', 'CD3', 'CD68', 'CD8', 'CD4', 'CD20', 'IBA1', 'GFAP', 'NeuN']]
        
        if 'region' not in self.ibex_data.columns:
            print("⚠ No region assignments found. Running tissue segmentation...")
            self.assign_cells_to_regions()
        
        regional_profiles = []
        
        for region in self.region_types:
            region_cells = self.ibex_data[self.ibex_data['region'] == region]
            
            if len(region_cells) == 0:
                continue
                
            profile = {'region': region, 'n_cells': len(region_cells)}
            
            # Calculate marker statistics
            for marker in marker_columns:
                if marker in region_cells.columns:
                    values = region_cells[marker]
                    profile.update({
                        f'{marker}_mean': values.mean(),
                        f'{marker}_std': values.std(),
                        f'{marker}_median': values.median(),
                        f'{marker}_positive_fraction': np.mean(values > np.percentile(values, 75))
                    })
            
            # Calculate spatial features
            if len(region_cells) >= 10:  # Need minimum cells for spatial analysis
                coords = region_cells[['X', 'Y']].values * self.pixel_size
                
                # Cell density
                if len(coords) > 1:
                    # Calculate area (convex hull)
                    try:
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(coords)
                        area = hull.volume  # 2D area
                        profile['cell_density'] = len(region_cells) / area
                    except:
                        profile['cell_density'] = np.nan
                
                # Nearest neighbor distances
                if len(coords) > 5:
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=6)
                    nn.fit(coords)
                    distances, _ = nn.kneighbors(coords)
                    profile['mean_nn_distance'] = np.mean(distances[:, 1])  # Exclude self
                    
            regional_profiles.append(profile)
            
        profiles_df = pd.DataFrame(regional_profiles)
        print(f"📋 Generated profiles for {len(profiles_df)} regions")
        
        return profiles_df
    
    def spatial_interaction_analysis(self, interaction_radius_um=50):
        """
        Analyze spatial interactions between regions and cell types
        
        Args:
            interaction_radius_um: Radius for interaction analysis
            
        Returns:
            DataFrame: Interaction analysis results
        """
        if 'region' not in self.ibex_data.columns:
            print("⚠ No region assignments found. Running tissue segmentation...")
            self.assign_cells_to_regions()
        
        # Load data into spatial analyzer
        self.spatial_analyzer.load_cell_data(
            self.ibex_data, coord_columns=['X', 'Y']
        )
        
        # Perform interaction analysis using regions as cell types
        interactions = self.spatial_analyzer.cell_interaction_analysis(
            distance_threshold_um=interaction_radius_um,
            cell_type_column='region'
        )
        
        # Calculate interaction enrichment
        if not interactions.empty:
            interaction_summary = interactions.groupby('interaction_type').agg({
                'distance_um': ['count', 'mean', 'std'],
                'cell1_id': 'count'
            }).round(3)
            
            interaction_summary.columns = ['count', 'mean_distance', 'std_distance', 'total_interactions']
            interaction_summary = interaction_summary.reset_index()
            
            print(f"📊 Found {len(interactions)} interactions between regions")
            
            return interaction_summary
        else:
            print("⚠ No interactions found within specified radius")
            return pd.DataFrame()
    
    def calculate_interface_metrics(self, interface_width_um=100):
        """
        Calculate metrics specific to tumor-normal interface regions
        
        Args:
            interface_width_um: Width of interface zone to analyze
            
        Returns:
            dict: Interface analysis metrics
        """
        if 'region' not in self.ibex_data.columns:
            self.assign_cells_to_regions()
        
        # Find interface cells
        interface_cells = self.ibex_data[self.ibex_data['region'] == 'interface']
        
        if len(interface_cells) == 0:
            print("⚠ No interface region found")
            return {}
        
        # Calculate interface metrics
        metrics = {
            'interface_cell_count': len(interface_cells),
            'interface_length_um': 0,  # Placeholder - would need boundary detection
            'interface_cell_density': len(interface_cells) / (interface_width_um ** 2)
        }
        
        # Marker enrichment in interface
        for col in self.ibex_data.columns:
            if 'intensity' in col.lower() or col in ['CD45', 'CD3', 'CD68']:
                if col in interface_cells.columns:
                    interface_mean = interface_cells[col].mean()
                    overall_mean = self.ibex_data[col].mean()
                    metrics[f'{col}_interface_enrichment'] = interface_mean / overall_mean if overall_mean > 0 else 1
        
        return metrics
    
    def generate_atlas_report(self):
        """
        Generate comprehensive meningioma atlas-style analysis report
        
        Returns:
            dict: Comprehensive analysis report
        """
        report = {}
        
        # Ensure regions are assigned
        if 'region' not in self.ibex_data.columns:
            self.assign_cells_to_regions()
        
        # Basic statistics
        report['summary'] = {
            'total_cells': len(self.ibex_data),
            'regions_identified': len(self.ibex_data['region'].unique()),
            'region_distribution': self.ibex_data['region'].value_counts().to_dict()
        }
        
        # Regional profiles
        profiles = self.calculate_regional_profiles()
        report['regional_profiles'] = profiles.to_dict('records')
        
        # Spatial interactions
        interactions = self.spatial_interaction_analysis()
        if not interactions.empty:
            report['spatial_interactions'] = interactions.to_dict('records')
        
        # Interface metrics
        interface_metrics = self.calculate_interface_metrics()
        report['interface_analysis'] = interface_metrics
        
        print("📋 Meningioma Atlas Report Generated")
        return report

def plot_tissue_segmentation(segmentation_mask, cell_coords=None, 
                           pixel_size_um=0.325, save_path=None):
    """
    Plot tissue segmentation results
    
    Args:
        segmentation_mask: Segmentation mask array
        cell_coords: Optional cell coordinates to overlay
        pixel_size_um: Pixel size for scaling
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot segmentation mask
    region_colors = ['blue', 'green', 'yellow', 'red']
    region_names = ['Normal', 'Interface', 'Tumor Edge', 'Tumor Core']
    
    im1 = axes[0].imshow(segmentation_mask, cmap='viridis', alpha=0.8)
    axes[0].set_title('Tissue Segmentation')
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')
    
    # Create custom colorbar
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=plt.cm.viridis(i/3), label=name) 
                      for i, name in enumerate(region_names)]
    axes[0].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    # Plot with cell overlay if provided
    if cell_coords is not None:
        axes[1].imshow(segmentation_mask, cmap='viridis', alpha=0.5)
        
        # Overlay cells
        x_coords = cell_coords[:, 0] / pixel_size_um
        y_coords = cell_coords[:, 1] / pixel_size_um
        axes[1].scatter(x_coords, y_coords, c='white', s=0.5, alpha=0.7)
        
        axes[1].set_title('Segmentation + Cells')
        axes[1].set_xlabel('X (pixels)')
        axes[1].set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig

def plot_regional_profiles(profiles_df, markers=None, save_path=None):
    """
    Plot marker expression profiles by tissue region
    
    Args:
        profiles_df: Regional profiles DataFrame
        markers: List of markers to plot
        save_path: Path to save figure
    """
    if markers is None:
        # Auto-detect marker mean columns
        markers = [col.replace('_mean', '') for col in profiles_df.columns 
                  if col.endswith('_mean') and 'CD' in col or 'IBA' in col or 'GFAP' in col]
    
    if len(markers) == 0:
        print("⚠ No marker columns found for plotting")
        return None
    
    # Prepare data for plotting
    plot_data = []
    for _, row in profiles_df.iterrows():
        for marker in markers:
            mean_col = f'{marker}_mean'
            if mean_col in row:
                plot_data.append({
                    'Region': row['region'],
                    'Marker': marker,
                    'Expression': row[mean_col]
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create heatmap
    pivot_df = plot_df.pivot(index='Region', columns='Marker', values='Expression')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', 
                cbar_kws={'label': 'Mean Expression'})
    plt.title('Marker Expression by Tissue Region')
    plt.xlabel('Markers')
    plt.ylabel('Tissue Regions')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return plt.gcf()

# Integration functions for IBEX pipeline
def integrate_with_ibex_alignment(aligned_data_path, segmentation_results):
    """
    Integrate tissue segmentation with IBEX alignment results
    
    Args:
        aligned_data_path: Path to aligned IBEX data
        segmentation_results: Tissue segmentation results
        
    Returns:
        dict: Integrated analysis results
    """
    print("🔄 Integrating with IBEX alignment pipeline...")
    # Placeholder for integration logic
    return {'status': 'integrated', 'data_path': aligned_data_path}

if __name__ == "__main__":
    print("🧠 Meningioma Atlas Integration")
    print("Available classes: MeningiomaAtlasAnalyzer")
    print("Available functions: plot_tissue_segmentation, plot_regional_profiles")