#!/usr/bin/env python3
"""
Germain Lab Methods for JR_IBEX_003
Multi-scale spatial features and cell interaction networks
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from scipy import spatial, stats
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "scripts"))

from spatial_utils import SpatialAnalyzer

class GermainAnalyzer:
    """
    Germain lab-style multi-scale spatial analysis
    Focus on cell interaction networks and multi-scale spatial organization
    """
    
    def __init__(self, ibex_data, pixel_size_um=0.325):
        """
        Initialize Germain analyzer
        
        Args:
            ibex_data: IBEX cell data DataFrame
            pixel_size_um: Pixel size in micrometers
        """
        self.ibex_data = ibex_data
        self.pixel_size = pixel_size_um
        self.spatial_analyzer = SpatialAnalyzer(pixel_size_um)
        
        # Multi-scale analysis parameters
        self.scale_levels = [10, 25, 50, 100, 200]  # microns
        
    def multi_scale_spatial_analysis(self, cell_data=None, cell_type_column=None):
        """
        Perform multi-scale spatial analysis across different length scales
        
        Args:
            cell_data: DataFrame with cell data
            cell_type_column: Column name for cell type classification
            
        Returns:
            dict: Multi-scale analysis results
        """
        if cell_data is None:
            cell_data = self.ibex_data
            
        # Load data into spatial analyzer
        self.spatial_analyzer.load_cell_data(
            cell_data, coord_columns=['X', 'Y']
        )
        
        multi_scale_results = {}
        
        for scale in self.scale_levels:
            print(f"🔬 Analyzing scale: {scale} μm")
            
            scale_results = {}
            
            # 1. Neighborhood density analysis
            neighborhoods = self.spatial_analyzer.spatial_neighborhoods(
                radius_um=scale, cell_type_column=cell_type_column
            )
            
            scale_results['neighborhood_analysis'] = {
                'mean_density': neighborhoods['neighborhood_density'].mean(),
                'std_density': neighborhoods['neighborhood_density'].std(),
                'cv_density': neighborhoods['neighborhood_density'].std() / neighborhoods['neighborhood_density'].mean()
            }
            
            # 2. Spatial clustering analysis
            clustering_results = self.spatial_clustering_analysis(
                cell_data, eps_um=scale/2, cell_type_column=cell_type_column
            )
            scale_results['clustering'] = clustering_results
            
            # 3. Local spatial autocorrelation
            if cell_type_column:
                autocorr = self.calculate_spatial_autocorrelation(
                    cell_data, radius_um=scale, cell_type_column=cell_type_column
                )
                scale_results['autocorrelation'] = autocorr
            
            # 4. Ripley's K function
            ripley_results = self.spatial_analyzer.ripley_k_function(
                radii_um=[scale], cell_type_column=cell_type_column
            )
            if not ripley_results.empty:
                scale_results['ripley_k'] = {
                    'K_observed': ripley_results['K_observed'].iloc[0],
                    'K_expected': ripley_results['K_expected'].iloc[0],
                    'L_deviation': ripley_results['L_deviation'].iloc[0]
                }
            
            multi_scale_results[f'scale_{scale}um'] = scale_results
        
        return multi_scale_results
    
    def spatial_clustering_analysis(self, cell_data, eps_um=25, min_samples=5, cell_type_column=None):
        """
        Perform detailed spatial clustering analysis
        
        Args:
            cell_data: DataFrame with cell data
            eps_um: Clustering radius in micrometers
            min_samples: Minimum samples for DBSCAN
            cell_type_column: Column for cell type analysis
            
        Returns:
            dict: Clustering analysis results
        """
        coordinates = cell_data[['X', 'Y']].values * self.pixel_size
        
        # Overall clustering
        clustering = DBSCAN(eps=eps_um, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(coordinates)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        results = {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'clustering_coefficient': n_clusters / len(cell_data),
            'noise_fraction': n_noise / len(cell_data)
        }
        
        # Calculate cluster properties
        if n_clusters > 0:
            cluster_sizes = []
            cluster_densities = []
            
            for cluster_id in set(cluster_labels):
                if cluster_id == -1:  # Skip noise
                    continue
                    
                cluster_points = coordinates[cluster_labels == cluster_id]
                cluster_size = len(cluster_points)
                
                # Calculate cluster area (convex hull)
                if cluster_size >= 3:
                    try:
                        hull = spatial.ConvexHull(cluster_points)
                        cluster_area = hull.volume  # 2D area
                        cluster_density = cluster_size / cluster_area
                    except:
                        cluster_density = np.nan
                else:
                    cluster_density = np.nan
                
                cluster_sizes.append(cluster_size)
                if not np.isnan(cluster_density):
                    cluster_densities.append(cluster_density)
            
            results.update({
                'mean_cluster_size': np.mean(cluster_sizes),
                'std_cluster_size': np.std(cluster_sizes),
                'mean_cluster_density': np.mean(cluster_densities) if cluster_densities else np.nan,
                'std_cluster_density': np.std(cluster_densities) if cluster_densities else np.nan
            })
        
        # Cell type-specific clustering
        if cell_type_column and cell_type_column in cell_data.columns:
            type_clustering = {}
            
            for cell_type in cell_data[cell_type_column].unique():
                type_mask = cell_data[cell_type_column] == cell_type
                type_coords = coordinates[type_mask]
                
                if len(type_coords) >= min_samples:
                    type_clustering_obj = DBSCAN(eps=eps_um, min_samples=min_samples)
                    type_labels = type_clustering_obj.fit_predict(type_coords)
                    
                    type_n_clusters = len(set(type_labels)) - (1 if -1 in type_labels else 0)
                    type_n_noise = list(type_labels).count(-1)
                    
                    type_clustering[cell_type] = {
                        'n_clusters': type_n_clusters,
                        'n_noise': type_n_noise,
                        'clustering_coefficient': type_n_clusters / len(type_coords)
                    }
            
            results['type_specific_clustering'] = type_clustering
        
        return results
    
    def calculate_spatial_autocorrelation(self, cell_data, radius_um=50, cell_type_column=None):
        """
        Calculate spatial autocorrelation (Moran's I) for cell types
        
        Args:
            cell_data: DataFrame with cell data
            radius_um: Radius for spatial weights
            cell_type_column: Column for cell type classification
            
        Returns:
            dict: Spatial autocorrelation results
        """
        if not cell_type_column or cell_type_column not in cell_data.columns:
            return {}
        
        coordinates = cell_data[['X', 'Y']].values * self.pixel_size
        
        # Build spatial weights matrix
        tree = spatial.cKDTree(coordinates)
        
        # For each cell type, calculate Moran's I
        autocorr_results = {}
        
        for cell_type in cell_data[cell_type_column].unique():
            # Create binary indicator for this cell type
            indicator = (cell_data[cell_type_column] == cell_type).astype(int)
            
            # Calculate Moran's I
            morans_i = self._calculate_morans_i(coordinates, indicator, radius_um)
            
            autocorr_results[cell_type] = {
                'morans_i': morans_i['I'],
                'expected_i': morans_i['EI'],
                'variance_i': morans_i['VI'],
                'z_score': morans_i['z']
            }
        
        return autocorr_results
    
    def _calculate_morans_i(self, coordinates, values, radius_um):
        """Calculate Moran's I statistic"""
        n = len(coordinates)
        
        # Build spatial weights matrix
        tree = spatial.cKDTree(coordinates)
        
        # Initialize weights matrix
        W = np.zeros((n, n))
        
        for i in range(n):
            neighbors = tree.query_ball_point(coordinates[i], radius_um)
            neighbors = [j for j in neighbors if j != i]  # Exclude self
            
            if neighbors:
                # Simple binary weights (1 if neighbor, 0 otherwise)
                W[i, neighbors] = 1
        
        # Calculate Moran's I
        y = np.array(values)
        y_mean = np.mean(y)
        
        # Numerator: sum of spatial weights * products of deviations
        numerator = 0
        for i in range(n):
            for j in range(n):
                numerator += W[i, j] * (y[i] - y_mean) * (y[j] - y_mean)
        
        # Denominator: sum of squared deviations
        denominator = np.sum((y - y_mean)**2)
        
        # Sum of all weights
        S0 = np.sum(W)
        
        if S0 == 0 or denominator == 0:
            return {'I': 0, 'EI': 0, 'VI': 0, 'z': 0}
        
        # Moran's I
        I = (n / S0) * (numerator / denominator)
        
        # Expected value under null hypothesis
        EI = -1 / (n - 1)
        
        # Variance (simplified)
        VI = 2 / ((n - 1) * S0)
        
        # Z-score
        z = (I - EI) / np.sqrt(VI) if VI > 0 else 0
        
        return {'I': I, 'EI': EI, 'VI': VI, 'z': z}
    
    def build_cell_interaction_network(self, cell_data, interaction_radius_um=50, 
                                     cell_type_column=None, min_interactions=1):
        """
        Build cell-cell interaction network
        
        Args:
            cell_data: DataFrame with cell data
            interaction_radius_um: Radius for defining interactions
            cell_type_column: Column for cell type classification
            min_interactions: Minimum interactions to include cell in network
            
        Returns:
            networkx.Graph: Cell interaction network
        """
        coordinates = cell_data[['X', 'Y']].values * self.pixel_size
        
        # Build spatial network
        G = nx.Graph()
        
        # Add nodes (cells)
        for i, (idx, cell) in enumerate(cell_data.iterrows()):
            node_attrs = {'x': coordinates[i, 0], 'y': coordinates[i, 1]}
            
            if cell_type_column and cell_type_column in cell:
                node_attrs['cell_type'] = cell[cell_type_column]
            
            G.add_node(idx, **node_attrs)
        
        # Add edges (interactions)
        tree = spatial.cKDTree(coordinates)
        
        for i, coord in enumerate(coordinates):
            neighbors = tree.query_ball_point(coord, interaction_radius_um)
            neighbors = [j for j in neighbors if j != i]  # Exclude self
            
            cell_idx = cell_data.index[i]
            
            for j in neighbors:
                neighbor_idx = cell_data.index[j]
                distance = np.linalg.norm(coordinates[i] - coordinates[j])
                
                # Add edge with distance weight
                G.add_edge(cell_idx, neighbor_idx, 
                          weight=1/distance if distance > 0 else 1,
                          distance=distance)
        
        # Filter nodes by minimum interactions
        nodes_to_remove = [node for node in G.nodes() if G.degree(node) < min_interactions]
        G.remove_nodes_from(nodes_to_remove)
        
        print(f"🕸️ Built interaction network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def analyze_network_properties(self, G, cell_type_column=None):
        """
        Analyze network properties and community structure
        
        Args:
            G: NetworkX graph
            cell_type_column: Column for cell type analysis
            
        Returns:
            dict: Network analysis results
        """
        if G.number_of_nodes() == 0:
            return {}
        
        results = {}
        
        # Basic network properties
        results['basic_properties'] = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G),
            'n_connected_components': nx.number_connected_components(G)
        }
        
        # Degree statistics
        degrees = [G.degree(n) for n in G.nodes()]
        results['degree_statistics'] = {
            'mean_degree': np.mean(degrees),
            'std_degree': np.std(degrees),
            'max_degree': max(degrees) if degrees else 0,
            'degree_assortativity': nx.degree_assortativity_coefficient(G)
        }
        
        # Centrality measures
        if G.number_of_nodes() <= 1000:  # Avoid computation for very large networks
            betweenness = nx.betweenness_centrality(G)
            closeness = nx.closeness_centrality(G)
            
            results['centrality'] = {
                'mean_betweenness': np.mean(list(betweenness.values())),
                'mean_closeness': np.mean(list(closeness.values())),
                'max_betweenness_node': max(betweenness, key=betweenness.get),
                'max_closeness_node': max(closeness, key=closeness.get)
            }
        
        # Community detection (Leiden algorithm approximation using Louvain)
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(G)
            modularity = community_louvain.modularity(communities, G)
            
            results['community_structure'] = {
                'n_communities': len(set(communities.values())),
                'modularity': modularity
            }
            
            # Add community info to nodes
            nx.set_node_attributes(G, communities, 'community')
            
        except ImportError:
            print("⚠ Community detection requires python-louvain package")
        
        # Cell type-specific network properties
        if cell_type_column:
            node_attrs = nx.get_node_attributes(G, 'cell_type')
            if node_attrs:
                type_specific = {}
                
                for cell_type in set(node_attrs.values()):
                    type_nodes = [n for n, t in node_attrs.items() if t == cell_type]
                    type_subgraph = G.subgraph(type_nodes)
                    
                    if type_subgraph.number_of_nodes() > 0:
                        type_specific[cell_type] = {
                            'n_nodes': type_subgraph.number_of_nodes(),
                            'n_edges': type_subgraph.number_of_edges(),
                            'density': nx.density(type_subgraph),
                            'average_clustering': nx.average_clustering(type_subgraph)
                        }
                
                results['cell_type_networks'] = type_specific
        
        return results
    
    def calculate_network_motifs(self, G, motif_size=3):
        """
        Calculate network motifs (small subgraph patterns)
        
        Args:
            G: NetworkX graph
            motif_size: Size of motifs to analyze
            
        Returns:
            dict: Motif counts
        """
        if G.number_of_nodes() < motif_size or motif_size > 4:
            return {}
        
        motif_counts = {}
        
        if motif_size == 3:
            # Count triangles
            triangles = [clique for clique in nx.enumerate_all_cliques(G) if len(clique) == 3]
            motif_counts['triangles'] = len(triangles)
            
            # Count open triplets (2-paths)
            open_triplets = 0
            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                if len(neighbors) >= 2:
                    # Count pairs of neighbors that are not connected
                    for i, n1 in enumerate(neighbors):
                        for n2 in neighbors[i+1:]:
                            if not G.has_edge(n1, n2):
                                open_triplets += 1
            
            motif_counts['open_triplets'] = open_triplets
            
            # Clustering coefficient (proportion of closed triplets)
            if open_triplets + motif_counts['triangles'] > 0:
                motif_counts['transitivity'] = (3 * motif_counts['triangles']) / (3 * motif_counts['triangles'] + open_triplets)
        
        return motif_counts
    
    def multi_scale_network_analysis(self, cell_data, cell_type_column=None):
        """
        Perform network analysis at multiple scales
        
        Args:
            cell_data: DataFrame with cell data
            cell_type_column: Column for cell type classification
            
        Returns:
            dict: Multi-scale network analysis results
        """
        multi_scale_networks = {}
        
        for scale in self.scale_levels:
            print(f"🕸️ Network analysis at scale: {scale} μm")
            
            # Build network at this scale
            G = self.build_cell_interaction_network(
                cell_data, 
                interaction_radius_um=scale,
                cell_type_column=cell_type_column
            )
            
            # Analyze network properties
            network_props = self.analyze_network_properties(G, cell_type_column)
            
            # Calculate motifs
            motifs = self.calculate_network_motifs(G)
            
            multi_scale_networks[f'scale_{scale}um'] = {
                'network_properties': network_props,
                'motifs': motifs,
                'graph': G  # Store graph for further analysis
            }
        
        return multi_scale_networks
    
    def generate_germain_report(self, cell_data=None, cell_type_column=None):
        """
        Generate comprehensive Germain lab-style analysis report
        
        Args:
            cell_data: DataFrame with cell data
            cell_type_column: Column for cell type classification
            
        Returns:
            dict: Comprehensive analysis report
        """
        if cell_data is None:
            cell_data = self.ibex_data
        
        report = {}
        
        # Multi-scale spatial analysis
        print("🔬 Running multi-scale spatial analysis...")
        multi_scale_spatial = self.multi_scale_spatial_analysis(cell_data, cell_type_column)
        report['multi_scale_spatial'] = multi_scale_spatial
        
        # Multi-scale network analysis
        print("🕸️ Running multi-scale network analysis...")
        multi_scale_networks = self.multi_scale_network_analysis(cell_data, cell_type_column)
        
        # Extract network properties (exclude graphs for serialization)
        network_summary = {}
        for scale, data in multi_scale_networks.items():
            network_summary[scale] = {
                'network_properties': data['network_properties'],
                'motifs': data['motifs']
            }
        
        report['multi_scale_networks'] = network_summary
        
        # Cross-scale analysis
        report['cross_scale_analysis'] = self._analyze_scale_dependencies(multi_scale_spatial, network_summary)
        
        print("📋 Germain Lab Analysis Report Generated")
        return report
    
    def _analyze_scale_dependencies(self, spatial_results, network_results):
        """Analyze how properties change across scales"""
        scale_analysis = {}
        
        # Extract density values across scales
        densities = []
        clustering_coeffs = []
        network_densities = []
        
        for scale in self.scale_levels:
            scale_key = f'scale_{scale}um'
            
            if scale_key in spatial_results:
                spatial_data = spatial_results[scale_key]
                if 'neighborhood_analysis' in spatial_data:
                    densities.append(spatial_data['neighborhood_analysis']['mean_density'])
            
            if scale_key in network_results:
                network_data = network_results[scale_key]
                if 'network_properties' in network_data:
                    net_props = network_data['network_properties']
                    if 'basic_properties' in net_props:
                        clustering_coeffs.append(net_props['basic_properties'].get('average_clustering', 0))
                        network_densities.append(net_props['basic_properties'].get('density', 0))
        
        if len(densities) >= 2:
            scale_analysis['density_scaling'] = {
                'scales': self.scale_levels[:len(densities)],
                'densities': densities,
                'correlation_with_scale': stats.pearsonr(self.scale_levels[:len(densities)], densities)[0]
            }
        
        if len(clustering_coeffs) >= 2:
            scale_analysis['clustering_scaling'] = {
                'scales': self.scale_levels[:len(clustering_coeffs)],
                'clustering_coefficients': clustering_coeffs
            }
        
        return scale_analysis

def plot_multi_scale_analysis(multi_scale_results, save_path=None):
    """
    Plot multi-scale analysis results
    
    Args:
        multi_scale_results: Multi-scale analysis results dict
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    scales = []
    densities = []
    clustering_coeffs = []
    l_deviations = []
    
    # Extract data across scales
    for scale_key, results in multi_scale_results.items():
        if 'scale_' in scale_key:
            scale = int(scale_key.split('_')[1].replace('um', ''))
            scales.append(scale)
            
            if 'neighborhood_analysis' in results:
                densities.append(results['neighborhood_analysis']['mean_density'])
            
            if 'clustering' in results:
                clustering_coeffs.append(results['clustering'].get('clustering_coefficient', 0))
            
            if 'ripley_k' in results:
                l_deviations.append(results['ripley_k'].get('L_deviation', 0))
    
    # Sort by scale
    sorted_data = sorted(zip(scales, densities, clustering_coeffs, l_deviations))
    if sorted_data:
        scales, densities, clustering_coeffs, l_deviations = zip(*sorted_data)
    
    # Plot 1: Density vs scale
    if densities:
        axes[0, 0].plot(scales, densities, 'o-', color='blue')
        axes[0, 0].set_xlabel('Scale (μm)')
        axes[0, 0].set_ylabel('Mean Density')
        axes[0, 0].set_title('Neighborhood Density vs Scale')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Clustering vs scale
    if clustering_coeffs:
        axes[0, 1].plot(scales, clustering_coeffs, 'o-', color='red')
        axes[0, 1].set_xlabel('Scale (μm)')
        axes[0, 1].set_ylabel('Clustering Coefficient')
        axes[0, 1].set_title('Clustering vs Scale')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: L-function deviation vs scale
    if l_deviations:
        axes[1, 0].plot(scales, l_deviations, 'o-', color='green')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Scale (μm)')
        axes[1, 0].set_ylabel('L(r) - r')
        axes[1, 0].set_title('Spatial Clustering (Ripley\'s L)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Scale relationships
    if len(scales) >= 3 and densities and clustering_coeffs:
        axes[1, 1].scatter(densities, clustering_coeffs, c=scales, cmap='viridis', s=50)
        axes[1, 1].set_xlabel('Density')
        axes[1, 1].set_ylabel('Clustering Coefficient')
        axes[1, 1].set_title('Density vs Clustering')
        
        # Add colorbar for scale
        scatter = axes[1, 1].scatter(densities, clustering_coeffs, c=scales, cmap='viridis', s=50)
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('Scale (μm)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_interaction_network(G, pos=None, node_colors=None, save_path=None, 
                           figsize=(10, 8), node_size=20):
    """
    Plot cell interaction network
    
    Args:
        G: NetworkX graph
        pos: Node positions dict (if None, will use spatial layout)
        node_colors: Node colors dict or array
        save_path: Path to save figure
        figsize: Figure size
        node_size: Node size for plotting
    """
    plt.figure(figsize=figsize)
    
    if pos is None:
        # Use actual spatial coordinates if available
        node_attrs = nx.get_node_attributes(G, 'x')
        if node_attrs:
            pos = {node: (attrs['x'], attrs['y']) 
                  for node, attrs in G.nodes(data=True) 
                  if 'x' in attrs and 'y' in attrs}
        else:
            # Use spring layout
            pos = nx.spring_layout(G)
    
    # Draw network
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, edge_color='gray')
    
    if node_colors is not None:
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_size, alpha=0.8)
    else:
        nx.draw_networkx_nodes(G, pos, node_size=node_size, alpha=0.8)
    
    plt.title(f'Cell Interaction Network ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)')
    plt.axis('equal')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

if __name__ == "__main__":
    print("🧬 Germain Lab Methods")
    print("Available classes: GermainAnalyzer")
    print("Available functions: plot_multi_scale_analysis, plot_interaction_network")