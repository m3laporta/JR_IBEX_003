#!/usr/bin/env python3
"""
JR_IBEX_003 Workspace Loader
Auto-loads complete environment with all necessary libraries and utilities
"""

import sys
import os
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
import warnings

# Suppress common warnings for cleaner startup
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def setup_project_paths():
    """Set up project paths and add to Python path"""
    # Get project root (go up from scripts directory)
    project_root = Path(__file__).parent.parent.absolute()
    
    # Add project directories to Python path
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "scripts"))
    sys.path.insert(0, str(project_root / "literature"))
    
    return project_root

def load_config():
    """Load master configuration"""
    project_root = Path(__file__).parent.parent.absolute()
    config_path = project_root / "config" / "master_config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded master configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"⚠ Warning: Master config not found at {config_path}")
        return {}

def setup_logging(config):
    """Set up logging configuration"""
    project_root = Path(__file__).parent.parent.absolute()
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_file = log_dir / "jribex003.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('JR_IBEX_003')
    logger.info(f"Logging initialized - Level: {log_level}")
    return logger

def import_core_libraries():
    """Import all essential libraries for IBEX analysis"""
    print("📚 Loading core libraries...")
    
    # Essential data science libraries
    global np, pd, plt, sns
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Image processing libraries
    global skimage, tifffile, sitk
    import skimage
    from skimage import measure, segmentation, morphology, filters
    import tifffile
    import SimpleITK as sitk
    
    # Registration and alignment
    global StackReg
    try:
        from pystackreg import StackReg
        print("  ✓ pystackreg (StackReg)")
    except ImportError:
        print("  ⚠ pystackreg not available")
    
    # Spatial analysis
    global scipy
    import scipy
    from scipy import spatial, stats, ndimage
    
    # Machine learning
    global sklearn
    import sklearn
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    
    # Visualization
    try:
        global napari
        import napari
        print("  ✓ napari")
    except ImportError:
        print("  ⚠ napari not available (GUI applications)")
    
    # Other utilities
    import joblib
    from pathlib import Path
    from datetime import datetime
    
    print("  ✓ All core libraries loaded successfully")

def create_analysis_functions():
    """Create commonly used analysis functions"""
    
    def read_ibex_stack(file_path):
        """Read IBEX OME-TIFF with proper dimension handling"""
        stack = tifffile.imread(file_path)
        
        # Ensure ZCYX format
        if stack.ndim == 4:
            return stack
        elif stack.ndim == 3 and stack.shape[0] <= 5:
            return stack[None]  # Add Z dimension
        elif stack.ndim == 3 and stack.shape[-1] <= 5:
            return stack[..., None].transpose(3, 2, 0, 1)
        elif stack.ndim == 2:
            return stack[None, None]
        else:
            raise ValueError(f"Unsupported stack shape: {stack.shape}")
    
    def make_mip(stack, axis=0):
        """Create maximum intensity projection"""
        return np.max(stack, axis=axis)
    
    def pad_to_match(img1, img2):
        """Pad images to match dimensions"""
        h_max = max(img1.shape[0], img2.shape[0])
        w_max = max(img1.shape[1], img2.shape[1])
        
        def pad_image(img):
            h_pad = h_max - img.shape[0]
            w_pad = w_max - img.shape[1]
            return np.pad(img, ((0, h_pad), (0, w_pad)), mode='constant')
        
        return pad_image(img1), pad_image(img2)
    
    def calculate_spatial_features(coords, values=None, radius=50):
        """Calculate spatial features for cell coordinates"""
        from sklearn.neighbors import NearestNeighbors
        
        # Nearest neighbor distances
        nn = NearestNeighbors(n_neighbors=6)
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)
        
        features = {
            'mean_nn_distance': np.mean(distances[:, 1:], axis=1),
            'std_nn_distance': np.std(distances[:, 1:], axis=1),
            'n_neighbors': np.sum(distances <= radius, axis=1) - 1
        }
        
        if values is not None:
            # Local environment features
            for i in range(len(coords)):
                neighbors_idx = indices[i][distances[i] <= radius]
                if len(neighbors_idx) > 1:
                    neighbor_values = values[neighbors_idx[1:]]  # Exclude self
                    features[f'local_mean_{i}'] = np.mean(neighbor_values)
        
        return pd.DataFrame(features)
    
    # Make functions globally available
    globals()['read_ibex_stack'] = read_ibex_stack
    globals()['make_mip'] = make_mip
    globals()['pad_to_match'] = pad_to_match
    globals()['calculate_spatial_features'] = calculate_spatial_features
    
    print("  ✓ Analysis functions created")

def load_existing_notebooks():
    """Import functions from existing notebooks"""
    project_root = Path(__file__).parent.parent.absolute()
    notebooks_dir = project_root / "notebooks"
    
    # Add notebooks directory to path for importing
    sys.path.insert(0, str(notebooks_dir))
    
    print("  ✓ Existing notebooks available for import")

def setup_matplotlib():
    """Configure matplotlib for high-quality figures"""
    plt.style.use('default')
    
    # Set publication-quality defaults
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.titlesize': 9,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight'
    })
    
    print("  ✓ Matplotlib configured for publication figures")

def check_environment():
    """Check environment and dependencies"""
    import platform
    
    print(f"\n🖥  Environment Check:")
    print(f"  Python: {sys.version}")
    print(f"  Platform: {platform.platform()}")
    print(f"  Working directory: {os.getcwd()}")
    
    # Check key packages
    key_packages = ['numpy', 'pandas', 'matplotlib', 'scikit-image', 'tifffile']
    for pkg in key_packages:
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            print(f"  {pkg}: {version}")
        except ImportError:
            print(f"  {pkg}: ⚠ NOT AVAILABLE")

def main():
    """Main workspace loading function"""
    print("🚀 JR_IBEX_003 Workspace Loader")
    print("=" * 50)
    
    # Set up project structure
    project_root = setup_project_paths()
    print(f"📁 Project root: {project_root}")
    
    # Load configuration
    config = load_config()
    
    # Set up logging
    logger = setup_logging(config)
    
    # Import libraries
    import_core_libraries()
    
    # Create analysis functions  
    create_analysis_functions()
    
    # Load existing notebooks
    load_existing_notebooks()
    
    # Set up visualization
    setup_matplotlib()
    
    # Environment check
    check_environment()
    
    print("\n✅ Workspace loaded successfully!")
    print("\n📋 Available functions:")
    print("  - read_ibex_stack(path)")
    print("  - make_mip(stack, axis=0)")
    print("  - pad_to_match(img1, img2)")
    print("  - calculate_spatial_features(coords, values=None, radius=50)")
    print("\n📚 Available modules: np, pd, plt, sns, skimage, tifffile, sitk, scipy, sklearn")
    print("\n🎯 Ready for spatial immunofluorescence analysis!")
    
    return {
        'project_root': project_root,
        'config': config,
        'logger': logger
    }

if __name__ == "__main__":
    workspace = main()