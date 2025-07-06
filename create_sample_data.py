#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create synthetic well log data for testing the automated well correlation system.
This script generates realistic gamma ray logs with geological patterns.
"""

import numpy as np
import pandas as pd
import lasio
import os
import argparse

def create_synthetic_well(api_number, x, y, depth_range=(1000, 5000), 
                         base_gr=50, noise_level=10, pattern_scale=100):
    """
    Create a synthetic well with realistic gamma ray patterns.
    
    Parameters:
    -----------
    api_number : int
        Well API number
    x, y : float
        Well coordinates
    depth_range : tuple
        Depth range (min, max) in feet
    base_gr : float
        Base gamma ray value
    noise_level : float
        Random noise level
    pattern_scale : float
        Scale of geological patterns
    
    Returns:
    --------
    lasio.LASFile
        LAS file object with synthetic data
    """
    
    # Create depth array
    depths = np.arange(depth_range[0], depth_range[1], 1)
    
    # Create realistic gamma ray patterns
    # Multiple sine waves to simulate different geological features
    gr = base_gr + \
         30 * np.sin(depths/pattern_scale) + \
         20 * np.sin(depths/(pattern_scale/2)) + \
         15 * np.sin(depths/(pattern_scale/4)) + \
         noise_level * np.random.random(len(depths))
    
    # Add some sharp boundaries (like unconformities)
    boundary_depths = [1500, 2500, 3500, 4500]
    for boundary in boundary_depths:
        if boundary in depths:
            idx = np.where(depths == boundary)[0][0]
            # Add sharp change at boundary
            gr[idx:] += 20 * np.random.random()
    
    # Ensure values are within reasonable range
    gr = np.clip(gr, 0, 200)
    
    # Create LAS file
    las = lasio.LASFile()
    las.well.WELL.data = f"WELL_{api_number}"
    las.well.DEPT.data = depths
    las.add_curve('GR', gr, unit='gAPI')
    
    return las

def create_sample_dataset(num_wells=5, output_dir="sample_data"):
    """
    Create a complete sample dataset with multiple wells.
    
    Parameters:
    -----------
    num_wells : int
        Number of wells to create
    output_dir : str
        Output directory for the dataset
    """
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "las_files"), exist_ok=True)
    
    # Define well locations (spread out in a grid)
    wells = []
    for i in range(num_wells):
        # Create a grid pattern
        row = i // 3
        col = i % 3
        x = -96.1234 + col * 0.001  # Longitude
        y = 29.5678 + row * 0.001   # Latitude
        api = 1234567890 + i
        wells.append((api, x, y))
    
    print(f"Creating {num_wells} synthetic wells...")
    
    # Create LAS files
    for api, x, y in wells:
        print(f"Creating well {api} at ({x:.4f}, {y:.4f})")
        las = create_synthetic_well(api, x, y)
        las.write(os.path.join(output_dir, "las_files", f"{api}.las"))
    
    # Create header file
    header_data = [{'APINo': api, 'Longitude': x, 'Latitude': y} for api, x, y in wells]
    header_df = pd.DataFrame(header_data)
    header_path = os.path.join(output_dir, "well_header.csv")
    header_df.to_csv(header_path, index=False)
    
    print(f"\n✅ Sample dataset created in '{output_dir}' directory")
    print(f"   - {num_wells} LAS files in '{output_dir}/las_files/'")
    print(f"   - Header file: '{header_path}'")
    
    return output_dir

def create_training_dataset(num_wells=20, output_dir="training_data"):
    """
    Create a larger dataset for training the neural network.
    
    Parameters:
    -----------
    num_wells : int
        Number of wells for training
    output_dir : str
        Output directory for training data
    """
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "las_files"), exist_ok=True)
    
    # Create more wells with varied patterns
    wells = []
    for i in range(num_wells):
        # Random positions within a reasonable area
        x = -96.1234 + np.random.uniform(-0.01, 0.01)
        y = 29.5678 + np.random.uniform(-0.01, 0.01)
        api = 2000000000 + i  # Different API range for training
        wells.append((api, x, y))
    
    print(f"Creating {num_wells} training wells...")
    
    # Create LAS files with more variation
    for api, x, y in wells:
        # Vary the parameters for more realistic training data
        base_gr = np.random.uniform(30, 70)
        noise_level = np.random.uniform(5, 15)
        pattern_scale = np.random.uniform(80, 120)
        
        las = create_synthetic_well(api, x, y, 
                                  base_gr=base_gr,
                                  noise_level=noise_level,
                                  pattern_scale=pattern_scale)
        las.write(os.path.join(output_dir, "las_files", f"{api}.las"))
    
    # Create header file
    header_data = [{'APINo': api, 'Longitude': x, 'Latitude': y} for api, x, y in wells]
    header_df = pd.DataFrame(header_data)
    header_path = os.path.join(output_dir, "well_header.csv")
    header_df.to_csv(header_path, index=False)
    
    print(f"\n✅ Training dataset created in '{output_dir}' directory")
    print(f"   - {num_wells} LAS files in '{output_dir}/las_files/'")
    print(f"   - Header file: '{header_path}'")
    
    return output_dir

def main():
    """Main function to create sample data"""
    
    parser = argparse.ArgumentParser(description="Create synthetic well log data for testing")
    parser.add_argument("--num-wells", type=int, default=5, 
                       help="Number of wells to create (default: 5)")
    parser.add_argument("--output-dir", type=str, default="sample_data",
                       help="Output directory (default: sample_data)")
    parser.add_argument("--training", action="store_true",
                       help="Create training dataset instead of test dataset")
    parser.add_argument("--num-training-wells", type=int, default=20,
                       help="Number of wells for training dataset (default: 20)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SYNTHETIC WELL LOG DATA GENERATOR")
    print("=" * 60)
    
    if args.training:
        print(f"Creating training dataset with {args.num_training_wells} wells...")
        create_training_dataset(args.num_training_wells, args.output_dir)
    else:
        print(f"Creating test dataset with {args.num_wells} wells...")
        create_sample_dataset(args.num_wells, args.output_dir)
    
    print("\nNext steps:")
    print("1. Run the installation test: python test_installation.py")
    print("2. Follow the workflow in setup_guide.md")
    print("3. Start with the sample data for testing")

if __name__ == "__main__":
    main() 