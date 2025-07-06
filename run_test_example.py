#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete test example for the automated well correlation system.
This script demonstrates the full workflow from data creation to correlation.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("Stdout:", e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr)
        return False

def create_directories():
    """Create necessary directories"""
    dirs = [
        "test_output",
        "test_output/training_data",
        "test_output/models",
        "test_output/projects",
        "test_output/images"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def main():
    """Main test function"""
    print("=" * 80)
    print("AUTOMATED WELL CORRELATION - COMPLETE TEST EXAMPLE")
    print("=" * 80)
    
    # Check if we're in the right directory
    if not os.path.exists("autoWellCorr.py"):
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating test directories...")
    create_directories()
    
    # Step 1: Create sample data
    print("\n📊 Step 1: Creating sample well log data...")
    if not run_command(
        "python create_sample_data.py --num-wells 5 --output-dir test_output/sample_data",
        "Create Sample Data"
    ):
        print("❌ Failed to create sample data. Exiting.")
        sys.exit(1)
    
    # Step 2: Create training data
    print("\n🔧 Step 2: Processing training data...")
    if not run_command(
        "python createTrainingData.py "
        "--data-name testWell "
        "--data-dir test_output/sample_data/las_files "
        "--output-dir test_output/training_data "
        "--log-name GR "
        "--patch-size 256 "
        "--skip-inc 10 "
        "--depth-inc 1 "
        "--clip-log True "
        "--cpu-count 2",
        "Create Training Data"
    ):
        print("❌ Failed to create training data. Exiting.")
        sys.exit(1)
    
    # Step 3: Train the model
    print("\n🧠 Step 3: Training neural network model...")
    if not run_command(
        "python trainModel.py "
        "--model-name testModel "
        "--data-dir test_output/training_data "
        "--output-dir test_output/models "
        "--imgs-dir test_output/images "
        "--save-imgs True "
        "--epochs 10 "
        "--batch-size 5 "
        "--split-ratio 0.8 "
        "--skip-inc 1",
        "Train Model"
    ):
        print("❌ Failed to train model. Exiting.")
        sys.exit(1)
    
    # Step 4: Create project file
    print("\n📋 Step 4: Creating project file...")
    if not run_command(
        "python createAutoWellProject.py "
        "--proj-name testProject "
        "--proj-dir test_output/projects "
        "--header-dir test_output/sample_data/well_header.csv "
        "--las-dir test_output/sample_data/las_files "
        "--uwi-col APINo "
        "--x-col Longitude "
        "--y-col Latitude "
        "--log-name GR",
        "Create Project"
    ):
        print("❌ Failed to create project. Exiting.")
        sys.exit(1)
    
    # Step 5: Run correlation
    print("\n🔗 Step 5: Running automated correlation...")
    
    # Create a Python script for correlation
    correlation_script = """
import sys
import os
sys.path.append('.')

from autoWellCorr import AutoWellCorrelation

print("Initializing correlation system...")

# Initialize the correlation system
correlator = AutoWellCorrelation(
    projPath='test_output/projects/testProject.eg',
    modelPath='test_output/models/testModel',
    resampleInc=2,
    minZ=None,
    maxZ=None,
    maxOffset=1000,
    smSigma=5,
    numNeighbors=3
)

print("Building connectivity graph...")
correlator.buildConnectivityGraph()

print(f"Correlation complete!")
print(f"Number of correlation points: {len(correlator.G.edges())}")
print(f"Number of well-depth nodes: {len(correlator.G.nodes())}")

# Show some correlation examples
if len(correlator.G.edges()) > 0:
    print("\\nSample correlations:")
    for i, (node1, node2, data) in enumerate(list(correlator.G.edges(data=True))[:3]):
        print(f"  {node1} <-> {node2} (weight: {data['weight']:.4f})")

print("\\n✅ Correlation test completed successfully!")
"""
    
    # Write and run the correlation script
    with open("test_output/run_correlation.py", "w") as f:
        f.write(correlation_script)
    
    if not run_command(
        "python test_output/run_correlation.py",
        "Run Correlation"
    ):
        print("❌ Failed to run correlation. Exiting.")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 80)
    print("🎉 TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    print("\n📁 Generated files:")
    print("  - Sample data: test_output/sample_data/")
    print("  - Training data: test_output/training_data/")
    print("  - Trained model: test_output/models/testModel")
    print("  - Project file: test_output/projects/testProject.eg")
    print("  - Training images: test_output/images/")
    
    print("\n📊 Test results:")
    print("  ✅ Sample well log data created")
    print("  ✅ Training data processed")
    print("  ✅ Neural network model trained")
    print("  ✅ Project file created")
    print("  ✅ Automated correlation completed")
    
    print("\n🚀 Next steps:")
    print("  1. Examine the generated files")
    print("  2. Try with your own well log data")
    print("  3. Adjust parameters for better results")
    print("  4. Scale up to larger datasets")
    
    print("\n📖 For more information:")
    print("  - Check setup_guide.md for detailed installation instructions")
    print("  - Read README.md for workflow details")
    print("  - Examine the generated images for training progress")

if __name__ == "__main__":
    main() 