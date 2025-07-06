# Setup Guide for Automated Well Correlation System

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Ubuntu 18.04+
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **CPU**: Multi-core processor (4+ cores recommended)
- **GPU**: Optional but recommended for faster training (NVIDIA GPU with CUDA support)

### Recommended Requirements
- **RAM**: 32GB
- **GPU**: NVIDIA RTX 3060 or better with 8GB+ VRAM
- **Storage**: SSD with 50GB+ free space

## Step 1: Install Python

### Option A: Anaconda (Recommended)
1. Download Anaconda from: https://www.anaconda.com/products/distribution
2. Install with default settings
3. Open Anaconda Prompt (Windows) or Terminal (macOS/Linux)

### Option B: Python.org
1. Download Python 3.8+ from: https://www.python.org/downloads/
2. Install with "Add Python to PATH" checked
3. Open Command Prompt (Windows) or Terminal (macOS/Linux)

## Step 2: Create Virtual Environment

```bash
# Create a new virtual environment
conda create -n wellcorr python=3.9
conda activate wellcorr

# OR using venv (if using Python.org)
python -m venv wellcorr
# On Windows:
wellcorr\Scripts\activate
# On macOS/Linux:
source wellcorr/bin/activate
```

## Step 3: Install PyTorch

### For CPU-only:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### For NVIDIA GPU (CUDA):
```bash
# Check your CUDA version first
nvidia-smi

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Step 4: Install Other Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

## Step 5: Verify Installation

Create a test script to verify everything is working:

```python
# test_installation.py
import torch
import numpy as np
import pandas as pd
import lasio
import networkx as nx
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.feature_extraction.image import extract_patches

print("✅ All packages imported successfully!")

# Test CUDA availability
if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  CUDA not available - will use CPU")

# Test basic functionality
x = torch.randn(10, 1, 256)
print(f"✅ PyTorch tensor created: {x.shape}")

# Test LAS file creation
las = lasio.LASFile()
las.well.WELL.data = "TEST_WELL"
las.well.DEPT.data = np.arange(1000, 2000, 1)
las.add_curve('GR', np.random.random(1000), unit='gAPI')
print("✅ LAS file created successfully")

print("\n🎉 Installation complete! You're ready to run the well correlation system.")
```

Run the test:
```bash
python test_installation.py
```

## Step 6: Install Additional Tools (Optional)

### Git (for version control)
- **Windows**: Download from https://git-scm.com/
- **macOS**: `brew install git`
- **Ubuntu**: `sudo apt install git`

### Visual Studio Code (Recommended IDE)
1. Download from: https://code.visualstudio.com/
2. Install Python extension
3. Install Jupyter extension

### Jupyter Lab (for interactive development)
```bash
pip install jupyterlab
jupyter lab
```

## Step 7: Download Sample Data (Optional)

If you don't have well log data, you can create synthetic data for testing:

```python
# create_sample_data.py
import numpy as np
import pandas as pd
import lasio
import os

# Create sample directory
os.makedirs("sample_data", exist_ok=True)
os.makedirs("sample_data/las_files", exist_ok=True)

# Create synthetic well data
def create_synthetic_well(api_number, x, y, depth_range=(1000, 5000)):
    depths = np.arange(depth_range[0], depth_range[1], 1)
    # Create realistic gamma ray patterns
    gr = 50 + 30 * np.sin(depths/100) + 20 * np.sin(depths/50) + 10 * np.random.random(len(depths))
    
    las = lasio.LASFile()
    las.well.WELL.data = f"WELL_{api_number}"
    las.well.DEPT.data = depths
    las.add_curve('GR', gr, unit='gAPI')
    
    return las

# Create multiple synthetic wells
wells = [
    (1234567890, -96.1234, 29.5678),
    (1234567891, -96.1235, 29.5679),
    (1234567892, -96.1236, 29.5680),
    (1234567893, -96.1237, 29.5681),
    (1234567894, -96.1238, 29.5682)
]

# Save LAS files
for api, x, y in wells:
    las = create_synthetic_well(api, x, y)
    las.write(f"sample_data/las_files/{api}.las")

# Create header file
header_data = [{'APINo': api, 'Longitude': x, 'Latitude': y} for api, x, y in wells]
header_df = pd.DataFrame(header_data)
header_df.to_csv("sample_data/well_header.csv", index=False)

print("✅ Sample data created in 'sample_data' directory")
```

## Troubleshooting

### Common Issues:

1. **CUDA not found**
   - Install NVIDIA drivers
   - Install CUDA toolkit
   - Reinstall PyTorch with correct CUDA version

2. **Memory errors**
   - Reduce batch size in training
   - Use smaller patch sizes
   - Close other applications

3. **LAS file reading errors**
   - Ensure LAS files are properly formatted
   - Check file encoding
   - Verify log curve names

4. **Import errors**
   - Activate virtual environment
   - Reinstall packages: `pip install --force-reinstall package_name`

### Getting Help:
- Check PyTorch documentation: https://pytorch.org/docs/
- LAS file format: https://www.cwls.org/products/software-las/
- Create issues on the GitHub repository

## Next Steps

After installation, you can:
1. Run the sample data creation script
2. Follow the workflow in the main README
3. Start with small datasets for testing
4. Gradually scale up to larger datasets

Happy well correlating! 🎯 