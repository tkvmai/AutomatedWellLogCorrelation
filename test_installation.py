#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify installation of all required packages for the 
automated well correlation system.
"""

import sys
import os
import matplotlib

def test_imports():
    """Test all required package imports"""
    print("Testing package imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import scipy
        print(f"✅ SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"❌ SciPy import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import lasio
        print(f"✅ LASIO {lasio.__version__}")
    except ImportError as e:
        print(f"❌ LASIO import failed: {e}")
        return False
    
    try:
        import networkx as nx
        print(f"✅ NetworkX {nx.__version__}")
    except ImportError as e:
        print(f"❌ NetworkX import failed: {e}")
        return False
    
    try:
        import joblib
        print(f"✅ Joblib {joblib.__version__}")
    except ImportError as e:
        print(f"❌ Joblib import failed: {e}")
        return False
    
    try:
        import h5py
        print(f"✅ H5Py {h5py.__version__}")
    except ImportError as e:
        print(f"❌ H5Py import failed: {e}")
        return False
    
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    return True

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA availability...")
    
    import torch
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        return True
    else:
        print("⚠️  CUDA not available - will use CPU")
        return False

def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\nTesting basic functionality...")
    
    try:
        import torch
        import numpy as np
        import pandas as pd
        import lasio
        from scipy import signal
        from sklearn.feature_extraction.image import extract_patches
        
        # Test PyTorch tensor operations
        x = torch.randn(10, 1, 256)
        print(f"✅ PyTorch tensor created: {x.shape}")
        
        # Test NumPy operations
        arr = np.random.random(100)
        print(f"✅ NumPy array created: {arr.shape}")
        
        # Test Pandas DataFrame
        df = pd.DataFrame({'depth': np.arange(1000, 2000), 'gr': np.random.random(1000)})
        print(f"✅ Pandas DataFrame created: {df.shape}")
        
        # Test LAS file creation
        las = lasio.LASFile()
        las.well.WELL.data = "TEST_WELL"
        las.well.DEPT.data = np.arange(1000, 2000, 1)
        las.add_curve('GR', np.random.random(1000), unit='gAPI')
        print("✅ LAS file created successfully")
        
        # Test patch extraction
        data = np.random.random(1000)
        patches = extract_patches(data, 256, 10)
        print(f"✅ Patch extraction: {patches.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_project_files():
    """Test if project files are accessible"""
    print("\nTesting project files...")
    
    required_files = [
        'autoWellCorr.py',
        'vae.py', 
        'createTrainingData.py',
        'trainModel.py',
        'createAutoWellProject.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        return False
    
    return True

def main():
    """Main test function"""
    print("=" * 60)
    print("AUTOMATED WELL CORRELATION SYSTEM - INSTALLATION TEST")
    print("=" * 60)
    
    # Test Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠️  Python 3.8+ recommended")
    else:
        print("✅ Python version OK")
    
    # Run all tests
    tests = [
        ("Package Imports", test_imports),
        ("CUDA Availability", test_cuda),
        ("Basic Functionality", test_basic_functionality),
        ("Project Files", test_project_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("You can now run the well correlation system.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print("Please check the installation guide and troubleshoot the issues.")
    
    print("\nNext steps:")
    print("1. Create sample data using create_sample_data.py")
    print("2. Follow the workflow in the README.md")
    print("3. Start with small datasets for testing")

if __name__ == "__main__":
    main() 