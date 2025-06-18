#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality
"""

import sys
import importlib.util

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            __import__(package_name)
            print(f"✓ {package_name} imported successfully")
        else:
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                print(f"✗ {module_name} not found")
                return False
            else:
                __import__(module_name)
                print(f"✓ {module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ {module_name} import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ {module_name} error: {e}")
        return False

def main():
    print("Testing 3D Reconstruction Project Installation")
    print("=" * 50)
    
    # Test required packages
    required_packages = [
        ('cv2', None),  # opencv-python installs as cv2
        ('numpy', None),
        ('scipy', None), 
        ('matplotlib', None),
        ('open3d', None),
        ('flask', None),
        ('sklearn', None)  # scikit-learn installs as sklearn
    ]
    
    all_good = True
    
    print("\n1. Testing required packages:")
    for module, package in required_packages:
        if not test_import(module):
            all_good = False
    
    print("\n2. Testing project modules:")
    
    # Add src to path for testing
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    project_modules = [
        'core.camera',
        'core.features', 
        'core.geometry',
        'visualization.pointcloud',
        'utils.helpers'
    ]
    
    for module in project_modules:
        if not test_import(module):
            all_good = False
    
    print("\n3. Testing basic functionality:")
    
    try:
        import cv2
        import numpy as np
        from core.features import FeatureDetector
        
        # Create a simple test image
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test feature detector
        detector = FeatureDetector('ORB')  # Use ORB as it's more likely to work
        kp, desc = detector.detect_and_compute(test_img)
        
        print(f"✓ Feature detection working (found {len(kp)} keypoints)")
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("✓ All tests passed! The project is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Generate sample images: python create_sample_images.py")
        print("3. Start web interface: python web/app.py")
        print("4. Or use command line: python main.py --images sample_images/")
    else:
        print("✗ Some tests failed. Please install missing dependencies.")
        print("Run: pip install -r requirements.txt")
    
    return all_good

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)