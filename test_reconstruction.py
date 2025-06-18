#!/usr/bin/env python3
"""
Test the 3D reconstruction with sample images
"""

import sys
import os
import cv2
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.features import FeatureDetector, FeatureMatcher
from core.geometry import StereoReconstructor
from utils.helpers import create_default_camera_matrix

def test_reconstruction():
    print("Testing 3D reconstruction with sample images...")
    
    # Check if sample images exist
    img1_path = "sample_images/textured_view1.jpg"
    img2_path = "sample_images/textured_view2.jpg"
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("Sample images not found. Run: python3 create_sample_images.py")
        return False
    
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Error: Could not load sample images")
        return False
    
    print(f"Loaded images: {img1.shape}, {img2.shape}")
    
    # Create camera matrix
    camera_matrix = create_default_camera_matrix(img1.shape[:2])
    print("Created default camera matrix")
    
    # Feature detection and matching
    try:
        detector = FeatureDetector('SIFT')
        matcher = FeatureMatcher(detector)
        
        pts1, pts2, matches = matcher.match_image_pair(img1, img2)
        print(f"Found {len(matches)} feature matches")
        
        if len(matches) < 10:
            print("Error: Not enough matches found")
            return False
        
        # Filter matches
        pts1_filtered, pts2_filtered, mask = matcher.filter_matches_with_fundamental(pts1, pts2)
        print(f"After filtering: {len(pts1_filtered)} good matches")
        
        # 3D reconstruction
        reconstructor = StereoReconstructor(camera_matrix)
        points_3d, R, t = reconstructor.reconstruct_from_stereo_pair(
            img1, img2, pts1_filtered, pts2_filtered
        )
        
        print(f"Reconstructed {len(points_3d)} 3D points")
        print(f"Mean depth: {np.mean(points_3d[:, 2]):.2f}")
        print(f"Depth range: [{np.min(points_3d[:, 2]):.2f}, {np.max(points_3d[:, 2]):.2f}]")
        
        print("✅ 3D reconstruction test successful!")
        return True
        
    except Exception as e:
        print(f"❌ 3D reconstruction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_reconstruction()
    sys.exit(0 if success else 1)