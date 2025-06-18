#!/usr/bin/env python3
"""
Debug the exact source of the numpy array conversion error
"""

import sys
import os
import cv2
import numpy as np
import tempfile

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.features import FeatureDetector, FeatureMatcher
from core.geometry import StereoReconstructor
from visualization.pointcloud import PointCloudProcessor
from utils.helpers import create_default_camera_matrix

def debug_reconstruction():
    print("Debugging 3D reconstruction...")
    
    # Load sample images
    img1 = cv2.imread("sample_images/textured_view1.jpg")
    img2 = cv2.imread("sample_images/textured_view2.jpg")
    
    if img1 is None or img2 is None:
        print("Sample images not found")
        return
    
    # Create camera matrix
    camera_matrix = create_default_camera_matrix(img1.shape[:2])
    
    # Feature detection and matching
    detector = FeatureDetector('SIFT')
    matcher = FeatureMatcher(detector)
    
    pts1, pts2, matches = matcher.match_image_pair(img1, img2)
    print(f"Found {len(matches)} matches")
    
    # Filter matches
    pts1_filtered, pts2_filtered, mask = matcher.filter_matches_with_fundamental(pts1, pts2)
    print(f"After filtering: {len(pts1_filtered)} matches")
    print(f"pts1_filtered shape: {pts1_filtered.shape}")
    print(f"pts1_filtered sample: {pts1_filtered[:2]}")
    
    # 3D reconstruction
    reconstructor = StereoReconstructor(camera_matrix)
    points_3d, R, t = reconstructor.reconstruct_from_stereo_pair(
        img1, img2, pts1_filtered, pts2_filtered
    )
    
    print(f"Reconstructed {len(points_3d)} 3D points")
    print(f"Points 3D shape: {points_3d.shape}, dtype: {points_3d.dtype}")
    print(f"Sample points: {points_3d[:3]}")
    
    # Test point cloud processing step by step
    processor = PointCloudProcessor()
    
    print("\n--- Testing color extraction ---")
    try:
        colors = processor.extract_colors_from_image(img1, pts1_filtered)
        print(f"✓ Color extraction successful: shape {colors.shape}")
    except Exception as e:
        print(f"✗ Color extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n--- Testing point cloud creation ---")
    try:
        pcd = processor.visualizer.create_point_cloud(points_3d, colors)
        print(f"✓ Point cloud creation successful: {len(pcd.points)} points")
    except Exception as e:
        print(f"✗ Point cloud creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n--- Testing outlier filtering ---")
    try:
        pcd_filtered = processor.visualizer.filter_outliers(pcd)
        print(f"✓ Outlier filtering successful: {len(pcd_filtered.points)} points")
    except Exception as e:
        print(f"✗ Outlier filtering failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n--- Testing visualization ---")
    try:
        output_dir = tempfile.mkdtemp()
        viz_path = processor.create_and_save_visualization(
            points_3d, img1, pts1_filtered, output_dir
        )
        print(f"✓ Visualization successful: {viz_path}")
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ All tests passed!")

if __name__ == '__main__':
    debug_reconstruction()