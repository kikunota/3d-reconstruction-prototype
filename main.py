#!/usr/bin/env python3
"""
3D Reconstruction using Structure from Motion (SfM)
Main script for command-line usage
"""

import sys
import os
import argparse
import cv2
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.camera import CameraCalibrator
from core.features import FeatureDetector, FeatureMatcher
from core.geometry import StereoReconstructor
from visualization.pointcloud import PointCloudProcessor
from utils.helpers import load_images_from_folder, resize_image, create_default_camera_matrix

def main():
    parser = argparse.ArgumentParser(description='3D Reconstruction using Structure from Motion')
    parser.add_argument('--images', '-i', required=True, help='Path to folder containing images')
    parser.add_argument('--calibration', '-c', help='Path to camera calibration file (.npz)')
    parser.add_argument('--output', '-o', default='output', help='Output directory for results')
    parser.add_argument('--feature-type', choices=['SIFT', 'ORB'], default='SIFT', help='Feature detector type')
    parser.add_argument('--max-size', type=int, default=800, help='Maximum image size for processing')
    parser.add_argument('--visualize', action='store_true', help='Show 3D visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load images
    print(f"Loading images from {args.images}")
    images = load_images_from_folder(args.images)
    
    if len(images) < 2:
        print("Error: Need at least 2 images for reconstruction")
        return
    
    print(f"Found {len(images)} images")
    
    # Take first two images for stereo reconstruction
    img1_name, img1 = images[0]
    img2_name, img2 = images[1]
    
    print(f"Using images: {img1_name} and {img2_name}")
    
    # Resize images for faster processing
    img1 = resize_image(img1, args.max_size)
    img2 = resize_image(img2, args.max_size)
    
    # Camera calibration
    calibrator = CameraCalibrator()
    if args.calibration and os.path.exists(args.calibration):
        print(f"Loading camera calibration from {args.calibration}")
        calibrator.load_calibration(args.calibration)
        camera_matrix, dist_coeffs = calibrator.get_camera_params()
    else:
        print("Using default camera matrix (no calibration provided)")
        camera_matrix = create_default_camera_matrix(img1.shape[:2])
        dist_coeffs = None
    
    print(f"Camera matrix:\n{camera_matrix}")
    
    # Feature detection and matching
    print(f"Detecting features using {args.feature_type}")
    detector = FeatureDetector(args.feature_type)
    matcher = FeatureMatcher(detector)
    
    pts1, pts2, matches = matcher.match_image_pair(img1, img2)
    print(f"Found {len(matches)} initial matches")
    
    if len(matches) < 20:
        print("Warning: Very few matches found. Results may be poor.")
    
    # Filter matches using fundamental matrix
    pts1_filtered, pts2_filtered, mask = matcher.filter_matches_with_fundamental(pts1, pts2)
    print(f"After filtering: {len(pts1_filtered)} good matches")
    
    # Save matches visualization
    kp1, _ = detector.detect_and_compute(img1)
    kp2, _ = detector.detect_and_compute(img2)
    matches_img = detector.visualize_matches(img1, kp1, img2, kp2, matches[:100])
    matches_path = os.path.join(args.output, 'feature_matches.jpg')
    cv2.imwrite(matches_path, matches_img)
    print(f"Feature matches saved to {matches_path}")
    
    # 3D reconstruction
    print("Starting 3D reconstruction...")
    reconstructor = StereoReconstructor(camera_matrix)
    
    try:
        points_3d, R, t = reconstructor.reconstruct_from_stereo_pair(
            img1, img2, pts1_filtered, pts2_filtered
        )
        
        print(f"Reconstructed {len(points_3d)} 3D points")
        print(f"Mean depth: {np.mean(points_3d[:, 2]):.2f}")
        print(f"Depth range: {np.min(points_3d[:, 2]):.2f} to {np.max(points_3d[:, 2]):.2f}")
        
        # Process and save point cloud
        processor = PointCloudProcessor()
        pcd = processor.process_reconstruction_result(
            points_3d, img1, pts1_filtered,
            output_file=os.path.join(args.output, 'reconstruction.ply')
        )
        
        print(f"Point cloud saved to {os.path.join(args.output, 'reconstruction.ply')}")
        
        # Create visualization image
        viz_path = processor.create_and_save_visualization(
            points_3d, img1, pts1_filtered, args.output
        )
        print(f"Visualization saved to {viz_path}")
        
        # Show interactive visualization if requested
        if args.visualize:
            print("Showing 3D visualization...")
            try:
                processor.visualizer.visualize_point_cloud(pcd)
            except Exception as e:
                print(f"Could not show visualization: {e}")
        
        print("Reconstruction completed successfully!")
        print(f"Results saved in {args.output}/")
        
    except Exception as e:
        print(f"Reconstruction failed: {e}")
        return

if __name__ == '__main__':
    main()