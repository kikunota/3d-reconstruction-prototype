import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
import tempfile
import os

from .bundle_adjustment import BundleAdjuster, CameraParameters, Observation
from .geometry import StereoReconstructor, GeometryEstimator
from .features import FeatureMatcher
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from visualization.bundle_adjustment_viz import BundleAdjustmentVisualizer


class EnhancedReconstructor:
    """Enhanced reconstruction pipeline with bundle adjustment"""
    
    def __init__(self, camera_matrix: np.ndarray, use_bundle_adjustment: bool = True,
                 use_distortion: bool = True, outlier_threshold: float = 2.0):
        self.camera_matrix = camera_matrix
        self.use_bundle_adjustment = use_bundle_adjustment
        self.use_distortion = use_distortion
        self.outlier_threshold = outlier_threshold
        
        # Components
        self.stereo_reconstructor = StereoReconstructor(camera_matrix)
        self.bundle_adjuster = None
        self.visualizer = BundleAdjustmentVisualizer()
        
        # Data storage
        self.cameras = []
        self.points_3d_global = []
        self.observations = []
        self.image_data = []
        self.reconstruction_graph = {}  # Track which points come from which image pairs
        
    def add_image_pair_reconstruction(self, img1: np.ndarray, img2: np.ndarray,
                                    pts1: np.ndarray, pts2: np.ndarray,
                                    camera1_id: Optional[int] = None,
                                    camera2_id: Optional[int] = None) -> Dict[str, Any]:
        """Add reconstruction from an image pair"""
        
        # Perform initial stereo reconstruction
        try:
            points_3d, R, t = self.stereo_reconstructor.reconstruct_from_stereo_pair(
                img1, img2, pts1, pts2
            )
        except Exception as e:
            return {'success': False, 'error': str(e)}
        
        # Create camera parameters
        if camera1_id is None:
            camera1_id = len(self.cameras)
            camera1 = self._create_camera_from_matrix(self.camera_matrix)
            self.cameras.append(camera1)
        
        if camera2_id is None:
            camera2_id = len(self.cameras)
            camera2 = self._create_camera_from_matrix(self.camera_matrix, R, t)
            self.cameras.append(camera2)
        
        # Add 3D points to global list
        point_start_id = len(self.points_3d_global)
        self.points_3d_global.extend(points_3d)
        
        # Create observations for bundle adjustment
        valid_pts1, valid_pts2 = self._filter_valid_correspondences(pts1, pts2, points_3d)
        
        for i, (pt1, pt2) in enumerate(zip(valid_pts1, valid_pts2)):
            point_id = point_start_id + i
            
            # Add observations for both cameras
            self.observations.append(Observation(camera1_id, point_id, pt1[0], pt1[1]))
            self.observations.append(Observation(camera2_id, point_id, pt2[0], pt2[1]))
        
        # Store image data
        self.image_data.append({
            'camera1_id': camera1_id,
            'camera2_id': camera2_id,
            'img1': img1,
            'img2': img2,
            'points_3d': points_3d,
            'pts1': valid_pts1,
            'pts2': valid_pts2
        })
        
        # Track reconstruction graph
        pair_key = f"{camera1_id}-{camera2_id}"
        self.reconstruction_graph[pair_key] = {
            'point_range': (point_start_id, point_start_id + len(points_3d)),
            'n_points': len(points_3d)
        }
        
        return {
            'success': True,
            'n_points_3d': len(points_3d),
            'camera1_id': camera1_id,
            'camera2_id': camera2_id,
            'total_points': len(self.points_3d_global),
            'total_cameras': len(self.cameras)
        }
    
    def _create_camera_from_matrix(self, K: np.ndarray, R: Optional[np.ndarray] = None, 
                                  t: Optional[np.ndarray] = None) -> CameraParameters:
        """Create CameraParameters from intrinsic matrix and pose"""
        if R is None:
            R = np.eye(3)
        if t is None:
            t = np.zeros(3)
        
        rvec, _ = cv2.Rodrigues(R)
        
        return CameraParameters(
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
            rvec=rvec.flatten(), tvec=t.flatten()
        )
    
    def _filter_valid_correspondences(self, pts1: np.ndarray, pts2: np.ndarray, 
                                    points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter correspondences to match valid 3D points"""
        if len(pts1) != len(points_3d):
            # Take first N correspondences matching number of 3D points
            n_points = min(len(pts1), len(points_3d))
            return pts1[:n_points].reshape(-1, 2), pts2[:n_points].reshape(-1, 2)
        
        return pts1.reshape(-1, 2), pts2.reshape(-1, 2)
    
    def perform_bundle_adjustment(self, max_iterations: int = 100, 
                                n_refinement_iterations: int = 3,
                                verbose: bool = True) -> Dict[str, Any]:
        """Perform bundle adjustment on all reconstructions"""
        
        if not self.use_bundle_adjustment:
            return {'success': False, 'message': 'Bundle adjustment disabled'}
        
        if not self.cameras or not self.points_3d_global or not self.observations:
            return {'success': False, 'message': 'Insufficient data for bundle adjustment'}
        
        try:
            # Initialize bundle adjuster
            self.bundle_adjuster = BundleAdjuster(
                use_distortion=self.use_distortion,
                outlier_threshold=self.outlier_threshold
            )
            
            # Add cameras
            for camera in self.cameras:
                self.bundle_adjuster.add_camera(camera)
            
            # Set 3D points
            points_3d_array = np.array(self.points_3d_global)
            self.bundle_adjuster.set_points_3d(points_3d_array)
            
            # Add observations
            for obs in self.observations:
                self.bundle_adjuster.add_observation(obs.camera_id, obs.point_id, obs.x, obs.y, obs.weight)
            
            # Get initial reprojection errors for comparison
            initial_errors, _ = self.bundle_adjuster.get_reprojection_errors()
            
            if verbose:
                print(f"\nBundle Adjustment Setup:")
                print(f"  Cameras: {len(self.cameras)}")
                print(f"  3D Points: {len(self.points_3d_global)}")
                print(f"  Observations: {len(self.observations)}")
                print(f"  Initial RMS error: {np.sqrt(np.mean(initial_errors**2)):.4f} pixels")
                print(f"  Use distortion: {self.use_distortion}")
                print(f"  Outlier threshold: {self.outlier_threshold}")
                
                # Estimate processing time
                estimated_time_per_iteration = len(self.observations) * 0.001  # Rough estimate
                estimated_total_time = estimated_time_per_iteration * n_refinement_iterations
                print(f"  Estimated time: ~{estimated_total_time:.1f}s ({n_refinement_iterations} iterations)")
            
            # Perform iterative refinement with progress tracking
            if verbose:
                print(f"\nStarting {n_refinement_iterations}-iteration bundle adjustment...")
                
            refinement_result = self.bundle_adjuster.iterative_refinement(
                n_iterations=n_refinement_iterations
            )
            
            # Update our data with optimized results
            self.cameras = self.bundle_adjuster.cameras.copy()
            self.points_3d_global = self.bundle_adjuster.points_3d.tolist()
            
            # Get final reprojection errors
            final_errors, _ = self.bundle_adjuster.get_reprojection_errors()
            
            result = {
                'success': True,
                'initial_rms_error': np.sqrt(np.mean(initial_errors**2)),
                'final_rms_error': np.sqrt(np.mean(final_errors**2)),
                'improvement_ratio': np.sqrt(np.mean(initial_errors**2)) / np.sqrt(np.mean(final_errors**2)),
                'n_outliers_removed': len(initial_errors) - len(final_errors),
                'refinement_result': refinement_result
            }
            
            if verbose:
                print(f"\nBundle Adjustment Results:")
                print(f"  Final RMS error: {result['final_rms_error']:.4f} pixels")
                print(f"  Improvement: {result['improvement_ratio']:.2f}x")
                print(f"  Outliers removed: {result['n_outliers_removed']}")
            
            return result
            
        except Exception as e:
            print(f"Bundle adjustment failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            
            return {
                'success': False,
                'message': f'Bundle adjustment failed: {str(e)}',
                'error': str(e)
            }
    
    def get_optimized_points_3d(self) -> np.ndarray:
        """Get the optimized 3D points after bundle adjustment"""
        return np.array(self.points_3d_global)
    
    def get_optimized_cameras(self) -> List[CameraParameters]:
        """Get the optimized camera parameters after bundle adjustment"""
        return self.cameras.copy()
    
    def create_detailed_visualizations(self, output_dir: str) -> Dict[str, str]:
        """Create comprehensive visualizations of the reconstruction and bundle adjustment"""
        
        if not self.bundle_adjuster:
            return {'error': 'Bundle adjustment not performed yet'}
        
        visualizations = {}
        
        try:
            # 1. Reprojection error analysis
            error_viz_path = os.path.join(output_dir, 'reprojection_errors.png')
            self.visualizer.plot_reprojection_errors(self.bundle_adjuster, error_viz_path)
            visualizations['reprojection_errors'] = error_viz_path
            
            # 2. Camera trajectory
            trajectory_path = os.path.join(output_dir, 'camera_trajectory.png')
            points_3d = np.array(self.points_3d_global) if self.points_3d_global else None
            self.visualizer.plot_camera_trajectory(self.bundle_adjuster, points_3d, trajectory_path)
            visualizations['camera_trajectory'] = trajectory_path
            
            # 3. Optimization convergence
            convergence_path = os.path.join(output_dir, 'optimization_convergence.png')
            self.visualizer.plot_optimization_convergence(self.bundle_adjuster, convergence_path)
            visualizations['optimization_convergence'] = convergence_path
            
        except Exception as e:
            print(f"Warning: Visualization creation failed: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def get_reconstruction_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the reconstruction"""
        
        def convert_numpy_types(obj):
            """Convert numpy types to Python native types for JSON serialization"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        summary = {
            'n_cameras': int(len(self.cameras)),
            'n_points_3d': int(len(self.points_3d_global)),
            'n_observations': int(len(self.observations)),
            'n_image_pairs': int(len(self.image_data)),
            'reconstruction_graph': convert_numpy_types(self.reconstruction_graph)
        }
        
        if self.bundle_adjuster:
            errors, _ = self.bundle_adjuster.get_reprojection_errors()
            if len(errors) > 0:
                summary.update({
                    'bundle_adjustment': {
                        'performed': True,
                        'rms_error': float(np.sqrt(np.mean(errors**2))),
                        'mean_error': float(np.mean(errors)),
                        'max_error': float(np.max(errors)),
                        'outlier_threshold': float(self.outlier_threshold),
                        'n_outliers': int(np.sum(errors > self.outlier_threshold)),
                        'outlier_ratio': float(np.sum(errors > self.outlier_threshold) / len(errors))
                    }
                })
            else:
                summary['bundle_adjustment'] = {
                    'performed': True,
                    'rms_error': 0.0,
                    'mean_error': 0.0,
                    'max_error': 0.0,
                    'outlier_threshold': float(self.outlier_threshold),
                    'n_outliers': 0,
                    'outlier_ratio': 0.0
                }
            
            # Camera trajectory statistics
            positions, _ = self.bundle_adjuster.get_camera_trajectory()
            if len(positions) > 1:
                distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                summary['camera_trajectory'] = {
                    'total_distance': float(np.sum(distances)),
                    'mean_spacing': float(np.mean(distances)),
                    'position_range': {
                        'x': [float(positions[:, 0].min()), float(positions[:, 0].max())],
                        'y': [float(positions[:, 1].min()), float(positions[:, 1].max())],
                        'z': [float(positions[:, 2].min()), float(positions[:, 2].max())]
                    }
                }
            else:
                summary['camera_trajectory'] = {
                    'total_distance': 0.0,
                    'mean_spacing': 0.0,
                    'position_range': {
                        'x': [0.0, 0.0],
                        'y': [0.0, 0.0],
                        'z': [0.0, 0.0]
                    }
                }
        else:
            summary['bundle_adjustment'] = {'performed': False}
        
        return summary