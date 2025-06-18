import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings


@dataclass
class CameraParameters:
    """Camera intrinsic and extrinsic parameters"""
    # Intrinsics
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0  # Radial distortion coefficient 1
    k2: float = 0.0  # Radial distortion coefficient 2
    p1: float = 0.0  # Tangential distortion coefficient 1
    p2: float = 0.0  # Tangential distortion coefficient 2
    k3: float = 0.0  # Radial distortion coefficient 3
    
    # Extrinsics (rotation vector and translation)
    rvec: np.ndarray = None  # 3x1 rotation vector
    tvec: np.ndarray = None  # 3x1 translation vector
    
    def __post_init__(self):
        if self.rvec is None:
            self.rvec = np.zeros(3)
        if self.tvec is None:
            self.tvec = np.zeros(3)
    
    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """Get 3x3 intrinsic matrix"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    @property
    def distortion_coeffs(self) -> np.ndarray:
        """Get distortion coefficients in OpenCV format"""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])


@dataclass
class Observation:
    """2D observation of a 3D point in a camera"""
    camera_id: int
    point_id: int
    x: float
    y: float
    weight: float = 1.0


class CameraModel:
    """Different camera projection models"""
    
    @staticmethod
    def project_pinhole(point_3d: np.ndarray, camera: CameraParameters) -> np.ndarray:
        """Project 3D point using pinhole camera model"""
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(camera.rvec)
        
        # Transform point to camera coordinates
        point_cam = R @ point_3d + camera.tvec
        
        # Project to image plane
        if point_cam[2] <= 0:
            return np.array([float('inf'), float('inf')])
        
        x = point_cam[0] / point_cam[2]
        y = point_cam[1] / point_cam[2]
        
        # Apply intrinsics
        u = camera.fx * x + camera.cx
        v = camera.fy * y + camera.cy
        
        return np.array([u, v])
    
    @staticmethod
    def project_with_distortion(point_3d: np.ndarray, camera: CameraParameters) -> np.ndarray:
        """Project 3D point with radial and tangential distortion"""
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(camera.rvec)
        
        # Transform point to camera coordinates
        point_cam = R @ point_3d + camera.tvec
        
        # Project to normalized image coordinates
        if point_cam[2] <= 0:
            return np.array([float('inf'), float('inf')])
        
        x = point_cam[0] / point_cam[2]
        y = point_cam[1] / point_cam[2]
        
        # Apply distortion
        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r4*r2
        
        # Radial distortion
        radial = 1 + camera.k1*r2 + camera.k2*r4 + camera.k3*r6
        
        # Tangential distortion
        tangential_x = 2*camera.p1*x*y + camera.p2*(r2 + 2*x*x)
        tangential_y = camera.p1*(r2 + 2*y*y) + 2*camera.p2*x*y
        
        # Apply distortion
        x_distorted = x * radial + tangential_x
        y_distorted = y * radial + tangential_y
        
        # Apply intrinsics
        u = camera.fx * x_distorted + camera.cx
        v = camera.fy * y_distorted + camera.cy
        
        return np.array([u, v])


class BundleAdjuster:
    """Robust bundle adjustment with outlier rejection"""
    
    def __init__(self, use_distortion: bool = True, outlier_threshold: float = 2.0):
        self.use_distortion = use_distortion
        self.outlier_threshold = outlier_threshold
        self.cameras: List[CameraParameters] = []
        self.points_3d: np.ndarray = None
        self.observations: List[Observation] = []
        self.optimization_history = []
        
    def add_camera(self, camera: CameraParameters) -> int:
        """Add a camera and return its ID"""
        camera_id = len(self.cameras)
        self.cameras.append(camera)
        return camera_id
    
    def set_points_3d(self, points_3d: np.ndarray):
        """Set the 3D points to optimize"""
        self.points_3d = points_3d.copy()
    
    def add_observation(self, camera_id: int, point_id: int, x: float, y: float, weight: float = 1.0):
        """Add a 2D observation"""
        self.observations.append(Observation(camera_id, point_id, x, y, weight))
    
    def project_point(self, point_3d: np.ndarray, camera: CameraParameters) -> np.ndarray:
        """Project 3D point using selected camera model"""
        if self.use_distortion:
            return CameraModel.project_with_distortion(point_3d, camera)
        else:
            return CameraModel.project_pinhole(point_3d, camera)
    
    def compute_residuals(self, params: np.ndarray) -> np.ndarray:
        """Compute reprojection residuals"""
        try:
            # Unpack parameters
            cameras, points_3d = self._unpack_parameters(params)
        except Exception as e:
            print(f"Warning: Parameter unpacking failed: {e}")
            # Return large residuals if unpacking fails
            return np.full(len(self.observations) * 2, 1000.0)
        
        residuals = []
        
        for obs in self.observations:
            if obs.point_id >= len(points_3d) or obs.camera_id >= len(cameras):
                residuals.extend([1000.0, 1000.0])  # Large error for invalid indices
                continue
                
            try:
                # Project 3D point
                projected = self.project_point(points_3d[obs.point_id], cameras[obs.camera_id])
                
                # Skip invalid projections
                if np.any(np.isinf(projected)) or np.any(np.isnan(projected)):
                    residuals.extend([1000.0, 1000.0])  # Large error for invalid projections
                    continue
                
                # Compute residual
                dx = projected[0] - obs.x
                dy = projected[1] - obs.y
                
                # Check for NaN/inf in residuals
                if np.isnan(dx) or np.isnan(dy) or np.isinf(dx) or np.isinf(dy):
                    residuals.extend([1000.0, 1000.0])
                    continue
                
                # Apply weight
                residuals.extend([dx * obs.weight, dy * obs.weight])
                
            except Exception as e:
                # Handle any projection errors
                residuals.extend([1000.0, 1000.0])
                continue
        
        return np.array(residuals)
    
    def compute_jacobian_sparsity(self) -> lil_matrix:
        """Compute sparsity pattern of Jacobian matrix"""
        n_cameras = len(self.cameras)
        n_points = len(self.points_3d) if self.points_3d is not None else 0
        n_residuals = len(self.observations) * 2
        
        # Camera parameters: 6 (pose) + intrinsics + distortion
        camera_param_size = 9 if self.use_distortion else 4  # fx, fy, cx, cy + distortion
        camera_param_size += 6  # rvec + tvec
        
        n_params = n_cameras * camera_param_size + n_points * 3
        
        # Create sparsity pattern
        sparsity = lil_matrix((n_residuals, n_params), dtype=int)
        
        residual_idx = 0
        for obs in self.observations:
            # Camera parameters affect this residual
            camera_start = obs.camera_id * camera_param_size
            camera_end = camera_start + camera_param_size
            sparsity[residual_idx:residual_idx+2, camera_start:camera_end] = 1
            
            # 3D point affects this residual
            point_start = n_cameras * camera_param_size + obs.point_id * 3
            point_end = point_start + 3
            sparsity[residual_idx:residual_idx+2, point_start:point_end] = 1
            
            residual_idx += 2
        
        return sparsity
    
    def _pack_parameters(self) -> np.ndarray:
        """Pack camera and 3D point parameters into a single vector"""
        params = []
        
        # Pack camera parameters
        for camera in self.cameras:
            params.extend(camera.rvec)
            params.extend(camera.tvec)
            params.extend([camera.fx, camera.fy, camera.cx, camera.cy])
            if self.use_distortion:
                params.extend([camera.k1, camera.k2, camera.p1, camera.p2, camera.k3])
        
        # Pack 3D points
        if self.points_3d is not None:
            params.extend(self.points_3d.flatten())
        
        return np.array(params)
    
    def _unpack_parameters(self, params: np.ndarray) -> Tuple[List[CameraParameters], np.ndarray]:
        """Unpack parameter vector into cameras and 3D points"""
        cameras = []
        param_idx = 0
        
        # Calculate expected parameter size
        camera_param_size = 10 + (5 if self.use_distortion else 0)
        expected_camera_params = len(self.cameras) * camera_param_size
        
        if len(params) < expected_camera_params:
            raise ValueError(f"Not enough parameters: expected at least {expected_camera_params}, got {len(params)}")
        
        # Unpack cameras
        for i in range(len(self.cameras)):
            if param_idx + camera_param_size > len(params):
                raise ValueError(f"Parameter array too short for camera {i}")
                
            rvec = params[param_idx:param_idx+3]
            param_idx += 3
            tvec = params[param_idx:param_idx+3]
            param_idx += 3
            fx, fy, cx, cy = params[param_idx:param_idx+4]
            param_idx += 4
            
            # Ensure reasonable focal lengths
            fx = max(abs(fx), 1.0)
            fy = max(abs(fy), 1.0)
            
            if self.use_distortion:
                k1, k2, p1, p2, k3 = params[param_idx:param_idx+5]
                param_idx += 5
                # Clamp distortion coefficients to reasonable ranges
                k1 = np.clip(k1, -1.0, 1.0)
                k2 = np.clip(k2, -1.0, 1.0)
                k3 = np.clip(k3, -1.0, 1.0)
                p1 = np.clip(p1, -0.1, 0.1)
                p2 = np.clip(p2, -0.1, 0.1)
            else:
                k1 = k2 = p1 = p2 = k3 = 0.0
            
            camera = CameraParameters(fx, fy, cx, cy, k1, k2, p1, p2, k3, rvec.copy(), tvec.copy())
            cameras.append(camera)
        
        # Unpack 3D points
        remaining_params = params[param_idx:]
        if len(remaining_params) % 3 != 0:
            raise ValueError(f"Invalid number of point parameters: {len(remaining_params)} (must be multiple of 3)")
        
        n_points = len(remaining_params) // 3
        if n_points == 0:
            points_3d = np.empty((0, 3))
        else:
            points_3d = remaining_params.reshape(n_points, 3)
        
        return cameras, points_3d
    
    def robust_loss(self, residuals: np.ndarray, loss_type: str = 'huber') -> np.ndarray:
        """Apply robust loss function for outlier rejection"""
        if loss_type == 'huber':
            delta = self.outlier_threshold
            abs_residuals = np.abs(residuals)
            mask = abs_residuals <= delta
            robust_residuals = residuals.copy()
            robust_residuals[~mask] = delta * np.sign(residuals[~mask]) * (1 + np.log(abs_residuals[~mask] / delta))
            return robust_residuals
        elif loss_type == 'cauchy':
            return residuals / (1 + (residuals / self.outlier_threshold)**2)
        else:
            return residuals  # No robust loss
    
    def optimize(self, max_iterations: int = 100, convergence_threshold: float = 1e-6, 
                verbose: bool = True) -> Dict[str, Any]:
        """Perform bundle adjustment optimization"""
        if not self.cameras or self.points_3d is None or not self.observations:
            raise ValueError("Bundle adjustment requires cameras, 3D points, and observations")
        
        # Pack initial parameters
        initial_params = self._pack_parameters()
        
        # Define residual function with robust loss
        def residual_func(params):
            residuals = self.compute_residuals(params)
            return self.robust_loss(residuals)
        
        # Choose optimization method based on problem size
        use_sparse = len(self.observations) > 1000
        
        # Enhanced progress tracking with timing
        import time
        start_time = time.time()
        iteration_times = []
        
        def callback(xk, *args):
            current_time = time.time()
            iteration_times.append(current_time)
            
            residuals = self.compute_residuals(xk)
            rms_error = np.sqrt(np.mean(residuals**2))
            self.optimization_history.append(rms_error)
            
            if verbose:
                iteration = len(self.optimization_history)
                elapsed = current_time - start_time
                
                if iteration == 1:
                    print(f"  Iteration {iteration:3d}: RMS error = {rms_error:.4f} pixels")
                elif iteration % 5 == 0 or iteration <= 10:
                    # Calculate iteration rate
                    if len(iteration_times) > 1:
                        recent_rate = 5.0 / (iteration_times[-1] - iteration_times[-6]) if len(iteration_times) >= 6 else iteration / elapsed
                    else:
                        recent_rate = iteration / elapsed
                    
                    improvement = ((self.optimization_history[0] - rms_error) / self.optimization_history[0] * 100) if self.optimization_history[0] > 0 else 0
                    print(f"  Iteration {iteration:3d}: RMS error = {rms_error:.4f} pixels | "
                          f"Rate: {recent_rate:.1f} it/s | Improvement: {improvement:.1f}% | "
                          f"Elapsed: {elapsed:.1f}s")
        
        # Run optimization with progress reporting
        if verbose:
            print("Starting bundle adjustment optimization...")
            print(f"  Method: {'Trust Region Reflective (sparse)' if use_sparse else 'Levenberg-Marquardt'}")
            print(f"  Parameters: {len(initial_params):,}")
            print(f"  Observations: {len(self.observations):,}")
            
        if use_sparse:
            # Use trust region method with sparse Jacobian for large problems
            if verbose:
                print("  Computing sparsity pattern...")
            jac_sparsity = self.compute_jacobian_sparsity()
            if verbose:
                print(f"  Sparsity: {jac_sparsity.nnz:,} non-zeros out of {jac_sparsity.shape[0] * jac_sparsity.shape[1]:,} ({100*jac_sparsity.nnz/(jac_sparsity.shape[0] * jac_sparsity.shape[1]):.2f}%)")
                
            result = least_squares(
                residual_func,
                initial_params,
                jac_sparsity=jac_sparsity,
                max_nfev=max_iterations * len(initial_params),
                ftol=convergence_threshold,
                xtol=convergence_threshold,
                method='trf',  # Trust Region Reflective
                verbose=0
            )
        else:
            # Use Levenberg-Marquardt for smaller problems (no sparse support)
            result = least_squares(
                residual_func,
                initial_params,
                max_nfev=max_iterations * len(initial_params),
                ftol=convergence_threshold,
                xtol=convergence_threshold,
                method='lm',  # Levenberg-Marquardt
                verbose=0
            )
        
        # Unpack optimized parameters
        self.cameras, self.points_3d = self._unpack_parameters(result.x)
        
        # Compute final statistics
        final_residuals = self.compute_residuals(result.x)
        rms_error = np.sqrt(np.mean(final_residuals**2))
        max_error = np.max(np.abs(final_residuals))
        
        # Identify outliers
        outlier_mask = np.abs(final_residuals) > self.outlier_threshold
        n_outliers = np.sum(outlier_mask) // 2  # Each observation has 2 residuals
        
        optimization_result = {
            'success': result.success,
            'rms_error': rms_error,
            'max_error': max_error,
            'n_iterations': result.nfev,
            'n_outliers': n_outliers,
            'outlier_ratio': n_outliers / len(self.observations),
            'cost_reduction': (self.optimization_history[0] - rms_error) / self.optimization_history[0] if self.optimization_history else 0,
            'message': result.message
        }
        
        if verbose:
            print(f"Bundle adjustment completed:")
            print(f"  Success: {result.success}")
            print(f"  RMS error: {rms_error:.4f} pixels")
            print(f"  Max error: {max_error:.4f} pixels")
            print(f"  Iterations: {result.nfev}")
            print(f"  Outliers: {n_outliers}/{len(self.observations)} ({optimization_result['outlier_ratio']:.1%})")
            print(f"  Cost reduction: {optimization_result['cost_reduction']:.1%}")
        
        return optimization_result
    
    def iterative_refinement(self, n_iterations: int = 3, outlier_percentile: float = 95) -> Dict[str, Any]:
        """Iterative bundle adjustment with progressive outlier removal"""
        import time
        
        results = []
        total_start_time = time.time()
        
        print(f"\n🔄 ITERATIVE BUNDLE ADJUSTMENT ({n_iterations} iterations)")
        print("=" * 60)
        
        for iteration in range(n_iterations):
            iteration_start_time = time.time()
            print(f"\n📊 ITERATION {iteration + 1}/{n_iterations}")
            print("-" * 40)
            print(f"Current observations: {len(self.observations):,}")
            print(f"Current 3D points: {len(self.points_3d):,}")
            print(f"Current cameras: {len(self.cameras)}")
            
            # Optimize
            print("\n🎯 Optimizing...")
            result = self.optimize(verbose=True)
            results.append(result)
            
            iteration_time = time.time() - iteration_start_time
            print(f"\n✅ Iteration {iteration + 1} completed in {iteration_time:.1f}s")
            print(f"   RMS Error: {result['rms_error']:.4f} pixels")
            print(f"   Max Error: {result['max_error']:.4f} pixels")
            print(f"   Outliers: {result['n_outliers']}/{len(self.observations)} ({result['outlier_ratio']:.1%})")
            
            if iteration < n_iterations - 1:  # Don't remove outliers on last iteration
                print(f"\n🧹 Removing outliers (keeping {outlier_percentile}% best observations)...")
                
                # Compute residuals and remove worst outliers
                residuals = self.compute_residuals(self._pack_parameters())
                residual_magnitudes = np.sqrt(residuals[::2]**2 + residuals[1::2]**2)
                
                threshold = np.percentile(residual_magnitudes, outlier_percentile)
                good_observations = []
                
                for i, obs in enumerate(self.observations):
                    if i < len(residual_magnitudes) and residual_magnitudes[i] <= threshold:
                        good_observations.append(obs)
                
                removed = len(self.observations) - len(good_observations)
                removal_percentage = removed / len(self.observations) * 100 if self.observations else 0
                
                print(f"   Removed: {removed:,} observations ({removal_percentage:.1f}%)")
                print(f"   Threshold: {threshold:.2f} pixels")
                print(f"   Remaining: {len(good_observations):,} observations")
                
                self.observations = good_observations
                
                if not self.observations:
                    print("⚠️  WARNING: All observations removed as outliers!")
                    break
                elif len(good_observations) < 10:
                    print("⚠️  WARNING: Very few observations remaining, stopping refinement")
                    break
        
        total_time = time.time() - total_start_time
        
        print(f"\n🎉 BUNDLE ADJUSTMENT COMPLETE")
        print("=" * 60)
        print(f"Total time: {total_time:.1f}s")
        
        if len(results) > 1:
            initial_error = results[0]['rms_error']
            final_error = results[-1]['rms_error']
            improvement_ratio = initial_error / final_error
            improvement_percent = (1 - final_error / initial_error) * 100
            
            print(f"Initial RMS error: {initial_error:.4f} pixels")
            print(f"Final RMS error: {final_error:.4f} pixels")
            print(f"Improvement: {improvement_ratio:.2f}x ({improvement_percent:.1f}% reduction)")
        
        return {
            'iterations': results,
            'final_result': results[-1] if results else None,
            'improvement': results[-1]['rms_error'] / results[0]['rms_error'] if len(results) > 1 else 1.0,
            'total_time': total_time,
            'n_iterations_completed': len(results)
        }
    
    def get_reprojection_errors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get reprojection errors for each observation"""
        residuals = self.compute_residuals(self._pack_parameters())
        errors_x = residuals[::2]
        errors_y = residuals[1::2]
        error_magnitudes = np.sqrt(errors_x**2 + errors_y**2)
        
        return error_magnitudes, residuals
    
    def get_camera_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get camera positions and orientations"""
        positions = []
        orientations = []
        
        for camera in self.cameras:
            # Camera position in world coordinates
            R, _ = cv2.Rodrigues(camera.rvec)
            position = -R.T @ camera.tvec
            positions.append(position)
            orientations.append(camera.rvec)
        
        return np.array(positions), np.array(orientations)