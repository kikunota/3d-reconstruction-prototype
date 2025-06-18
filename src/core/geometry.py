import cv2
import numpy as np
from typing import Tuple, Optional

class GeometryEstimator:
    def __init__(self):
        pass
    
    def estimate_essential_matrix(self, pts1: np.ndarray, pts2: np.ndarray, 
                                camera_matrix: np.ndarray, 
                                threshold: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate essential matrix between two views
        """
        if len(pts1) < 5:
            raise ValueError("Need at least 5 point correspondences")
        
        E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, 
                                      cv2.RANSAC, 0.999, threshold, maxIters=5000)
        
        if E is None:
            raise ValueError("Could not estimate essential matrix")
            
        return E, mask
    
    def recover_pose(self, essential_matrix: np.ndarray, pts1: np.ndarray, 
                    pts2: np.ndarray, camera_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Recover camera pose from essential matrix
        """
        _, R, t, mask = cv2.recoverPose(essential_matrix, pts1, pts2, camera_matrix)
        return R, t, mask
    
    def triangulate_points(self, pts1: np.ndarray, pts2: np.ndarray,
                          camera_matrix: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points from two views
        """
        P1 = camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = camera_matrix @ np.hstack([R, t])
        
        # Ensure points are in correct format
        pts1_norm = pts1.reshape(-1, 2).T
        pts2_norm = pts2.reshape(-1, 2).T
        
        points_4d = cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm)
        
        # Handle division by zero
        w = points_4d[3]
        valid_mask = np.abs(w) > 1e-8
        
        points_3d = np.zeros((4, points_4d.shape[1]))
        points_3d[:3, valid_mask] = points_4d[:3, valid_mask] / w[valid_mask]
        points_3d[3, :] = 1
        
        # Return only valid points
        valid_points = points_3d[:3, valid_mask].T
        
        if len(valid_points) == 0:
            raise ValueError("No valid 3D points could be triangulated")
        
        return valid_points
    
    def check_triangulation_quality(self, points_3d: np.ndarray, 
                                   camera_matrix: np.ndarray, 
                                   R: np.ndarray, t: np.ndarray,
                                   pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Check quality of triangulated points by reprojection error
        """
        P1 = camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = camera_matrix @ np.hstack([R, t])
        
        points_3d_hom = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        
        proj1 = (P1 @ points_3d_hom.T).T
        proj1 = proj1[:, :2] / proj1[:, 2:3]
        
        proj2 = (P2 @ points_3d_hom.T).T
        proj2 = proj2[:, :2] / proj2[:, 2:3]
        
        error1 = np.sqrt(np.sum((proj1 - pts1.reshape(-1, 2))**2, axis=1))
        error2 = np.sqrt(np.sum((proj2 - pts2.reshape(-1, 2))**2, axis=1))
        
        total_error = error1 + error2
        
        # Safely convert to Python scalar
        if len(total_error) > 0:
            mean_error = float(np.mean(total_error))
        else:
            mean_error = 0.0
        
        return total_error, mean_error
    
    def filter_points_by_depth(self, points_3d: np.ndarray, 
                              max_depth: float = 200.0, 
                              min_depth: float = 0.01) -> np.ndarray:
        """
        Filter 3D points based on reasonable depth values
        """
        depths = points_3d[:, 2]
        valid_mask = (depths > min_depth) & (depths < max_depth)
        return valid_mask

class StereoReconstructor:
    def __init__(self, camera_matrix: np.ndarray):
        self.camera_matrix = camera_matrix
        self.geometry_estimator = GeometryEstimator()
    
    def reconstruct_from_stereo_pair(self, img1: np.ndarray, img2: np.ndarray,
                                   pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full stereo reconstruction pipeline
        """
        print(f"Starting reconstruction with {len(pts1)} point correspondences")
        
        E, mask = self.geometry_estimator.estimate_essential_matrix(
            pts1, pts2, self.camera_matrix
        )
        
        pts1_filtered = pts1[mask.ravel().astype(bool)]
        pts2_filtered = pts2[mask.ravel().astype(bool)]
        
        print(f"After essential matrix filtering: {len(pts1_filtered)} points")
        
        R, t, pose_mask = self.geometry_estimator.recover_pose(
            E, pts1_filtered, pts2_filtered, self.camera_matrix
        )
        
        points_3d = self.geometry_estimator.triangulate_points(
            pts1_filtered, pts2_filtered, self.camera_matrix, R, t
        )
        
        if len(points_3d) == 0:
            raise ValueError("No 3D points could be triangulated")
        
        depth_mask = self.geometry_estimator.filter_points_by_depth(points_3d)
        points_3d_filtered = points_3d[depth_mask]
        
        print(f"After depth filtering: {len(points_3d_filtered)} 3D points")
        
        if len(points_3d_filtered) == 0:
            print("Warning: All points filtered out by depth filter, using unfiltered points")
            points_3d_filtered = points_3d
        
        # Check quality with corresponding filtered points
        if len(points_3d_filtered) == len(points_3d):
            # No depth filtering was applied
            corresponding_pts1 = pts1_filtered
            corresponding_pts2 = pts2_filtered
        else:
            # Depth filtering was applied
            corresponding_pts1 = pts1_filtered[depth_mask]
            corresponding_pts2 = pts2_filtered[depth_mask]
        
        reprojection_errors, mean_error = self.geometry_estimator.check_triangulation_quality(
            points_3d_filtered, self.camera_matrix, R, t,
            corresponding_pts1, corresponding_pts2
        )
        
        print(f"Mean reprojection error: {mean_error:.2f} pixels")
        
        # Filter out points with very high reprojection error
        if len(reprojection_errors) > 0:
            error_threshold = np.percentile(reprojection_errors, 90)  # Keep 90% of points
            good_points_mask = reprojection_errors <= error_threshold
            if np.sum(good_points_mask) > 3:  # Ensure we have some points left
                points_3d_filtered = points_3d_filtered[good_points_mask]
                print(f"After reprojection error filtering: {len(points_3d_filtered)} 3D points")
        
        return points_3d_filtered, R, t