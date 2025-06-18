import cv2
import numpy as np
import os
from typing import Tuple, List, Optional

class CameraCalibrator:
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_error = None
        
    def calibrate_camera(self, images_folder: str, pattern_size: Tuple[int, int] = (9, 6)) -> bool:
        """
        Calibrate camera using chessboard pattern images
        """
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        
        objpoints = []
        imgpoints = []
        
        images = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(images) == 0:
            print("No images found for calibration")
            return False
            
        img_shape = None
        
        for fname in images:
            img_path = os.path.join(images_folder, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if img_shape is None:
                img_shape = gray.shape[::-1]
            
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
        
        if len(objpoints) < 5:
            print(f"Not enough valid images for calibration. Found {len(objpoints)}, need at least 5")
            return False
            
        ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None
        )
        
        self.calibration_error = ret
        print(f"Camera calibration completed with RMS error: {ret:.4f}")
        return True
    
    def load_calibration(self, calibration_file: str) -> bool:
        """Load camera calibration from file"""
        try:
            data = np.load(calibration_file)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.calibration_error = float(data['calibration_error'])
            return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def save_calibration(self, calibration_file: str):
        """Save camera calibration to file"""
        if self.camera_matrix is not None:
            np.savez(calibration_file,
                    camera_matrix=self.camera_matrix,
                    dist_coeffs=self.dist_coeffs,
                    calibration_error=self.calibration_error)
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Remove distortion from image"""
        if self.camera_matrix is None or self.dist_coeffs is None:
            return image
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
    
    def get_camera_params(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get camera matrix and distortion coefficients"""
        return self.camera_matrix, self.dist_coeffs