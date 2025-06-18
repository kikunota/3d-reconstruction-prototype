import cv2
import numpy as np
from typing import Tuple, List, Optional

class FeatureDetector:
    def __init__(self, feature_type: str = 'SIFT'):
        """
        Initialize feature detector
        Args:
            feature_type: 'SIFT' or 'ORB'
        """
        self.feature_type = feature_type
        if feature_type.upper() == 'SIFT':
            # Increase SIFT features for better reconstruction
            self.detector = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.03, edgeThreshold=8)
        elif feature_type.upper() == 'ORB':
            # Increase ORB features significantly
            self.detector = cv2.ORB_create(nfeatures=15000, scaleFactor=1.2, nlevels=12)
        else:
            raise ValueError("Feature type must be 'SIFT' or 'ORB'")
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detect keypoints and compute descriptors
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if descriptors is None:
            descriptors = np.array([])
        
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, 
                      ratio_threshold: float = 0.75) -> List[cv2.DMatch]:
        """
        Match features between two images using ratio test
        """
        if desc1.size == 0 or desc2.size == 0:
            return []
        
        if self.feature_type.upper() == 'SIFT':
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        matches = matcher.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def visualize_matches(self, img1: np.ndarray, kp1: List[cv2.KeyPoint],
                         img2: np.ndarray, kp2: List[cv2.KeyPoint],
                         matches: List[cv2.DMatch]) -> np.ndarray:
        """
        Create visualization of feature matches
        """
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img_matches

class FeatureMatcher:
    def __init__(self, detector: FeatureDetector):
        self.detector = detector
    
    def match_image_pair(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[cv2.DMatch]]:
        """
        Match features between two images and return corresponding points
        """
        kp1, desc1 = self.detector.detect_and_compute(img1)
        kp2, desc2 = self.detector.detect_and_compute(img2)
        
        matches = self.detector.match_features(desc1, desc2)
        
        if len(matches) < 8:
            print(f"Warning: Only {len(matches)} matches found, need at least 8 for robust estimation")
            return np.array([]), np.array([]), matches
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        return pts1, pts2, matches
    
    def filter_matches_with_fundamental(self, pts1: np.ndarray, pts2: np.ndarray,
                                      threshold: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter matches using fundamental matrix and RANSAC
        """
        if len(pts1) < 8:
            return pts1, pts2, np.ones(len(pts1), dtype=bool)
        
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, threshold, 0.99, maxIters=5000)
        
        if mask is None:
            mask = np.ones(len(pts1), dtype=bool)
        else:
            mask = mask.ravel().astype(bool)
        
        return pts1[mask], pts2[mask], mask
    
    def find_best_image_pairs(self, images: List[np.ndarray], min_matches: int = 50) -> List[Tuple[int, int, int]]:
        """
        Find the best image pairs for reconstruction based on feature matches
        Returns: List of (img1_idx, img2_idx, num_matches) sorted by match count
        """
        pairs = []
        
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                try:
                    pts1, pts2, matches = self.match_image_pair(images[i], images[j])
                    if len(matches) >= min_matches:
                        pairs.append((i, j, len(matches)))
                        print(f"Pair {i}-{j}: {len(matches)} matches")
                except Exception as e:
                    print(f"Failed to match pair {i}-{j}: {e}")
                    continue
        
        # Sort by number of matches (descending)
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs