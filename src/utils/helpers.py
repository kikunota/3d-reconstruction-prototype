import cv2
import numpy as np
import os
from typing import List, Tuple

def load_images_from_folder(folder_path: str) -> List[Tuple[str, np.ndarray]]:
    """
    Load all images from a folder
    """
    images = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist")
        return images
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append((filename, img))
    
    return images

def resize_image(image: np.ndarray, max_size: int = 800) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    
    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)
    
    return cv2.resize(image, (new_w, new_h))

def create_default_camera_matrix(image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a default camera matrix when calibration is not available
    """
    h, w = image_shape
    focal_length = max(w, h)  # Rough estimate
    cx, cy = w / 2, h / 2
    
    camera_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return camera_matrix

def ensure_directory_exists(directory: str):
    """
    Create directory if it doesn't exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def draw_keypoints_on_image(image: np.ndarray, keypoints: List[cv2.KeyPoint]) -> np.ndarray:
    """
    Draw keypoints on image for visualization
    """
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, 
                                          color=(0, 255, 0), 
                                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img_with_keypoints