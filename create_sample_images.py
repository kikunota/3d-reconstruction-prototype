#!/usr/bin/env python3
"""
Create sample images for testing 3D reconstruction
This script generates synthetic stereo images of a simple 3D scene
"""

import cv2
import numpy as np
import os

def create_simple_3d_scene():
    """
    Create a simple 3D scene with textured objects
    """
    # Create a canvas
    height, width = 480, 640
    
    # Create two slightly different views of the same scene
    scenes = []
    
    for view in range(2):
        img = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add some textured objects
        # Checkerboard pattern
        for i in range(0, height, 40):
            for j in range(0, width, 40):
                if (i//40 + j//40) % 2 == 0:
                    cv2.rectangle(img, (j, i), (j+40, i+40), (200, 200, 200), -1)
        
        # Add some geometric shapes with different positions for each view
        offset = view * 10  # Small offset for second view
        
        # Red rectangle
        cv2.rectangle(img, (100 + offset, 100), (200 + offset, 200), (0, 0, 255), -1)
        cv2.rectangle(img, (105 + offset, 105), (195 + offset, 195), (100, 100, 255), -1)
        
        # Green circle
        cv2.circle(img, (350 + offset, 150), 50, (0, 255, 0), -1)
        cv2.circle(img, (350 + offset, 150), 30, (100, 255, 100), -1)
        
        # Blue triangle
        pts = np.array([[450 + offset, 80], [400 + offset, 180], [500 + offset, 180]], np.int32)
        cv2.fillPoly(img, [pts], (255, 0, 0))
        pts_inner = np.array([[440 + offset, 100], [420 + offset, 160], [460 + offset, 160]], np.int32)
        cv2.fillPoly(img, [pts_inner], (255, 100, 100))
        
        # Add some random texture points
        np.random.seed(42 + view)  # Different seed for each view
        for _ in range(100):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            cv2.circle(img, (x, y), 2, color, -1)
        
        # Add some text for texture
        cv2.putText(img, f'View {view + 1}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, 'Sample 3D Scene', (50 + offset, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        
        scenes.append(img)
    
    return scenes

def create_real_world_sample():
    """
    Create a more realistic textured pattern
    """
    height, width = 480, 640
    scenes = []
    
    for view in range(2):
        img = np.ones((height, width, 3), dtype=np.uint8) * 250
        
        # Create a brick-like pattern
        brick_h, brick_w = 30, 60
        offset = view * 5
        
        for i in range(0, height, brick_h):
            for j in range(0, width, brick_w):
                # Alternate brick pattern
                x_offset = (brick_w // 2) if (i // brick_h) % 2 else 0
                x = j + x_offset + offset
                
                if x < width - brick_w:
                    # Brick color variation
                    base_color = 180 + np.random.randint(-30, 30)
                    color = (base_color - 50, base_color - 30, base_color)
                    cv2.rectangle(img, (x, i), (x + brick_w - 2, i + brick_h - 2), color, -1)
                    
                    # Add some mortar lines
                    cv2.rectangle(img, (x, i), (x + brick_w, i + brick_h), (200, 200, 200), 2)
        
        # Add some distinctive features
        # Large circle
        cv2.circle(img, (320 + offset, 240), 80, (100, 150, 200), 3)
        cv2.circle(img, (320 + offset, 240), 60, (150, 100, 200), 2)
        
        # Cross pattern
        cv2.line(img, (200 + offset, 100), (200 + offset, 200), (50, 50, 200), 3)
        cv2.line(img, (150 + offset, 150), (250 + offset, 150), (50, 50, 200), 3)
        
        # Add corner markers
        corners = [(50, 50), (width-50, 50), (50, height-50), (width-50, height-50)]
        for i, (x, y) in enumerate(corners):
            cv2.rectangle(img, (x-10+offset, y-10), (x+10+offset, y+10), (255, 0, 255), -1)
            cv2.putText(img, str(i), (x-5+offset, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        scenes.append(img)
    
    return scenes

def main():
    # Create sample_images directory
    os.makedirs('sample_images', exist_ok=True)
    
    print("Creating sample images for 3D reconstruction testing...")
    
    # Create simple synthetic scene
    simple_scenes = create_simple_3d_scene()
    cv2.imwrite('sample_images/simple_view1.jpg', simple_scenes[0])
    cv2.imwrite('sample_images/simple_view2.jpg', simple_scenes[1])
    print("Created simple synthetic stereo pair: simple_view1.jpg, simple_view2.jpg")
    
    # Create more realistic textured scene
    realistic_scenes = create_real_world_sample()
    cv2.imwrite('sample_images/textured_view1.jpg', realistic_scenes[0])
    cv2.imwrite('sample_images/textured_view2.jpg', realistic_scenes[1])
    print("Created textured stereo pair: textured_view1.jpg, textured_view2.jpg")
    
    # Create a calibration pattern (chessboard)
    chessboard = np.ones((480, 640), dtype=np.uint8) * 255
    square_size = 40
    
    for i in range(0, 480, square_size):
        for j in range(0, 640, square_size):
            if (i//square_size + j//square_size) % 2 == 1:
                cv2.rectangle(chessboard, (j, i), (j+square_size, i+square_size), 0, -1)
    
    cv2.imwrite('sample_images/chessboard_calibration.jpg', chessboard)
    print("Created chessboard calibration pattern: chessboard_calibration.jpg")
    
    print("\nSample images created successfully!")
    print("You can now test the 3D reconstruction with:")
    print("  python main.py --images sample_images --visualize")
    print("  Or use the web interface: python web/app.py")

if __name__ == '__main__':
    main()