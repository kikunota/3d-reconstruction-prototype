#!/usr/bin/env python3
"""
Test the web API with sample images
"""

import requests
import os

def test_web_api():
    """Test the web API upload endpoint"""
    
    url = "http://localhost:5004/upload"
    
    # Check if sample images exist
    img1_path = "sample_images/textured_view1.jpg"
    img2_path = "sample_images/textured_view2.jpg"
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("Sample images not found. Run: python3 create_sample_images.py")
        return False
    
    try:
        # Prepare files for upload
        with open(img1_path, 'rb') as f1, open(img2_path, 'rb') as f2:
            files = [
                ('images', ('view1.jpg', f1, 'image/jpeg')),
                ('images', ('view2.jpg', f2, 'image/jpeg'))
            ]
            
            print("Uploading sample images to web API...")
            response = requests.post(url, files=files, timeout=60)
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print("✅ Web API test successful!")
                    print(f"  - Matches: {result.get('num_matches')}")
                    print(f"  - Filtered matches: {result.get('num_filtered_matches')}")
                    print(f"  - 3D points: {result.get('num_3d_points')}")
                    print(f"  - Mean depth: {result.get('mean_depth'):.2f}")
                    return True
                else:
                    print(f"❌ API returned error: {result.get('error')}")
                    return False
            else:
                print(f"❌ HTTP error {response.status_code}: {response.text}")
                return False
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == '__main__':
    success = test_web_api()
    exit(0 if success else 1)