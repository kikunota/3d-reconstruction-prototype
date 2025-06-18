from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
import tempfile
import base64
from werkzeug.utils import secure_filename
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.camera import CameraCalibrator
from core.features import FeatureDetector, FeatureMatcher
from core.geometry import StereoReconstructor
from core.enhanced_reconstruction import EnhancedReconstructor
from visualization.pointcloud import PointCloudProcessor
from utils.helpers import resize_image, create_default_camera_matrix, ensure_directory_exists

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 512MB max file size for 500 images
app.config['UPLOAD_FOLDER'] = 'uploads'

# Flask app configuration

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Template error: {e}")
        return f"Template error: {str(e)}", 500

@app.route('/test')
def test():
    return "Flask app is working!"

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content response for favicon

@app.route('/upload', methods=['POST'])
def upload_images():
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    files = request.files.getlist('images')
    
    if len(files) < 2:
        return jsonify({'error': 'Please provide at least 2 images'}), 400
    
    if len(files) > 500:
        return jsonify({'error': 'Maximum 500 images allowed'}), 400
    
    ensure_directory_exists(app.config['UPLOAD_FOLDER'])
    
    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_files.append(filepath)
    
    if len(uploaded_files) < 2:
        return jsonify({'error': 'Need at least 2 valid images'}), 400
    
    try:
        result = process_reconstruction(uploaded_files)  # Use all uploaded images
        return jsonify(result)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Reconstruction error: {error_details}")
        return jsonify({
            'success': False,
            'error': f'Reconstruction failed: {str(e)}'
        }), 500

def process_reconstruction(image_paths):
    """
    Enhanced reconstruction pipeline with bundle adjustment
    """
    if len(image_paths) < 2:
        raise ValueError("Need at least 2 images for reconstruction")
    
    print(f"Processing {len(image_paths)} images for enhanced reconstruction with bundle adjustment")
    
    # Load and resize all images
    images = []
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not load image {path}, skipping")
            continue
        # Resize for faster processing
        img = resize_image(img, max_size=800)
        images.append(img)
        print(f"Loaded image {i+1}/{len(image_paths)}: {img.shape}")
    
    if len(images) < 2:
        raise ValueError("Need at least 2 valid images after loading")
    
    # Create default camera matrix (in real application, use calibrated camera)
    camera_matrix = create_default_camera_matrix(images[0].shape[:2])
    
    # Initialize enhanced reconstructor with bundle adjustment
    enhanced_reconstructor = EnhancedReconstructor(
        camera_matrix=camera_matrix,
        use_bundle_adjustment=True,
        use_distortion=True,  # Enable distortion correction
        outlier_threshold=2.0
    )
    
    # Feature detection and matching
    detector = FeatureDetector('SIFT')
    matcher = FeatureMatcher(detector)
    
    # Find best image pairs for reconstruction
    print("Finding best image pairs...")
    best_pairs = matcher.find_best_image_pairs(images, min_matches=30)
    
    if not best_pairs:
        print("No suitable image pairs found, falling back to consecutive pairs")
        best_pairs = [(i, i+1, 0) for i in range(len(images) - 1)]
    
    print(f"Found {len(best_pairs)} suitable image pairs")
    
    # Process image pairs and add to enhanced reconstructor
    successful_pairs = 0
    max_pairs = min(len(best_pairs), len(images) * 2)  # Process more pairs for better reconstruction
    
    for pair_idx, (i, j, num_matches) in enumerate(best_pairs[:max_pairs]):
        print(f"Processing image pair {i+1}-{j+1} ({pair_idx+1}/{max_pairs}): {num_matches} matches")
        img1, img2 = images[i], images[j]
        
        try:
            pts1, pts2, matches = matcher.match_image_pair(img1, img2)
            
            if len(matches) < 15:
                print(f"Warning: Not enough matches for pair {i+1}-{j+1}: {len(matches)}, skipping")
                continue
            
            # Filter matches using fundamental matrix
            pts1_filtered, pts2_filtered, mask = matcher.filter_matches_with_fundamental(pts1, pts2)
            
            if len(pts1_filtered) < 8:
                print(f"Warning: Not enough good matches after filtering for pair {i+1}-{j+1}: {len(pts1_filtered)}, skipping")
                continue
            
            # Add this pair to the enhanced reconstructor
            result = enhanced_reconstructor.add_image_pair_reconstruction(
                img1, img2, pts1_filtered, pts2_filtered
            )
            
            if result['success']:
                successful_pairs += 1
                print(f"Pair {i+1}-{j+1}: {result['n_points_3d']} 3D points reconstructed")
                print(f"Total cameras: {result['total_cameras']}, Total points: {result['total_points']}")
            else:
                print(f"Failed to reconstruct pair {i+1}-{j+1}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Error processing pair {i+1}-{j+1}: {e}")
            continue
    
    if successful_pairs == 0:
        raise ValueError("No successful reconstructions from any image pair")
    
    print(f"\nSuccessfully processed {successful_pairs} image pairs")
    
    # Perform bundle adjustment
    print("\n" + "="*50)
    print("PERFORMING BUNDLE ADJUSTMENT")
    print("="*50)
    
    ba_result = enhanced_reconstructor.perform_bundle_adjustment(
        max_iterations=50,  # Reduced for faster processing
        n_refinement_iterations=2,  # Reduced from 3 to 2 for speed
        verbose=True
    )
    
    if not ba_result['success']:
        print(f"Bundle adjustment failed: {ba_result.get('message', 'Unknown error')}")
        print("Continuing with non-optimized reconstruction...")
    
    # Get final optimized 3D points
    combined_points_3d = enhanced_reconstructor.get_optimized_points_3d()
    print(f"\nFinal optimized reconstruction: {len(combined_points_3d)} 3D points")
    
    # Process and visualize point cloud
    processor = PointCloudProcessor()
    
    # Create output directory - use persistent directory instead of tempfile
    output_dir = os.path.join(os.path.dirname(__file__), 'downloads')
    ensure_directory_exists(output_dir)
    
    # Create unique subdirectory with timestamp
    import time
    timestamp = str(int(time.time()))
    output_dir = os.path.join(output_dir, f"reconstruction_{timestamp}")
    ensure_directory_exists(output_dir)
    
    # Save point cloud
    try:
        print("Processing optimized point cloud...")
        pcd = processor.process_reconstruction_result(
            combined_points_3d, images[0], None,  # Use None for pts1 since we're combining multiple pairs
            output_file=os.path.join(output_dir, 'reconstruction.ply')
        )
        print("Point cloud processing completed")
    except Exception as e:
        print(f"Point cloud processing error: {e}")
        raise
    
    # Create enhanced visualizations including bundle adjustment analysis
    try:
        print("Creating enhanced visualizations...")
        viz_path = processor.create_and_save_visualization(
            combined_points_3d, images[0], None, output_dir
        )
        print(f"3D visualization created: {viz_path}")
        
        # Create bundle adjustment visualizations
        ba_visualizations = enhanced_reconstructor.create_detailed_visualizations(output_dir)
        print(f"Bundle adjustment visualizations: {list(ba_visualizations.keys())}")
        
    except Exception as e:
        import traceback
        print(f"Warning: Could not create visualization: {e}")
        traceback.print_exc()
        viz_path = None
        ba_visualizations = {}
    
    # Create matches visualization for first successful pair
    matches_path = None
    if successful_pairs > 0:
        try:
            # Use first two images for matches visualization
            img1, img2 = images[0], images[1]
            pts1, pts2, matches = matcher.match_image_pair(img1, img2)
            kp1, _ = detector.detect_and_compute(img1)
            kp2, _ = detector.detect_and_compute(img2)
            matches_img = detector.visualize_matches(img1, kp1, img2, kp2, matches[:50])  # Show first 50 matches
            
            matches_path = os.path.join(output_dir, 'matches.jpg')
            cv2.imwrite(matches_path, matches_img)
        except Exception as e:
            print(f"Warning: Could not create matches visualization: {e}")
            matches_path = None
    
    # Convert images to base64 for web display
    def image_to_base64(img_path):
        if img_path and os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        return None
    
    # Get reconstruction summary
    reconstruction_summary = enhanced_reconstructor.get_reconstruction_summary()
    
    # Convert numpy values safely to Python scalars
    mean_depth = np.mean(combined_points_3d[:, 2])
    if hasattr(mean_depth, 'item'):
        mean_depth = mean_depth.item()
    else:
        mean_depth = float(mean_depth)
    
    # Prepare bundle adjustment visualization URLs (converted to base64)
    ba_viz_data = {}
    for viz_name, viz_path in ba_visualizations.items():
        if viz_path and os.path.exists(viz_path):
            ba_viz_data[viz_name] = image_to_base64(viz_path)
    
    # Check for both PLY and LAS files
    ply_file = os.path.join(output_dir, 'reconstruction.ply')
    las_file = os.path.join(output_dir, 'reconstruction.las')
    
    download_files = {}
    subpath = f"reconstruction_{timestamp}"  # Use the timestamp from above
    
    if os.path.exists(ply_file):
        download_files['ply'] = f"/download/{subpath}/reconstruction.ply"
    if os.path.exists(las_file):
        download_files['las'] = f"/download/{subpath}/reconstruction.las"
    
    result = {
        'success': True,
        'num_images_processed': len(images),
        'num_successful_pairs': successful_pairs,
        'num_cameras': reconstruction_summary['n_cameras'],
        'num_observations': reconstruction_summary['n_observations'],
        'num_3d_points': int(len(combined_points_3d)),
        'mean_depth': mean_depth,
        'bundle_adjustment': reconstruction_summary.get('bundle_adjustment', {'performed': False}),
        'camera_trajectory': reconstruction_summary.get('camera_trajectory', {}),
        'matches_image': image_to_base64(matches_path) if matches_path else None,
        'reconstruction_image': image_to_base64(viz_path) if viz_path else None,
        'bundle_adjustment_visualizations': ba_viz_data,
        'download_files': download_files,
        'point_cloud_file': ply_file,  # Keep for backward compatibility
        'reconstruction_summary': reconstruction_summary
    }
    
    return result

@app.route('/download/<path:subpath>/<filename>')
def download_file(subpath, filename):
    # Security check - construct safe path
    downloads_dir = os.path.join(os.path.dirname(__file__), 'downloads')
    file_path = os.path.join(downloads_dir, subpath, filename)
    
    # Ensure the path is within our downloads directory
    if not os.path.abspath(file_path).startswith(os.path.abspath(downloads_dir)):
        return "Unauthorized", 403
    
    # Check if file exists
    if not os.path.exists(file_path):
        return "File not found", 404
    
    try:
        return send_file(file_path, as_attachment=True, download_name=filename)
    except Exception as e:
        print(f"Download error: {e}")
        return f"Download failed: {str(e)}", 500

def cleanup_old_downloads():
    """Clean up download files older than 1 hour"""
    downloads_dir = os.path.join(os.path.dirname(__file__), 'downloads')
    if not os.path.exists(downloads_dir):
        return
    
    import time
    current_time = time.time()
    one_hour_ago = current_time - 3600  # 1 hour in seconds
    
    try:
        for item in os.listdir(downloads_dir):
            item_path = os.path.join(downloads_dir, item)
            if os.path.isdir(item_path):
                # Check if directory name matches our pattern and is old
                if item.startswith('reconstruction_'):
                    try:
                        timestamp = int(item.split('_')[1])
                        if timestamp < one_hour_ago:
                            import shutil
                            shutil.rmtree(item_path)
                            print(f"Cleaned up old download directory: {item}")
                    except (ValueError, IndexError):
                        pass  # Skip invalid directory names
    except Exception as e:
        print(f"Cleanup error: {e}")

if __name__ == '__main__':
    ensure_directory_exists('uploads')
    ensure_directory_exists('static')
    ensure_directory_exists(os.path.join(os.path.dirname(__file__), 'downloads'))
    
    # Clean up old downloads on startup
    cleanup_old_downloads()
    
    # Try different ports if 5000 is occupied (common on macOS with AirPlay)
    import socket
    
    def is_port_available(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) != 0
    
    port = 5000
    while not is_port_available(port) and port < 5010:
        port += 1
    
    print(f"Starting server on http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)