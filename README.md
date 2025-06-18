# 3D Reconstruction with Bundle Adjustment

<img width="400" alt="image" src="https://github.com/user-attachments/assets/299a5c90-8805-48d5-a807-3b391be181e0" />


<img width="400" alt="image" src="https://github.com/user-attachments/assets/50ef9a9f-3e2b-4f9d-b7b1-5b698752d73e" />


A comprehensive 3D reconstruction system that creates point clouds from multiple images using Structure from Motion (SfM) with advanced bundle adjustment optimization.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

## 🌟 Features

### Core Capabilities
- **Multi-Image Reconstruction**: Process 2-500 images for dense 3D point clouds
- **Advanced Bundle Adjustment**: Iterative optimization with outlier rejection
- **Multiple Camera Models**: Pinhole and radial distortion correction
- **Dual Export Formats**: PLY (3D software) and LAS (GIS applications)
- **Web Interface**: User-friendly browser-based interface
- **CLI Support**: Command-line interface for batch processing

### Advanced Algorithms
- **SIFT/ORB Feature Detection**: Up to 15,000 features per image
- **Robust Matching**: Fundamental matrix filtering with RANSAC
- **Smart Pair Selection**: Analyzes all image combinations for optimal reconstruction
- **Bundle Adjustment**: Levenberg-Marquardt and Trust Region optimization
- **Outlier Rejection**: Progressive outlier removal with quality metrics

### Visualization & Analysis
- **4-Panel Point Cloud Views**: 3D perspective + orthographic projections
- **Reprojection Error Analysis**: Comprehensive error distribution plots
- **Camera Trajectory Visualization**: 3D camera path analysis
- **Optimization Convergence**: Real-time progress tracking
- **Bundle Adjustment Statistics**: Detailed accuracy metrics

## Installation

1. **Clone or download this project**
   ```bash
   cd 3d-reconstruction
   ```

2. **Install Python dependencies**
   ```bash
   pip3 install -r requirements.txt
   ```

   **macOS Additional Step**: If you get Open3D import errors, install OpenMP:
   ```bash
   brew install libomp
   ```

   Required packages:
   - OpenCV (opencv-python, opencv-contrib-python)
   - NumPy, SciPy, Matplotlib
   - Open3D (for 3D visualization)
   - Flask (for web interface)

## Quick Start

### Option 1: Web Interface (Recommended)

1. **Start the web server**
   ```bash
   python3 web/app.py
   ```

2. **Open your browser**
   - Go to `http://localhost:5001` (or the port shown in the terminal)
   - Upload 2-10 images of the same object/scene from different viewpoints
   - Click "Start 3D Reconstruction"
   - View results including feature matches and 3D point cloud

### Option 2: Command Line

1. **Prepare your images**
   - Place images in a folder (e.g., `my_images/`)
   - Images should be of the same object from different angles

2. **Run reconstruction**
   ```bash
   python3 main.py --images my_images/ --output results/ --visualize
   ```

3. **View results**
   - Point cloud: `results/reconstruction.ply`
   - Feature matches: `results/feature_matches.jpg`
   - 3D visualization (if --visualize flag used)

## Project Structure

```
3d-reconstruction/
├── src/
│   ├── core/
│   │   ├── camera.py          # Camera calibration
│   │   ├── features.py        # Feature detection and matching
│   │   └── geometry.py        # 3D geometry estimation
│   ├── visualization/
│   │   └── pointcloud.py      # Point cloud processing and visualization
│   └── utils/
│       └── helpers.py         # Utility functions
├── web/
│   ├── app.py                 # Flask web application
│   └── templates/
│       └── index.html         # Web interface
├── sample_images/             # Sample test images
├── main.py                    # Command line interface
├── create_sample_images.py    # Generate test images
└── requirements.txt           # Python dependencies
```

## How It Works

### 1. Feature Detection and Matching
- Detect keypoints in images using SIFT or ORB
- Match features between image pairs
- Filter matches using fundamental matrix estimation

### 2. Camera Pose Estimation
- Estimate essential matrix from feature correspondences
- Recover camera rotation and translation
- Validate pose using triangulation

### 3. 3D Triangulation
- Convert 2D point correspondences to 3D points
- Filter points based on reprojection error and depth
- Generate colored point cloud

### 4. Visualization and Export
- Create 3D point cloud visualization
- Export results in standard formats (PLY)
- Generate analysis images and statistics

## Usage Tips

### For Best Results:
- **Image Quality**: Use high-resolution, well-lit images
- **Overlap**: Ensure 60-80% overlap between consecutive images
- **Texture**: Objects should have sufficient texture for feature matching
- **Viewpoint**: Take images from different angles around the object
- **Stability**: Avoid blurry images - use tripod if possible

### Supported Formats:
- **Input**: JPG, PNG, BMP images
- **Output**: PLY point clouds, JPG visualizations

## Advanced Usage

### Camera Calibration
If you have a calibrated camera, you can provide calibration parameters:

```bash
python3 main.py --images my_images/ --calibration camera_calibration.npz
```

To create calibration data, take several photos of a chessboard pattern and use the camera calibration module.

### Feature Detector Options
Choose between SIFT (more accurate) and ORB (faster):

```bash
python3 main.py --images my_images/ --feature-type ORB
```

### Batch Processing
Process multiple image sets:

```bash
for folder in image_set_*; do
    python3 main.py --images "$folder" --output "results_$(basename $folder)"
done
```

## Sample Images

The project includes sample synthetic images for testing:

```bash
python3 create_sample_images.py  # Generate sample images
python3 main.py --images sample_images/ --visualize  # Test reconstruction
```

## Troubleshooting

### Common Issues:

1. **"Not enough matches found"**
   - Ensure images have sufficient texture and overlap
   - Try different feature detector (SIFT vs ORB)
   - Check image quality and lighting

2. **"Module not found" errors**
   - Install all requirements: `pip3 install -r requirements.txt`
   - Check Python version (3.7+ recommended)

3. **Open3D import error (macOS)**: `Library not loaded: libomp.dylib`
   - Install OpenMP: `brew install libomp`
   - This is required for Open3D on macOS

4. **Poor reconstruction quality**
   - Increase image overlap between views
   - Use higher resolution images
   - Ensure good lighting and focus
   - Try calibrating your camera

5. **Web interface not loading**
   - The app automatically finds an available port (usually 5001 on macOS)
   - Check the terminal output for the correct URL
   - On macOS, port 5000 is often used by AirPlay Receiver

## Technical Details

### Algorithm Pipeline:
1. **Feature Detection**: SIFT/ORB keypoint detection
2. **Feature Matching**: Ratio test + RANSAC filtering
3. **Fundamental Matrix**: Robust estimation using RANSAC
4. **Essential Matrix**: From fundamental matrix and camera parameters
5. **Pose Recovery**: SVD decomposition of essential matrix
6. **Triangulation**: Linear triangulation of 3D points
7. **Bundle Adjustment**: (Future enhancement)

### Dependencies:
- **OpenCV**: Computer vision operations
- **NumPy**: Numerical computations
- **Open3D**: 3D data processing and visualization
- **Flask**: Web interface framework
- **Matplotlib**: 2D plotting and visualization

## Future Enhancements

- [ ] Multi-view reconstruction (>2 images)
- [ ] Bundle adjustment optimization
- [ ] Automatic camera calibration
- [ ] Dense reconstruction methods
- [ ] Mesh generation from point clouds
- [ ] Real-time reconstruction

## Contributing

Feel free to contribute improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## References

- Hartley, R. & Zisserman, A. "Multiple View Geometry in Computer Vision"
- Szeliski, R. "Computer Vision: Algorithms and Applications"
- OpenCV Documentation: https://docs.opencv.org/
- Open3D Documentation: http://www.open3d.org/
