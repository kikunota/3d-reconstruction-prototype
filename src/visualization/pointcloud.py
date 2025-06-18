import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import cv2

class PointCloudVisualizer:
    def __init__(self):
        self.point_cloud = None
    
    def create_point_cloud(self, points_3d: np.ndarray, colors: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        """
        Create Open3D point cloud from 3D points
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        if colors is not None:
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            colors = np.random.rand(len(points_3d), 3)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        self.point_cloud = pcd
        return pcd
    
    def filter_outliers(self, pcd: o3d.geometry.PointCloud, 
                       nb_neighbors: int = 20, std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
        """
        Remove outliers from point cloud
        """
        pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
        return pcd_filtered
    
    def downsample_point_cloud(self, pcd: o3d.geometry.PointCloud, voxel_size: float = 0.1) -> o3d.geometry.PointCloud:
        """
        Downsample point cloud using voxel grid
        """
        return pcd.voxel_down_sample(voxel_size)
    
    def estimate_normals(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Estimate normals for the point cloud
        """
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        return pcd
    
    def visualize_point_cloud(self, pcd: o3d.geometry.PointCloud, window_name: str = "3D Reconstruction"):
        """
        Visualize point cloud in interactive window
        """
        o3d.visualization.draw_geometries([pcd], window_name=window_name)
    
    def save_point_cloud(self, pcd: o3d.geometry.PointCloud, filename: str):
        """
        Save point cloud to file (PLY, PCD, etc.)
        """
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Point cloud saved to {filename}")
    
    def save_point_cloud_las(self, points_3d: np.ndarray, colors: Optional[np.ndarray] = None, 
                            filename: str = None) -> str:
        """
        Save point cloud in LAS format for GIS applications
        """
        try:
            import laspy
        except ImportError:
            print("Warning: laspy not installed. Install with: pip install laspy")
            return None
        
        if filename is None:
            filename = "reconstruction.las"
        
        try:
            # Create LAS file
            header = laspy.LasHeader(point_format=3, version="1.2")
            header.add_extra_dim(laspy.ExtraBytesParams(name="reconstruction_quality", type=np.float32))
            
            # Create LAS data
            las = laspy.LasData(header)
            
            # Set coordinates (LAS uses integer coordinates with scale factors)
            # Calculate appropriate scale and offset
            x_min, x_max = points_3d[:, 0].min(), points_3d[:, 0].max()
            y_min, y_max = points_3d[:, 1].min(), points_3d[:, 1].max()
            z_min, z_max = points_3d[:, 2].min(), points_3d[:, 2].max()
            
            # Set scale to achieve good precision (typically 0.001 for mm precision)
            scale = 0.001
            header.scale = [scale, scale, scale]
            header.offset = [x_min, y_min, z_min]
            
            # Set coordinate ranges
            header.min = [x_min, y_min, z_min]
            header.max = [x_max, y_max, z_max]
            
            # Assign coordinates
            las.x = points_3d[:, 0]
            las.y = points_3d[:, 1]
            las.z = points_3d[:, 2]
            
            # Set colors if available
            if colors is not None:
                # LAS colors are 16-bit, scale from 8-bit if needed
                if colors.max() <= 1.0:
                    # Colors are in 0-1 range
                    las.red = (colors[:, 0] * 65535).astype(np.uint16)
                    las.green = (colors[:, 1] * 65535).astype(np.uint16)
                    las.blue = (colors[:, 2] * 65535).astype(np.uint16)
                else:
                    # Colors are in 0-255 range
                    las.red = (colors[:, 0] * 257).astype(np.uint16)  # 257 = 65535/255
                    las.green = (colors[:, 1] * 257).astype(np.uint16)
                    las.blue = (colors[:, 2] * 257).astype(np.uint16)
            else:
                # Default gray color
                default_color = 32767  # Mid-gray in 16-bit
                las.red = np.full(len(points_3d), default_color, dtype=np.uint16)
                las.green = np.full(len(points_3d), default_color, dtype=np.uint16)
                las.blue = np.full(len(points_3d), default_color, dtype=np.uint16)
            
            # Set classification (default to unclassified)
            las.classification = np.zeros(len(points_3d), dtype=np.uint8)
            
            # Set intensity based on Z coordinate for visualization
            z_normalized = (points_3d[:, 2] - z_min) / (z_max - z_min) if z_max > z_min else np.zeros(len(points_3d))
            las.intensity = (z_normalized * 65535).astype(np.uint16)
            
            # Add reconstruction quality metric
            quality_scores = np.ones(len(points_3d), dtype=np.float32)  # Default high quality
            las.reconstruction_quality = quality_scores
            
            # Write LAS file
            las.write(filename)
            print(f"LAS point cloud saved to {filename}")
            return filename
            
        except Exception as e:
            print(f"Error saving LAS file: {e}")
            return None
    
    def create_mesh(self, pcd: o3d.geometry.PointCloud, 
                   method: str = 'poisson') -> o3d.geometry.TriangleMesh:
        """
        Create mesh from point cloud
        """
        if not pcd.has_normals():
            pcd = self.estimate_normals(pcd)
        
        if method == 'poisson':
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        elif method == 'ball_pivoting':
            radii = [0.005, 0.01, 0.02, 0.04]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
        else:
            raise ValueError("Method must be 'poisson' or 'ball_pivoting'")
        
        return mesh
    
    def visualize_mesh(self, mesh: o3d.geometry.TriangleMesh, window_name: str = "3D Mesh"):
        """
        Visualize mesh in interactive window
        """
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], window_name=window_name)

class PointCloudProcessor:
    def __init__(self):
        self.visualizer = PointCloudVisualizer()
    
    def extract_colors_from_image(self, img: np.ndarray, pts2d: np.ndarray) -> np.ndarray:
        """
        Extract RGB colors from image at 2D point locations
        """
        h, w = img.shape[:2]
        colors = []
        
        for pt in pts2d:
            # Handle different point formats: (N, 2), (N, 1, 2), etc.
            if len(pt.shape) > 1 and pt.shape[0] == 1:
                # Shape is (1, 2) - flatten it
                pt = pt.flatten()
            
            # Safely convert to integers
            if hasattr(pt[0], 'item'):
                x, y = int(pt[0].item()), int(pt[1].item())
            else:
                x, y = int(pt[0]), int(pt[1])
                
            if 0 <= x < w and 0 <= y < h:
                if len(img.shape) == 3:
                    color = img[y, x][::-1]  # BGR to RGB
                    color = [int(c) for c in color]  # Ensure integers
                else:
                    gray_val = int(img[y, x])
                    color = [gray_val] * 3
                colors.append(color)
            else:
                colors.append([128, 128, 128])  # Gray for out-of-bounds
        
        return np.array(colors)
    
    def process_reconstruction_result(self, points_3d: np.ndarray, 
                                    img1: np.ndarray, pts1: Optional[np.ndarray] = None,
                                    output_file: Optional[str] = None) -> o3d.geometry.PointCloud:
        """
        Process and visualize 3D reconstruction results
        """
        print(f"Processing {len(points_3d)} 3D points")
        
        if pts1 is not None:
            colors = self.extract_colors_from_image(img1, pts1)
        else:
            # Generate random colors when no 2D points available
            colors = np.random.randint(0, 255, (len(points_3d), 3))
        
        pcd = self.visualizer.create_point_cloud(points_3d, colors)
        
        pcd_filtered = self.visualizer.filter_outliers(pcd)
        print(f"After outlier removal: {len(pcd_filtered.points)} points")
        
        if len(pcd_filtered.points) > 1000:
            pcd_filtered = self.visualizer.downsample_point_cloud(pcd_filtered, voxel_size=0.05)
            print(f"After downsampling: {len(pcd_filtered.points)} points")
        
        if output_file:
            # Save in PLY format
            self.visualizer.save_point_cloud(pcd_filtered, output_file)
            
            # Also save in LAS format
            las_file = output_file.replace('.ply', '.las')
            
            # Get points and colors from filtered point cloud
            filtered_points = np.asarray(pcd_filtered.points)
            filtered_colors = np.asarray(pcd_filtered.colors)
            
            # Convert colors from 0-1 range to 0-255 if needed
            if filtered_colors.max() <= 1.0:
                filtered_colors = (filtered_colors * 255).astype(np.uint8)
            
            las_path = self.visualizer.save_point_cloud_las(filtered_points, filtered_colors, las_file)
            if las_path:
                print(f"Point cloud also saved in LAS format: {las_path}")
        
        return pcd_filtered
    
    def create_and_save_visualization(self, points_3d: np.ndarray, 
                                    img1: np.ndarray, pts1: Optional[np.ndarray] = None,
                                    output_dir: str = ".") -> str:
        """
        Create point cloud visualization and save as image
        """
        pcd = self.process_reconstruction_result(points_3d, img1, pts1)
        
        # Skip Open3D visualization on macOS due to GUI issues, use matplotlib directly
        import platform
        if platform.system() == "Darwin":  # macOS
            print("Using matplotlib visualization on macOS to avoid GUI issues")
            return self.create_matplotlib_visualization(points_3d, output_dir)
        
        try:
            # Try to create headless visualization on other platforms
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=800, height=600)
            vis.add_geometry(pcd)
            
            # Set camera view
            view_control = vis.get_view_control()
            view_control.set_front([0, 0, -1])
            view_control.set_up([0, -1, 0])
            
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            
            output_path = f"{output_dir}/reconstruction_result.png"
            vis.capture_screen_image(output_path)
            vis.destroy_window()
            
            return output_path
            
        except Exception as e:
            print(f"Could not create Open3D visualization: {e}")
            # Fallback: create a simple matplotlib plot
            return self.create_matplotlib_visualization(points_3d, output_dir)
    
    def create_matplotlib_visualization(self, points_3d: np.ndarray, output_dir: str) -> str:
        """
        Enhanced matplotlib visualization for 3D point clouds
        """
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        try:
            # Create figure with multiple views
            fig = plt.figure(figsize=(16, 12))
            
            # Sample points if too many for better performance
            max_points = 2000
            if len(points_3d) > max_points:
                indices = np.random.choice(len(points_3d), max_points, replace=False)
                points_sample = points_3d[indices]
            else:
                points_sample = points_3d
            
            # Ensure we have valid points
            if len(points_sample) == 0:
                raise ValueError("No points to visualize")
            
            # Main 3D view
            ax1 = fig.add_subplot(221, projection='3d')
            scatter = ax1.scatter(points_sample[:, 0], points_sample[:, 1], points_sample[:, 2], 
                                c=points_sample[:, 2], cmap='viridis', s=2, alpha=0.8)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y') 
            ax1.set_zlabel('Z (Depth)')
            ax1.set_title('3D Point Cloud - Perspective View')
            
            # Top view (X-Y plane)
            ax2 = fig.add_subplot(222)
            ax2.scatter(points_sample[:, 0], points_sample[:, 1], 
                       c=points_sample[:, 2], cmap='viridis', s=1, alpha=0.7)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title('Top View (X-Y Plane)')
            ax2.set_aspect('equal')
            
            # Side view (X-Z plane)
            ax3 = fig.add_subplot(223)
            ax3.scatter(points_sample[:, 0], points_sample[:, 2], 
                       c=points_sample[:, 1], cmap='plasma', s=1, alpha=0.7)
            ax3.set_xlabel('X')
            ax3.set_ylabel('Z (Depth)')
            ax3.set_title('Side View (X-Z Plane)')
            ax3.set_aspect('equal')
            
            # Front view (Y-Z plane)
            ax4 = fig.add_subplot(224)
            ax4.scatter(points_sample[:, 1], points_sample[:, 2], 
                       c=points_sample[:, 0], cmap='coolwarm', s=1, alpha=0.7)
            ax4.set_xlabel('Y')
            ax4.set_ylabel('Z (Depth)')
            ax4.set_title('Front View (Y-Z Plane)')
            ax4.set_aspect('equal')
            
            # Add colorbar for depth
            plt.colorbar(scatter, ax=ax1, shrink=0.8, label='Depth (Z)')
            
            # Add statistics text
            stats_text = f"""Point Cloud Statistics:
Total Points: {len(points_3d):,}
Displayed: {len(points_sample):,}
X Range: [{points_sample[:, 0].min():.2f}, {points_sample[:, 0].max():.2f}]
Y Range: [{points_sample[:, 1].min():.2f}, {points_sample[:, 1].max():.2f}]
Z Range: [{points_sample[:, 2].min():.2f}, {points_sample[:, 2].max():.2f}]"""
            
            fig.suptitle('3D Reconstruction Results', fontsize=16, fontweight='bold')
            fig.text(0.02, 0.02, stats_text, fontsize=10, fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            plt.tight_layout()
            
            output_path = f"{output_dir}/reconstruction_result.png"  
            plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Enhanced visualization saved: {output_path}")
            return output_path
            
        except Exception as e:
            plt.close('all')  # Clean up any open figures
            print(f"Matplotlib visualization failed: {e}")
            return None