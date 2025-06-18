import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.bundle_adjustment import BundleAdjuster, CameraParameters


class BundleAdjustmentVisualizer:
    """Visualization tools for bundle adjustment results"""
    
    def __init__(self):
        self.fig_size = (16, 12)
        self.dpi = 150
    
    def plot_reprojection_errors(self, bundle_adjuster: BundleAdjuster, 
                                output_path: str, bins: int = 50) -> str:
        """Create comprehensive reprojection error analysis plots"""
        errors, residuals = bundle_adjuster.get_reprojection_errors()
        
        fig, axes = plt.subplots(2, 3, figsize=self.fig_size)
        fig.suptitle('Bundle Adjustment: Reprojection Error Analysis', fontsize=16, fontweight='bold')
        
        # 1. Error histogram
        ax = axes[0, 0]
        ax.hist(errors, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.2f}')
        ax.axvline(np.median(errors), color='green', linestyle='--', label=f'Median: {np.median(errors):.2f}')
        ax.axvline(bundle_adjuster.outlier_threshold, color='orange', linestyle='--', 
                  label=f'Outlier threshold: {bundle_adjuster.outlier_threshold:.2f}')
        ax.set_xlabel('Reprojection Error (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Error vs observation index
        ax = axes[0, 1]
        ax.plot(errors, 'o', markersize=2, alpha=0.6)
        ax.axhline(np.mean(errors), color='red', linestyle='--', label='Mean')
        ax.axhline(bundle_adjuster.outlier_threshold, color='orange', linestyle='--', label='Threshold')
        ax.set_xlabel('Observation Index')
        ax.set_ylabel('Reprojection Error (pixels)')
        ax.set_title('Error per Observation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Cumulative distribution
        ax = axes[0, 2]
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(sorted_errors, cumulative, linewidth=2)
        ax.axvline(np.percentile(errors, 50), color='green', linestyle='--', label='50th percentile')
        ax.axvline(np.percentile(errors, 90), color='orange', linestyle='--', label='90th percentile')
        ax.axvline(np.percentile(errors, 95), color='red', linestyle='--', label='95th percentile')
        ax.set_xlabel('Reprojection Error (pixels)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. X vs Y residuals scatter
        ax = axes[1, 0]
        residuals_x = residuals[::2]
        residuals_y = residuals[1::2]
        scatter = ax.scatter(residuals_x, residuals_y, c=errors, cmap='viridis', alpha=0.6, s=20)
        ax.set_xlabel('X Residual (pixels)')
        ax.set_ylabel('Y Residual (pixels)')
        ax.set_title('X vs Y Residuals')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Error Magnitude')
        
        # 5. Error by camera
        ax = axes[1, 1]
        camera_errors = {}
        for i, obs in enumerate(bundle_adjuster.observations):
            if obs.camera_id not in camera_errors:
                camera_errors[obs.camera_id] = []
            camera_errors[obs.camera_id].append(errors[i])
        
        camera_ids = list(camera_errors.keys())
        mean_errors = [np.mean(camera_errors[cid]) for cid in camera_ids]
        std_errors = [np.std(camera_errors[cid]) for cid in camera_ids]
        
        ax.errorbar(camera_ids, mean_errors, yerr=std_errors, 
                   fmt='o-', capsize=5, linewidth=2, markersize=6)
        ax.set_xlabel('Camera ID')
        ax.set_ylabel('Mean Reprojection Error (pixels)')
        ax.set_title('Error by Camera')
        ax.grid(True, alpha=0.3)
        
        # 6. Statistics summary
        ax = axes[1, 2]
        ax.axis('off')
        
        stats_text = f"""Reprojection Error Statistics:
        
Total Observations: {len(errors):,}
Mean Error: {np.mean(errors):.3f} pixels
Median Error: {np.median(errors):.3f} pixels
Std Dev: {np.std(errors):.3f} pixels
Min Error: {np.min(errors):.3f} pixels
Max Error: {np.max(errors):.3f} pixels

Percentiles:
50th: {np.percentile(errors, 50):.3f} pixels
90th: {np.percentile(errors, 90):.3f} pixels
95th: {np.percentile(errors, 95):.3f} pixels
99th: {np.percentile(errors, 99):.3f} pixels

Outliers (>{bundle_adjuster.outlier_threshold:.1f}px):
Count: {np.sum(errors > bundle_adjuster.outlier_threshold)}
Percentage: {100 * np.sum(errors > bundle_adjuster.outlier_threshold) / len(errors):.1f}%"""
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path
    
    def plot_camera_trajectory(self, bundle_adjuster: BundleAdjuster, 
                              points_3d: Optional[np.ndarray], output_path: str) -> str:
        """Visualize camera trajectory and 3D points"""
        positions, orientations = bundle_adjuster.get_camera_trajectory()
        
        fig = plt.figure(figsize=self.fig_size)
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Plot camera positions
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'o-', linewidth=2, markersize=8, label='Camera trajectory')
        
        # Plot camera orientations as arrows
        for i, (pos, rvec) in enumerate(zip(positions, orientations)):
            R, _ = cv2.Rodrigues(rvec)
            # Direction vector (camera looking direction)
            direction = R @ np.array([0, 0, 1])  # Camera looks down Z-axis
            ax1.quiver(pos[0], pos[1], pos[2], 
                      direction[0], direction[1], direction[2],
                      length=1.0, color='red', alpha=0.7)
        
        # Plot 3D points if provided
        if points_3d is not None and len(points_3d) > 0:
            # Sample points for visualization
            if len(points_3d) > 1000:
                indices = np.random.choice(len(points_3d), 1000, replace=False)
                sample_points = points_3d[indices]
            else:
                sample_points = points_3d
            
            ax1.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2],
                       c='blue', s=1, alpha=0.3, label='3D Points')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Camera Trajectory')
        ax1.legend()
        
        # Top view (X-Y)
        ax2 = fig.add_subplot(222)
        ax2.plot(positions[:, 0], positions[:, 1], 'o-', linewidth=2, markersize=6)
        for i, pos in enumerate(positions):
            ax2.annotate(f'C{i}', (pos[0], pos[1]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        if points_3d is not None and len(points_3d) > 0:
            ax2.scatter(sample_points[:, 0], sample_points[:, 1], 
                       c='blue', s=1, alpha=0.2)
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Top View (X-Y)')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Side view (X-Z)
        ax3 = fig.add_subplot(223)
        ax3.plot(positions[:, 0], positions[:, 2], 'o-', linewidth=2, markersize=6)
        for i, pos in enumerate(positions):
            ax3.annotate(f'C{i}', (pos[0], pos[2]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        if points_3d is not None and len(points_3d) > 0:
            ax3.scatter(sample_points[:, 0], sample_points[:, 2], 
                       c='blue', s=1, alpha=0.2)
        
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_title('Side View (X-Z)')
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        
        # Camera parameters summary
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        # Calculate trajectory statistics
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_distance = np.sum(distances)
        
        camera_info = f"""Camera Trajectory Analysis:

Number of cameras: {len(positions)}
Total trajectory distance: {total_distance:.2f} units

Camera positions range:
X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]
Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]
Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]

Average camera spacing: {np.mean(distances):.2f} units
Max camera spacing: {np.max(distances) if len(distances) > 0 else 0:.2f} units
Min camera spacing: {np.min(distances) if len(distances) > 0 else 0:.2f} units

Camera intrinsics (first camera):
fx: {bundle_adjuster.cameras[0].fx:.1f}
fy: {bundle_adjuster.cameras[0].fy:.1f}
cx: {bundle_adjuster.cameras[0].cx:.1f}
cy: {bundle_adjuster.cameras[0].cy:.1f}"""
        
        if bundle_adjuster.use_distortion:
            camera_info += f"""
Distortion coefficients:
k1: {bundle_adjuster.cameras[0].k1:.6f}
k2: {bundle_adjuster.cameras[0].k2:.6f}
p1: {bundle_adjuster.cameras[0].p1:.6f}
p2: {bundle_adjuster.cameras[0].p2:.6f}"""
        
        ax4.text(0.05, 0.95, camera_info, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        
        plt.suptitle('Camera Trajectory and 3D Scene', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path
    
    def plot_optimization_convergence(self, bundle_adjuster: BundleAdjuster, 
                                    output_path: str) -> str:
        """Plot optimization convergence history"""
        if not bundle_adjuster.optimization_history:
            print("No optimization history available, creating placeholder convergence plot")
            # Create a simple placeholder plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, 'No Optimization History Available\n\nThis can happen if:\n• Bundle adjustment was skipped\n• Optimization converged in 1 iteration\n• An error occurred during optimization', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
            ax.set_title('Bundle Adjustment Convergence', fontsize=14, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            return output_path
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Convergence plot
        ax = axes[0]
        iterations = range(len(bundle_adjuster.optimization_history))
        errors = bundle_adjuster.optimization_history
        
        ax.plot(iterations, errors, 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMS Reprojection Error (pixels)')
        ax.set_title('Bundle Adjustment Convergence')
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        if len(errors) > 1:
            improvement = (errors[0] - errors[-1]) / errors[0] * 100
            ax.text(0.05, 0.95, f'Initial error: {errors[0]:.4f}\nFinal error: {errors[-1]:.4f}\nImprovement: {improvement:.1f}%',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        # Log scale plot if useful
        ax = axes[1]
        ax.semilogy(iterations, errors, 'r-', linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMS Reprojection Error (pixels, log scale)')
        ax.set_title('Convergence (Log Scale)')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Bundle Adjustment Optimization Progress', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path
    
    def create_before_after_comparison(self, errors_before: np.ndarray, errors_after: np.ndarray,
                                     output_path: str) -> str:
        """Create before/after comparison of bundle adjustment"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Before histogram
        ax = axes[0, 0]
        ax.hist(errors_before, bins=50, alpha=0.7, color='red', label='Before BA')
        ax.axvline(np.mean(errors_before), color='darkred', linestyle='--', 
                  label=f'Mean: {np.mean(errors_before):.2f}')
        ax.set_xlabel('Reprojection Error (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title('Before Bundle Adjustment')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # After histogram
        ax = axes[0, 1]
        ax.hist(errors_after, bins=50, alpha=0.7, color='green', label='After BA')
        ax.axvline(np.mean(errors_after), color='darkgreen', linestyle='--', 
                  label=f'Mean: {np.mean(errors_after):.2f}')
        ax.set_xlabel('Reprojection Error (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title('After Bundle Adjustment')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Comparison plot
        ax = axes[1, 0]
        ax.hist(errors_before, bins=50, alpha=0.5, color='red', label='Before BA')
        ax.hist(errors_after, bins=50, alpha=0.5, color='green', label='After BA')
        ax.set_xlabel('Reprojection Error (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title('Before vs After Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Statistics comparison
        ax = axes[1, 1]
        ax.axis('off')
        
        improvement = (np.mean(errors_before) - np.mean(errors_after)) / np.mean(errors_before) * 100
        
        stats_text = f"""Bundle Adjustment Results:

BEFORE:
Mean Error: {np.mean(errors_before):.3f} pixels
Median Error: {np.median(errors_before):.3f} pixels
Std Dev: {np.std(errors_before):.3f} pixels
95th Percentile: {np.percentile(errors_before, 95):.3f} pixels

AFTER:
Mean Error: {np.mean(errors_after):.3f} pixels
Median Error: {np.median(errors_after):.3f} pixels
Std Dev: {np.std(errors_after):.3f} pixels
95th Percentile: {np.percentile(errors_after, 95):.3f} pixels

IMPROVEMENT:
Mean Error Reduction: {improvement:.1f}%
Std Dev Reduction: {(np.std(errors_before) - np.std(errors_after)) / np.std(errors_before) * 100:.1f}%"""
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        
        plt.suptitle('Bundle Adjustment: Before vs After Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path