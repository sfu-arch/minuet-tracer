import numpy as np
import os
import struct
import open3d as o3d
from pathlib import Path
import pandas as pd
import argparse

def read_point_cloud(file_path, stride=None, max_points=None):
    """
    Read point cloud data from various formats and prepare for Minuet processing
    
    Args:
        file_path: Path to point cloud file (.ply, .pcd, .txt, .bin, .npy)
        stride: Voxel size for quantization
        max_points: Maximum number of points to use (for memory constraints)
    
    Returns:
        coords: List of (x,y,z) coordinates quantized by stride
        features: List of feature vectors (if available)
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    print(f"Reading point cloud from {file_path}")
    
    # Handle different file formats
    if extension in ['.ply', '.pcd']:
        # Use Open3D to read standard point cloud formats
        pcd = o3d.io.read_point_cloud(str(file_path))
        points = np.asarray(pcd.points)
        
        # Extract features if available
        if pcd.has_colors():
            features = np.asarray(pcd.colors)
        elif pcd.has_normals():
            features = np.asarray(pcd.normals)
        else:
            features = None
            
    elif extension == '.bin':
        # Handle KITTI binary format (float32 values)
        data = np.fromfile(file_path, dtype=np.float32)
        points = data.reshape(-1, 4)[:, :3]  # X,Y,Z (drop intensity)
        features = data.reshape(-1, 4)[:, 3:]  # Intensity as feature
        
    elif extension == '.npy':
        # Handle NumPy format
        data = np.load(file_path)
        if data.shape[1] >= 3:
            points = data[:, :3]  # X,Y,Z
            if data.shape[1] > 3:
                features = data[:, 3:]  # Additional features
            else:
                features = None
        else:
            raise ValueError(f"NumPy array must have at least 3 columns for XYZ, got {data.shape[1]}")
            
    elif extension == '.txt' or extension == '.csv':
        # Handle text formats
        try:
            if extension == '.csv':
                data = pd.read_csv(file_path).values
            else:
                data = np.loadtxt(file_path, delimiter=None)
            
            if data.shape[1] >= 3:
                points = data[:, :3]  # X,Y,Z
                if data.shape[1] > 3:
                    features = data[:, 3:]  # Additional features
                else:
                    features = None
            else:
                raise ValueError(f"Text file must have at least 3 columns for XYZ, got {data.shape[1]}")
        except Exception as e:
            raise ValueError(f"Failed to parse text file: {e}")
    else:
        raise ValueError(f"Unsupported file extension: {extension}")
    
    # Limit number of points if specified
    if max_points is not None and points.shape[0] > max_points:
        # Randomly sample points
        indices = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[indices]
        if features is not None:
            features = features[indices]
    
     # Auto-determine stride if not provided
    if stride is None:
        # Calculate bounding box
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        dimensions = max_vals - min_vals
        
        # Set stride to 1% of the smallest non-zero dimension
        non_zero_dims = dimensions[dimensions > 0]
        if len(non_zero_dims) > 0:
            stride = float(np.min(non_zero_dims)) / 100.0
        else:
            stride = 0.01  # Default if all dimensions are zero
            
    print(f"Auto-selected stride: {stride}")
    
    
    # Convert to list of tuples and quantize coordinates
    coords = []
    for point in points:
        # Quantize coordinates by stride
        x, y, z = point
        qx, qy, qz = int(x/stride), int(y/stride), int(z/stride)
        coords.append((qx, qy, qz))
        print(f"Quantized point: ({qx}, {qy}, {qz})")
        print(f"Original point: ({x}, {y}, {z})")
    
    print(f"Loaded {len(coords)} points")
    return coords, features

def visualize_point_cloud(coords, features=None):
    """Visualize point cloud using Open3D"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(coords))
  
    if features is not None and features.shape[1] >= 3:
        # Normalize RGB colors if needed
        if features.max() > 1:
            rgb = features[:, :3] / 255.0
        else:
            rgb = features[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and visualize point cloud data")
    parser.add_argument("--file", type=str, help="Path to point cloud file")
    parser.add_argument("--stride", type=float, default=None, help="Voxel size for quantization")

    args = parser.parse_args()
    if args.stride:
        coords, features = read_point_cloud(args.file, args.stride)
    else:
        coords, features = read_point_cloud(args.file)
    print(coords)
    visualize_point_cloud(coords, features)