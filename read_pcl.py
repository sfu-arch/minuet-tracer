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
    
    already_quantized = False
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
        file_size = file_path.stat().st_size
        if file_size == 262144:
            # SemanticKITTI voxelized format
            voxels = []
            with open(file_path, 'rb') as f:
                compressed = f.read(262144)
            for k in range(262144):
                byte = compressed[k]
                for b in range(8):
                    if byte & (0x80 >> b):
                        l = k * 8 + b
                        x = l // 8192
                        y = (l // 32) % 256
                        z = l % 32
                        voxels.append([x, y, z])
            points = np.array(voxels, dtype=np.int16)
            features = None
            already_quantized = True
        else:
            # KITTI LiDAR format (float32)
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
        if already_quantized:
            stride = 1
        else:
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

    # Quantize coordinates
    quantized_coords = (points / stride).astype(np.int32)
    
    # Deduplicate and average features
    coords_flat = [tuple(coord) for coord in quantized_coords]
    if features is not None:
        coord_map = {}
        for coord, feat in zip(coords_flat, features):
            if coord not in coord_map:
                coord_map[coord] = [feat]
            else:
                coord_map[coord].append(feat)
        unique_coords = []
        averaged_features = []
        for coord, feats in coord_map.items():
            unique_coords.append(coord)
            averaged_features.append(np.mean(feats, axis=0))
        coords = np.array(unique_coords, dtype=np.int32)
        features = np.array(averaged_features)
    else:
        # Only deduplicate coordinates
        coords = np.unique(quantized_coords, axis=0)
        features = None
    
    print(f"Loaded {len(coords)} points")
    return coords, features

def visualize_point_cloud(coords, features=None):
    """Visualize point cloud using Open3D"""
    coords = np.asarray(coords)

    if coords.size == 0:
        print("Empty point cloud. Nothing to visualize.")
        return

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float64))

        if features is not None:
            features = np.asarray(features)
            if features.shape[0] == coords.shape[0] and features.shape[1] >= 3:
                rgb = features[:, :3]
                if rgb.max() > 1:
                    rgb = rgb / 255.0
                pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
            else:
                print("Feature shape mismatch or insufficient channels; using default color.")
                default_color = np.full((coords.shape[0], 3), 0.5, dtype=np.float64)
                pcd.colors = o3d.utility.Vector3dVector(default_color)
        else:
            default_color = np.full((coords.shape[0], 3), 0.5, dtype=np.float64)
            pcd.colors = o3d.utility.Vector3dVector(default_color)

        o3d.visualization.draw_geometries([pcd])
    except Exception as e:
        print(f"Visualization failed: {e}")


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