import numpy as np
import os
import struct
import open3d as o3d
from pathlib import Path
import pandas as pd
import argparse
<<<<<<< HEAD
import shutil
import random
||||||| parent of 2a43493 (Adding support for simbin)
=======
def write_simbin_file(file_path, coords, features=None):
    """
    Write point cloud data to a simbin file format
    
    Args:
        file_path: Path to output simbin file
        coords: List of (x,y,z) coordinates
        features: List of feature vectors (optional)
    """
    with open(file_path, 'wb') as f:
        # Write header
        f.write(b'SIMBIN')
        f.write(struct.pack('I', len(coords)))  # Number of points
        
        # Write coordinates
        for coord in coords:
            f.write(struct.pack('fff', *coord))
        
        # Write features if available
        if features is not None:
            for feat in features:
                f.write(struct.pack('f' * len(feat), *feat))
    return 
>>>>>>> 2a43493 (Adding support for simbin)


def read_point_cloud(file_path, stride=None, max_points=None, write_simbin=True):
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
    # Write coordinates and features to simbin file
    write_simbin_file("output.simbin", coords, features)

    return coords, features

def write_simbin(file_path, coordinates, features=None):
    """
    Write 3D point cloud coordinates and optional features to a simbin file.

    Args:
        file_path (str): Path to the simbin file to write.
        coordinates (np.ndarray): Array of shape (N, 3), dtype float32 or int32.
        features (np.ndarray, optional): Array of shape (N, D), dtype float32, or None.

    Raises:
        ValueError: If coordinates or features have unsupported dtypes or shapes.
    """
    # Validate inputs
    if not isinstance(coordinates, np.ndarray) or coordinates.shape[1] != 3:
        raise ValueError("Coordinates must be a numpy array of shape (N, 3).")
    if coordinates.dtype not in [np.float32, np.int32]:
        raise ValueError("Coordinates must be of dtype float32 or int32.")
    if features is not None:
        if not isinstance(features, np.ndarray) or features.shape[0] != coordinates.shape[0]:
            raise ValueError("Features must be a numpy array with shape (N, D) matching coordinates.")
        if features.dtype != np.float32:
            raise ValueError("Features must be of dtype float32.")

    N = coordinates.shape[0]
    coord_type = 0 if coordinates.dtype == np.float32 else 1
    feature_dim = 0 if features is None else features.shape[1]

    with open(file_path, 'wb') as f:
        # Write header
        header = struct.pack('<6sBBII', b'SIMBIN', 1, coord_type, N, feature_dim)
        f.write(header)
        # Write coordinates
        f.write(coordinates.tobytes())
        # Write features if present
        if features is not None:
            f.write(features.tobytes())

def read_simbin(file_path):
    """
    Read 3D point cloud coordinates and optional features from a simbin file.

    Args:
        file_path (str): Path to the simbin file to read.

    Returns:
        tuple: (coordinates, features)
            - coordinates (np.ndarray): Array of shape (N, 3), dtype float32 or int32.
            - features (np.ndarray or None): Array of shape (N, D), dtype float32, or None.

    Raises:
        ValueError: If the file is not a valid simbin file.
    """
    with open(file_path, 'rb') as f:
        # Read header
        header_size = 6 + 1 + 1 + 4 + 4  # Total bytes in header
        header = f.read(header_size)
        if len(header) != header_size:
            raise ValueError("File is too short to contain a valid simbin header.")
        
        magic, version, coord_type, N, feature_dim = struct.unpack('<6sBBII', header)
        if magic != b'SIMBIN':
            raise ValueError("File does not start with 'SIMBIN' magic number.")
        if version != 1:
            raise ValueError("Unsupported simbin version.")

        # Determine coordinate dtype
        dtype = np.float32 if coord_type == 0 else np.int32

        # Read coordinates
        coord_bytes = f.read(N * 3 * 4)
        if len(coord_bytes) != N * 3 * 4:
            raise ValueError("File does not contain expected coordinate data.")
        coordinates = np.frombuffer(coord_bytes, dtype=dtype).reshape(N, 3)

        # Read features if present
        if feature_dim > 0:
            feature_bytes = f.read(N * feature_dim * 4)
            if len(feature_bytes) != N * feature_dim * 4:
                raise ValueError("File does not contain expected feature data.")
            features = np.frombuffer(feature_bytes, dtype=np.float32).reshape(N, feature_dim)
        else:
            features = None

        return coordinates, features

def sample_point_clouds(source_dir, dest_dir, samples_per_category):
    """
    Samples point clouds from a source directory, categorizes them by size,
    and saves them to a destination directory, including a .simbin version.

    Args:
        source_dir (str): The directory containing the original point cloud files.
        dest_dir (str): The directory where sampled files will be saved.
        samples_per_category (int): The number of samples to take from each category.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    # 1. Find all point cloud files and get their point counts
    all_files = []
    supported_extensions = ['.ply', '.pcd', '.txt', '.bin', '.npy', '.csv']
    print("Scanning for point cloud files...")
    for file_path in source_path.rglob('*'):
        if file_path.suffix.lower() in supported_extensions:
            try:
                coords, _ = read_point_cloud(str(file_path))
                point_count = len(coords)
                all_files.append({'path': file_path, 'count': point_count})
            except Exception as e:
                print(f"Could not process {file_path}: {e}")

    if not all_files:
        print("No compatible point cloud files found.")
        return

    # 2. Sort files by point count
    all_files.sort(key=lambda x: x['count'])

    # 3. Determine categories and ranges
    num_files = len(all_files)
    third = num_files // 3
    
    small_files = all_files[:third]
    medium_files = all_files[third:2*third]
    large_files = all_files[2*third:]

    categories = {
        'small': small_files,
        'medium': medium_files,
        'large': large_files
    }

    # Print category ranges
    if small_files:
        print(f"\nSmall samples range: {small_files[0]['count']} to {small_files[-1]['count']} points.")
    if medium_files:
        print(f"Medium samples range: {medium_files[0]['count']} to {medium_files[-1]['count']} points.")
    if large_files:
        print(f"Large samples range: {large_files[0]['count']} to {large_files[-1]['count']} points.")
    print("-" * 30)

    # 4. Sample, copy, and convert files
    for category_name, file_list in categories.items():
        category_path = dest_path / category_name
        category_path.mkdir(exist_ok=True)

        if not file_list:
            print(f"No files in category: {category_name}")
            continue

        # Randomly sample files from the category
        num_to_sample = min(samples_per_category, len(file_list))
        sampled_files = random.sample(file_list, num_to_sample)
        print(f"Sampling {num_to_sample} files for '{category_name}' category...")

        for file_info in sampled_files:
            original_path = file_info['path']
            dest_file_path = category_path / original_path.name
            
            # a. Copy the original file
            print(f"  Copying {original_path.name} to {category_path}")
            shutil.copy(original_path, dest_file_path)

            # b. Convert to .simbin and save
            try:
                coords, features = read_point_cloud(str(original_path))
                simbin_path = category_path / (original_path.stem + '.simbin')
                print(f"  Converting {original_path.name} to {simbin_path.name}")
                write_simbin(simbin_path, coords, features)
            except Exception as e:
                print(f"    Failed to convert {original_path.name} to simbin: {e}")

    print("\nSampling process completed.")


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
    parser.add_argument("--write-simbin", action='store_true', help="Write output to simbin format", default=False)

    args = parser.parse_args()
    if args.stride:
        coords, features = read_point_cloud(args.file, args.stride, write_simbin=args.write_simbin)
    else:
        coords, features = read_point_cloud(args.file, write_simbin=args.write_simbin)
    print(coords)
    visualize_point_cloud(coords, features)