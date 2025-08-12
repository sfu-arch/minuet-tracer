import numpy as np
import os
import struct
import open3d as o3d
from pathlib import Path
import pandas as pd
from typing import Tuple
import argparse

def write_simbin_file(file_path, coords, features=None):
    """
    Backward-compatible wrapper that writes using the new simbin format.

    This delegates to write_simbin() to ensure the file can be read by read_simbin().
    """
    # Normalize inputs to expected dtypes
    coords_np = np.asarray(coords)
    if coords_np.dtype not in [np.float32, np.int32]:
        # Prefer integer grid for quantized coords
        coords_np = coords_np.astype(np.int32, copy=False)
    feats_np = None if features is None else np.asarray(features, dtype=np.float32)
    write_simbin(file_path, coords_np, feats_np)
    return

def voxelize_points(points, min_bound, max_bound, voxel_size, offset):
    """
    Voxelize a list of 3D points within specified boundaries, applying an offset.

    Parameters:
    - points: numpy array of shape (N, 3) representing the 3D points (X, Y, Z).
    - min_bound: numpy array of shape (3,) representing the minimum boundaries (min_x, min_y, min_z).
    - max_bound: numpy array of shape (3,) representing the maximum boundaries (max_x, max_y, max_z).
    - voxel_size: scalar or numpy array of shape (3,) representing the size of each voxel in meters.
    - offset: numpy array of shape (3,) representing the offset to apply (offset_x, offset_y, offset_z).

    Returns:
    - voxels: numpy array of shape (M, 3) representing the set of the occupied voxels.
    """
    # Ensure inputs are numpy arrays
    points = np.asarray(points)
    min_bound = np.asarray(min_bound)
    max_bound = np.asarray(max_bound)
    voxel_size = np.asarray(voxel_size)
    offset = np.asarray(offset)
    
    # Filter points within boundaries: min_bound < points < max_bound
    mask = np.all(points > min_bound, axis=1) & np.all(points < max_bound, axis=1)
    points_filtered = points[mask]
    
    # Shift points by subtracting the offset
    p_shifted = points_filtered + offset
    
    # Compute voxel indices
    voxel_indices = np.floor(p_shifted / voxel_size).astype(int)
    
    # Find unique voxel indices to remove duplicates (sorts inputs)
    # unique_voxel_indices = np.unique(voxel_indices, axis=0)
    # return unique_voxel_indices

    # Deduplicate while preserving first occurrence
    seen = set()
    unique_voxels = []
    for voxel in voxel_indices:
        key = tuple(voxel)
        if key not in seen:
            seen.add(key)
            unique_voxels.append(voxel)

    return np.array(unique_voxels)

def downsample_voxels(voxels, voxelization_stride):
    """
    Downsample a list of voxels by dividing coordinates by stride and deduplicating.

    Parameters:
    - voxels: numpy array of shape (M, 3) representing the voxel coordinates.
    - voxelization_stride: integer representing the downsampling factor.

    Returns:
    - downsampled_voxels: numpy array of shape (K, 3) representing the downsampled voxels.
    """
    # Ensure inputs are numpy arrays
    voxels = np.asarray(voxels)
    
    # Divide voxel coordinates by stride (using integer division)
    downsampled = voxels // voxelization_stride
    
    # Deduplicate while preserving first occurrence (like in voxelize_points)
    seen = set()
    unique_voxels = []
    for voxel in downsampled:
        key = tuple(voxel)
        if key not in seen:
            seen.add(key)
            unique_voxels.append(voxel)
    
    return np.array(unique_voxels).astype(int)

import numpy as np
from typing import Set, Tuple

def dilate_voxels_by_large_kernel_group_reduction(
    voxels: np.ndarray,
    kernel_size: int,
    stride: int = 1
) -> np.ndarray:
    """
    Dilates a set of voxels based on a group reduction rule using NumPy arrays.

    For each stride-aligned voxel, it inspects a large kernel_size^3 volume,
    partitioned into 27 groups. If a group's outer shell contains any voxels,
    the corresponding adjacent voxel to the center is created if it doesn't exist.

    Args:
        voxels: A NumPy array of shape (N, 3) representing voxel coordinates.
        kernel_size: The odd-valued size of the large inspection kernel.
        stride: The alignment requirement for a voxel to be a kernel center.

    Returns:
        A new NumPy array of shape (M, 3) with the original and new voxels.
    """
    # A kernel size of 3 or less would do nothing with the improved logic,
    # and it must be odd for a clear center.
    assert kernel_size > 1 and kernel_size % 2 != 0, \
        "kernel_size must be an odd integer greater than 1."
    assert stride > 0, "stride must be positive."

    # For efficient O(1) lookups, convert the NumPy array to a set of tuples.
    # NumPy arrays are mutable and cannot be stored in a set directly.
    voxels_set: Set[Tuple[int, int, int]] = {tuple(v) for v in voxels}

    # Calculate the extent of the kernel from its center.
    half_kernel = (kernel_size - 1) // 2

    # A helper lambda to get the sign of a number (-1, 0, or 1).
    sgn = lambda val: (val > 0) - (val < 0)

    # Iterate through the original NumPy array rows to find kernel centers.
    for center_voxel in voxels:
        cx, cy, cz = center_voxel[0], center_voxel[1], center_voxel[2]
        
        # A voxel is a center only if its coordinates are aligned to the stride.
        if cx % stride != 0 or cy % stride != 0 or cz % stride != 0:
            continue

        # 1. For this center, determine which of the 26 groups are populated.
        populated_groups: Set[Tuple[int, int, int]] = set()

        # Iterate through the large kernel_size^3 volume around the center.
        for i in range(-half_kernel, half_kernel + 1):
            for j in range(-half_kernel, half_kernel + 1):
                for k in range(-half_kernel, half_kernel + 1):
                    # Skip the 3x3x3 center of the large kernel.
                    if abs(i) <= 1 and abs(j) <= 1 and abs(k) <= 1:
                        continue
                    
                    check_pos = (cx + i, cy + j, cz + k)
                    if check_pos in voxels_set:
                        # This position has a voxel. Add its group index to the set.
                        populated_groups.add((sgn(i), sgn(j), sgn(k)))

        # 2. For each populated group, add the corresponding neighbor voxel to the set.
        for group_idx in populated_groups:
            gx, gy, gz = group_idx
            neighbor_pos = (cx + gx, cy + gy, cz + gz)
            voxels_set.add(neighbor_pos)
            
    print(f"Dialated {len(voxels)} voxels to {len(voxels_set)} voxels by group-reduced assignment")
    # Convert the final set of tuples back to a NumPy array for the return value.
    return np.array(list(voxels_set)).astype(int)

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

            min_bound = np.array([-50.0, -50.0, -5.0])
            max_bound = np.array([50.0, 50.0, 3.0])
            voxel_size = 0.05  # Scalar, same size for all dimensions
            offset = np.array([50.0, 50.0, 5.0])
            voxelized_points = voxelize_points(points, min_bound, max_bound, voxel_size, offset)

            if (stride == None):
                stride = 1
            voxelized_points = downsample_voxels(voxelized_points, stride)
            print(f"Loaded {len(voxelized_points)} points")

            # we don't care about actual feature values
            return voxelized_points, None
        
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
        
        min_bound = np.array([0.0, 15.0, -0.5])
        max_bound = np.array([5.0, 22.0, 4.0])
        voxel_size = 0.02  # Scalar, same size for all dimensions
        offset = np.array([0.0, -15.0, 0.5])
        voxelized_points = voxelize_points(points, min_bound, max_bound, voxel_size, offset)

        if (stride == None):
            stride = 1
        voxelized_points = downsample_voxels(voxelized_points, stride)
        print(f"Loaded {len(voxelized_points)} points")

        # we don't care about actual feature values
        return voxelized_points, None
    
    elif extension == '.simbin':
        # Read coordinates/features directly from simbin
        coords, features = read_simbin(str(file_path))
        print(f"Loaded {len(coords)} points from simbin")
        return coords, features
    
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
    file_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f:
        # Try new header first (16 bytes)
        header_size_new = 6 + 1 + 1 + 4 + 4
        header = f.read(header_size_new)
        if len(header) < 10:
            raise ValueError("File too short to be a simbin file.")
        magic = header[:6]
        if magic != b'SIMBIN':
            raise ValueError("File does not start with 'SIMBIN' magic number.")

        coordinates = None
        features = None

        if len(header) == header_size_new:
            # Interpret as new header and validate size math; else fallback
            version, coord_type, N, feature_dim = struct.unpack('<B B I I', header[6:])
            # Validate version and basic fields
            valid_basic = (version == 1 and coord_type in (0, 1) and N >= 0 and feature_dim >= 0)
            # Check if the remaining file size matches expected layout
            expected_payload = N * 3 * 4 + N * feature_dim * 4
            remaining = file_size - header_size_new
            size_ok = remaining == expected_payload
            if valid_basic and size_ok:
                dtype = np.float32 if coord_type == 0 else np.int32
                coord_bytes = f.read(N * 3 * 4)
                coordinates = np.frombuffer(coord_bytes, dtype=dtype).reshape(N, 3)
                if feature_dim > 0:
                    feature_bytes = f.read(N * feature_dim * 4)
                    features = np.frombuffer(feature_bytes, dtype=np.float32).reshape(N, feature_dim)
                return coordinates, features
            else:
                # Fallback to legacy parse
                f.seek(0)

        # Legacy format: 6s magic + uint32 N, then N*3 float32 coords, optional features contiguous
        f.seek(6)
        n_bytes = f.read(4)
        if len(n_bytes) != 4:
            raise ValueError("Legacy simbin: missing point count.")
        (N_legacy,) = struct.unpack('<I', n_bytes)
        # After legacy header, remaining bytes are coords + optional features
        remaining = file_size - 10
        coord_bytes_len = N_legacy * 3 * 4
        if remaining < coord_bytes_len:
            raise ValueError("Legacy simbin: file smaller than expected for coordinates.")
        coord_bytes = f.read(coord_bytes_len)
        coordinates = np.frombuffer(coord_bytes, dtype=np.float32).reshape(N_legacy, 3)
        # Determine feature_dim from leftover bytes if any
        leftover = remaining - coord_bytes_len
        if leftover == 0:
            features = None
        else:
            # If leftover is divisible by N*4, interpret as float32 features with D columns
            if N_legacy == 0 or (leftover % (N_legacy * 4)) != 0:
                raise ValueError("Legacy simbin: leftover bytes not compatible with features array.")
            feature_dim = leftover // (N_legacy * 4)
            feature_bytes = f.read(leftover)
            features = np.frombuffer(feature_bytes, dtype=np.float32).reshape(N_legacy, feature_dim)
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

def spatially_pruning_downsample(
    points: np.ndarray,
    kernel_size: Tuple[int, int, int],
    suppression_ratio: float,
    stride: int = 2
) -> np.ndarray:
    """
    Simulates the VoxelNeXt spatially-pruning downsampling layer.

    Since feature magnitudes are unknown without a trained model, this function
    randomly selects a subset of points to "dilate" before applying a
    strided downsampling operation. This models the data-dependent nature
    of the layer for performance simulation.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 3) representing the
                             integer coordinates of active 3D voxels.
        kernel_size (Tuple[int, int, int]): The size of the dilation kernel,
                                            e.g., (3, 3, 3).
        suppression_ratio (float): The ratio of points to treat as "unimportant"
                                   and not dilate. For VoxelNeXt, this is 0.5.
        stride (int): The stride for the final downsampling operation.
                      Defaults to 2.

    Returns:
        np.ndarray: A new NumPy array of shape (M, 3) with the coordinates
                    of the active voxels after the operation. M <= N.
    """
    if not (0.0 <= suppression_ratio <= 1.0):
        raise ValueError("suppression_ratio must be between 0.0 and 1.0")

    num_points = points.shape[0]
    if num_points == 0:
        return np.array([], dtype=int).reshape(0, 3)

    # 1. Randomly select points for dilation (the "important" points)
    # This simulates the feature magnitude check without needing the actual features.
    num_important = int(np.ceil(num_points * (1 - suppression_ratio)))
    
    # Create a random permutation of indices and split them
    shuffled_indices = np.random.permutation(num_points)
    important_indices = shuffled_indices[:num_important]
    unimportant_indices = shuffled_indices[num_important:]

    important_points = points[important_indices]
    unimportant_points = points[unimportant_indices]

    # 2. Perform explicit dilation on the "important" points
    # For each important point, generate all neighbors within the kernel window.
    kx, ky, kz = kernel_size
    offset_x = np.arange(-(kx // 2), (kx // 2) + 1)
    offset_y = np.arange(-(ky // 2), (ky // 2) + 1)
    offset_z = np.arange(-(kz // 2), (kz // 2) + 1)
    
    # Create a grid of all possible offsets
    offsets = np.stack(np.meshgrid(offset_x, offset_y, offset_z), axis=-1).reshape(-1, 3)

    # Replicate each important point for each offset and add the offsets
    # This creates the full set of dilated points
    dilated_points = important_points[:, np.newaxis, :] + offsets

    # Reshape to a flat list of points
    dilated_points = dilated_points.reshape(-1, 3)

    # 3. Combine unimportant points with the new dilated set
    # We use np.unique to ensure each coordinate appears only once.
    # This is more efficient than a set for large numpy arrays.
    combined_points = np.vstack([dilated_points, unimportant_points])
    unique_combined_points = np.unique(combined_points, axis=0)

    # 4. Apply the strided downsampling
    # A point survives if all its coordinates are divisible by the stride.
    downsampled_mask = np.all(unique_combined_points % stride == 0, axis=1)
    final_points = unique_combined_points[downsampled_mask]

    return final_points


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

    # coords = dilate_voxels_by_large_kernel_group_reduction(coords, 5)
    visualize_point_cloud(coords, None)