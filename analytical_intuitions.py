import numpy as np
from itertools import product
from typing import Dict, Any, Tuple, List
from read_pcl import *

def kernel_neighbor_search(
    input_voxels: np.ndarray,
    kernel_size: int,
    output_downsample: bool
) -> dict:
    """
    Performs a kernel mapping neighbor search on a set of unique input voxels.

    The function finds, for each offset in a kernel, which input points map to a
    valid neighbor. A mapping is considered valid if the potential neighbor's
    coordinate exists in the original input voxel set.

    Args:
        input_voxels: A NumPy array of shape (N, 3) containing unique integer
                      voxel coordinates.
        kernel_size: An integer specifying the dimensions of the cubic kernel
                     (e.g., a value of 3 means a 3x3x3 kernel).
        output_downsample: A boolean flag. If True, output coordinates are
                           calculated by clearing the least significant bit
                           of the input coordinates. If False, output
                           coordinates are the same as the input.

    Returns:
        A dictionary where each key is a 3D tuple representing a kernel offset,
        and the value is a list of indices of the input voxels that map to a
        valid neighbor at that offset.
    """
    # Create a fast lookup table from voxel coordinates to their index
    voxel_lookup = {tuple(v): i for i, v in enumerate(input_voxels)}

    # Determine the output voxel coordinates based on the downsampling flag
    if output_downsample:
        # Clear the least significant bit of each coordinate component
        # This is equivalent to floor division by 2 and multiplication by 2
        output_voxels = input_voxels & ~1
    else:
        output_voxels = input_voxels

    # Generate all possible offsets for the given kernel size
    offset_range = range(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    kernel_offsets = list(product(offset_range, repeat=3))

    # Initialize the results dictionary with empty lists for each offset
    kernel_map = {offset: [] for offset in kernel_offsets}

    # Iterate through each output voxel and check for neighbors
    for i, p_out in enumerate(output_voxels):
        for offset in kernel_offsets:
            # Calculate the potential neighbor's coordinate
            neighbor_coord = tuple(p_out + np.array(offset))

            # If the neighbor exists in the original set of input voxels,
            # record the index of the point that generated this mapping.
            if neighbor_coord in voxel_lookup:
                kernel_map[offset].append(i)

    return kernel_map


def get_gemm_intuitions(
    input_voxels: np.ndarray,
    kernel_size: int,
    output_downsample: bool,
    input_channels_count: int,
    filters_count: int
) -> Dict[str, Any]:
    """
    Calculates GEMM (General Matrix Multiplication) statistics for a sparse convolution.

    This function determines, for each kernel offset (delta), the size of the
    resulting matrix multiplication and the computational cost in terms of MACs.

    Args:
        input_voxels: A NumPy array of shape (N, 3) with unique integer voxel coordinates.
        kernel_size: The dimension of the cubic kernel (e.g., 3 for a 3x3x3 kernel).
        output_downsample: Flag to determine if output coordinates are downsampled.
        input_channels_count: The number of feature channels for each input voxel.
        filters_count: The number of output filters (output channels).

    Returns:
        A dictionary containing detailed statistics, including:
        - 'kernel_map': The mapping from each delta to the list of input point indices.
        - 'inputs_per_delta': The number of active input points for each delta.
        - 'gemm_sizes_per_delta': The dimensions (M, K) x (K, N) for each delta's GEMM.
        - 'macs_per_delta': The number of Multiply-Accumulate operations for each delta.
        - 'total_macs': The total number of MACs for the entire sparse convolution.
    """
    # 1. Get the mapping of input points for each kernel offset (delta)
    kernel_map = kernel_neighbor_search(
        input_voxels, kernel_size, output_downsample
    )

    # --- Initialize dictionaries to store our results ---
    inputs_per_delta = {}
    gemm_sizes_per_delta = {}
    macs_per_delta = {}
    total_macs = 0

    # 2. Iterate through each delta (kernel offset) and its corresponding input points
    for delta, input_indices in kernel_map.items():
        # Number of input points that contribute to this delta's GEMM
        num_inputs_for_delta = len(input_indices)
        inputs_per_delta[delta] = num_inputs_for_delta

        # Define the dimensions of the GEMM operation for this delta.
        # Matrix A: [num_inputs, input_channels]
        # Matrix B (Filter): [input_channels, filters_count]
        # Result C: [num_inputs, filters_count]
        m = num_inputs_for_delta
        k = input_channels_count
        n = filters_count
        
        gemm_sizes_per_delta[delta] = {
            'A_dims': (m, k),
            'B_dims': (k, n),
            'GEMM_shape': f"({m}x{k}) x ({k}x{n})"
        }

        # 3. Calculate the number of MACs for this delta's GEMM
        # A MAC operation is one multiplication and one addition.
        # The total number of multiplications (and additions) is M * K * N.
        num_macs = m * k * n
        macs_per_delta[delta] = num_macs

        # 4. Add to the total MAC count
        total_macs += num_macs

    # 5. Compile all stats into a final dictionary
    stats = {
        'kernel_map': kernel_map,
        'inputs_per_delta': inputs_per_delta,
        'gemm_sizes_per_delta': gemm_sizes_per_delta,
        'macs_per_delta': macs_per_delta,
        'total_macs': total_macs
    }

    return stats

if __name__ == "__main__":
    # --- Example Usage ---

    # Define a sample set of unique input voxels

    voxels, _ = read_point_cloud('examples/000000.bin')

    # Set convolution parameters
    k_size = 3
    in_channels = 32
    out_filters = 64
    downsample = False

    # Get the GEMM statistics
    gemm_stats = get_gemm_intuitions(
        input_voxels=voxels,
        kernel_size=k_size,
        output_downsample=downsample,
        input_channels_count=in_channels,
        filters_count=out_filters
    )

    # --- Print the results in a readable format ---
    print(f"--- GEMM Intuitions for a {k_size}x{k_size}x{k_size} Sparse Convolution ---")
    print(f"Input Channels: {in_channels}, Output Filters: {out_filters}, Downsampling: {downsample}\n")

    print("--- Statistics Per Kernel Offset (Delta) ---")
    for delta, num_inputs in gemm_stats['inputs_per_delta'].items():
        if num_inputs > 0:  # Only show deltas that have computations
            gemm_shape = gemm_stats['gemm_sizes_per_delta'][delta]['GEMM_shape']
            macs = gemm_stats['macs_per_delta'][delta]
            print(f"Delta {delta}:")
            print(f"  - Active Input Points: {num_inputs}")
            print(f"  - GEMM Shape: {gemm_shape}")
            print(f"  - MACs: {macs:,}")

    print("\n--- Overall Statistics ---")
    print(f"Total Combined MACs for all GEMMs: {gemm_stats['total_macs']:,}")