from itertools import islice
import threading # Import the threading module
import matplotlib.pyplot as plt
import gzip
import math
import threading # Import the threading module
import matplotlib.pyplot as plt
import gzip
import struct
import concurrent.futures # Add this import
from hashlib import sha256
from minuet_config import output_dir

def file_checksum(filename):
    hash_lib = sha256()
    with open(filename, 'rb') as f:
        # Read the file in chunks to handle large files efficiently
        chunk = f.read(4096)
        while chunk:
            hash_lib.update(chunk)
            chunk = f.read(4096)
    return hash_lib.hexdigest()


import numpy as np
from typing import Sequence, Any, Mapping

def create_slot_array(sorted_kmap_idx, kernel_map):
    slot_array = np.zeros(len(kernel_map.keys()), dtype=np.int32)
    offsets_active = []
    for i in sorted_kmap_idx:
        o = i
        matches = kernel_map[o]
        if len(matches) > 0:
            slot_array[i] = len(matches)
            offsets_active.append(o)
    return slot_array, offsets_active




def create_in_out_masks(kernel_map, slot_dict, num_offsets, num_sources):
    """
    Args:
      kernel_map   : List[tuple[int, int]]
                     Each key 'o' (offset) maps to a list of (in_idx, out_idx) tuples.
                     Note: The original docstring type hint for matches was more complex.
                     This implementation assumes 'matches' is List[(in_idx, out_idx)]
                     based on the loop structure `for local_i, (in_idx,out_idx) in enumerate(matches):`.
      num_offsets  : total number of offsets (O)
      num_sources  : total number of source points (N_in)
  
    Returns:
      out_mask     : List[int]  of length (match_offsets * matches)
      in_mask      : List[int] of length (match_offsets * matches)
      cumsum       : Cumulative sum of counts for allocating slots
    """ 
    # initialize both to -1
    
    in_mask = np.full(num_offsets*num_sources, -1, dtype=np.int32)
    out_mask = np.full(num_offsets*num_sources, -1, dtype=np.int32)
    def _process_kernel_item(item_tuple):
        o, matches_list = item_tuple 
        # Based on the original loop: for local_i, (in_idx, out_idx) in enumerate(matches_list):
        # item_in_idx is used with num_targets for in_mask.
        # item_out_idx is used with num_sources for out_mask.
        for local_i, (in_idx, out_idx) in enumerate(matches_list):
            # Update in_mask (related to targets)
            in_mask_actual_idx = o * num_sources + in_idx
            in_mask[in_mask_actual_idx] =  slot_dict[o] + local_i  
            # Update out_mask (related to sources)
            out_mask_actual_idx = o * num_sources + out_idx
            out_mask[out_mask_actual_idx] = slot_dict[o] + local_i

        return out_mask, in_mask

    processed_results = []
    # Use ThreadPoolExecutor to parallelize the processing of kernel_map items.
    # The default number of worker threads will be used.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all items from kernel_map for processing.
        # kernel_map.items() yields (offset, matches_list) tuples.
        futures = [executor.submit(_process_kernel_item, item) for item in kernel_map.items()]
        
    return out_mask, in_mask


def read_metadata(filename):
    """
    Read the metadata from a gzipped binary file (Python implementation).
    
    Args:
        filename (str): The path to the metadata file.
        
    Returns:
        dict: A dictionary containing the metadata with keys:
              'version', 'num_total_system_offsets', 'num_total_system_sources',
              'total_slots_in_gemm_buffer', 'num_active_offsets_in_map',
              'active_offsets_details' (list of dicts), 
              'out_mask' (numpy.ndarray), 'in_mask' (numpy.ndarray).
    
    Raises:
        ValueError: If the file format is invalid or version is unsupported.
        EOFError: If the file is truncated.
    """
    contents = {}
    active_offsets_details_list = []

    with gzip.open(filename, 'rb') as f:
        # Helper to read and unpack data
        def read_unpack(fmt):
            size = struct.calcsize(fmt)
            data = f.read(size)
            if len(data) < size:
                raise EOFError(f"Unexpected EOF while trying to read {size} bytes for format '{fmt}'.")
            return struct.unpack(fmt, data)

        # Read and verify magic number and version
        magic_bytes, version_num = read_unpack('<4sI')
        if magic_bytes != b'MINU':
            raise ValueError(f"Invalid metadata file format: magic number mismatch. Expected b'MINU', got {magic_bytes}")
        if version_num != 1:
            raise ValueError(f"Unsupported metadata file version: {version_num}. Expected 1.")
        contents['version'] = version_num

        # Read num_total_system_offsets, num_total_system_sources
        num_sys_offsets, num_sys_sources = read_unpack('<II')
        contents['num_total_system_offsets'] = num_sys_offsets
        contents['num_total_system_sources'] = num_sys_sources

        # Read total_slots_in_gemm_buffer
        total_gemm_slots, = read_unpack('<I')
        contents['total_slots_in_gemm_buffer'] = total_gemm_slots

        # Read number of active offsets
        num_active_offsets, = read_unpack('<I')
        contents['num_active_offsets_in_map'] = num_active_offsets

        # Read each active offset's details
        for _ in range(num_active_offsets):
            offset_key, base_addr, num_matches = read_unpack('<III')
            active_offsets_details_list.append({
                'offset_key': offset_key,
                'base_address': base_addr,
                'num_matches': num_matches
            })
        contents['active_offsets_details'] = active_offsets_details_list

        # Calculate mask size and read masks
        if num_sys_offsets > 0 and num_sys_sources > 0:
            mask_elements = num_sys_offsets * num_sys_sources
            mask_bytes = mask_elements * np.dtype(np.int32).itemsize

            out_mask_data = f.read(mask_bytes)
            if len(out_mask_data) < mask_bytes:
                raise EOFError(f"Unexpected EOF while reading out_mask data. Expected {mask_bytes} bytes.")
            contents['out_mask'] = np.frombuffer(out_mask_data, dtype=np.int32)

            in_mask_data = f.read(mask_bytes)
            if len(in_mask_data) < mask_bytes:
                raise EOFError(f"Unexpected EOF while reading in_mask data. Expected {mask_bytes} bytes.")
            contents['in_mask'] = np.frombuffer(in_mask_data, dtype=np.int32)
        else:
            contents['out_mask'] = np.array([], dtype=np.int32)
            contents['in_mask'] = np.array([], dtype=np.int32)
            
    return contents

    




"""
######## Algorithm for gemm grouping ########
# We want to partition a sequence of positions into contiguous “groups” that each:
- Start at an address divisible by the alignment (e.g. 4)
- Contain at most max_group positions
- Require at most max_slots raw slots before padding
Minimize:
  - The total number of groups
  - Subject to the total wasted slots (padding)

We employ a dynamic programming approach to solve this problem.
The algorithm works as follows:
- We define a DP array where dp[i] = (num_groups, wasted_slots) for the first i positions.
- We iterate through the positions and for each position, we try to form groups of size k (1 <= k <= max_group).
- For each group, we calculate the required slots and wasted slots.
- We update the DP array with the minimum number of groups and wasted slots.
- Finally, we reconstruct the groups and their positions from the DP array.
- The output is a list of positions and their corresponding groups.
- The algorithm is efficient and runs in O(n^2) time complexity, where n is the number of positions.
- The algorithm is designed to be flexible and can handle different alignment, max_group, and max_slots constraints.
"""

def dp_group(
    slots,
    alignment=4,
    max_group=6,
    max_slots=None
):
    """
    Groups positions under constraints:
      - alignment boundary
      - max positions per group
      - optional max raw slots per group
    Returns:
      - pos_indices: list of absolute slot index where each position begins
      - groups: list of (start_pos, end_pos, base_addr, required_slots, allocated_slots)
    """
    n = len(slots)
    dp = [(float('inf'), float('inf'))] * (n + 1)
    dp[n] = (0, 0)
    choice = [0] * n

    # DP to choose optimal group size at each index
    for i in range(n - 1, -1, -1):
        best = (float('inf'), float('inf'))
        best_k = 1
        for k in range(1, max_group + 1):
            if i + k > n:
                break
            req = sum(slots[i:i+k])
            if max_slots is not None and req > max_slots:
                break
            alloc = ((req + alignment - 1) // alignment) * alignment
            waste = alloc - req
            ng, nw = dp[i + k]
            cand = (1 + ng, waste + nw)
            if cand < best:
                best = cand
                best_k = k
        dp[i] = best
        choice[i] = best_k

    # Reconstruct groups, per-position indices, and membership
    pos_indices = []
    groups = []
    addr = 0
    i = 0
    while i < n:
        k = choice[i]
        req = sum(slots[i:i+k])
        alloc = ((req + alignment - 1) // alignment) * alignment
        groups.append((i, i+k-1, addr, req, alloc))
        offset = 0
        for s in slots[i:i+k]:
            pos_indices.append(addr + offset)
            offset += s
        addr += alloc
        i += k

    # Build explicit membership lists
    membership = [list(range(s, e+1)) for s, e, *_ in groups]

    return pos_indices, groups, membership


def greedy_group(slots, alignment=4, max_group=6, max_slots=None):
    """
    Greedy grouping after sorting positions by descending slot requirement.

    Args:
      slots      : list of slot‐counts per position
      alignment  : alignment boundary (e.g. 4)
      max_group  : maximum number of positions per group
      max_slots  : optional cap on raw slots per group (None to disable)

    Returns:
      pos_indices : list of absolute slot index where each original position begins
      groups      : list of tuples (members, base_addr, req, alloc)
      membership  : list of lists, each inner list is the positions in that group
    """
    n = len(slots)
    # Verify list is sorted
    if not all(slots[i] >= slots[i + 1] for i in range(n - 1)):
        raise ValueError("slots must be sorted in descending order")
    
    # pair each slot with its original index, sort descending
    # Slot is already sorted, so we can just enumerate
    # DO NOT SORT AGAIN
    indexed = list(enumerate(slots))
    addr = 0
    cur_sum = 0
    cur_count = 0
    cur_members = []

    groups = []
    pos_indices = [None] * n

    def flush_group():
        nonlocal addr, cur_sum, cur_count, cur_members
        if not cur_members:
            return
        req = cur_sum
        alloc = ((req + alignment - 1) // alignment) * alignment
        if len(cur_members) == 1:
            cur_members.append(cur_members[0])
        groups.append((cur_members.copy(), addr, req, alloc))
        addr += alloc
        cur_sum = 0
        cur_count = 0
        cur_members.clear()

    # process each position in descending‐slot order
    for idx, size in indexed:
        # if adding this would violate constraints, flush current
        if (cur_count >= max_group) or \
           (max_slots is not None and cur_sum + size > max_slots):
            flush_group()

        # record this position’s start index
        pos_indices[idx] = addr + cur_sum

        # add to current group
        cur_members.append(idx)
        cur_sum += size
        cur_count += 1

    # flush any trailing group
    flush_group()

    # build membership lists
    membership = [members for members, _, _, _ in groups]
    # Calcualate total slots
    total_slots = sum(g[3] for g in groups)
 
    gemm_list = []
    for g in groups:
            gemm_list.append({
            'num_offsets': g[0][1]-g[0][0]+1,
            'gemm_M': g[3],
            'slots': g[2],
            'padding': g[3]-g[2]
        })
            
    print(gemm_list)
    # Write out the gemm list to file.
    checksum = write_gemm_list(gemm_list)

    return pos_indices, groups, membership, gemm_list, total_slots, checksum


def write_gemm_list(gemm_data_list, filename = output_dir+"gemms.bin.gz"):
    """
    Write the gemm list to a file in a packed binary format,
    compressed with gzip.
    Each GEMM entry is structured as:
    - num_offsets (unsigned int)
    - gemm_M (unsigned int)
    - padding (unsigned int)
    All integers are packed in little-endian byte order (<).
    """
    with gzip.open(filename, 'wb') as f: # Open in binary write mode
        for gemm in gemm_data_list:
            num_offsets = gemm['num_offsets']
            gemm_M = gemm['gemm_M']
            # gemm['gemm_N'] is the same as num_offsets, so it's implicitly covered
            padding = gemm['padding']
            packed_header = struct.pack("III",  # Changed from !III to <III
                                        num_offsets, 
                                        gemm_M, 
                                        padding)
            f.write(packed_header)
    checksum = file_checksum(filename)
    return checksum

def read_gemm_list(filename):
    """
    Read the gemm list from a packed binary, gzipped file.
    Assumes data is in little-endian byte order.
    Returns a list of GEMM dictionaries.
    """
    gemm_data_list = []
    
    # Define the format and size of the fixed-size header
    # num_offsets, gemm_M, padding, len_inputs, len_outs
    header_format = '<IIIII'  # Changed from !IIIII to <IIIII
    header_size = struct.calcsize(header_format)

    with gzip.open(filename, 'rb') as f: # Open in binary read mode
        while True:
            # Read the header for the next GEMM entry
            packed_header = f.read(header_size)
            if not packed_header:
                break # End of file
            if len(packed_header) < header_size:
                raise EOFError("Incomplete GEMM header found. File might be corrupted.")

            num_offsets, gemm_M, padding, len_inputs, len_outs = struct.unpack(header_format, packed_header)

            gemm_entry = {
                'num_offsets': num_offsets,
                'gemm_M': gemm_M,
                'gemm_N': num_offsets, # Reconstruct gemm_N, as it's num_offsets
                'padding': padding,
                'offsets': [],
                'inputs': (),
                'outs': ()
            }

            gemm_data_list.append(gemm_entry)
            
    return gemm_data_list


def compact_bar_chart(groups):
    """Plot one bar per group, labeled by [start_pos–end_pos]@start_addr."""
    fig, ax = plt.subplots(figsize=(10, len(groups) * 0.5))

    plot_groups = []
    for idx, (se, addr, req, alloc) in enumerate(groups):
        plot_groups.append((se[0], se[1], addr, req, alloc))

    for idx, (s, e, addr, req, alloc) in enumerate(plot_groups):
        ax.broken_barh([(addr, alloc)], (idx - 0.4, 0.8))
        ax.text(addr + alloc / 2, idx, f"[{s}-{e}] @{addr}",
                ha='center', va='center')
    ax.set_xlabel('Memory address (slots)')
    ax.set_yticks([])
    ax.set_title('Compact Group Bar Chart')
    plt.tight_layout()
    plt.show()


def gather_thread(
    # Arguments defining this thread's share of work
    log_tid: int, 
    num_threads: int, 
    # Original processing parameters (passed through)
    num_points: int,
    num_offsets: int,
    num_tiles_per_pt: int, 
    tile_feat_size: int, 
    bulk_feat_size: int,     
    source_masks: list,
    sources: list,
    gemm_buffers: list # Shared output list/array, modified by this worker
):
    """
    The function executed by each Python thread.
    It processes an interleaved subset of points. For each point, it processes all its tiles.
    """
    
    # --- Derived Constants (can be recalculated or passed if complex) ---
    num_bulks_per_tile = 0
    assert(tile_feat_size % bulk_feat_size == 0)
    assert(tile_feat_size > 0)
    assert (bulk_feat_size > 0)    
    
    num_bulks_per_tile = tile_feat_size // bulk_feat_size
    total_feats_per_pt = num_tiles_per_pt * tile_feat_size 

    # Each log_tid processes points in an interleaved fashion
    # e.g., if num_threads = 4, log_tid = 0, it processes points 0, 4, 8, ...
    # This is a simulation of the warp-level parallelism in CUDA.
    # Each thread will process a subset of points based on its logical ID
    # and the total number of threads.
    # This is to ensure that neighboring threads process adjacent memory locations at a time.
    
    for pt_idx in range(log_tid, num_points, num_threads): 
        
        tile_data = [0.0] * tile_feat_size 

        for tile_idx_in_pt in range(num_tiles_per_pt): 
            # 1. LOAD PHASE for one tile
            pt_base = pt_idx * total_feats_per_pt 
            tile_base_in_pt = tile_idx_in_pt * tile_feat_size 
            tile_first_bulk_idx_in_pt = tile_base_in_pt // bulk_feat_size

            for bulk_offset_in_tile in range(num_bulks_per_tile): 
                bulk_idx_in_pt = tile_first_bulk_idx_in_pt + bulk_offset_in_tile
                vload_addr = pt_base + bulk_idx_in_pt * bulk_feat_size
                tile_data_bulk_lstart = bulk_offset_in_tile * bulk_feat_size

                # Records as single access. 
                # Simulation of a vector load in CUDA.
                for elem_idx_in_bulk in range(bulk_feat_size): 
                    gsrc_idx = vload_addr + elem_idx_in_bulk 
                    tile_data_idx = tile_data_bulk_lstart + elem_idx_in_bulk
                    assert(0 <= gsrc_idx < len(sources))
                    assert(0 <= tile_data_idx < len(tile_data))
                    tile_data[tile_data_idx] = sources[gsrc_idx]
            
            # 2. STORE PHASE for the currently loaded tile
            for off_idx in range(num_offsets): 
                mask_idx = off_idx * num_points + pt_idx 
                
                assert(0 <= mask_idx < len(source_masks))
                
                dest_slot = source_masks[mask_idx] 
                
                if dest_slot == -1:  continue # Offset not present
                
                dest_tile_base_in_slot = tile_idx_in_pt * tile_feat_size
                dest_tile_first_bulk_idx_in_slot = dest_tile_base_in_slot // bulk_feat_size

                for bulk_idx in range(num_bulks_per_tile): 
                    dest_bulk_idx_in_slot = dest_tile_first_bulk_idx_in_slot + bulk_idx
                    tile_data_bulk_lsrc = bulk_idx * bulk_feat_size
                    vstore_addr = dest_slot*total_feats_per_pt + dest_bulk_idx_in_slot * bulk_feat_size
                    
                    for elem_idx_in_bulk in range(bulk_feat_size): 
                        tile_data_idx = tile_data_bulk_lsrc + elem_idx_in_bulk 
                        out_buf_gdest_idx = vstore_addr + elem_idx_in_bulk 
                        assert(0 <= out_buf_gdest_idx < len(gemm_buffers))                             
                        assert(0 <= tile_data_idx < len(tile_data))
                        #    0 <= tile_data_idx < len(tile_data):
                        gemm_buffers[out_buf_gdest_idx] = tile_data[tile_data_idx]

# Main function to orchestrate the threaded execution
def python_threaded_gather_simulation(
    # Parameters defining the data and processing
        num_threads: int,
        num_points: int,
        num_offsets: int, 
        num_tiles_per_pt: int,
        tile_feat_size: int,
        bulk_feat_size: int,
        source_masks: list,
        sources: list,
        gemm_buffers: list
    ) -> None:
    """
    Orchestrates the gather operation using multiple Python threads.
    """

    # --- Input Validations ---
    if num_threads <= 0: 
        raise ValueError("num_threads must be positive.") # Updated variable name
    if num_points < 0: raise ValueError("num_points cannot be negative.")
    if num_tiles_per_pt < 0: raise ValueError("num_tiles_per_pt cannot be negative.")
    if tile_feat_size < 0: raise ValueError("tile_feat_size cannot be negative.")
    
    if tile_feat_size > 0:
        if bulk_feat_size <= 0:
            raise ValueError("bulk_feat_size must be positive if tile_feat_size > 0.")
        if tile_feat_size % bulk_feat_size != 0:
            raise ValueError("tile_feat_size must be divisible by bulk_feat_size.")
    elif bulk_feat_size < 0:
        raise ValueError("bulk_feat_size cannot be negative.")

    # --- Thread Management ---
    threads_list = [] 

    # Create and start each thread
    for i in range(num_threads): # Use updated num_threads
        log_tid = i 
        # Create a thread object, targeting the gather_thread function
        thread_obj = threading.Thread( 
            target=gather_thread,
            args=(
                log_tid,
                num_threads,
                num_points,
                num_offsets,
                num_tiles_per_pt,
                tile_feat_size,
                bulk_feat_size,
                source_masks,
                sources,
                gemm_buffers 
            )
        )
        
        
        threads_list.append(thread_obj)
        thread_obj.start() # Start the thread's execution

    # Wait for all threads to complete their work
    for thread_obj in threads_list: # Use updated threads_list
        thread_obj.join()

    # At this point, 'gemm_buffers' has been modified by the worker threads.
    # No explicit return is needed if 'gemm_buffers' was intended to be modified in-place.

# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    # Define some example parameters and data
    # Note: For a real test, these would need to be appropriately sized and valued.
    _num_pts = 8  
    _num_tiles = 4 
    _tile_feats = 16 
    _bulk_feats = 4  
    _n_threads = 1 
    _n_offsets = 2 
    _total_feats_pt = _num_tiles * _tile_feats 
    
    # Initialize dummy data
    # For 'sources', ensure enough data for all points and their features
    _src_data = [float(i) for i in range(_num_pts * _total_feats_pt)] 
    
    # 'source_masks' needs to be [num_offsets * num_points]
    # Example: if create_in_out_masks is fixed, this would be populated by local indices.
    # For this standalone example, let's simulate source_masks providing local slot indices (0 or -1 for simplicity here)
    
    _src_masks_data = [-1]*(_num_pts * _n_offsets) # Placeholder for source masks
    # First offset has local indices for the first 4 points, rest are -1
    # Second offset has local indices for the last 4 points, rest are -1
    _src_masks_data = [0, -1, -1, -1, -1, -1, -1,-1,
                       4,  1,  2,  3,  -1, -1, -1, -1]
    
    _src_bufs_data = [0.0] * (5 * _num_tiles * _tile_feats)  # Initialize output buffers     

    print(f"Starting Python threaded simulation with {_n_threads} threads...")
    print(f"Initial sum of gemm_buffers: {sum(_src_bufs_data)}")

    python_threaded_gather_simulation(
        num_threads=_n_threads,  # Updated parameter name
        num_points=_num_pts,
        num_offsets = _n_offsets,
        num_tiles_per_pt=_num_tiles,
        tile_feat_size=_tile_feats,
        bulk_feat_size=_bulk_feats,
        source_masks=_src_masks_data,
        sources=_src_data,
        gemm_buffers=_src_bufs_data
    )

    print("Python threaded simulation finished.")
    print(f"Final sum of gemm_buffers: {sum(_src_bufs_data)}")
    # Add more checks here if needed, e.g., print parts of the buffer.
    
    
