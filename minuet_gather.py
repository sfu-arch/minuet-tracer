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
import numpy as np
from typing import Sequence, Any, Mapping
import minuet_config  # Import minuet_config for configuration settings
# from minuet_config import output_dir # Original import
import minuet_mapping as mapping_module # To access mapping_module.curr_phase
from minuet_utils import file_checksum
from pprint import pprint

# def file_checksum(filename):
#     """Calculate SHA-256 checksum of a file"""
#     hash_lib = sha256()
#     with open(filename, 'rb') as f:
#         # Read in chunks of 4K
#         for chunk in iter(lambda: f.read(4096), b''):
#             hash_lib.update(chunk)
#     return hash_lib.hexdigest()


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

    future_to_item = {}
    # Use ThreadPoolExecutor to parallelize the processing of kernel_map items.
    # The default number of worker threads will be used.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all items from kernel_map for processing.
        # kernel_map.items() yields (offset, matches_list) tuples.
        futures = [executor.submit(_process_kernel_item, item) for item in kernel_map.items()]
        future_to_item.update({future: item[0] for future, item in zip(futures, kernel_map.items())})

        # Check for failures in processing.
        failed = False
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing kernel_map item {future_to_item[future]}: {e}")
        #         failed = True
        # if failed:
        #     raise RuntimeError("create_in_out_masks: Failed to process some items in kernel_map.")

    return out_mask, in_mask


def write_metadata(out_mask, in_mask, slot_dict, slot_array, num_offsets, num_sources, total_slots, filename=minuet_config.output_dir+'metadata.bin.gz'):
    """
    Write the metadata to a gzipped binary file.
    
    Format:
    - Magic number "MINU" + version (uint32)
    - Number of offsets (uint32)
    - Number of sources (uint32)
    - Total slots allocated for gemm buffer (uint32)
    - Number of active offsets (uint32)
    - For each active offset:
        - Offset value (uint32)
        - Base address (uint32)
        - Actual size without padding (uint32)
        - Actual size with padding (uint32)
    - Masks:
        - Output mask (bytes)
        - Input mask (bytes)
    """
    with gzip.open(filename, 'wb') as f:
        # Write magic number ("MINU") and version (1)
        f.write(struct.pack('4sI', b'MINU', 1))

        # Write number of offsets and sources
        f.write(struct.pack('II', num_offsets, num_sources))

        # Write total slots allocated for gemm buffer
        f.write(struct.pack('I', total_slots))

        # Write number of active offsets
        f.write(struct.pack('I', len(slot_dict)))

        # Write each active offset, base address and actual size (without padding, with padding)
        for i, (offset, addr) in enumerate(slot_dict.items()):     # Write offset value and entry counts
            f.write(struct.pack('III', offset, addr, slot_array[i]))        
       
        # Write masks
        f.write(out_mask.tobytes())
        f.write(in_mask.tobytes())
    
    checksum = file_checksum(filename)
    return checksum


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
    for i in range(n - 1):
        if slots[i] < slots[i + 1]:
            raise ValueError("slots must be sorted in descending order")
        
    # if not all(slots[i] >= slots[i + 1] for i in range(n - 1)):
        # raise ValueError("slots must be sorted in descending order")
    
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
            
    pprint(gemm_list)

    return pos_indices, groups, membership, gemm_list, total_slots


def write_gemm_list(gemm_data_list, filename = minuet_config.output_dir+"gemms.bin.gz"):
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
    # num_offsets, gemm_M, padding
    header_format = '<III'  # Updated format to match write_gemm_list
    header_size = struct.calcsize(header_format)

    with gzip.open(filename, 'rb') as f: # Open in binary read mode
        while True:
            # Read the header for the next GEMM entry
            packed_header = f.read(header_size)
            if not packed_header:
                break # End of file
            if len(packed_header) < header_size:
                raise EOFError("Incomplete GEMM header found. File might be corrupted.")

            num_offsets, gemm_M, padding = struct.unpack(header_format, packed_header) # Updated unpacking

            gemm_entry = {
                'num_offsets': num_offsets,
                'gemm_M': gemm_M,
                'gemm_N': num_offsets, # Reconstruct gemm_N, as it's num_offsets
                'padding': padding,
                # 'offsets': [], # These were not written, so remove or handle if needed later
                # 'inputs': (),  # These were not written
                # 'outs': ()    # These were not written
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
    thread_id: int, 
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
    # Thread synchronization
    gmem_lock = threading.Lock()
    t_local = threading.local()

    def record_local(thread_id, op, addr):
        """Record a memory access in thread-local storage"""
        if not hasattr(t_local, 'local_trace'):
            t_local.local_trace = []
            
        tensor = mapping_module.addr_to_tensor(addr)
        # Assuming mapping_module.curr_phase is set appropriately for the gather operation
        entry = (mapping_module.curr_phase, thread_id, op, tensor, addr)
        t_local.local_trace.append(entry)

    def flush_local_trace():
        """Transfer thread-local trace to global mem_trace and clear local trace"""
        if hasattr(t_local, 'local_trace') and t_local.local_trace:
            with gmem_lock:
                minuet_config.mem_trace.extend(t_local.local_trace)
                t_local.local_trace = [] # Clear after flushing

    assert(tile_feat_size % bulk_feat_size == 0)    
    num_bulks = tile_feat_size // bulk_feat_size
    total_feats_per_pt = num_tiles_per_pt * tile_feat_size
    # Each thread_id processes points in an interleaved fashion
    # e.g., if num_threads = 4, thread_id = 0, it processes points 0, 4, 8, ...
    # This is a simulation of the warp-level parallelism in CUDA.
    # Each thread will process a subset of points based on its logical ID
    # and the total number of threads.
    # This is to ensure that neighboring threads process adjacent memory locations at a time.
    
    for pt_idx in range(thread_id, num_points, num_threads): 
        pt_base = pt_idx * total_feats_per_pt
        
        for tile_idx in range(num_tiles_per_pt):
            # --- LOAD: cut the tile into bulks in one go ---
            tile_start = pt_base + tile_idx * tile_feat_size
            
            for b in range(num_bulks):
                bulk_start_addr = tile_start + b * bulk_feat_size
                bulk_end_addr = tile_start + (b + 1) * bulk_feat_size

                # Get the bulk data if data is available 
                # Else it is just a simulation.
                bulk = sources[bulk_start_addr:bulk_end_addr] if sources else None

                # Record the load operation in local trace
                record_local(thread_id, mapping_module.OPS['R'], minuet_config.IV_BASE + bulk_start_addr*minuet_config.SIZE_FEAT)

                
            # --- STORE: for each offset, copy each bulk slice into the right place ---
            for off_idx in range(num_offsets):
                mask_idx = off_idx * num_points + pt_idx
                dest_slot = source_masks[mask_idx]
                if dest_slot < 0:
                    continue

                dest_base = dest_slot * total_feats_per_pt + tile_idx * tile_feat_size
                for b in range(num_bulks):
                    bulk_start_addr = dest_base + b * bulk_feat_size
                    if bulk is not None:
                        gemm_buffers[bulk_start_addr : bulk_start_addr + bulk_feat_size] = bulk
                    record_local(thread_id, mapping_module.OPS['W'], int(np.uint64(minuet_config.GM_BASE) + bulk_start_addr*minuet_config.SIZE_FEAT))
    flush_local_trace()
    


# Main function to orchestrate the threaded execution
def mt_gather(
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
    mapping_module.curr_phase = mapping_module.PHASES['GTH']  # Set the current 

    # --- Thread Management ---
    threads_list = [] 

    # Create and start each thread
    for i in range(num_threads): # Use updated num_threads
        thread_id = i 
        # Create a thread object, targeting the gather_thread function
        thread_obj = threading.Thread( 
            target=gather_thread,
            args=(
                thread_id,
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


# scatter operation: thread function
def scatter_thread(
    thread_id: int,
    num_threads: int,
    num_points: int,          # Number of *output* points
    num_offsets: int,
    num_tiles_per_pt: int,
    tile_feat_size: int,
    bulk_feat_size: int,
    out_mask: list, # This is the in_mask from create_in_out_masks
    gemm_buffers: list,       # Source of data (indexed by gemm_slot)
    outputs: list             # Destination array (indexed by output_pt_idx)
):
    """
    The function executed by each Python thread for scatter operation.
    It processes an interleaved subset of output points.
    Reads an entire source tile from gemm_buffers into a temporary buffer,
    then writes this temporary buffer to the corresponding destination tile in outputs.
    Mirrors the bulk handling logic of gather_thread.
    """
    # Thread synchronization and local trace setup (same as gather_thread)
    gmem_lock = threading.Lock()
    t_local = threading.local()

    def record_local(thread_id, op, addr):
        if not hasattr(t_local, 'local_trace'):
            t_local.local_trace = []
        tensor = mapping_module.addr_to_tensor(addr)
        entry = (mapping_module.curr_phase, thread_id, op, tensor, addr)
        t_local.local_trace.append(entry)

    def flush_local_trace():
        if hasattr(t_local, 'local_trace') and t_local.local_trace:
            with gmem_lock:
                minuet_config.mem_trace.extend(t_local.local_trace)
                t_local.local_trace = []

    # Assumptions: num_bulks > 0, and tile_feat_size % bulk_feat_size == 0.
    num_bulks = tile_feat_size // bulk_feat_size
    total_feats_per_pt = num_tiles_per_pt * tile_feat_size

    # pt_idx here refers to an *output* point index
    for pt_idx in range(thread_id, num_points, num_threads):
        dest_pt_base = pt_idx * total_feats_per_pt
        
        for off_idx in range(num_offsets):
            mask_idx = off_idx * num_points + pt_idx
            source_slot = out_mask[mask_idx]
            if source_slot == -1:
                    continue

            for tile_idx in range(num_tiles_per_pt):
                dest_tile_base = dest_pt_base + tile_idx * tile_feat_size # Base for current output tile

                # mask_idx uses num_points, which is the number of output points for scatter

                source_tile_base = source_slot * total_feats_per_pt + tile_idx * tile_feat_size
                
                # Initialize a temporary buffer for the tile
                tile_data = [0.0] * tile_feat_size

                # --- Phase 1: Read entire source tile from gemm_buffers into tile_data ---
                for b in range(num_bulks):
                    bulk_offset = b * bulk_feat_size
                    source_bulk_addr = source_tile_base + bulk_offset
                    
                    # Record read access
                    record_local(thread_id, mapping_module.OPS['R'], int(np.uint64(minuet_config.GM_BASE) + source_bulk_addr * minuet_config.SIZE_FEAT))
                    
                    # Actual data loading into tile_data if gemm_buffers is available
                    if gemm_buffers is not None:
                        source_bulk_start = source_bulk_addr
                        source_bulk_end = source_bulk_addr + bulk_feat_size
                        
                        # Assuming source indices are valid as per original logic
                        tile_data[bulk_offset : bulk_offset + bulk_feat_size] = gemm_buffers[source_bulk_start:source_bulk_end]

                # --- Phase 2: Write from tile_data to outputs array ---
                for b in range(num_bulks):
                    bulk_offset = b * bulk_feat_size
                    dest_bulk_addr = dest_tile_base + bulk_offset
                    
                    # Record write access
                    record_local(thread_id, mapping_module.OPS['W'], minuet_config.IV_BASE + dest_bulk_addr * minuet_config.SIZE_FEAT)
                    
                    # Actual data storing only if outputs is not None AND gemm_buffers was not None (i.e., tile_data has valid data)
                    if outputs is not None and gemm_buffers is not None: # gemm_buffers check implies tile_data is valid
                        bulk_to_write = tile_data[bulk_offset : bulk_offset + bulk_feat_size]
                        
                        dest_bulk_start = dest_bulk_addr
                        dest_bulk_end = dest_bulk_addr + bulk_feat_size
                        for i, update in enumerate(bulk_to_write):
                            outputs[dest_bulk_start + i] += update
    flush_local_trace()

# Main function to orchestrate the threaded scatter execution
def mt_scatter(
    num_threads: int,
    num_points: int,          # Number of *output* points
    num_offsets: int,
    num_tiles_per_pt: int,
    tile_feat_size: int,
    bulk_feat_size: int,
    out_mask: list, # This is the in_mask from create_in_out_masks
    gemm_buffers: list,       # Source of data
    outputs: list             # Destination array
) -> None:
    """
    Orchestrates the scatter operation using multiple Python threads.
    """
    # Ensure 'SCT' phase is defined in mapping_module.PHASES
    mapping_module.curr_phase = mapping_module.PHASES['SCT']

    # --- Thread Management ---
    threads_list = []

    for i in range(num_threads):
        thread_id = i
        thread_obj = threading.Thread(
            target=scatter_thread,
            args=(
                thread_id,
                num_threads,
                num_points,
                num_offsets,
                num_tiles_per_pt,
                tile_feat_size,
                bulk_feat_size,
                out_mask,
                gemm_buffers,
                outputs
            )
        )
        threads_list.append(thread_obj)
        thread_obj.start()

    for thread_obj in threads_list:
        thread_obj.join()

    # 'outputs' array has been modified by the worker threads.





# --- Example Usage (Illustrative) ---
# if __name__ == '__main__':
#    # ... (original content will be moved to test_gather_original) ...

def test_gather_original():
    """Wraps the original gather test from the if __name__ == '__main__' block."""
    print(f"Starting Python threaded simulation for GATHER...")
    if hasattr(minuet_config, 'mem_trace'):
        minuet_config.mem_trace = [] # Clear trace for this test
    else:
        # This case should ideally be handled by the main block's default setup
        print("Warning: minuet_config.mem_trace not found for clearing.")


    # Define some example parameters and data
    _num_pts = 8  
    _num_tiles = 4 
    _tile_feats = 16 
    _bulk_feats = 4  
    _n_threads = 2 
    _n_offsets = 2 
    _total_feats_pt = _num_tiles * _tile_feats 
    
    # Initialize dummy data
    _src_data = [float(i) for i in range(_num_pts * _total_feats_pt)] 
    
    _src_masks_data = [-1]*(_num_pts * _n_offsets) 
    _src_masks_data = [0, -1, -1, -1, -1, -1, -1,-1,
                       4,  1,  2,  3,  -1, -1, -1, -1] # Example mask
    
    # Assuming 5 slots are enough for the active items in _src_masks_data
    # Max slot index used is 4, so 5 slots (0-4) needed.
    _num_active_slots_example = 5 
    _src_bufs_data = [0.0] * (_num_active_slots_example * _total_feats_pt)

    print(f"  Parameters: num_points={_num_pts}, num_offsets={_n_offsets}, num_tiles={_num_tiles}, tile_feats={_tile_feats}, bulk_feats={_bulk_feats}, n_threads={_n_threads}")
    print(f"  Initial sum of gemm_buffers: {sum(_src_bufs_data)}")

    mt_gather(
        num_threads=_n_threads,
        num_points=_num_pts,
        num_offsets = _n_offsets,
        num_tiles_per_pt=_num_tiles,
        tile_feat_size=_tile_feats,
        bulk_feat_size=_bulk_feats,
        source_masks=_src_masks_data,
        sources=_src_data,
        gemm_buffers=_src_bufs_data
    )

    print("  Python threaded GATHER simulation finished.")
    print(f"  Final sum of gemm_buffers: {sum(_src_bufs_data)}")
    
    if hasattr(minuet_config, 'mem_trace') and minuet_config.mem_trace:
        print("  Gather test mem_trace (first 5 entries):")
        for entry in islice(minuet_config.mem_trace, 5):
            print(f"    {entry}")
        if len(minuet_config.mem_trace) > 5:
            print(f"    ... and {len(minuet_config.mem_trace) - 5} more entries.")
    else:
        print("  Gather test mem_trace: No trace recorded or trace is empty.")


def test_scatter():
    """Tests the mt_scatter function."""
    print(f"Starting Python threaded simulation for SCATTER...")
    if hasattr(minuet_config, 'mem_trace'):
        minuet_config.mem_trace = [] # Clear trace for this test
    else:
        print("Warning: minuet_config.mem_trace not found for clearing.")

    # Scatter test parameters
    _num_pts_out = 2  # Number of output points
    _num_tiles = 1
    _tile_feats = 4
    _bulk_feats = 2   # 2 bulks per tile
    _n_threads = 1    # Simpler for verification
    _n_offsets = 2    # Two offsets contributing to an output point

    _total_feats_pt_out = _num_tiles * _tile_feats  # 1 * 4 = 4

    # GEMM buffer setup
    _num_slots_gemm = 2 # Only one unique slot of data from GEMM buffer for this test
    _gemm_buffers_data = [float(i+10) for i in range(_total_feats_pt_out*_num_slots_gemm)] # [10.0, 11.0, 12.0, 13.0]
    # _gemm_buffers_data = [float(i+10) for i in range(_total_feats_pt_out)] + [float(i+20) for i in range(_total_feats_pt_out)] # Effectively just _gemm_slot_data if _num_slots_gemm is 1

    # Output mask (this is the `in_mask` from `create_in_out_masks` conceptually)
    # Format: _out_mask_data[off_idx * _num_pts_out + pt_idx] = gemm_slot_idx
    # Output Point 0 gets data from GEMM slot 0 via offset 0 AND offset 1
    # Output Point 1 gets no data
    _out_mask_data = [-1] * (_n_offsets * _num_pts_out) # Initialize with -1
    # Offset 0:
    _out_mask_data[0 * _num_pts_out + 0] = 0  # Off0, PtOut0 -> Slot0
    # _out_mask_data[0 * _num_pts_out + 1] = -1 (already -1)
    # Offset 1:
    _out_mask_data[1 * _num_pts_out + 0] = 1  # Off1, PtOut0 -> Slot0
    # _out_mask_data[1 * _num_pts_out + 1] = -1 (already -1)
    # Resulting _out_mask_data = [0, -1, 0, -1]

    # Outputs array (destination)
    _outputs_data = [0.0] * (_num_pts_out * _total_feats_pt_out)
    # Expected initial _outputs_data = [0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0]

    print(f"  Parameters: num_output_points={_num_pts_out}, num_offsets={_n_offsets}, num_tiles={_num_tiles}, tile_feats={_tile_feats}, bulk_feats={_bulk_feats}, n_threads={_n_threads}")
    print(f"  GEMM buffer (slot 0 data): {_gemm_buffers_data}")
    print(f"  Out Mask (for scatter): {_out_mask_data}")
    print(f"  Initial outputs: {_outputs_data}")

    mt_scatter(
        num_threads=_n_threads,
        num_points=_num_pts_out,
        num_offsets=_n_offsets,
        num_tiles_per_pt=_num_tiles,
        tile_feat_size=_tile_feats,
        bulk_feat_size=_bulk_feats,
        out_mask=_out_mask_data,
        gemm_buffers=None,
        outputs=_outputs_data
    )

    print(f"  Final outputs: {_outputs_data}")

    # Verification
    expected_outputs_data = [0.0] * len(_outputs_data)
    # Point 0, Offset 0 contribution
    if _out_mask_data[0 * _num_pts_out + 0] != -1:
        for i in range(_total_feats_pt_out):
            expected_outputs_data[0 * _total_feats_pt_out + i] += _gemm_buffers_data[0 * _total_feats_pt_out + i]
    # Point 0, Offset 1 contribution
    if _out_mask_data[1 * _num_pts_out + 0] != -1:
        for i in range(_total_feats_pt_out):
            expected_outputs_data[0 * _total_feats_pt_out + i] += _gemm_buffers_data[1 * _total_feats_pt_out + i]
    # Point 1 receives no data as per mask, so its part in expected_outputs_data remains 0.0

    print(f"  Expected outputs (element-wise sum): {expected_outputs_data}")

    successful = True
    if len(_outputs_data) == len(expected_outputs_data):
        for i in range(len(_outputs_data)):
            if abs(_outputs_data[i] - expected_outputs_data[i]) > 1e-9:
                successful = False
                break
    else:
        successful = False # Length mismatch itself is a failure

    if successful:
        print("  Scatter test: PASSED (based on element-wise sum expectation)")
    else:
        print("  Scatter test: FAILED")

    if hasattr(minuet_config, 'mem_trace') and minuet_config.mem_trace:
        print("  Scatter test mem_trace (first 5 entries):")
        for entry in islice(minuet_config.mem_trace, 5):
            print(f"    {entry}")
        if len(minuet_config.mem_trace) > 5:
            print(f"    ... and {len(minuet_config.mem_trace) - 5} more entries.")
    else:
        print("  Scatter test mem_trace: No trace recorded or trace is empty.")


if __name__ == '__main__':
    test_to_run = "scatter"  # Options: "gather", "scatter", "both", "none"

    if test_to_run.lower() == "gather" or test_to_run.lower() == "both":
        print("\\n" + "="*30 + " GATHER TEST START " + "="*30 + "\\n")
        test_gather_original()
        print("="*30 + " GATHER TEST END " + "="*30 + "\\n")

    if test_to_run.lower() == "scatter" or test_to_run.lower() == "both":
        print("\\n" + "="*30 + " SCATTER TEST START " + "="*30 + "\\n")
        test_scatter()
        print("="*30 + " SCATTER TEST END " + "="*30 + "\\n")

    if test_to_run.lower() == "none":
        print("No tests selected to run.")
