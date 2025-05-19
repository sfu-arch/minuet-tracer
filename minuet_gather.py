from itertools import islice
import math
import threading # Import the threading module


def create_in_out_masks(kernel_map, num_sources, num_targets):
    """
    Args:
      kernel_map   : dict[int, List[
                       ((tgt_coord, tgt_idx),
                        (src_coord, src_idx))
                     ]]
      num_sources  : total number of source points (N_in)
      num_targets  : total number of target points (N_out)

    Returns:
      out_mask     : List[int] of length (num_offsets * num_sources)
                     where out_mask[o * num_sources + s] = local index
                     of source s in kernel_map[o], or -1 if absent.

      in_mask     : List[int] of length (num_offsets * num_targets)
                     where in_mask[o * num_targets + t] = local index
                     of target t in kernel_map[o], or -1 if absent.
    """
    num_offsets = max(kernel_map.keys()) + 1

    # initialize both to -1
    out_mask = [-1] * (num_offsets * num_sources)
    in_mask = [-1] * (num_offsets * num_targets)

    # Count slot_array for allocation
    slot_array = []

    # fill in masks
    for o, matches in kernel_map.items():
        if len(matches) > 0:
            slot_array.append(len(matches))
        for local_i, (in_idx,out_idx) in enumerate(matches):
            # record where in the list this target appears
            # Preallocates a slot within the source buffer for the array.
            in_mask[o * num_targets + in_idx] = local_i # Corrected to store local_i
            # record where in the list this source appears
            out_mask[o * num_sources + out_idx] = local_i # Corrected to store local_i


    slot_array = [0] + [sum(slot_array[:i+1]) for i in range(len(slot_array))]
    offsets_active = [off_idx for off_idx, matches in kernel_map.items() if len(matches) > 0]
    return out_mask, in_mask, offsets_active, slot_array





    return out_mask, in_mask


def worker_thread_task(
    # Arguments defining this thread's share of work
    log_tid: int, 
    num_threads: int, 

    # Original processing parameters (passed through)
    num_points: int,
    num_tiles_per_pt: int, 
    tile_feat_size: int, 
    bulk_feat_size: int,     
    num_offsets: int,
    offset_cumsum_pad: list, 
    source_masks: list,
    sources: list,
    source_buffers: list # Shared output list/array, modified by this worker
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
            for off_cfg_idx in range(num_offsets): 
                mask_idx = off_cfg_idx * num_points + pt_idx 
                
                assert(0 <= mask_idx < len(source_masks))
                
                dest_slot_relidx = source_masks[mask_idx] 
                
                if dest_slot_relidx == -1: continue # Offset not present for this point
                # This assert remains valid as offset_cumsum_pad is still indexed by off_cfg_idx
                assert(0 <= off_cfg_idx < len(offset_cumsum_pad))
                
                # MODIFIED LOGIC FOR dest_slot_base:
                # Assumes offset_cumsum_pad[off_cfg_idx] is the base *feature* address for this offset config's block,
                # and dest_slot_relidx is the local *slot* index (0-indexed) of the point within this block.
                dest_slot_base = (offset_cumsum_pad[off_cfg_idx] + dest_slot_relidx) * total_feats_per_pt

                if tile_feat_size > 0:

                    dest_tile_base_in_slot = tile_idx_in_pt * tile_feat_size
                    dest_tile_first_bulk_idx_in_slot = 0
                    
                    if bulk_feat_size > 0: # Avoid division by zero
                        dest_tile_first_bulk_idx_in_slot = dest_tile_base_in_slot // bulk_feat_size

                    for bulk_offset_in_tile in range(num_bulks_per_tile): 
                        dest_bulk_idx_in_slot = dest_tile_first_bulk_idx_in_slot + bulk_offset_in_tile
                        tile_data_bulk_lsrc = bulk_offset_in_tile * bulk_feat_size
                        vstore_addr = dest_slot_base + dest_bulk_idx_in_slot * bulk_feat_size
                        
                        for elem_idx_in_bulk in range(bulk_feat_size): 
                            tile_data_idx = tile_data_bulk_lsrc + elem_idx_in_bulk 
                            out_buf_gdest_idx = vstore_addr + elem_idx_in_bulk 
                            assert(0 <= out_buf_gdest_idx < len(source_buffers))                             
                            assert(0 <= tile_data_idx < len(tile_data))
                            #    0 <= tile_data_idx < len(tile_data):
                            source_buffers[out_buf_gdest_idx] = tile_data[tile_data_idx]

# Main function to orchestrate the threaded execution
def python_threaded_gather_simulation(
    # Parameters defining the data and processing
    num_points: int,
    num_tiles_per_pt: int, 
    tile_feat_size: int, 
    bulk_feat_size: int, 
    num_threads: int, 
    # Data arrays
    num_offsets: int,
    offset_cumsum_pad: list, 
    source_masks: list,
    sources: list,
    source_buffers: list # Output buffer that will be modified
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
        # Create a thread object, targeting the worker_thread_task function
        thread_obj = threading.Thread( 
            target=worker_thread_task,
            args=(
                log_tid,
                num_threads,
                num_points,
                num_tiles_per_pt,
                tile_feat_size,
                bulk_feat_size,
                num_offsets,
                offset_cumsum_pad,
                source_masks,
                sources,
                source_buffers 
            )
        )
        threads_list.append(thread_obj)
        thread_obj.start() # Start the thread's execution

    # Wait for all threads to complete their work
    for thread_obj in threads_list: # Use updated threads_list
        thread_obj.join()

    # At this point, 'source_buffers' has been modified by the worker threads.
    # No explicit return is needed if 'source_buffers' was intended to be modified in-place.

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
    _src_masks_data = [0,1,2,3,-1,-1,-1,-1, \
                       0,-1,-1,-1,-1,-1,-1,-1]                  
    _offsets_cumsum = [0]*(_n_offsets+1) # Placeholder for cumulative offsets

        # Populate the cumulative offsets based on the source masks
    for o in range(_n_offsets):
        # Count the number of active slots for this offset
        num_active_slots = sum(1 for i in _src_masks_data[o * _num_pts:(o + 1) * _num_pts] if i != -1)
        _offsets_cumsum[o + 1] = _offsets_cumsum[o] + (num_active_slots)
    required_buffer_size = _offsets_cumsum[-1] # Total size needed for source_buffers
    
    _src_bufs_data = [0.0] * (required_buffer_size * _total_feats_pt + _total_feats_pt) # Add some padding


    print(f"Starting Python threaded simulation with {_n_threads} threads...")
    print(f"Initial sum of source_buffers: {sum(_src_bufs_data)}")

    python_threaded_gather_simulation(
        num_points=_num_pts,
        num_tiles_per_pt=_num_tiles,
        tile_feat_size=_tile_feats,
        bulk_feat_size=_bulk_feats,
        num_threads=_n_threads, # Updated parameter name
        num_offsets=_n_offsets,
        offset_cumsum_pad=_offsets_cumsum, # Updated parameter name
        source_masks=_src_masks_data,
        sources=_src_data,
        source_buffers=_src_bufs_data
    )

    print("Python threaded simulation finished.")
    print(f"Final sum of source_buffers: {sum(_src_bufs_data)}")
    # Add more checks here if needed, e.g., print parts of the buffer.