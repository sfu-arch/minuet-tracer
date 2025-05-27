import os
import sys
import json
import argparse # Added for command-line arguments
import numpy as np # Added for gemm_buffer

# Attempt to determine the build directory relative to this script
# This is a common setup but might need adjustment based on your actual build process
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming c++ build directory is ../c++/build relative to this script's parent directory
# If minuet/python_tracer.py and minuet/c++/build
cpp_build_dir_guess1 = os.path.join(script_dir, 'c++', 'build')
# If minuet/python_tracer.py and minuet/build (if CMake builds in a top-level build dir)
cpp_build_dir_guess2 = os.path.join(os.path.dirname(script_dir), 'build')
# If minuet/python_tracer.py and minuet/c++/build (if script is in root of project)
cpp_build_dir_guess3 = os.path.join(script_dir, '..', 'c++', 'build')


# Add the guessed build directory to Python's path if it exists
added_to_path = False
for guess_dir in [cpp_build_dir_guess1, cpp_build_dir_guess2, cpp_build_dir_guess3]:
    if os.path.isdir(guess_dir):
        print(f"Attempting to add {guess_dir} to sys.path")
        sys.path.insert(0, os.path.abspath(guess_dir))
        added_to_path = True
        break

if not added_to_path:
    print("Could not automatically find C++ build directory. Ensure it's in PYTHONPATH or sys.path.")

try:
    import minuet_cpp_module as minuet_cpp
    print("Successfully imported minuet_cpp_module.")
except ImportError as e:
    print(f"Error importing minuet_cpp_module: {e}")
    print("Please ensure the module is built and in the Python path.")
    print("Checked common locations but failed. Verify your CMake build output directory.")
    sys.exit(1)

# --- Configuration Loading ---
# Setup argument parser
parser = argparse.ArgumentParser(description='Minuet Python Tracer Script')
parser.add_argument('--config', type=str, help='Path to the configuration JSON file')
args = parser.parse_args()

# Determine the path to config.json
if args.config:
    config_file_path = args.config
    print(f"Using configuration file from command line: {config_file_path}")
else:
    # Default behavior: try to find config.json in the project root
    # (parent directory of the directory containing this script if this script is in c++)
    script_dir_for_config = os.path.dirname(os.path.abspath(__file__))
    project_root_dir_for_config = os.path.abspath(os.path.join(script_dir_for_config, '..'))
    config_file_path = os.path.join(project_root_dir_for_config, "config.json")
    print(f"Config file not provided via command line. Using default path: {config_file_path}")

# Add project_root_dir to sys.path for Python utility modules like minuet_mapping
# This needs to be defined before the try-except block for imports
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir) # Insert at high priority
    print(f"Added project root {project_root_dir} to sys.path for utility modules.")

if os.path.exists(config_file_path):
    print(f"Attempting to load configuration from: {config_file_path}")
    if minuet_cpp.load_config_from_file(config_file_path):
        print("Successfully loaded configuration into C++ module.")
        # Optionally, retrieve and print some config values to verify
        cpp_config = minuet_cpp.get_global_config()
        print(f"  C++ Config NUM_THREADS: {cpp_config.NUM_THREADS}")
        print(f"  C++ Config I_BASE: {hex(cpp_config.I_BASE)}")
        # print(f"  C++ Config output_dir: {cpp_config.output_dir}") # output_dir is not directly on MinuetConfigReader yet
    else:
        print(f"Failed to load configuration from {config_file_path} into C++ module.")
        sys.exit(1)
else:
    print(f"Configuration file not found at {config_file_path}. Please ensure it exists.")
    sys.exit(1)


# Import other necessary Minuet Python modules (if they are in the same directory or PYTHONPATH)
# These are needed for constants and Python-side logic not (yet) in C++
try:
    # These Python-side configs might now be redundant if C++ config is the source of truth
    # Or they might be used for Python-specific parts of the script
    from minuet_mapping import PHASES as PY_PHASES, OPS as PY_OPS, TENSORS as PY_TENSORS #, output_dir as PY_output_dir, debug as PY_debug, NUM_THREADS as PY_NUM_THREADS, I_BASE as PY_I_BASE, TILE_BASE as PY_TILE_BASE, QK_BASE as PY_QK_BASE, QI_BASE as PY_QI_BASE, QO_BASE as PY_QO_BASE, PIV_BASE as PY_PIV_BASE, KM_BASE as PY_KM_BASE, WO_BASE as PY_WO_BASE, SIZE_KEY as PY_SIZE_KEY, SIZE_INT as PY_SIZE_INT
    from minuet_gather import create_in_out_masks, compact_bar_chart # Removed write_metadata import from minuet_gather
    # from minuet_config import GEMM_ALIGNMENT, GEMM_WT_GROUP, GEMM_SIZE, NUM_TILES, TILE_FEATS, BULK_FEATS, TOTAL_FEATS_PT
except ImportError as e:
    print(f"Failed to import Minuet Python utility modules (minuet_mapping, etc.): {e}")
    print("Ensure these Python files are in the same directory or accessible via PYTHONPATH.")
    sys.exit(1)


# Helper to convert Python list of tuples to list of C++ Coord3D objects
def py_coords_to_cpp_coords(py_coords_list):
    cpp_coords_list = []
    for item in py_coords_list:
        if isinstance(item, minuet_cpp.Coord3D): # If already a C++ Coord3D object
            cpp_coords_list.append(item)
        elif isinstance(item, tuple) and len(item) == 3: # If a Python tuple
            cpp_coords_list.append(minuet_cpp.Coord3D(item[0], item[1], item[2]))
        else:
            raise TypeError(f"Invalid item type for coordinate conversion: {type(item)}")
    return cpp_coords_list

def main():
    # Retrieve the global config object from C++
    cpp_global_config = minuet_cpp.get_global_config()

    # --- Initial Data Setup (matches Python script's example) ---
    initial_coords_tuples_raw = [
        (1, 5, 0), (0, 0, 2), (0, 1, 1), (0, 0, 3)
    ]
    initial_coords_cpp = py_coords_to_cpp_coords(initial_coords_tuples_raw)

    stride = 1
    offset_coords_tuples_raw = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                offset_coords_tuples_raw.append((dx, dy, dz))
    offset_coords_cpp = py_coords_to_cpp_coords(offset_coords_tuples_raw)

    # Set debug flag (optional, default is false in C++)
    minuet_cpp.set_debug_flag(cpp_global_config.debug) # Use config's debug flag
    
    # Clear any previous trace data
    minuet_cpp.clear_mem_trace()

    # --- Phase 1: Radix Sort (Unique Sorted Input Coords with Original Indices) ---
    print(f"\\n--- Phase: {'RDX'} (RDX) with {cpp_global_config.NUM_THREADS} threads ---")
    unique_indexed_coords_cpp = minuet_cpp.compute_unique_sorted_coords(initial_coords_cpp, stride)

    # --- Phase 2: Build Queries ---
    print(f"--- Phase: {'QRY'} (QRY) ---")
    query_data_cpp = minuet_cpp.build_coordinate_queries(unique_indexed_coords_cpp, stride, offset_coords_cpp)

    # --- Phase 3: Sort Query Keys ---
    minuet_cpp.set_curr_phase('SRT') # "SRT"
    print(f"--- Phase: {minuet_cpp.get_curr_phase()} (SRT) ---")

    # --- Phase 4: Tile and Pivot Generation ---
    print(f"--- Phase: {'PVT'} (PVT) ---")
    # Assuming cpp_global_config.NUM_TILES means "items per tile" based on Python default and usage.
    tiles_pivots_data_cpp = minuet_cpp.create_tiles_and_pivots(unique_indexed_coords_cpp, cpp_global_config.NUM_TILES)

    # --- Phase 5: Lookup ---
    print(f"--- Phase: {'LKP'} (LKP) ---")
    # The last argument to perform_coordinate_lookup is tile_size_param, which should be items_per_tile.
    kernel_map_result_cpp = minuet_cpp.perform_coordinate_lookup(
        unique_indexed_coords_cpp, query_data_cpp.qry_keys, query_data_cpp.qry_in_idx,
        query_data_cpp.qry_off_idx, query_data_cpp.wt_offsets,
        tiles_pivots_data_cpp.tiles, tiles_pivots_data_cpp.pivots, cpp_global_config.NUM_TILES
    )
    minuet_cpp.set_curr_phase("") # Clear phase

    # --- Print Debug Information (if C++ debug was enabled) ---
    if minuet_cpp.get_debug_flag():
        print("\nSorted Source Array (Coordinate, Original Index) from C++:")
        for idxc_item in unique_indexed_coords_cpp:
            print(f"  {idxc_item}") 

        print("\nQuery Segments (Example of accessing qry_keys):")
        print(f"  Total query keys: {len(query_data_cpp.qry_keys)}")
        for i in range(min(5, len(query_data_cpp.qry_keys))):
             print(f"  Query key {i}: {query_data_cpp.qry_keys[i]}")


        print("\nKernel Map from C++:")
        if not kernel_map_result_cpp: # kernel_map_result_cpp is a map
            print("  Kernel map is empty.")
        for offset_key_int, matches_list in kernel_map_result_cpp.items():
            # Convert offset_key (uint32_t) back to Coord3D for display
            # Find the original Python tuple offset for display if needed, or use from_signed_key
            offset_as_coord_cpp = minuet_cpp.Coord3D.from_signed_key(offset_key_int)
            print(f"  Offset {offset_as_coord_cpp} (Key: {hex(offset_key_int)}):")
            if not matches_list:
                print("    No matches")
            else:
                for match_pair in matches_list: # match_pair is std::pair<int, int>
                    print(f"    Match: Input original_idx: {match_pair[0]} -> Query source_original_idx: {match_pair[1]}")
    
    # --- Retrieve and Write Memory Trace ---
    mem_trace_cpp = minuet_cpp.get_mem_trace()
    print(f"\nMemory Trace Entries from C++ ({len(mem_trace_cpp)} total):")
    for i in range(min(len(mem_trace_cpp), 10)):
        print(f"  {mem_trace_cpp[i]}") 
    if len(mem_trace_cpp) > 10:
        print(f"... and {len(mem_trace_cpp) - 10} more entries")

    # Create output directory if it doesn't exist
    # Use output_dir from C++ module if available, else from Python config
    current_output_dir = cpp_global_config.output_dir # Use config's output_dir
    
    if not os.path.exists(current_output_dir):
        os.makedirs(current_output_dir)
        print(f"Created output directory: {current_output_dir}")

    map_trace_filename = os.path.join(current_output_dir, "map_trace_cpp.bin.gz")
    kernel_map_filename = os.path.join(current_output_dir, "kernel_map_cpp.bin.gz")
    metadata_filename = os.path.join(current_output_dir, "metadata.bin.gz") # Added for metadata file
    
    map_trace_checksum_cpp = 0 # Initialize with a default value (e.g., 0 or -1)
    kernel_map_checksum_cpp = 0 # Initialize

    try:
        print(f"Writing gmem trace to {map_trace_filename}")
        map_trace_checksum_cpp = minuet_cpp.write_gmem_trace(map_trace_filename)
        print(f"C++ calculated CRC32 for gmem trace: {hex(map_trace_checksum_cpp)}")
        
        print(f"Writing kernel map to {kernel_map_filename}")
        kernel_map_checksum_cpp = minuet_cpp.write_kernel_map_to_gz(kernel_map_result_cpp, kernel_map_filename, offset_coords_cpp)
        print(f"C++ calculated CRC32 for kernel map: {hex(kernel_map_checksum_cpp)}")

    except Exception as e:
        print(f"Error during file writing: {e}")
        sys.exit(1)

    print("\nC++ Minuet mapping trace generation (via Python) complete.")

    ####################### Phase 2 Gather/Scatter (Python side) 
    offsets_active = []
    slot_array = []
    # The .items() from the bound C++ map (kernel_map_result_cpp) 
    # should respect the C++ map's iteration order.
    for offset_key_int, matches_list in kernel_map_result_cpp.items():
        # matches_list is a list of std::pair<int, int> from C++ (bound as list of pair-like objects)
        if matches_list: # Only include if there are matches (i.e., list is not empty)
            offsets_active.append(offset_key_int)
            slot_array.append(len(matches_list))
        
    print("\\n--- Phase: Gather/Scatter Metadata (Python using C++ KMap order) ---")
    if cpp_global_config.debug: 
        print("Offsets active (derived from C++ KernelMapType iteration order):")
        for o_idx, o_key in enumerate(offsets_active):
            # Accessing kernel_map_result_cpp[o_key] to get length for debug print
            print(f"  [{o_idx}] {hex(o_key)} (Matches: {len(kernel_map_result_cpp[o_key])})")
        print("Slot array (derived from C++ KernelMapType iteration order):", slot_array)

    # Python's greedy_group and create_in_out_masks will be used.
    # They need the kmap (now kernel_map_result_cpp), offsets_active, and slot_array.
    # The values in kernel_map_result_cpp are lists of C++ pair-like objects,
    # which should be usable by the Python functions if they access elements via indexing (e.g., pair[0], pair[1]).
    
    # Prepare 'slots' argument for greedy_group_cpp. It expects a list of integers.
    # slot_array was previously created as: slot_array.append(len(matches_list))
    # This 'slot_array' is the list of slot sizes.
    # Ensure slot_array is sorted in descending order as expected by greedy_group_cpp
    # (The C++ version currently assumes this precondition based on Python's behavior)
    # If kernel_map_result_cpp.items() gives items sorted by key (offset_key_int),
    # and we want to group based on slot sizes (match list lengths), we need to sort slot_array.
    # However, the Python greedy_group took the kmap, offsets_active, and slot_array.
    # The C++ `greedy_group_cpp` takes only the list of slot sizes.
    # Let's use the 'slot_array' which contains lengths of match lists.
    # The Python greedy_group sorts this internally if it's passed as a dictionary.
    # The C++ version expects a pre-sorted list of slot sizes.
    
    # Create the list of slot sizes (lengths of match lists)
    # The original python greedy_group took `kernel_map_result_cpp, offsets_active, slot_array`
    # The C++ `greedy_group_cpp` takes `slots` (a list of integers, assumed sorted descending).
    # `slot_array` here is already the list of match list lengths, corresponding to `offsets_active`.
    # We need to sort `slot_array` in descending order for the C++ function.
    cpp_slots_arg = sorted(slot_array, reverse=True)

    print(f"Calling C++ greedy_group_cpp with {len(cpp_slots_arg)} slot sizes.")
    if cpp_global_config.debug:
        print(f"  Slot sizes for C++ greedy_group_cpp (sorted desc): {cpp_slots_arg[:10]}...")


    # Call C++ greedy_group_cpp
    greedy_group_result_cpp = minuet_cpp.greedy_group_cpp(
        cpp_slots_arg, # Pass the sorted list of slot sizes
        alignment=cpp_global_config.GEMM_ALIGNMENT,
        max_group_items=cpp_global_config.GEMM_WT_GROUP, # Renamed from max_group
        max_raw_slots=cpp_global_config.GEMM_SIZE       # Renamed from max_slots
    )

    # Extract results from the GreedyGroupResult object
    slot_indices = greedy_group_result_cpp.pos_indices
    # The 'groups' in C++ GreedyGroupResult contains GroupInfo objects.
    # Python's greedy_group returned tuples: (members, base_addr, req, alloc)
    # We need to adapt if subsequent Python code expects this tuple structure.
    # For now, let's assume subsequent code can handle the GroupInfo objects or we adapt it.
    groups_cpp_objects = greedy_group_result_cpp.groups 
    membership = greedy_group_result_cpp.membership
    gemm_list_cpp = greedy_group_result_cpp.gemm_list # List of GemmInfo objects
    total_slots = greedy_group_result_cpp.total_slots_allocated
    gemm_checksum_cpp = greedy_group_result_cpp.checksum # Checksum from C++ write_gemm_list_cpp

    # Adapt groups_cpp_objects if necessary for compact_bar_chart or other Python consumers
    # compact_bar_chart expects: list of (se, addr, req, alloc)
    # se is (start_pos, end_pos) which corresponds to group.members[0] and group.members[-1]
    # if members are original indices.
    # The C++ GroupInfo.members are indices into the *input `slots` to greedy_group_cpp*.
    # This needs careful handling if compact_bar_chart relies on original global offset indices.
    # For now, let's construct a compatible 'groups_for_python' list.
    # The 'members' in C++ GroupInfo are indices relative to the `cpp_slots_arg` passed to it.
    # This is different from Python's `groups` which had original offset indices.
    # This part might need significant rework depending on how `compact_bar_chart` uses `groups`.
    # Let's assume for now that `compact_bar_chart` can be adapted or the C++ group structure is sufficient.
    # The `groups` variable for `compact_bar_chart` in Python was:
    # groups.append((cur_members.copy(), addr, req, alloc)) where cur_members were original indices.
    # The C++ GroupInfo has `members` (indices into sorted slot list), `base_addr`, `required_slots`, `allocated_slots`.
    
    # For compact_bar_chart, it expects: (se, addr, req, alloc)
    # se = (start_pos, end_pos)
    # Let's try to make a simplified groups list for the chart, using the C++ group info.
    # The 'members' in C++ GroupInfo are indices into the sorted slot list.
    # This is not directly (start_pos, end_pos) of original items.
    # For charting, we might just use a group index or a simplified representation.
    # Using (first_member_idx, last_member_idx) from the sorted list as a proxy for 'se'.
    groups_for_chart = []
    for g_info in groups_cpp_objects:
        # g_info.members are indices into the sorted list of slot sizes.
        # This is not directly (start_pos, end_pos) of original items.
        # For charting, we might just use a group index or a simplified representation.
        # Using (first_member_idx, last_member_idx) from the sorted list as a proxy for 'se'.
        if g_info.members:
            se_proxy = (g_info.members[0], g_info.members[-1]) 
        else:
            se_proxy = (-1, -1)
        groups_for_chart.append(
            (se_proxy, g_info.base_addr, g_info.required_slots, g_info.allocated_slots)
        )


    # slot_dict maps original offset indices to their starting slot index in the GEMM buffer.
    # The `pos_indices` from C++ `greedy_group_cpp` are for the `cpp_slots_arg` (sorted slot sizes).
    # We need to map these back to the original `offsets_active`.
    
    # Create pairs of (slot_size, original_index_in_offsets_active) from the unsorted slot_array
    # This helps map the sorted results from greedy_group_cpp back to original offset keys.
    indexed_original_slot_array = list(enumerate(slot_array)) # slot_array is unsorted here (ordered by offset_key)
    
    # Sort these pairs by slot_size descending, to match the order of cpp_slots_arg
    # Each item in sorted_indexed_slot_array is (original_index_in_offsets_active, slot_size)
    sorted_indexed_original_slot_array = sorted(indexed_original_slot_array, key=lambda x: x[1], reverse=True)

    # Now, slot_indices from C++ (which are for cpp_slots_arg) can be mapped.
    # slot_dict maps original_offset_key to its starting slot index in the GEMM buffer.
    slot_dict = {}
    for i in range(len(sorted_indexed_original_slot_array)):
        # original_idx_in_offsets_active is the index into the original offsets_active list
        original_idx_in_offsets_active = sorted_indexed_original_slot_array[i][0]
        original_offset_key = offsets_active[original_idx_in_offsets_active]
        # slot_indices[i] is the base address for the i-th item in the sorted list (cpp_slots_arg)
        slot_dict[original_offset_key] = slot_indices[i]


    # Generate masks with global idx.
    num_total_system_offsets = len(offset_coords_tuples_raw)
    num_total_system_sources = len(unique_indexed_coords_cpp)
    out_mask, in_mask = create_in_out_masks(
        kernel_map_result_cpp, 
        slot_dict, # slot_dict keys are original offset keys, values are base addresses
        num_total_system_offsets, 
        num_total_system_sources
    )
    
    print(out_mask, in_mask) # Debug print of masks
    
    print(f"Buffer size: {total_slots}")
    gemm_buffer = np.zeros(total_slots * cpp_global_config.TOTAL_FEATS_PT, dtype=np.uint16) # Use config
   
    if cpp_global_config.debug:
        print("In_mask (first few elements):", in_mask.ravel()[:20]) # Print a flattened sample
        # for i in range(in_mask.size):
        #     if in_mask.ravel()[i] != -1:
        #         offset_idx = i // len(unique_indexed_coords_cpp)
        #         point_idx = i % len(unique_indexed_coords_cpp)
        #         print(f"Writing to buffer: offset_idx={offset_idx}, point_idx={point_idx}, in_mask_val={in_mask.ravel()[i]}")
        
        print("Groups metadata (from C++ GreedyGroupResult, adapted for chart):")
        for g_chart_item in groups_for_chart: # Use the adapted list
            print(g_chart_item)    
        print("GEMM List (from C++ GreedyGroupResult):")
        # gemm_list_cpp contains GemmInfo objects. Access their fields.
        for g_info_cpp in gemm_list_cpp: 
            print(f"  NumOffsets: {g_info_cpp.num_offsets}, Gemm_M: {g_info_cpp.gemm_M}, Slots: {g_info_cpp.slots}, Padding: {g_info_cpp.padding}")

        # Print total space allocated by groups
        print(f"\nTotal allocated space (from C++): {total_slots} slots")

        print("\nPer-position slot indices (from C++ for sorted slot sizes):")
        print(slot_indices)

        print("\nGroup membership lists (from C++ for sorted slot sizes):")
        print(membership)

    # Write all checksums to file as json
    # Call write_metadata_cpp
    print(f"Writing metadata to {metadata_filename} using C++ implementation.")

    # Prepare active_offset_data for C++: list of (offset_key, base_addr, num_matches)
    # slot_dict iterates by slot size descending (due to its construction order)
    # cpp_slots_arg is num_matches, also by slot size descending
    active_offset_data_for_cpp = []
    for i, (offset_key, base_addr) in enumerate(slot_dict.items()):
        num_matches = cpp_slots_arg[i] # cpp_slots_arg is sorted list of match counts
        active_offset_data_for_cpp.append((offset_key, base_addr, num_matches))
    
    # Convert numpy masks to lists of int32_t for C++ vector binding
    # pybind11 can often handle numpy arrays directly if they are C-contiguous and of the correct type.
    # Let's try passing them directly first. If it fails, convert to .tolist().
    # out_mask and in_mask are already np.int32.
    
    metadata_checksum_cpp = minuet_cpp.write_metadata_cpp(
        out_mask, # Pass NumPy array directly
        in_mask,  # Pass NumPy array directly
        active_offset_data_for_cpp,
        num_total_system_offsets,
        num_total_system_sources,
        total_slots, # This is total_slots_allocated from greedy_group_result_cpp
        metadata_filename
    )
    print(f"C++ calculated CRC32 for metadata: {hex(metadata_checksum_cpp)}")


    checksums = {
        "map_trace_cpp.bin.gz": hex(map_trace_checksum_cpp), 
        "kernel_map_cpp.bin.gz": hex(kernel_map_checksum_cpp), 
        "gemms.bin.gz": hex(gemm_checksum_cpp),
        "metadata.bin.gz": hex(metadata_checksum_cpp) # Use CRC32 from C++
    }
    
    checksum_filename = os.path.join(current_output_dir, 'checksums_cpp.json')
    with open(checksum_filename, 'w') as f:
        json.dump(checksums, f, indent=2)
    print(f"Checksums written to {checksum_filename}")

    compact_bar_chart(groups_for_chart) # Use the adapted groups list
    print("\nPython tracer script finished.")

if __name__ == '__main__':
    main()
