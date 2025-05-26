from minuet_mapping import *
from minuet_gather import *
from minuet_config import *
from read_pcl import *
import os
# ── Example Test with Phases ──
if __name__ == '__main__':
    global phase
    # Input data
    in_coords = [(1,5,0), (0,0,2), (0,1,1), (0,0,3)]
    # in_coords, _ = read_point_cloud("/Path/To/000000.bin")
    
    # visualize_point_cloud(in_coords)

    stride = 1
    off_coords = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)]
    # off_coords = [(0,1,-1)]
    
    ####################### Phase 1 Mapping #######################
    
    # Phase 1: Sort and deduplicate input coordinates
    print(f"\n--- Phase: {curr_phase} with {NUM_THREADS} threads ---")
    uniq_coords = compute_unique_sorted_coords(in_coords, stride)

    # Phase 2: Build query data structures
    print('--- Phase: Build Queries ---')
    qry_keys, qry_in_idx, qry_off_idx, wt_offsets = build_coordinate_queries(
        uniq_coords, stride, off_coords
    )

    # Phase 3: Sort query keys (using existing sorted keys)
    curr_phase = PHASES['SRT']
    print('--- Phase: Sort QKeys ---')
    # No sorting needed in this implementation

    # Phase 4: Create tiles and pivots for lookup optimization
    print('--- Phase: Make Tiles & Pivots ---')
    coord_tiles, pivs = create_tiles_and_pivots(uniq_coords, NUM_PIVOTS) # Renamed c_tiles to coord_tiles
    
    # Phase 5: Perform coordinate lookup
    print('--- Phase: Lookup ---')
    kmap = lookup(
        uniq_coords, qry_keys, qry_in_idx,
        qry_off_idx, wt_offsets, coord_tiles, pivs, 2
    )
    
    curr_phase = None
    
    # Print debug information
    if debug:
        print('\nSorted Source Array (Coordinate, Original Index):')
        for idxc_item in uniq_coords: # Renamed idxc_item to idxc_item
            coord = idxc_item.coord # Renamed coord_obj to coord
            print(f"  key={hex(coord.to_key())}, coords=({coord.x}, {coord.y}, {coord.z}), index={idxc_item.orig_idx}")
            
        print('\nQuery Segments:')
        for off_idx in range(len(off_coords)):
            segment = [qry_keys[i] for i in range(len(qry_keys)) 
                       if qry_off_idx[i] == off_idx]
            # segment contains IndexedCoord objects where .coord is now a Coord3D object
            # To print the (x,y,z) of these query coordinates:
            print(f"  Offset {off_coords[off_idx]}: {[(idxc.coord.x, idxc.coord.y, idxc.coord.z) for idxc in segment]}") # Renamed ic to idxc

    if debug:
        print('\nKernel Map:')
        for off_idx, matches in kmap.items():
            print(f"  Offset {off_coords[off_idx]}: {matches}")
    
    # Write memory trace to file
    print('\nMemory Trace Entries:')
    for e in mem_trace[:-1]:  # Show all entries except the last one
        print(e)
    print(f"... and {len(mem_trace)-10} more entries")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    map_trace_checksum = write_gmem_trace(output_dir+'map_trace.bin.gz')
    write_kernel_map_to_gz(kmap, output_dir+'kernel_map.bin.gz', off_coords)

    ########################################################################################################################################################
    
    ############## Phase 2: Gather/Scatter Metadata Generation ##############

### Create Metadatas for kernel map
#### - First sort the list of offsets based on the length of matches
#### - Create an in, out mask for gather scatter [#Total slots * #Total points]
#### - mask[offset_idx][point_idx] = -1 (if not matched) otherwise the position of input in original input array. The points in slot array are listed based on sorted order of coordinates.
#### - offsets_active is a sparse list of offsets that have atleast one match
#### - slot_array is number of slots for each offset.

    # No need to sort kmap as it's already a SortedByValueLengthDict
    # Initialize offsets_active directly from keys of kmap (they're already sorted by match count)
    offsets_active = list(kmap._get_sorted_keys())
    
    # Initialize slot_array with lengths of match lists
    slot_array = [len(kmap[off_idx]) for off_idx in offsets_active]
    
    debug = True
    if debug:
        print("Offsets sorted by matches count:", offsets_active)
        print("Slot array:", slot_array)

    # Perform greedy grouping and padding.
    from minuet_gather import greedy_group
    slot_indices, groups, membership, gemm_list, total_slots, gemm_checksum = greedy_group(
        slot_array,
        alignment=GEMM_ALIGNMENT,
        max_group=GEMM_WT_GROUP,
        max_slots=GEMM_SIZE,
    )
    slot_dict = {offsets_active[i]: slot_indices[i] for i in range(len(slot_indices))}

    # Generate masks with global idx.
    out_mask, in_mask = create_in_out_masks(kmap, slot_dict, len(off_coords), len(uniq_coords))
    
    print(out_mask, in_mask)

    # Calculate buffer from slot_dict and slot_array
    print(f"Buffer size: {total_slots}")
    gemm_buffer = np.zeros(total_slots*TOTAL_FEATS_PT, dtype=np.uint16)
   
    print(in_mask)
    for i in range(in_mask.size):
        offset_idx = i//len(uniq_coords)
        point_idx = i%len(uniq_coords)
        if in_mask[i] != -1:
            print(f"Writing to buffer: {offset_idx}, {point_idx}, {in_mask[i]}")
        


    # python_threaded_gather_simulation(
    #     num_points=len(uniq_coords),
    #     num_tiles_per_pt=NUM_TILES,
    #     tile_feat_size=TILE_FEATS,
    #     bulk_feat_size=BULK_FEATS,
    #     num_threads=N_THREADS, # Updated parameter name
    #     offsets=offsets_active,
    #     offset_cumsum_pad=_offsets_cumsum, # Updated parameter name
    #     source_masks=in_mask,
    #     sources=
    #     source_buffers=gemm_buffer,
    # )


    # write_metadata_to_gz(out_mask, in_mask, slot_dict, slot_array, total_slots) 
    # from minuet_gather import create_in_out_masks
    # out_mask, in_mask, offsets_active, slot_array, metadata_checksum = create_in_out_masks(idx_kmap, len(off_coords), len(in_coords), len(uniq_coords))
    


    # if debug:
    #     print(out_mask)
    #     print(in_mask)
    #     print(offsets_active)
    #     print(slot_array)
        
    # matches = 0
    # for entry, item in out_mask.items():
    #     matches += len(item)
    # print(f"Total matches: {matches} out of {len(off_coords)*len(in_coords)}")
    # print(f"Metadata Checksum: {metadata_checksum}")

    
    

    debug = True
    
    if debug:
        print("Groups metadata ([start, end], base, req, alloc):")
    for g in groups:
        print(g)    
    print("GEMM List:")
    for g in gemm_list:
        print(g)

    if debug:
        print(gemm_list)

        # Print total space allocated by groups
        total_alloc = sum(g[3] for g in groups)
        print(f"\nTotal allocated space: {total_alloc} slots")

        print("\nPer-position slot indices:")
        print(slot_indices)

        print("\nGroup membership lists:")
        print(membership)


    # Write all checksums to file as json
    checksums = {
        "map_trace.bin.gz": map_trace_checksum,
        # "metadata.bin.gz": metadata_checksum,
        "gemms.bin.gz": gemm_checksum,
    }
    
    import json    
    with open(output_dir+'checksums.json', 'w') as f:
        json.dump(checksums, f, indent=2)


    from minuet_gather import compact_bar_chart
    compact_bar_chart(groups)



    # print("\nSorted Kernel Map by Length of Matches:")
    # for off_idx, matches in sorted_kmap:
    #     if matches:
    #         print(f"  Offset {off_coords[off_idx]}: {matches}")
    # print(f"Total entries in sorted kernel map: {len(sorted_kmap)}")
    #     if matches:
    #         print(f"  Offset {off_coords[off_idx]}: {matches}")
    # print(f"Total entries in sorted kernel map: {len(sorted_kmap)}")
    from minuet_gather import compact_bar_chart
    compact_bar_chart(groups)



    # print("\nSorted Kernel Map by Length of Matches:")
    # for off_idx, matches in sorted_kmap:
    #     if matches:
    #         print(f"  Offset {off_coords[off_idx]}: {matches}")
    # print(f"Total entries in sorted kernel map: {len(sorted_kmap)}")
    #     if matches:
    #         print(f"  Offset {off_coords[off_idx]}: {matches}")
    # print(f"Total entries in sorted kernel map: {len(sorted_kmap)}")
