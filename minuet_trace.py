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

    
    ############## Phase 2: Gather/Scatter Metadata Generation ##############

    # No need to sort kmap as it's already a SortedByValueLengthDict
    # Sparse list of offsets with matches
    offsets_active = list(kmap._get_sorted_keys())

    # Number of slots required by each offset
    slot_array = [len(kmap[off_idx]) for off_idx in offsets_active]
    
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
    
    # Dictionary with offsets active and position in global buffer.
    slot_dict = {offsets_active[i]: slot_indices[i] for i in range(len(slot_indices))}

    # Generate masks with global idx.
    out_mask, in_mask = create_in_out_masks(kmap, slot_dict, len(off_coords), len(uniq_coords))

    # Write metadata to file
    metadata_checksum = write_metadata(out_mask, in_mask, slot_dict, slot_array, len(off_coords), len(uniq_coords), total_slots, filename=output_dir+'metadata.bin.gz')


    # Ready for scatter simulation


    # Calculate buffer from slot_dict and slot_array
    print(f"Buffer size: {total_slots}")
    gemm_buffer = np.zeros(total_slots*TOTAL_FEATS_PT, dtype=np.uint16)
   
    
    
            

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
        "metadata.bin.gz": metadata_checksum,
        "gemms.bin.gz": gemm_checksum,
    }
    
    import json    
    with open(output_dir+'checksums.json', 'w') as f:
        json.dump(checksums, f, indent=2)


    from minuet_gather import compact_bar_chart
    compact_bar_chart(groups)


    from minuet_gather import mt_gather
    mt_gather(
        num_threads=1,  # Updated parameter name
        num_points=len(uniq_coords),
        num_offsets = len(off_coords),
        num_tiles_per_pt=minuet_config.NUM_TILES_GATHER,
        tile_feat_size=minuet_config.TILE_FEATS_GATHER,
        bulk_feat_size=minuet_config.BULK_FEATS_GATHER,
        source_masks= in_mask,
        sources=None,
        gemm_buffers= gemm_buffer
    )
    
        # Write memory trace to file
    print('\nGather Memory Trace Entries:')
    for e in mem_trace[:-1]:  # Show all entries except the last one
        print(e)
    print(f"... and {len(mem_trace)-10} more entries")
    write_gmem_trace(output_dir+'gather_trace.bin.gz', sizeof_addr=8)
    
 
    # print("\nSorted Kernel Map by Length of Matches:")
    # for off_idx, matches in sorted_kmap:
    #     if matches:
    #         print(f"  Offset {off_coords[off_idx]}: {matches}")
    # print(f"Total entries in sorted kernel map: {len(sorted_kmap)}")
    #     if matches:
    #         print(f"  Offset {off_coords[off_idx]}: {matches}")
    # print(f"Total entries in sorted kernel map: {len(sorted_kmap)}")




    # print("\nSorted Kernel Map by Length of Matches:")
    # for off_idx, matches in sorted_kmap:
    #     if matches:
    #         print(f"  Offset {off_coords[off_idx]}: {matches}")
    # print(f"Total entries in sorted kernel map: {len(sorted_kmap)}")
    #     if matches:
    #         print(f"  Offset {off_coords[off_idx]}: {matches}")
    # print(f"Total entries in sorted kernel map: {len(sorted_kmap)}")
