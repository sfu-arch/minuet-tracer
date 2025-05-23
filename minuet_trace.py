from minuet_mapping import *
from minuet_gather import *
from minuet_config import *
from read_pcl import *
import os
# ── Example Test with Phases ──
if __name__ == '__main__':
    global phase
    # Input data
    #in_coords = [(1,5,0), (0,0,2), (0,1,1), (0,0,3)]
    in_coords, _ = read_point_cloud("/Path/To/000000.bin")
    
    visualize_point_cloud(in_coords)

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
    curr_phase = 'Sort-QKeys'
    print('--- Phase: Sort QKeys ---')
    # No sorting needed in this implementation

    # Phase 4: Create tiles and pivots for lookup optimization
    print('--- Phase: Make Tiles & Pivots ---')
    coord_tiles, pivs = create_tiles_and_pivots(uniq_coords, 2) # Renamed c_tiles to coord_tiles
    
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
    for e in mem_trace[:10]:  # Show first 10 entries only
        print(e)
    print(f"... and {len(mem_trace)-10} more entries")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    write_gmem_trace(output_dir+'map_trace.bin.gz')
    write_kernel_map_to_gz(kmap, output_dir+'kernel_map.bin.gz', off_coords)

### End of minuet_mapping.py
    ########################################################################################################################################################
    
    ############## Phase 2: Gather/Scatter Metadata Generation ##############

### Create Metadatas for kernel map
#### - First sort the list of offsets based on the length of matches
#### - Create an in, out mask for gather scatter [#Total slots * #Total points]
#### - mask[offset_idx][point_idx] = -1 (if not matched) otherwise the position of input in original input array. The points in slot array are listed based on sorted order of coordinates.
#### - offsets_active is a sparse list of offsets that have atleast one match
#### - slot array is number of slots for each offset.

    sorted_kmap = sorted(kmap.items(), key=lambda item: len(item[1]), reverse=True)
    
    if debug:
        print(sorted_kmap)

    # Count slot_array for allocation
    slot_array = []
    for off_idx, matches in sorted_kmap:
        if len(matches) > 0:
            slot_array.append(len(matches))

    # Create kernel map like this: (offset_idx, source_original_idx, target_original_idx)
    # Example kmap entry for Offset 1: [(((target_coord_X, target_coord_Y, target_coord_Z), target_orig_idx=1), ((source_coord_X, source_coord_Y, source_coord_Z), source_orig_idx=2))]
    # dict[offset, List[(src_idx, dst_idx)]]
    idx_kmap = {}
    for off_idx, matches in sorted_kmap:
        for in_coord, out_coord in matches:
            src_idx = in_coord[1]
            dst_idx = out_coord[1]
            if off_idx not in idx_kmap:
                idx_kmap[off_idx] = []
            idx_kmap[off_idx].append((src_idx, dst_idx))
    
    from minuet_gather import create_in_out_masks
    out_mask, in_mask, offsets_active, slot_addr, metadata_checksum = create_in_out_masks(idx_kmap, len(off_coords), len(in_coords), len(uniq_coords))


    if debug:
        print(slot_addr)
        print(offsets_active)
        print(out_mask)
        print(in_mask)
    
    matches = 0
    for entry, item in out_mask.items():
        matches += len(item)
    print(f"Total matches: {matches} out of {len(off_coords)*len(in_coords)}")
    print(f"Metadata Checksum: {metadata_checksum}")

    # print("Out Mask:")
    # for i in range(len(out_mask)):
    #     print(off_coords[i // len(in_coords)], out_mask[i])
    # print("In Mask:")
    # for i in range(len(in_mask)):
    #     print(off_coords[i // len(uniq_coords)], in_mask[i])
       # Replace the current mask printing with this more readable version
    # print("\n=== Out Mask (showing non-default entries only) ===")
    # print("Format: Offset (x,y,z) -> [source_idx] = (local_idx, source_idx)")
    
    # print("\n=== In Mask (showing non-default entries only) ===")
    # print("Format: Offset (x,y,z) -> [target_idx] = (local_idx, target_idx)")
    # non_default_count = 0
    
    # # Group by offset for readability
    # for o in range(len(off_coords)):
    #     entries = []
    #     for t in range(len(uniq_coords)):
    #         mask_idx = o * len(uniq_coords) + t
    #         if mask_idx < len(in_mask) and in_mask[mask_idx] != (-1, -1):
    #             entries.append(f"[{t}] = {in_mask[mask_idx]}")
    #             non_default_count += 1
                
    #     if entries:  # Only print offsets that have matches
    #         print(f"Offset {off_coords[o]}: {', '.join(entries)}")
    
    # print(f"Total non-default entries: {non_default_count} out of {len(in_mask)}")



    from minuet_gather import greedy_group
    
    pos_indices, groups, membership, gemm_list, gemm_checksum = greedy_group(
        idx_kmap,
        offsets_active,
        slot_array,
        alignment=GEMM_ALIGNMENT,
        max_group=GEMM_WT_GROUP,
        max_slots=GEMM_SIZE,
    )
    
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
        print(pos_indices)

        print("\nGroup membership lists:")
        print(membership)


    # Write all checksums to file as json
    checksums = {
        "metadata.bin.gz": metadata_checksum,
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
