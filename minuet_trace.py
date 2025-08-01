
from minuet_mapping import *
from minuet_gather import *
from minuet_config import *
from read_pcl import *
import os
import argparse


# ── Example Test with Phases ──
if __name__ == '__main__':
    global phase
    # Input data
    
    in_coords = []
    parser = argparse.ArgumentParser(description="Minuet Mapping and Gathering Simulation")
    parser.add_argument('--pcl-file', type=str)
    parser.add_argument('--kernel', type=int, default=3, help="Kernel size for mapping",default=3)
    args = parser.parse_args()
    if args.pcl_file:
        in_coords, _ = read_point_cloud(args.pcl_file)
    else:
        in_coords = [(1,5,0), (0,0,2), (0,1,1), (0,0,3)]  
        
    stride = 1
    off_coords = []
    if args.kernel == 3:
        off_coords = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)]
    elif args.kernel == 5:
        off_coords = [(dx,dy,dz) for dx in (-2,-1,0,1,2) for dy in (-2,-1,0,1,2) for dz in (-2,-1,0,1,2)]
    # off_coords = [(0,1,-1)]
    # for i in range(len(off_coords)):
        # print(f"Offset {i}: {off_coords[i]}")
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
    
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    map_trace_checksum = write_gmem_trace(output_dir+'map_trace.bin.gz', sizeof_addr=8)
    write_kernel_map_to_gz(kmap, output_dir+'kernel_map.bin.gz', off_coords)

    
    ############## Phase 2: Gather/Scatter Metadata Generation ##############

    # No need to sort kmap as it's already a SortedByValueLengthDict
    # Sparse list of offsets with matches
    # For some reason cache is not none even if we don't use it
    # So invalidate it first.
    kmap._invalidate_cache()
    kmap._get_sorted_keys()
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

    ############## Phase 3: Gather/Scatter Simulation ##############


    # Calculate buffer from slot_dict and slot_array
    print(f"Buffer size: {total_slots}")
    gemm_buffer = np.zeros(total_slots*TOTAL_FEATS_PT, dtype=np.uint16)
   
    
    if debug:
        print("Groups metadata ([start, end], base, req, alloc):")
        for g in groups:
            print(g)    
        print("GEMM List:")
        for g in gemm_list:
            print(g)


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
    
    
    gather_checksum = write_gmem_trace(output_dir+'gather_trace.bin.gz', sizeof_addr=8)
    
    from minuet_gather import mt_gather
    mt_scatter(
        num_threads=2,  # Updated parameter name
        num_points=len(uniq_coords),
        num_offsets = len(off_coords),
        num_tiles_per_pt=minuet_config.NUM_TILES_GATHER,
        tile_feat_size=minuet_config.TILE_FEATS_GATHER,
        bulk_feat_size=minuet_config.BULK_FEATS_GATHER,
        out_mask=out_mask,
        gemm_buffers=None,
        outputs=None
        )
    
    scatter_checksum = write_gmem_trace(output_dir+'scatter_trace.bin.gz', sizeof_addr=8)

    # Write all checksums to file as json
    checksums = {
        "map_trace.bin.gz": map_trace_checksum,
        "metadata.bin.gz": metadata_checksum,
        "gemms.bin.gz": gemm_checksum,
        "gather_trace.bin.gz": gather_checksum,
        "scatter_trace.bin.gz": scatter_checksum,
    }
    
    import json    
    with open(output_dir+'checksums.json', 'w') as f:
        json.dump(checksums, f, indent=2)

 
    