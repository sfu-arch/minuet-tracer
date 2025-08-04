from minuet_mapping import *
from minuet_gather import *
import minuet_config 
from read_pcl import *
import os
import argparse


# ── Example Test with Phases ──
if __name__ == '__main__':
    global phase
    # Input data
    
    in_coords = []
    # This is ugly but what the hell
    parser = argparse.ArgumentParser(description="Minuet Mapping and Gathering Simulation")
    parser.add_argument('--pcl-file', type=str, required=True, help="Path to the point cloud file")
    parser.add_argument('--kernel', type=int, default=3, help="Kernel size for mapping")
    parser.add_argument('--channel', type=int, default=16, help="Number of Channels")
    parser.add_argument('--downsample-stride', type=int, default=1, help="Stride for downsample")
    parser.add_argument('--conv-stride', type=int, default=1, help="Stride for convolution")
    parser.add_argument('--output-dir', type=str, help="Output directory for traces")
    parser.add_argument('--config', type=str, default='config.json', help="Path to the Minuet configuration file", required=True)
    args = parser.parse_args()
    minuet_config.get_config(args.config)

    # Updating configuration based on command line arguments
    if args.output_dir:
        minuet_config.output_dir = args.output_dir
    if args.channel < 16:
        minuet_config.TILE_FEATS_GATHER = args.channel
    minuet_config.NUM_TILES_GATHER = args.channel // minuet_config.TILE_FEATS_GATHER

    # Show that configuration has been loaded
    print(f"Configuration loaded from {args.config}")
    print(f"NUM_THREADS: {minuet_config.NUM_THREADS}")
    print(f"GEMM_SIZE: {minuet_config.GEMM_SIZE}")
    print(f"Output directory: {minuet_config.output_dir}")
    
    # Load configuration
    
    
    
    if args.pcl_file:
        in_coords, _ = read_point_cloud(args.pcl_file)
        
    stride = args.downsample_stride
    off_coords = [(0,0,0)]
    if args.kernel == 3:
        off_coords = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)]
    elif args.kernel == 5:
        off_coords = [(dx,dy,dz) for dx in (-2,-1,0,1,2) for dy in (-2,-1,0,1,2) for dz in (-2,-1,0,1,2)]

    ####################### Phase 1 Mapping #######################
    
    # Phase 1: Sort and deduplicate input coordinates
    print(f"\n--- Phase: {curr_phase} with {minuet_config.NUM_THREADS} threads ---")
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
    coord_tiles, pivs = create_tiles_and_pivots(uniq_coords, minuet_config.NUM_PIVOTS) # Renamed c_tiles to coord_tiles
    
    # Phase 5: Perform coordinate lookup
    print('--- Phase: Lookup ---')
    kmap = lookup(
        uniq_coords, qry_keys, qry_in_idx,
        qry_off_idx, wt_offsets, coord_tiles, pivs, 2
    )
    
    curr_phase = None
        
    # Create output directory if it doesn't exist
    if not os.path.exists(minuet_config.output_dir):
        os.makedirs(minuet_config.output_dir)
    

    map_trace_checksum = write_gmem_trace(os.path.join(minuet_config.output_dir, 'map_trace.bin.gz'), sizeof_addr=8)
    write_kernel_map_to_gz(kmap, os.path.join(minuet_config.output_dir, 'kernel_map.bin.gz'), off_coords)

    
    ############## Phase 2: Gather/Scatter Metadata Generation ##############

    kmap._invalidate_cache()
    kmap._get_sorted_keys()
    offsets_active = list(kmap._get_sorted_keys())
    # Number of slots required by each offset
    slot_array = [len(kmap[off_idx]) for off_idx in offsets_active]
    

    # Perform greedy grouping and padding.
    from minuet_gather import greedy_group
    slot_indices, groups, membership, gemm_list, total_slots = greedy_group(
        slot_array,
        alignment=minuet_config.GEMM_ALIGNMENT,
        max_group=minuet_config.GEMM_WT_GROUP,
        max_slots=minuet_config.GEMM_SIZE,
    )
    gemm_checksum = write_gemm_list(gemm_list, os.path.join(minuet_config.output_dir, 'gemms.bin.gz'))

    # Dictionary with offsets active and position in global buffer.
    slot_dict = {offsets_active[i]: slot_indices[i] for i in range(len(slot_indices))}

    # Generate masks with global idx.
    out_mask, in_mask = create_in_out_masks(kmap, slot_dict, len(off_coords), len(uniq_coords))

    # Write metadata to file
    metadata_checksum = write_metadata(out_mask, in_mask, slot_dict, slot_array, len(off_coords), len(uniq_coords), total_slots, filename=os.path.join(minuet_config.output_dir, 'metadata.bin.gz'))

    ############## Phase 3: Gather/Scatter Simulation ##############


    # Calculate buffer from slot_dict and slot_array
    print(f"Buffer size: {total_slots}")
    gemm_buffer = np.zeros(total_slots*minuet_config.TOTAL_FEATS_PT, dtype=np.uint16)
   
    
    from minuet_gather import mt_gather
    mt_gather(
        num_threads=minuet_config.N_THREADS_GATHER,  # Updated parameter name
        num_points=len(uniq_coords),
        num_offsets = len(off_coords),
        num_tiles_per_pt=minuet_config.NUM_TILES_GATHER,
        tile_feat_size=minuet_config.TILE_FEATS_GATHER,
        bulk_feat_size=minuet_config.BULK_FEATS_GATHER,
        source_masks= in_mask,
        sources=None,
        gemm_buffers= gemm_buffer
    )


    gather_checksum = write_gmem_trace(os.path.join(minuet_config.output_dir, 'gather_trace.bin.gz'), sizeof_addr=8)

    from minuet_gather import mt_gather
    mt_scatter(
        num_threads=minuet_config.N_THREADS_GATHER,  # Updated parameter name
        num_points=len(uniq_coords),
        num_offsets = len(off_coords),
        num_tiles_per_pt=minuet_config.NUM_TILES_GATHER,
        tile_feat_size=minuet_config.TILE_FEATS_GATHER,
        bulk_feat_size=minuet_config.BULK_FEATS_GATHER,
        out_mask=out_mask,
        gemm_buffers=None,
        outputs=None
        )

    scatter_checksum = write_gmem_trace(os.path.join(minuet_config.output_dir, 'scatter_trace.bin.gz'), sizeof_addr=8)

    # Write all checksums to file as json
    checksums = {
        "map_trace.bin.gz": map_trace_checksum,
        "metadata.bin.gz": metadata_checksum,
        "gemms.bin.gz": gemm_checksum,
        "gather_trace.bin.gz": gather_checksum,
        "scatter_trace.bin.gz": scatter_checksum,
    }
    
    import json    
    with open(os.path.join(minuet_config.output_dir, 'checksums.json'), 'w') as f:
        json.dump(checksums, f, indent=2)

 
    