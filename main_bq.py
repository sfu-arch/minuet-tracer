import shutil
import math
from bq_mapping import dense_ball_query_mapping, IndexedCoord, write_gmem_trace, write_ball_query_results_to_gz
from bq_gather import mt_inverted_gather, ball_query_to_input_mapping, write_gemm_list
import bq_config 
from read_pcl import read_point_cloud
from coord import Coord3Df
import os
import argparse
import json
import numpy as np


def main():
    """Main entry point for ball query operations"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ball Query Mapping and Analysis")
    parser.add_argument('--pcl-file', type=str, required=True, help="Path to the point cloud file")
    parser.add_argument('--config', type=str, default='bq_config.json', help="Path to the ball query configuration file")
    parser.add_argument('--output-dir', type=str, default=None, help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    bq_config.get_config(args.config)
    
    # Override config with command line arguments if provided
    if args.output_dir is not None:
        bq_config.output_dir = args.output_dir
    
    # Load point cloud data
    print(f"Loading point cloud from: {args.pcl_file}")
    in_coords, _ = read_point_cloud(args.pcl_file)
    print(f"Loaded {len(in_coords)} points")
    
    # Initialize memory tracing
    print("Initializing memory tracing...")
    
    # Create octree for ball query operations
    print("Building octree for ball query acceleration...")
    
    # Convert points to IndexedCoord format
    indexed_coords = []
    for i, point in enumerate(in_coords):
        coord = Coord3Df(point[0], point[1], point[2])
        indexed_coords.append(IndexedCoord(coord, i))
    
    print(f"Converted {len(indexed_coords)} points to IndexedCoord format")
    
    # Create query coordinates (same as input coordinates - each point queries around itself)
    query_coords = [(point[0], point[1], point[2]) for point in in_coords]
    
    # Perform ball query operations using dense_ball_query_mapping
    print("Performing ball query operations...")
    query_results = dense_ball_query_mapping(
        NUM_THREADS=bq_config.NUM_THREADS,
        uniq_coords=indexed_coords,
        query_coords=query_coords,
        radius=bq_config.DEFAULT_RADIUS,
        max_points=bq_config.MAX_NEIGHBORHOOD,
        use_octree=True
    )
    
    # Create output directory if it doesn't exist
    if os.path.exists(bq_config.output_dir):
        # Delete existing output directory if it exists
        shutil.rmtree(bq_config.output_dir)    
    if not os.path.exists(bq_config.output_dir):
        os.makedirs(bq_config.output_dir)

    # Write ball query memory traces first
    print("Writing ball query memory traces...")
    map_checksum = write_gmem_trace(bq_config.output_dir + "/bq_map_trace.bin.gz", sizeof_addr=8)
    
    # ── Inverted Gather Operations ──
    print("Performing inverted gather operations...")
    
    # Create dummy feature data for testing (in real application, this would be actual point features)
    num_tiles_per_pt = getattr(bq_config, 'NUM_TILES_GATHER', 4)
    tile_feat_size = getattr(bq_config, 'TILE_FEATS_GATHER', 16)
    bulk_feat_size = getattr(bq_config, 'BULK_FEATS_GATHER', 4)
    
    total_feats_per_pt = num_tiles_per_pt * tile_feat_size
    dummy_features = [float(i % 256) for i in range(len(in_coords) * total_feats_per_pt)]
    
    # Calculate unique inputs needed for gather buffer allocation
    input_to_queries = ball_query_to_input_mapping(query_results)
    num_active_inputs = len(input_to_queries)
    
    print(f"Gather parameters:")
    print(f"  Total inputs: {len(in_coords)}")
    print(f"  Active inputs for gather: {num_active_inputs}")
    print(f"  Memory efficiency: {sum(len(neighbors) for neighbors in query_results.values()) / num_active_inputs:.2f}x")
    print(f"  Features per point: {total_feats_per_pt} ({num_tiles_per_pt} tiles × {tile_feat_size} features)")
    
    # Calculate buffer size needed for query-contiguous slots
    max_neighbors = max(len(neighbors) for neighbors in query_results.values()) if query_results else 0
    slots_per_query = 1 if max_neighbors == 0 else 2 ** math.ceil(math.log2(max_neighbors))
    total_slots_needed = len(query_results) * slots_per_query
    
    # Create properly sized gather buffer
    gather_buffer_size = total_slots_needed * total_feats_per_pt
    
    print(f"Gather buffer allocation:")
    print(f"  Max neighbors per query: {max_neighbors}")
    print(f"  Slots per query (power of 2): {slots_per_query}")
    print(f"  Total slots needed: {total_slots_needed}")
    print(f"  Buffer size: {gather_buffer_size} elements")
    
    # Perform inverted gather operation
    mt_inverted_gather(
        num_threads=bq_config.N_THREADS_GATHER,
        ball_query_results=query_results,
        num_tiles_per_pt=num_tiles_per_pt,
        tile_feat_size=tile_feat_size,
        bulk_feat_size=bulk_feat_size,
        sources=dummy_features,
        gemm_buffers=None
    )

    print(f"Gather created with {len(bq_config.mem_trace)} entries")
        
    
    # Write gather memory traces
    print("Writing gather memory traces...")
    gather_checksum = write_gmem_trace(bq_config.output_dir + "/bq_gather_trace.bin.gz", sizeof_addr=8)
    
    # Write gather buffer to disk
    # print("Writing gather buffer...")
    # gather_buffer_file = bq_config.output_dir + "/gather_buffer.bin"
    # with open(gather_buffer_file, 'wb') as f:
    #     # Convert to numpy array and save as binary
    #     buffer_array = np.array(gather_buffer, dtype=np.float32)
    #     f.write(buffer_array.tobytes())
    # print(f"Gather buffer written: {len(gather_buffer)} elements ({len(gather_buffer) * 4} bytes)")
    
    # Write ball query results
    print("Writing ball query results...")
    write_ball_query_results_to_gz(query_results, bq_config.output_dir + "/bq_kmap.gz")
    

    # Generate GEMM information based on gather results
    gemm_data = [{
        'num_offsets': total_slots_needed,  # Total slots with query-contiguous layout
        'gemm_M': len(query_results),  # Number of queries
        'gemm_N': slots_per_query,     # Max slots per query (power of 2)
        'padding': slots_per_query - max_neighbors if max_neighbors > 0 else 0  # Padding per query
    }]
    
    gemm_checksum = write_gemm_list(gemm_data, bq_config.output_dir + "/bq_gemms.bin.gz")
    
    # Generate comprehensive statistics
    print("Generating statistics...")
    total_queries = len(query_results)
    total_results = sum(len(results) for results in query_results.values())
    avg_results_per_query = total_results / total_queries if total_queries > 0 else 0
    
    # Calculate memory access statistics
    memory_reduction = total_results / num_active_inputs if num_active_inputs > 0 else 1.0
    bandwidth_saved = ((total_results - num_active_inputs) / total_results * 100) if total_results > 0 else 0.0
    
    stats = {
        'ball_query': {
            'total_queries': total_queries,
            'total_results': total_results,
            'avg_results_per_query': avg_results_per_query,
            'radius': bq_config.DEFAULT_RADIUS,
            'max_points': bq_config.MAX_NEIGHBORHOOD,
            'threads': bq_config.NUM_THREADS
        },
        'gather': {
            'total_inputs': len(in_coords),
            'active_inputs': num_active_inputs,
            'memory_reduction_factor': memory_reduction,
            'bandwidth_saved_percent': bandwidth_saved,
            'tiles_per_point': num_tiles_per_pt,
            'tile_feature_size': tile_feat_size,
            'bulk_feature_size': bulk_feat_size,
            'max_neighbors_per_query': max_neighbors,
            'slots_per_query_padded': slots_per_query,
            'total_slots_allocated': total_slots_needed,
            'padding_per_query': slots_per_query - max_neighbors if max_neighbors > 0 else 0
        },
        'checksums': {
            'ball_query_trace': map_checksum,
            'gather_trace': gather_checksum,
            'gemm_list': gemm_checksum
        }
    }
    
    # Save statistics and checksums
    print("Saving statistics and checksums...")
    with open(os.path.join(bq_config.output_dir, 'ball_query_gather_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Ball query and gather operations completed successfully!")
    print(f"Results saved to: {bq_config.output_dir}")
    print(f"")
    print(f"Performance Summary:")
    print(f"  Ball Query: {stats['ball_query']['total_queries']} queries, {stats['ball_query']['total_results']} total neighbors")
    print(f"  Gather Efficiency: {stats['gather']['memory_reduction_factor']:.2f}x memory access reduction")
    print(f"  Bandwidth Saved: {stats['gather']['bandwidth_saved_percent']:.1f}%")
    print(f"  Active Inputs: {stats['gather']['active_inputs']} / {stats['gather']['total_inputs']}")
    print(f"  Query-Contiguous Layout: {stats['gather']['max_neighbors_per_query']} max neighbors → {stats['gather']['slots_per_query_padded']} slots per query (power of 2)")
    print(f"  Total Slots Allocated: {stats['gather']['total_slots_allocated']} ({stats['gather']['padding_per_query']} padding per query)")
    print(f"")
    print(f"Generated Files:")
    print(f"  - bq_map_trace.bin.gz (ball query memory trace)")
    print(f"  - bq_gather_trace.bin.gz (gather memory trace)")
    print(f"  - bq_kmap.gz (ball query results)")
    print(f"  - bq_gemms.bin.gz (GEMM metadata)")
    print(f"  - ball_query_gather_stats.json (comprehensive statistics)")



if __name__ == '__main__':
    main()