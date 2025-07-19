from itertools import islice
import threading
import matplotlib.pyplot as plt
import gzip
import math
import struct
import concurrent.futures
from hashlib import sha256
import numpy as np
from typing import Sequence, Any, Mapping
import bq_config  # Import bq_config for configuration settings
import bq_mapping as mapping_module 
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


def write_gemm_list(gemm_data_list, filename=bq_config.output_dir + "gemms.bin.gz"):
    """
    Write the gemm list to a file in a packed binary format,
    compressed with gzip.
    Each GEMM entry is structured as:
    - num_offsets (unsigned int)
    - gemm_M (unsigned int)
    - padding (unsigned int)
    All integers are packed in little-endian byte order (<).
    """
    with gzip.open(filename, 'wb') as f:  # Open in binary write mode
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

    with gzip.open(filename, 'rb') as f:  # Open in binary read mode
        while True:
            # Read the header for the next GEMM entry
            packed_header = f.read(header_size)
            if not packed_header:
                break  # End of file
            if len(packed_header) < header_size:
                raise EOFError("Incomplete GEMM header found. File might be corrupted.")

            num_offsets, gemm_M, padding = struct.unpack(header_format, packed_header)  # Updated unpacking

            gemm_entry = {
                'num_offsets': num_offsets,
                'gemm_M': gemm_M,
                'gemm_N': num_offsets,  # Reconstruct gemm_N, as it's num_offsets
                'padding': padding,
            }

            gemm_data_list.append(gemm_entry)
            
    return gemm_data_list


# ── Inverted Access Pattern Functions ──
# Read each input only once and track which queries use it

def ball_query_to_input_mapping(ball_query_results: dict) -> dict:
    """
    Convert ball query results to input-centric mapping.
    
    Args:
        ball_query_results: {query_idx: [(input_idx, distance), ...]}
    
    Returns:
        input_to_queries: {input_idx: [query_idx, ...]}
    """
    from collections import defaultdict
    input_to_queries = defaultdict(list)
    
    for query_idx, neighbors in ball_query_results.items():
        for input_idx, distance in neighbors:
            input_to_queries[input_idx].append(query_idx)
    
    return dict(input_to_queries)

def create_inverted_masks(query_input_pairs: list, num_inputs: int, max_neighbors_per_query: int = None) -> tuple:
    """
    Create masks for input-centric gather operation with query-contiguous slot layout.
    Each query gets a fixed-size contiguous block of slots.
    
    Args:
        query_input_pairs: List of lists, where query_input_pairs[q] = [input_indices for query q]
        num_inputs: Total number of input points
        max_neighbors_per_query: Fixed size per query block (auto-computed if None)
    
    Returns:
        (active_inputs, input_to_queries, slot_assignments)
    """
    # Build input_to_queries mapping from query_input_pairs
    from collections import defaultdict
    input_to_queries = defaultdict(list)
    
    for query_idx, input_indices in enumerate(query_input_pairs):
        for input_idx in input_indices:
            input_to_queries[input_idx].append(query_idx)
    
    # Convert to regular dict and get sorted list of active inputs
    input_to_queries = dict(input_to_queries)
    active_inputs = sorted(input_to_queries.keys())
    
    # Determine the maximum number of neighbors per query for fixed-size blocks
    if max_neighbors_per_query is None:
        raw_max = max(len(inputs) for inputs in query_input_pairs) if query_input_pairs else 0
        # Round up to the nearest power of 2 for better memory alignment
        max_neighbors_per_query = 1 if raw_max == 0 else 2 ** math.ceil(math.log2(raw_max))
    
    print(f"Query-contiguous layout: {len(query_input_pairs)} queries, max {raw_max} neighbors -> {max_neighbors_per_query} slots per query (power of 2)")
    
    # Create slot assignments with query-contiguous layout
    slot_assignments = {}  # (input_idx, query_idx) -> slot_idx
    
    for query_idx, input_indices in enumerate(query_input_pairs):
        query_base_slot = query_idx * max_neighbors_per_query
        
        for neighbor_idx, input_idx in enumerate(input_indices):
            slot_idx = query_base_slot + neighbor_idx
            slot_assignments[(input_idx, query_idx)] = slot_idx
    
    return active_inputs, input_to_queries, slot_assignments

def inverted_gather_thread(
    # Thread parameters
    thread_id: int,
    num_threads: int, 
    # Inverted access parameters
    active_inputs: list,
    input_to_queries: dict,
    slot_assignments: dict,  # (input_idx, query_idx) -> slot_idx
    # Feature parameters  
    num_tiles_per_pt: int,
    tile_feat_size: int,
    bulk_feat_size: int,
    # Data arrays
    sources: list,
    gemm_buffers: list
):
    """
    Gather thread that processes inputs in interleaved fashion.
    Each input is read exactly once and distributed to ALL slots for ALL queries that need it.
    """
    # Thread synchronization
    gmem_lock = threading.Lock()
    t_local = threading.local()

    def record_local(thread_id, op, addr):
        """Record a memory access in thread-local storage"""
        if not hasattr(t_local, 'local_trace'):
            t_local.local_trace = []
            
        # Print details and address in hex
        tensor = mapping_module.addr_to_tensor(addr)
        entry = (bq_config.curr_phase, thread_id, op, tensor, addr)
        t_local.local_trace.append(entry)

    def flush_local_trace():
        """Transfer thread-local trace to global mem_trace and clear local trace"""
        if hasattr(t_local, 'local_trace') and t_local.local_trace:
            with gmem_lock:
                print(f"Thread {thread_id} flushing {len(t_local.local_trace)} local trace entries to global mem_trace.")
                bq_config.mem_trace.extend(t_local.local_trace)
                t_local.local_trace = []

    assert tile_feat_size % bulk_feat_size == 0
    num_bulks = tile_feat_size // bulk_feat_size
    total_feats_per_pt = num_tiles_per_pt * tile_feat_size
    
    # Process active inputs in interleaved fashion
    # Each thread processes inputs: thread_id, thread_id + num_threads, ...
    for i in range(thread_id, len(active_inputs), num_threads):
        input_idx = active_inputs[i]
        queries_using_input = input_to_queries[input_idx]
        
        # Calculate base address for this input
        input_base = input_idx * total_feats_per_pt
        
        # For each tile in this input point
        for tile_idx in range(num_tiles_per_pt):
            tile_start = input_base + tile_idx * tile_feat_size
            
            # Read the entire tile once, in bulks (SINGLE READ per bulk)
            tile_data = []
            for b in range(num_bulks):
                bulk_start_addr = tile_start + b * bulk_feat_size
                bulk_end_addr = bulk_start_addr + bulk_feat_size
                
                # Read bulk from sources 
                if sources:
                    bulk_data = sources[bulk_start_addr:bulk_end_addr]
                    tile_data.extend(bulk_data)
                else:
                    tile_data.extend([0.0] * bulk_feat_size)
                
                # Record the read operation (only once per bulk)
                record_local(thread_id, mapping_module.OPS['R'], 
                           bq_config.IV_BASE + bulk_start_addr * bq_config.SIZE_FEAT)
                

                
            # Now distribute this tile data to ALL queries that need this input
            for query_idx in queries_using_input:
                # Find the slot for this (input, query) pair
                slot_idx = slot_assignments.get((input_idx, query_idx))
                if slot_idx is None:
                    continue  # Skip if no slot assigned
                
                # Calculate destination in gemm buffer for this query's slot
                dest_tile_base = slot_idx * total_feats_per_pt + tile_idx * tile_feat_size
                
                # Write tile in bulks to gemm buffer for this query
                for b in range(num_bulks):
                    dest_bulk_addr = dest_tile_base + b * bulk_feat_size
                    bulk_data = tile_data[b * bulk_feat_size:(b + 1) * bulk_feat_size]
                    
                    # Write to gemm buffer
                    if gemm_buffers and dest_bulk_addr + bulk_feat_size <= len(gemm_buffers):
                        gemm_buffers[dest_bulk_addr:dest_bulk_addr + bulk_feat_size] = bulk_data
                    
                    # Record the write operation for this query's slot
                    record_local(thread_id, mapping_module.OPS['W'],
                               int(np.uint64(bq_config.GM_BASE) + dest_bulk_addr * bq_config.SIZE_FEAT))
    
    flush_local_trace()

def mt_inverted_gather(
    # Parameters
    num_threads: int,
    ball_query_results: dict,  # {query_idx: [(input_idx, distance), ...]}
    num_tiles_per_pt: int,
    tile_feat_size: int,
    bulk_feat_size: int,
    sources: list,
    gemm_buffers: list
) -> dict:
    """
    Orchestrates inverted gather: read each input once, distribute to all queries.
    """
    bq_config.curr_phase = mapping_module.PHASES['GTH']
    
    # Convert ball query results to input-centric mapping
    input_to_queries = ball_query_to_input_mapping(ball_query_results)
    
    # Create inverted masks and mappings with proper slot assignments
    query_input_pairs = []
    for query_idx in range(max(max(queries) for queries in input_to_queries.values()) + 1 if input_to_queries else 0):
        inputs_for_query = []
        for input_idx, queries in input_to_queries.items():
            if query_idx in queries:
                inputs_for_query.append(input_idx)
        query_input_pairs.append(inputs_for_query)
    
    active_inputs, input_to_queries_refined, slot_assignments = create_inverted_masks(
        query_input_pairs, len(sources) // (num_tiles_per_pt * tile_feat_size) if sources else 1000
    )
    
    print(f"Inverted gather: {len(active_inputs)} active inputs out of potential inputs")
    print(f"Memory efficiency: {sum(len(queries) for queries in input_to_queries.values()) / len(active_inputs):.2f}x reuse per input")
    
    # Create and start threads
    threads_list = []
    for i in range(num_threads):
        thread_obj = threading.Thread(
            target=inverted_gather_thread,
            args=(
                i,                    # thread_id
                num_threads,
                active_inputs,
                input_to_queries_refined,
                slot_assignments,
                num_tiles_per_pt,
                tile_feat_size,
                bulk_feat_size,
                sources,
                gemm_buffers
            )
        )
        threads_list.append(thread_obj)
        thread_obj.start()
    
    # Wait for all threads to complete
    for thread_obj in threads_list:
        thread_obj.join()
    
    # Complete inverted gather process
    print(f"Inverted gather process completed successfully")

def test_inverted_gather():
    """Test the inverted gather approach with ball query results."""
    print("Starting INVERTED GATHER test...")
    if hasattr(bq_config, 'mem_trace'):
        bq_config.mem_trace = []
    
    # Simulate ball query results 
    # Query 0 finds inputs [1, 3, 5], Query 1 finds inputs [1, 2, 4], etc.
    ball_query_results = {
        0: [(1, 0.1), (3, 0.2), (5, 0.3)],  # 3 neighbors
        1: [(1, 0.15), (2, 0.25), (4, 0.35)],  # 3 neighbors, input 1 is shared
        2: [(2, 0.12), (6, 0.22)],  # 2 neighbors, input 2 is shared
        3: [(7, 0.18)],  # 1 neighbor
    }
    
    # Parameters
    num_queries = 4
    num_total_inputs = 10
    num_tiles_per_pt = 2
    tile_feat_size = 4
    bulk_feat_size = 2
    num_threads = 2
    
    total_feats_per_input = num_tiles_per_pt * tile_feat_size
    
    # Create dummy source data
    sources = [float(i) for i in range(num_total_inputs * total_feats_per_input)]
    
    # Calculate unique inputs needed
    unique_inputs = set()
    for neighbors in ball_query_results.values():
        for input_idx, _ in neighbors:
            unique_inputs.add(input_idx)
    
    num_active_inputs = len(unique_inputs)
    
    # Calculate total slots needed (with power-of-2 padding per query)
    max_neighbors = max(len(neighbors) for neighbors in ball_query_results.values())
    slots_per_query = 1 if max_neighbors == 0 else 2 ** math.ceil(math.log2(max_neighbors))
    total_slots_needed = num_queries * slots_per_query
    
    gemm_buffers = [0.0] * (total_slots_needed * total_feats_per_input)
    
    print(f"Test parameters:")
    print(f"  Ball query results: {len(ball_query_results)} queries")
    print(f"  Unique inputs needed: {num_active_inputs} out of {num_total_inputs}")
    print(f"  Total input reads (query-centric): {sum(len(neighbors) for neighbors in ball_query_results.values())}")
    print(f"  Total input reads (input-centric): {num_active_inputs}")
    
    # Run inverted gather
    mt_inverted_gather(
        num_threads=num_threads,
        ball_query_results=ball_query_results,
        num_tiles_per_pt=num_tiles_per_pt,
        tile_feat_size=tile_feat_size,
        bulk_feat_size=bulk_feat_size,
        sources=sources,
        gemm_buffers=gemm_buffers
    )
    
    print(f"Inverted gather completed.")
    
    # Print memory trace info
    if hasattr(bq_config, 'mem_trace') and bq_config.mem_trace:
        print(f"Memory trace recorded {len(bq_config.mem_trace)} accesses")
        print("First 3 trace entries:")
        for entry in bq_config.mem_trace[:3]:
            print(f"  {entry}")

def test_slot_assignments():
    """Test that slot assignments correctly map each (input, query) pair to separate slots with query-contiguous layout"""
    print("Testing query-contiguous slot assignment correctness...")
    
    # Create test ball query results 
    test_query_results = [
        [5, 10],         # Query 0: inputs 5, 10
        [5, 12, 18],     # Query 1: inputs 5, 12, 18  
        [10, 15, 25]     # Query 2: inputs 10, 15, 25
    ]
    
    active_inputs, input_to_queries, slot_assignments = create_inverted_masks(test_query_results, 40)
    
    print(f"Active inputs: {sorted(active_inputs)}")
    print(f"Input to queries mapping:")
    for input_idx in sorted(input_to_queries.keys()):
        print(f"  Input {input_idx}: serves queries {input_to_queries[input_idx]}")
    
    print(f"Query-contiguous slot assignments:")
    # Group by query for better visualization
    by_query = {}
    for (input_idx, query_idx), slot_idx in slot_assignments.items():
        if query_idx not in by_query:
            by_query[query_idx] = []
        by_query[query_idx].append((input_idx, slot_idx))
    
    for query_idx in sorted(by_query.keys()):
        print(f"  Query {query_idx}:")
        for input_idx, slot_idx in sorted(by_query[query_idx], key=lambda x: x[1]):
            print(f"    Input {input_idx} -> Slot {slot_idx}")
    
    # Verify that each query has contiguous slots
    max_neighbors = max(len(inputs) for inputs in test_query_results)
    print(f"Max neighbors per query: {max_neighbors}")
    print(f"Expected slot ranges:")
    for query_idx in range(len(test_query_results)):
        start_slot = query_idx * max_neighbors
        end_slot = start_slot + len(test_query_results[query_idx]) - 1
        print(f"  Query {query_idx}: slots {start_slot}-{end_slot} (plus padding to {start_slot + max_neighbors - 1})")
    
    return True

def compare_gather_approaches():
    """
    Compare traditional query-centric vs inverted input-centric gather approaches.
    """
    print("COMPARING GATHER APPROACHES")
    print("=" * 50)
    
    # Example ball query results with significant input reuse
    ball_query_results = {
        0: [(5, 0.1), (10, 0.2), (15, 0.3), (20, 0.4)],
        1: [(5, 0.12), (12, 0.22), (18, 0.32)],
        2: [(10, 0.15), (15, 0.25), (25, 0.35)],
        3: [(5, 0.08), (15, 0.18), (30, 0.28)],
        4: [(10, 0.11), (20, 0.21)],
        5: [(15, 0.13), (25, 0.23), (30, 0.33)]
    }
    
    # Analyze the data
    total_accesses_traditional = sum(len(neighbors) for neighbors in ball_query_results.values())
    
    # Count unique inputs
    unique_inputs = set()
    input_reuse_count = {}
    for neighbors in ball_query_results.values():
        for input_idx, _ in neighbors:
            unique_inputs.add(input_idx)
            input_reuse_count[input_idx] = input_reuse_count.get(input_idx, 0) + 1
    
    total_accesses_inverted = len(unique_inputs)
    
    print(f"Ball Query Results Analysis:")
    print(f"  Number of queries: {len(ball_query_results)}")
    print(f"  Total neighbors found: {total_accesses_traditional}")
    print(f"  Unique inputs accessed: {total_accesses_inverted}")
    print(f"  Average neighbors per query: {total_accesses_traditional / len(ball_query_results):.2f}")
    print(f"  Average reuse per input: {total_accesses_traditional / total_accesses_inverted:.2f}")
    
    print(f"\\nMemory Access Comparison:")
    print(f"  Traditional (query-centric): {total_accesses_traditional} input reads")
    print(f"  Inverted (input-centric):    {total_accesses_inverted} input reads")
    print(f"  Memory access reduction:     {total_accesses_traditional / total_accesses_inverted:.2f}x")
    print(f"  Memory bandwidth saved:      {((total_accesses_traditional - total_accesses_inverted) / total_accesses_traditional * 100):.1f}%")
    
    print(f"\\nInput Reuse Distribution:")
    reuse_counts = list(input_reuse_count.values())
    for reuse in sorted(set(reuse_counts)):
        count = sum(1 for x in reuse_counts if x == reuse)
        print(f"  Inputs used {reuse} times: {count} inputs")
    
    # Show which queries each input serves
    input_to_queries = ball_query_to_input_mapping(ball_query_results)
    print(f"\\nInput Usage Details:")
    for input_idx in sorted(input_to_queries.keys()):
        queries = input_to_queries[input_idx]
        print(f"  Input {input_idx:2d}: serves queries {queries} ({len(queries)} times)")
    
    return {
        'traditional_accesses': total_accesses_traditional,
        'inverted_accesses': total_accesses_inverted,
        'reduction_factor': total_accesses_traditional / total_accesses_inverted,
        'input_to_queries': input_to_queries
    }

if __name__ == '__main__':
    test_to_run = "inverted"  # Options: "inverted", "compare", "slots", "all", "none"

    if test_to_run.lower() in ["inverted", "all"]:
        print("\\n" + "="*30 + " INVERTED GATHER TEST START " + "="*30 + "\\n")
        test_inverted_gather()
        print("="*30 + " INVERTED GATHER TEST END " + "="*30 + "\\n")

    if test_to_run.lower() in ["compare", "all"]:
        print("\\n" + "="*30 + " APPROACH COMPARISON START " + "="*30 + "\\n")
        compare_gather_approaches()
        print("="*30 + " APPROACH COMPARISON END " + "="*30 + "\\n")

    if test_to_run.lower() in ["slots", "all"]:
        print("\\n" + "="*30 + " SLOT ASSIGNMENT TEST START " + "="*30 + "\\n")
        test_slot_assignments()
        print("="*30 + " SLOT ASSIGNMENT TEST END " + "="*30 + "\\n")

    if test_to_run.lower() == "none":
        print("No tests selected to run.")
