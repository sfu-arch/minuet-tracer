#!/usr/bin/env python3
# trace_reader.py - Read and analyze minuet memory trace files

import gzip
import struct
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from minuet_mapping import bidict

PHASES = bidict({
    'RDX': 0,
    'QRY': 1,
    'SRT': 2,
    'PVT': 3,
    'LKP': 4,
    'GTH': 5,
    'SCT': 6,
    'OCTREE_BUILD': 7,    # New phase for octree construction
    'BALL_QUERY': 8       # New phase for ball query operations
})
TENSORS = bidict({
    'I': 0,
    'QK': 1,
    'QI': 2,
    'QO': 3,
    'PIV': 4,
    'KM': 5,
    'WC': 6,
    'TILE': 7,
    'IV': 8,
    'GM': 9,
    'WV': 10,
    'OCTREE_NODES': 11,   # New tensor for octree node data
    'OCTREE_INDICES': 12, # New tensor for octree point indices
    'BALL_RESULTS': 13    # New tensor for ball query results
})

OPS = bidict({
    'R': 0,
    'W': 1
})

def read_trace(filename,sizeof_addr=4):
    """Read a compressed memory trace file and return the entries."""
    entries = []
    
    try:
        with gzip.open(filename, 'rb') as f:
            # Read number of entries
            num_entries_data = f.read(4)
            if not num_entries_data:
                print(f"Error: Trace file {filename} appears to be empty or corrupted (could not read num_entries).")
                return []
            num_entries = struct.unpack('I', num_entries_data)[0]
            print(f"Reading {num_entries} trace entries from {filename}...")
            
            # Read each entry
            for i in range(num_entries):
                if sizeof_addr == 4:
                    entry_data = f.read(8) # Read 8 bytes for a <BBBBI structure
                else:
                    entry_data = f.read(12) # Read 12 bytes for a <BBBBQ structure
                if len(entry_data) < 4 + sizeof_addr:
                    # Corrected expected byte count in the error message
                    print(f"Error: Trace file {filename} is truncated. Expected {4 + sizeof_addr} bytes for entry {i+1}, got {len(entry_data)}.")
                    break
                # Use '<BBBBQ' for little-endian, 12-byte packed structure
                if sizeof_addr == 4:
                    entry = struct.unpack('<BBBBI', entry_data)  # Format is <BBBBI (little-endian)
                else:
                    entry = struct.unpack('<BBBBQ', entry_data)  # Format is <BBBBQ (little-endian)
                phase_id, thread_id, op_id, tensor_id, addr = entry
                                # Convert numeric IDs to strings
                phase = PHASES.inverse[phase_id][0] if phase_id in PHASES.inverse else f"Unknown-{phase_id}"
                op = OPS.inverse[op_id][0] if op_id in OPS.inverse else f"Unknown-{op_id}"
                tensor = TENSORS.inverse[tensor_id][0] if tensor_id in TENSORS.inverse else f"Unknown-{tensor_id}"
                entries.append({
                    'phase': phase,
                    'thread_id': thread_id,
                    'op': op,
                    'tensor': tensor,
                    'addr': addr
                })
        return entries
    
    except gzip.BadGzipFile:
        print(f"Error: File {filename} is not a valid GZIP file or is corrupted.")
        return []
    except Exception as e:
        print(f"Error reading trace file: {e}")
        return []

def analyze_trace(entries):
    """Analyze trace entries and print statistics."""
    if not entries:
        print("No trace entries to analyze")
        return
    
    # Count operations by phase
    phase_ops = defaultdict(lambda: {'R': 0, 'W': 0})
    for entry in entries:  
        print(entry)
        # print(entry)
        # print(entry['phase'], entry['op'])
        phase_ops[entry['phase']][entry['op']] += 1
    
    # Count operations by tensor
    tensor_ops = defaultdict(lambda: {'R': 0, 'W': 0})
    for entry in entries:
        tensor_ops[entry['tensor']][entry['op']] += 1
    
    # Count operations by thread
    thread_entries = defaultdict(list)
    thread_ops = defaultdict(lambda: {'R': 0, 'W': 0})
    for entry in entries:
        thread_ops[entry['thread_id']][entry['op']] += 1
        thread_entries[entry['thread_id']].append(entry)
        
    # Print statistics
    print("\n===== Memory Trace Statistics =====")
    print(f"Total entries: {len(entries)}")
    
    print("\n--- Operations by Phase ---")
    for phase, ops in sorted(phase_ops.items()):
        total = ops['R'] + ops['W']
        print(f"{phase}: {total} ops ({ops['R']} reads, {ops['W']} writes)")
    
    print("\n--- Operations by Tensor ---")
    for tensor, ops in sorted(tensor_ops.items()):
        total = ops['R'] + ops['W']
        print(f"{tensor}: {total} ops ({ops['R']} reads, {ops['W']} writes)")
    
    print("\n--- Operations by Thread ---")
    for thread_id, ops in sorted(thread_ops.items()):
        total = ops['R'] + ops['W']
        print(f"Thread {thread_id}: {total} ops ({ops['R']} reads, {ops['W']} writes)")
    # print("\n--- Thread Entries ---")
    # for thread_id, entries in sorted(thread_entries.items()):
    #     print(f"Thread {thread_id} has {len(entries)} entries:")
    #     for entry in entries:
    #         print(f"  {entry['phase']} - {entry['op']} - {entry['tensor']} - Addr: {entry['addr']}")

def plot_memory_access_patterns(entries, output_file=None):
    """Plot memory access patterns from trace entries."""
    if not entries:
        print("No trace entries to plot")
        return
    
    plt.figure(figsize=(12, 10))
    
    # Plot memory accesses by address and time
    addresses = [entry['addr'] for entry in entries]
    times = range(len(addresses))
    colors = ['blue' if entry['op'] == 'R' else 'red' for entry in entries]
    
    plt.subplot(2, 1, 1)
    plt.scatter(times, addresses, c=colors, s=5, alpha=0.6)
    plt.title('Memory Access Pattern')
    plt.xlabel('Access Sequence')
    plt.ylabel('Memory Address')
    plt.legend(['Read', 'Write'], loc='upper right')
    
    # Plot memory accesses by tensor type
    plt.subplot(2, 1, 2)
    tensor_counts = Counter([entry['tensor'] for entry in entries])
    tensors = [tensor for tensor, _ in tensor_counts.most_common()]
    counts = [count for _, count in tensor_counts.most_common()]
    
    plt.bar(tensors, counts)
    plt.title('Memory Accesses by Tensor Type')
    plt.xlabel('Tensor Type')
    plt.ylabel('Number of Accesses')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def diff_trace(file1_path, file2_path):
    """Compare two trace files and print the differences."""
    print(f"\n===== Diffing Trace Files =====")
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")

    entries1 = read_trace(file1_path)
    entries2 = read_trace(file2_path)

    if not entries1 and not entries2:
        print("Both trace files are empty or could not be read.")
        return
    if not entries1:
        print(f"Could not read entries from {file1_path} or it's empty.")
        print(f"{file2_path} contains {len(entries2)} entries.")
        # Optionally print all entries from file2 if desired
        # for i, entry in enumerate(entries2):
        #     print(f"  Entry {i} in {file2_path} (unique): {entry}")
        return
    if not entries2:
        print(f"Could not read entries from {file2_path} or it's empty.")
        print(f"{file1_path} contains {len(entries1)} entries.")
        # Optionally print all entries from file1 if desired
        # for i, entry in enumerate(entries1):
        #     print(f"  Entry {i} in {file1_path} (unique): {entry}")
        return

    len1, len2 = len(entries1), len(entries2)
    differences_found = False

    if len1 != len2:
        print(f"\nNumber of entries differ: File 1 has {len1}, File 2 has {len2}")
        differences_found = True

    common_len = min(len1, len2)
    print(f"\nComparing the first {common_len} entries...")

    for i in range(common_len):
        entry1 = entries1[i]
        entry2 = entries2[i]
        if entry1 != entry2:
            differences_found = True
            print(f"\nDifference at entry index {i}:")
            print(f"  File 1: {entry1}")
            print(f"  File 2: {entry2}")
            # Detailed field comparison
            for key in entry1.keys():
                if entry1.get(key) != entry2.get(key):
                    print(f"    Field '{key}': '{entry1.get(key)}' (File 1) vs '{entry2.get(key)}' (File 2)")


    if len1 > common_len:
        print(f"\nEntries unique to File 1 (from index {common_len}):")
        for i in range(common_len, len1):
            differences_found = True
            print(f"  Index {i}: {entries1[i]}")

    if len2 > common_len:
        print(f"\nEntries unique to File 2 (from index {common_len}):")
        for i in range(common_len, len2):
            differences_found = True
            print(f"  Index {i}: {entries2[i]}")

    if not differences_found:
        print("\nNo differences found between the trace files (up to common length if sizes differ, or full if same size).")
    else:
        print("\n===== End of Diff =====")


def main():
    parser = argparse.ArgumentParser(description='Read, analyze, and diff minuet memory trace files')
    parser.add_argument('--trace-file', required=True, help='Path to memory trace file (file1 for diff)')
    parser.add_argument('--diff-file2', help='Path to the second memory trace file for diff (file2 for diff)')
    parser.add_argument('--filter-phase', help='Filter by phase name (only for single file analysis)')
    parser.add_argument('--filter-op', choices=['R', 'W'], help='Filter by operation type (only for single file analysis)')
    parser.add_argument('--filter-tensor', help='Filter by tensor type (only for single file analysis)')
    parser.add_argument('--plot', action='store_true', help='Generate memory access plots (only for single file analysis)')
    parser.add_argument('--plot-file', help='Save plot to file instead of displaying (only for single file analysis)')
    parser.add_argument('--sizeof-addr', type=int, choices=[4, 8], default=4,
                        help='Size of address in bytes (4 for 32-bit, 8 for 64-bit; default: 4)')
    
    args = parser.parse_args()
    
    if args.diff_file2:
        diff_trace(args.trace_file, args.diff_file2)
    else:
        # Read trace file for single file analysis
        entries = read_trace(args.trace_file, args.sizeof_addr)
        
        # Apply filters if specified
        if args.filter_phase:
            entries = [e for e in entries if e['phase'] == args.filter_phase]
        if args.filter_op:
            entries = [e for e in entries if e['op'] == args.filter_op]
        if args.filter_tensor:
            entries = [e for e in entries if e['tensor'] == args.filter_tensor]
        
        # Print analysis
        analyze_trace(entries)
        
        # Generate plots if requested
        if args.plot:
            plot_memory_access_patterns(entries, args.plot_file)

if __name__ == '__main__':
    main()