#!/usr/bin/env python3
# trace_reader.py - Read and analyze minuet memory trace files

import gzip
import struct
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# Reverse mappings (to convert integers back to strings)
PHASES = {
    0: 'Radix-Sort',
    1: 'Build-Queries',
    2: 'Sort-QKeys',
    3: 'Tile-Pivots',
    4: 'Lookup',
    5: 'Lookup-Backward',
    6: 'Lookup-Forward',
    7: 'Dedup'
}

TENSORS = {
    0: 'I',    # Input keys
    1: 'QK',   # Query keys
    2: 'QI',   # Query input-index array
    3: 'QO',   # Query offset-index array
    4: 'PIV',  # Tile pivot keys
    5: 'KM',   # Kernel-map writes
    6: 'WO',   # Weight-offset keys
    7: 'WV'    # Weight values
}

OPS = {
    0: 'R',    # Read
    1: 'W'     # Write
}

def read_trace(filename):
    """Read a compressed memory trace file and return the entries."""
    entries = []
    
    try:
        with gzip.open(filename, 'rb') as f:
            # Read number of entries
            num_entries = struct.unpack('I', f.read(4))[0]
            print(f"Reading {num_entries} trace entries...")
            
            # Read each entry
            for _ in range(num_entries):
                entry = struct.unpack('BBBBI', f.read(8))  # Format is BBBBI
                phase_id, thread_id, op_id, tensor_id, addr = entry
                
                # Convert numeric IDs to strings
                phase = PHASES.get(phase_id, f"Unknown-{phase_id}")
                op = OPS.get(op_id, f"Unknown-{op_id}")
                tensor = TENSORS.get(tensor_id, f"Unknown-{tensor_id}")
                
                entries.append({
                    'phase': phase,
                    'thread_id': thread_id,
                    'op': op,
                    'tensor': tensor,
                    'addr': addr
                })
                
        return entries
    
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
        phase_ops[entry['phase']][entry['op']] += 1
    
    # Count operations by tensor
    tensor_ops = defaultdict(lambda: {'R': 0, 'W': 0})
    for entry in entries:
        tensor_ops[entry['tensor']][entry['op']] += 1
    
    # Count operations by thread
    thread_ops = defaultdict(lambda: {'R': 0, 'W': 0})
    for entry in entries:
        thread_ops[entry['thread_id']][entry['op']] += 1
    
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

def main():
    parser = argparse.ArgumentParser(description='Read and analyze minuet memory trace files')
    parser.add_argument('trace_file', help='Path to memory trace file')
    parser.add_argument('--filter-phase', help='Filter by phase name')
    parser.add_argument('--filter-op', choices=['R', 'W'], help='Filter by operation type')
    parser.add_argument('--filter-tensor', help='Filter by tensor type')
    parser.add_argument('--plot', action='store_true', help='Generate memory access plots')
    parser.add_argument('--plot-file', help='Save plot to file instead of displaying')
    
    args = parser.parse_args()
    
    # Read trace file
    entries = read_trace(args.trace_file)
    
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