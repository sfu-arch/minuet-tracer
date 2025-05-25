import gzip
import struct
from typing import Dict, List, Tuple
import argparse
from coord import Coord3D, unpack32, unpack32s

def read_kernel_map_from_gz(filename: str) -> Dict[int, List[Tuple[int, int]]]:
    """
    Read the kernel map from a gzipped binary file.
    Returns a dictionary where:
    - Keys are offset indices
    - Values are lists of (input_idx, query_src_orig_idx) tuples
    """
    kernel_map_data: Dict[int, List[Tuple[int, int]]] = {}
    
    try:
        with gzip.open(filename, 'rb') as f:
            # Read header with entry count
            num_total_entries_bytes = f.read(4)
            if not num_total_entries_bytes or len(num_total_entries_bytes) < 4:
                print(f"Error: Kernel map file '{filename}' is empty or header is missing/incomplete.")
                return kernel_map_data
            num_total_entries = struct.unpack('I', num_total_entries_bytes)[0]
            
            print(f"Reading {num_total_entries} entries from {filename}...")
            
            for i in range(num_total_entries):
                entry_data_bytes = f.read(12)  # III = 4*3 = 12 bytes
                if len(entry_data_bytes) < 12:
                    print(f"Error: Unexpected end of file while reading entry {i+1}/{num_total_entries}.")
                    break

                # Unpack the three integers: offset_key, input_idx, query_src_orig_idx
                offset_key, input_idx, query_src_orig_idx = struct.unpack('III', entry_data_bytes)

                # Use the offset_key as dictionary key
                if offset_key not in kernel_map_data:
                    kernel_map_data[offset_key] = []
                
                # Store just the two indices as tuple
                kernel_map_data[offset_key].append((input_idx, query_src_orig_idx))
        
        entries_read = sum(len(v) for v in kernel_map_data.values())
        print(f"Successfully read {entries_read} entries and reconstructed kernel map.")
        if entries_read != num_total_entries and (not entry_data_bytes or len(entry_data_bytes) >= 12): 
            print(f"Warning: Expected {num_total_entries} entries based on header, but read {entries_read}.")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return {}
    except gzip.BadGzipFile:
        print(f"Error: File '{filename}' is not a valid gzip file or is corrupted.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}
        
    return kernel_map_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read and display a kernel map from a .gz file.")
    parser.add_argument(
        "filepath", 
        nargs='?', 
        default="kernel_map.bin.gz", 
        help="Path to the kernel_map.bin.gz file (default: kernel_map.bin.gz)"
    )
    args = parser.parse_args()

    print(f"--- Reading Kernel Map from File: {args.filepath} ---")
    kernel_map = read_kernel_map_from_gz(args.filepath)
    
    if kernel_map:
        print('\n--- Kernel Map Contents ---')
        for offset_key, matches in sorted(kernel_map.items()):
            if matches:
                offset_coords = unpack32s(offset_key)
                print(f"  Offset {offset_coords}:")
                for match_idx, (input_idx, query_src_orig_idx) in enumerate(matches):
                    print(f"    Match {match_idx + 1}: Input idx: {input_idx} -> Source orig idx: {query_src_orig_idx}")
            else:
                offset_coords = unpack32s(offset_key)
                print(f"  Offset {offset_coords}: No matches")
    else:
        print("No kernel map data was read or the map is empty.")