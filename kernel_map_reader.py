import gzip
import struct
from typing import Dict, List, Tuple
import argparse
from coord import Coord3D # Import Coord3D
from coord import unpack32s # Import unpack32s for unpacking keys
# Removed local unpack32 function as Coord3D.from_key will be used

def read_kernel_map_from_gz(filename: str) -> Dict[int, List[Tuple[Tuple[Tuple[int,int,int], int], Tuple[Tuple[int,int,int], int]]]]:
    """
    Read the kernel map from a gzipped binary file.
    Returns a dictionary structured similarly to the original kernel_map.
    """
    kernel_map_data: Dict[int, List[Tuple[Tuple[Tuple[int,int,int], int], Tuple[Tuple[int,int,int], int]]]] = {}
    
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
                entry_data_bytes = f.read(4 + 4 + 4 + 4 + 4) # IIIII = 4*5 = 20 bytes
                if len(entry_data_bytes) < 20:
                    print(f"Error: Unexpected end of file while reading entry {i+1}/{num_total_entries}.")
                    break

                offset_idx_val, input_key, input_pos, output_key, output_pos = struct.unpack('IIIII', entry_data_bytes)

                # Use Coord3D.from_key and then convert to tuple
                input_coord_obj = Coord3D.from_key(input_key)
                input_coord_tuple = (input_coord_obj.x, input_coord_obj.y, input_coord_obj.z)
                
                output_coord_obj = Coord3D.from_key(output_key)
                output_coord_tuple = (output_coord_obj.x, output_coord_obj.y, output_coord_obj.z)
                
                if offset_idx_val not in kernel_map_data:
                    kernel_map_data[offset_idx_val] = []
                
                kernel_map_data[offset_idx_val].append(
                    ((input_coord_tuple, input_pos), (output_coord_tuple, output_pos))
                )
        
        entries_read = sum(len(v) for v in kernel_map_data.values())
        print(f"Successfully read {entries_read} entries and reconstructed kernel map.")
        if entries_read != num_total_entries and (not entry_data_bytes or len(entry_data_bytes) >= 17) : # Check if loop broke early due to EOF but not because of incomplete last read
             print(f"Warning: Expected {num_total_entries} entries based on header, but read {entries_read}.")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return {}
    except gzip.BadGzipFile:
        print(f"Error: File '{filename}' is not a valid gzip file or is corrupted.")
        return {}
    except ImportError:
        print(f"Error: Could not import Coord3D from coord.py. Make sure coord.py is in the same directory or accessible in PYTHONPATH.")
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
        for offset_idx, matches in sorted(kernel_map.items()): # Sort by offset_idx for consistent output
            if matches:
                print(f"  Offset Index {unpack32s(offset_idx)}:")
                for match_item_idx, match_item in enumerate(matches):
                    (in_coord_tuple, in_pos), (out_coord_tuple, out_pos) = match_item
                    print(f"    Match {match_item_idx + 1}: Input: {in_coord_tuple} (orig_idx: {in_pos}) -> Output: {out_coord_tuple} (orig_idx: {out_pos})")
            else:
                print(f"  Offset Index {offset_idx}: No matches")
    else:
        print("No kernel map data was read or the map is empty.")