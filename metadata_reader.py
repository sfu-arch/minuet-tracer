import argparse
import os
import sys
import json
import numpy as np

# --- Setup sys.path to find Minuet utility modules ---
# Assume this script is in the project root /Users/ashriram/Desktop/minuet/
# and minuet_gather.py, minuet_config.py are also in this directory or a subdirectory.
# If minuet_gather is in the same directory, direct import should work.
# If it's in a subdirectory (e.g., 'utils'), adjust sys.path.
# For this example, assuming minuet_gather.py is in the same directory or PYTHONPATH is set.

script_dir = os.path.dirname(os.path.abspath(__file__))
# If minuet_gather.py is in the same directory as this script:
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
    print(f"Added {script_dir} to sys.path for Minuet modules.")

# Try to import minuet_gather
try:
    from minuet_gather import read_metadata
    from minuet_utils import file_checksum # Import file_checksum from minuet_utils
    from minuet_config import output_dir as default_output_dir # For default filename
except ImportError as e:
    print(f"Error importing Minuet modules (minuet_gather, minuet_utils, minuet_config): {e}")
    print("Please ensure these Python files are in the same directory as metadata_reader.py, "
          "or accessible via PYTHONPATH.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Read and display Minuet metadata file.")
    default_metadata_file = os.path.join(default_output_dir, 'metadata.bin.gz')
    parser.add_argument(
        "metadata_file",
        nargs='?', # Makes the argument optional
        default=default_metadata_file,
        help=f"Path to the metadata.bin.gz file (default: {default_metadata_file})"
    )
    args = parser.parse_args()

    metadata_filename = args.metadata_file

    if not os.path.exists(metadata_filename):
        print(f"Error: Metadata file not found at {metadata_filename}")
        sys.exit(1)

    print(f"Reading metadata from: {metadata_filename}")
    
    try:
        metadata_contents = read_metadata(metadata_filename)
        checksum = file_checksum(metadata_filename) # Calculate checksum for verification
    except Exception as e:
        print(f"Error reading or processing metadata file: {e}")
        sys.exit(1)

    print("\n--- Metadata Contents ---")
    print(f"  File Checksum (SHA256): {checksum}")
    print(f"  Version: {metadata_contents.get('version')}")
    print(f"  Num Total System Offsets: {metadata_contents.get('num_total_system_offsets')}")
    print(f"  Num Total System Sources: {metadata_contents.get('num_total_system_sources')}")
    print(f"  Total Slots in GEMM Buffer: {metadata_contents.get('total_slots_in_gemm_buffer')}")
    print(f"  Num Active Offsets in Map: {metadata_contents.get('num_active_offsets_in_map')}")

    active_offsets = metadata_contents.get('active_offsets_details', [])
    print(f"\n  Active Offsets Details (first 5 of {len(active_offsets)}):")
    for i, detail in enumerate(active_offsets[:5]):
        print(f"    Offset Key: {hex(detail.get('offset_key'))}, "
              f"Base Address: {detail.get('base_address')}, "
              f"Num Matches: {detail.get('num_matches')}")
    if len(active_offsets) > 5:
        print("    ...")

    out_mask = metadata_contents.get('out_mask')
    in_mask = metadata_contents.get('in_mask')

    print("\n  Masks Information:")
    if out_mask is not None:
        print(f"    Output Mask Shape: {out_mask.shape}, Dtype: {out_mask.dtype}")
        # print(f"    Output Mask (first 10 elements): {out_mask.ravel()[:10]}")
    else:
        print("    Output Mask: Not found or empty.")
        
    if in_mask is not None:
        print(f"    Input Mask Shape: {in_mask.shape}, Dtype: {in_mask.dtype}")
        # print(f"    Input Mask (first 10 elements): {in_mask.ravel()[:10]}")
    else:
        print("    Input Mask: Not found or empty.")
        
    print("\nMetadata reading complete.")

if __name__ == "__main__":
    main()
