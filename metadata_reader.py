import gzip
import struct
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import argparse
import os
from minuet_gather import read_metadata
from coord import Coord3D # Import Coord3D



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the distribution of the number of reads per gene.")
    parser.add_argument("--trace-file", help="Input file in .gz format")
    args = parser.parse_args()

    out_mask, in_mask, offsets_active, slot_array = read_metadata(args.trace_file)
    print(out_mask)
    print(in_mask)
    print(offsets_active)
    print(slot_array)
