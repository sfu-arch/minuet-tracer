import gzip
import struct
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import argparse
import os
from minuet_gather import read_gemm_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the distribution of the number of reads per gene.")
    parser.add_argument("--trace-file", help="Input file in .gz format")
    args = parser.parse_args()

    gemm_list = read_gemm_list(args.trace_file)
    for g in gemm_list:
        print(g)
