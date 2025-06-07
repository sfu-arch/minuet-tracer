import gzip
import struct
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import argparse
import os
import minuet_config
from minuet_gather import read_gemm_list


def decipher_gemms_trace(num_input_feature_channels, num_filters_per_offset, gemms_bin_path = minuet_config.output_dir+"gemms.bin.gz"):
    """Reads a gemms.bin.gz file, deciphers the dimensions of the batched GEMM operations"""

    gemm_list = read_gemm_list(gemms_bin_path)

    print("Dimensions of all batched GEMM operations:")

    # Deciphering gemm_list values to batched GEMM parameters
    for i, gemm in enumerate(gemm_list):
        num_offsets = gemm['num_offsets']
        gemm_m = gemm['gemm_M']

        m_dim = gemm_m // num_offsets
        k_dim = num_input_feature_channels
        n_dim = num_filters_per_offset
        
        print(f"--- Batched GEMM Operation #{i+1} ---")
        print(f"    Batch Size: {num_offsets}")
        print(f"    GEMM Dimensions (M, K, N): ({m_dim}, {k_dim}, {n_dim})")
        print(f"     - Matrix A (Input):  [{m_dim}, {k_dim}]  (Number of (Points + Padding) per Offset x Input Channels)")
        print(f"     - Matrix B (Weights): [{k_dim}, {n_dim}]  (Input Channels x Number of Filters per Offset)")
        print("-" * 33 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the distribution of the number of reads per gene.")
    parser.add_argument("--trace-file", help="Input file in .gz format")
    args = parser.parse_args()

    gemm_list = read_gemm_list(args.trace_file)
    for g in gemm_list:
        print(g)
