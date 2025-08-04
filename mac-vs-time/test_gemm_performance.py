#!/usr/bin/env python3
"""
Test script for the new GEMM vs Gather performance model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pcl_macs import GEMMPerformanceModel

def test_performance_model():
    print("Testing GEMM vs Gather Performance Model")
    print("=" * 50)
    
    # Initialize performance model
    model = GEMMPerformanceModel()
    
    # Test cases from VoxelNeXt (different M values)
    test_cases = [
        {"name": "patchify_conv", "M": 1100000, "N": 48, "K": 256},
        {"name": "patchify_expand", "M": 100000, "N": 96, "K": 48},
        {"name": "stage_block_dw", "M": 6000000, "N": 96, "K": 343},
        {"name": "stage_block_mlp", "M": 100000, "N": 384, "K": 96},
    ]
    
    for case in test_cases:
        print(f"\nLayer: {case['name']}")
        print(f"M={case['M']:,}, N={case['N']}, K={case['K']}")
        
        perf = model.estimate_performance(
            case['M'], case['N'], case['K'],
            channel_sparsity=0.35, 
            feature_sparsity=0.55
        )
        
        print(f"  GEMM time:   {perf['gemm_time_ms']:.3f} ms")
        print(f"  Gather time: {perf['gather_time_ms']:.3f} ms")
        print(f"  Total time:  {perf['total_time_ms']:.3f} ms")
        print(f"  Bottleneck:  {perf['bottleneck']}")
        print(f"  Ratio (G/C): {perf['time_ratio_gather_to_gemm']:.2f}x")
        
        # Show utilization
        print(f"  GEMM util:   {perf['gemm_compute_utilization']*100:.1f}%")
        print(f"  Memory util: {perf['gather_memory_utilization']*100:.1f}%")
        
        # Show memory transfer
        print(f"  Gather bytes: {perf['gather_bytes']:,} bytes")
        print(f"  GEMM FLOPs:   {perf['gemm_flops']:,}")

if __name__ == "__main__":
    test_performance_model()
