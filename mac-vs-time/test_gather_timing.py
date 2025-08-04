#!/usr/bin/env python3
"""
Test script to debug gather timing calculations
"""

# Simplified performance model for testing
class TestGEMMPerformanceModel:
    def __init__(self):
        # Hardware parameters (A100-like)
        self.peak_flops = 312e12  # Peak FLOPs/s (312 TFLOPS)
        self.memory_bandwidth = 1555e9  # Memory bandwidth (1555 GB/s)
    
    def estimate_performance(self, M, N, K):
        """Simple performance estimation"""
        
        # 1. GEMM COMPUTATION (Compute-bound)
        gemm_flops = 2 * M * N * K  # MAC operations
        gemm_time_seconds = gemm_flops / self.peak_flops
        gemm_time_ms = gemm_time_seconds * 1000
        
        # 2. GATHER OPERATION (Memory-bound)
        gather_bytes = M * 128  # M * 128 bytes as specified
        gather_time_seconds = gather_bytes / self.memory_bandwidth
        gather_time_ms = gather_time_seconds * 1000
        
        return {
            'M': M, 'N': N, 'K': K,
            'gemm_flops': gemm_flops,
            'gemm_time_ms': gemm_time_ms,
            'gather_bytes': gather_bytes,
            'gather_time_ms': gather_time_ms,
            'total_time_ms': gemm_time_ms + gather_time_ms
        }

def test_minkowski_layers():
    """Test MinkowskiNet layers"""
    model = TestGEMMPerformanceModel()
    
    # MinkowskiNet layers from CSV
    layers = [
        ('conv0', 400000, 32, 108),
        ('encoder_stage_1', 400000, 64, 864),
        ('encoder_stage_2', 400000, 128, 1728),
        ('encoder_stage_3', 400000, 256, 3456),
        ('encoder_stage_4', 400000, 512, 6912),
        ('classification_head', 100000, 20, 32),
    ]
    
    print("MinkowskiNet GEMM vs Gather Timing Analysis")
    print("=" * 80)
    print(f"{'Layer':<20} {'M':<8} {'N':<4} {'K':<6} {'GEMM(ms)':<10} {'Gather(ms)':<12} {'Total(ms)':<10} {'Bottleneck':<12}")
    print("-" * 80)
    
    total_gemm_time = 0
    total_gather_time = 0
    
    for layer_name, M, N, K in layers:
        result = model.estimate_performance(M, N, K)
        
        bottleneck = 'GEMM' if result['gemm_time_ms'] > result['gather_time_ms'] else 'Gather'
        
        print(f"{layer_name:<20} {M:<8} {N:<4} {K:<6} {result['gemm_time_ms']:<10.3f} {result['gather_time_ms']:<12.3f} {result['total_time_ms']:<10.3f} {bottleneck:<12}")
        
        total_gemm_time += result['gemm_time_ms']
        total_gather_time += result['gather_time_ms']
    
    print("-" * 80)
    print(f"{'TOTAL':<20} {'':<8} {'':<4} {'':<6} {total_gemm_time:<10.3f} {total_gather_time:<12.3f} {total_gemm_time + total_gather_time:<10.3f}")
    
    print(f"\nSummary:")
    print(f"Total GEMM time: {total_gemm_time:.3f} ms")
    print(f"Total Gather time: {total_gather_time:.3f} ms")
    print(f"Gather/GEMM ratio: {total_gather_time/total_gemm_time:.3f}")
    
    # Check individual layer calculations
    print(f"\nDetailed calculation for encoder_stage_4 (largest layer):")
    M, N, K = 400000, 512, 6912
    result = model.estimate_performance(M, N, K)
    
    print(f"M = {M:,}")
    print(f"N = {N}")
    print(f"K = {K}")
    print(f"GEMM FLOPs = 2 * {M:,} * {N} * {K} = {result['gemm_flops']:,}")
    print(f"GEMM time = {result['gemm_flops']:,} / {model.peak_flops:.0e} = {result['gemm_time_ms']:.6f} ms")
    print(f"Gather bytes = {M:,} * 128 = {result['gather_bytes']:,}")
    print(f"Gather time = {result['gather_bytes']:,} / {model.memory_bandwidth:.0e} = {result['gather_time_ms']:.6f} ms")

if __name__ == "__main__":
    test_minkowski_layers()
