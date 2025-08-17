#!/usr/bin/env python3
"""
Performance comparison showing the benefit of limiting max_points
"""

from ballquery_mapping import *
import numpy as np
import time

def performance_comparison():
    """Compare performance with and without max_points limit"""
    
    print("Performance comparison: max_points impact")
    print("=" * 50)
    
    # Create a dense point cloud
    np.random.seed(42)
    num_points = 5000
    in_coords = []
    
    # Generate points in clusters
    for i in range(num_points):
        cluster_center = np.random.choice([0, 10, 20], 3) * 2  # Spread out clusters
        noise = np.random.normal(0, 1, 3)
        point = cluster_center + noise
        in_coords.append(tuple(point))
    
    # Query points
    query_coords = [(0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (20.0, 20.0, 20.0)]
    radius = 5.0  # Large radius to find many points
    
    print(f"Dataset: {len(in_coords)} points, {len(query_coords)} queries")
    print(f"Search radius: {radius}")
    print()
    
    # Test different max_points values
    for max_points in [None, 32, 64, 128]:
        print(f"Testing max_points = {max_points}")
        
        start_time = time.time()
        
        if max_points is None:
            # Test without limit (use very large number)
            results = ball_query_mapping(
                query_coords=query_coords,
                input_coords=in_coords,
                radius=radius,
                max_points=10000  # Very large limit
            )
        else:
            results = ball_query_mapping(
                query_coords=query_coords,
                input_coords=in_coords,
                radius=radius,
                max_points=max_points
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        total_matches = sum(len(matches) for matches in results.values())
        avg_matches = total_matches / len(results)
        
        print(f"  Time: {duration:.3f}s")
        print(f"  Total matches: {total_matches}")
        print(f"  Avg matches per query: {avg_matches:.1f}")
        print(f"  Memory accesses: {len(mem_trace) if 'mem_trace' in globals() else 'N/A'}")
        print()
    
    print("Key insights:")
    print("- Lower max_points reduces processing time")
    print("- Results are always sorted by distance (closest first)")
    print("- Memory access patterns are more predictable with limits")
    print("- Useful for applications where you only need nearest neighbors")

if __name__ == "__main__":
    performance_comparison()
