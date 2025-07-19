#!/usr/bin/env python3
"""
Test script to verify that the max_points parameter is working correctly
"""

from ballquery_mapping import *
import numpy as np

def test_max_points_limit():
    """Test that max_points parameter correctly limits the number of results"""
    
    print("Testing max_points parameter...")
    
    # Create a dense cluster of points around origin
    np.random.seed(42)
    center = np.array([0.0, 0.0, 0.0])
    in_coords = []
    
    # Generate 100 points in a tight cluster (radius 0.5)
    for i in range(100):
        # Create points very close to origin
        noise = np.random.normal(0, 0.2, 3)
        point = center + noise
        in_coords.append(tuple(point))
    
    # Query at the center with large radius (should find all points)
    query_coords = [(0.0, 0.0, 0.0)]
    radius = 2.0  # Large radius to capture all points
    
    print(f"Generated {len(in_coords)} input points in tight cluster")
    print(f"Querying at center with radius {radius}")
    
    # Test with different max_points values
    for max_points in [10, 20, 50, 100]:
        results = ball_query_mapping(
            query_coords=query_coords,
            input_coords=in_coords,
            radius=radius,
            max_points=max_points
        )
        
        num_results = len(results[0]) if 0 in results else 0
        print(f"max_points={max_points:3d}: Got {num_results:3d} results")
        
        # Verify that we don't exceed max_points
        assert num_results <= max_points, f"Got {num_results} results, expected <= {max_points}"
        
        # Verify results are sorted by distance (closest first)
        if num_results > 1:
            distances = [result[1] for result in results[0]]
            assert distances == sorted(distances), "Results should be sorted by distance"
    
    print("✓ max_points parameter is working correctly!")
    print("✓ Results are sorted by distance (closest first)")

def test_max_points_edge_cases():
    """Test edge cases for max_points parameter"""
    
    print("\nTesting max_points edge cases...")
    
    # Create a few scattered points
    in_coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    query_coords = [(0.0, 0.0, 0.0)]
    radius = 2.0
    
    # Test max_points = 0 (should return empty)
    results = ball_query_mapping(
        query_coords=query_coords,
        input_coords=in_coords,
        radius=radius,
        max_points=0
    )
    assert len(results[0]) == 0, "max_points=0 should return no results"
    print("✓ max_points=0 returns empty results")
    
    # Test max_points = 1 (should return only closest point)
    results = ball_query_mapping(
        query_coords=query_coords,
        input_coords=in_coords,
        radius=radius,
        max_points=1
    )
    assert len(results[0]) == 1, "max_points=1 should return exactly 1 result"
    assert results[0][0][1] == 0.0, "Closest point should be at distance 0"
    print("✓ max_points=1 returns only closest point")
    
    # Test max_points larger than available points
    results = ball_query_mapping(
        query_coords=query_coords,
        input_coords=in_coords,
        radius=radius,
        max_points=100
    )
    assert len(results[0]) == 4, "Should return all 4 available points"
    print("✓ max_points larger than available points works correctly")

if __name__ == "__main__":
    test_max_points_limit()
    test_max_points_edge_cases()
    print("\nAll tests passed! 🎉")
