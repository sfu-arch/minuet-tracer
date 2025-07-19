#!/usr/bin/env python3
"""
Ball Query Results Reader

This module provides functions to read and analyze ball query results from gzipped binary files.
It includes validation and statistical analysis capabilities for ball query kernel maps.

Usage:
    from ballquery_reader import kernel_map_reader_ballquery, validate_ball_query_results
    
    # Read ball query results
    results = kernel_map_reader_ballquery("bq_kmap.gz", verbose=True)
    
    # Validate results
    is_valid = validate_ball_query_results(results, max_radius=2.0)
"""

import struct
import gzip
from typing import Dict, List, Tuple, Optional
import argparse

def read_ball_query_results_from_gz(filename: str) -> Dict[int, List[Tuple[int, float]]]:
    """
    Read ball query results from a gzipped binary file.
    This is the low-level reader for ball query results.
    
    Format:
    - num_queries (uint32_t)
    For each query:
        - query_idx (uint32_t)
        - num_matches (uint32_t)
        For each match:
            - input_idx (uint32_t)
            - distance (float)
    
    Args:
        filename: Path to the gzipped binary file
        
    Returns:
        Dictionary mapping query indices to lists of (input_index, distance) tuples
    """
    results = {}
    
    try:
        with gzip.open(filename, 'rb') as f:
            # Read header
            num_queries_bytes = f.read(4)
            if len(num_queries_bytes) < 4:
                raise ValueError("File too short to contain valid header")
            
            num_queries = struct.unpack('I', num_queries_bytes)[0]
            print(f"Reading {num_queries} queries from {filename}")
            
            total_matches = 0
            
            # Read each query's results
            for _ in range(num_queries):
                # Read query index and number of matches
                query_header = f.read(8)  # 4 bytes for query_idx + 4 bytes for num_matches
                if len(query_header) < 8:
                    raise ValueError("Unexpected end of file while reading query header")
                
                query_idx, num_matches = struct.unpack('II', query_header)
                
                # Read matches for this query
                matches = []
                for _ in range(num_matches):
                    match_data = f.read(8)  # 4 bytes for input_idx + 4 bytes for distance
                    if len(match_data) < 8:
                        raise ValueError("Unexpected end of file while reading match data")
                    
                    input_idx, distance = struct.unpack('If', match_data)
                    matches.append((input_idx, distance))
                
                results[query_idx] = matches
                total_matches += num_matches
            
            print(f"Successfully read {len(results)} queries with {total_matches} total matches")
            
    except Exception as e:
        print(f"Error reading ball query results from {filename}: {e}")
        return {}
    
    return results


def kernel_map_reader_ballquery(filename: str, 
                                verbose: bool = True, 
                                max_queries_to_show: int = 5) -> Dict[int, List[Tuple[int, float]]]:
    """
    Kernel map reader specifically for ball query results.
    This function provides a user-friendly interface for reading ball query kernel maps.
    
    Args:
        filename: Path to the ball query results file (.gz format)
        verbose: Whether to print detailed statistics
        max_queries_to_show: Maximum number of queries to display in verbose mode
        
    Returns:
        Dictionary mapping query indices to lists of (input_index, distance) tuples
    """
    results = read_ball_query_results_from_gz(filename)
    
    if not results:
        print("No results found or error reading file")
        return {}
    
    if verbose:
        print(f"\n{'='*60}")
        print("Ball Query Kernel Map Statistics")
        print(f"{'='*60}")
        
        # Basic statistics
        num_queries = len(results)
        total_matches = sum(len(matches) for matches in results.values())
        avg_matches_per_query = total_matches / num_queries if num_queries > 0 else 0
        
        print(f"Total queries: {num_queries}")
        print(f"Total matches: {total_matches}")
        print(f"Average matches per query: {avg_matches_per_query:.2f}")
        
        # Distance statistics
        all_distances = []
        for matches in results.values():
            all_distances.extend([distance for _, distance in matches])
        
        if all_distances:
            print(f"Distance range: {min(all_distances):.3f} to {max(all_distances):.3f}")
            print(f"Average distance: {sum(all_distances)/len(all_distances):.3f}")
        
        # Match count distribution
        match_counts = [len(matches) for matches in results.values()]
        if match_counts:
            print(f"Matches per query range: {min(match_counts)} to {max(match_counts)}")
        
        # Show sample queries
        print(f"\nSample queries (showing first {max_queries_to_show}):")
        for i, (query_idx, matches) in enumerate(sorted(results.items())[:max_queries_to_show]):
            print(f"  Query {query_idx}: {len(matches)} matches")
            if matches:
                # Show first few matches
                sample_matches = matches[:3]
                for input_idx, distance in sample_matches:
                    print(f"    -> Input {input_idx} (distance: {distance:.3f})")
                if len(matches) > 3:
                    print(f"    ... and {len(matches) - 3} more matches")
        
        print(f"{'='*60}")
    
    return results


def validate_ball_query_results(results: Dict[int, List[Tuple[int, float]]], 
                               max_radius: Optional[float] = None) -> bool:
    """
    Validate ball query results for consistency and correctness.
    
    Args:
        results: Ball query results dictionary
        max_radius: Maximum expected radius (optional validation)
        
    Returns:
        True if results are valid, False otherwise
    """
    if not results:
        print("Validation failed: Empty results")
        return False
    
    issues = []
    
    # Check for consistent data types and ranges
    for query_idx, matches in results.items():
        if not isinstance(query_idx, int):
            issues.append(f"Query index {query_idx} is not an integer")
        
        if not isinstance(matches, list):
            issues.append(f"Matches for query {query_idx} is not a list")
            continue
        
        for match_idx, (input_idx, distance) in enumerate(matches):
            if not isinstance(input_idx, int):
                issues.append(f"Input index {input_idx} in query {query_idx} is not an integer")
            
            if not isinstance(distance, (int, float)):
                issues.append(f"Distance {distance} in query {query_idx} is not numeric")
            
            if distance < 0:
                issues.append(f"Negative distance {distance} in query {query_idx}")
            
            if max_radius is not None and distance > max_radius:
                issues.append(f"Distance {distance} exceeds max radius {max_radius} in query {query_idx}")
    
    if issues:
        print("Validation issues found:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
        return False
    
    print("Ball query results validation passed")
    return True


def analyze_ball_query_statistics(results: Dict[int, List[Tuple[int, float]]]) -> Dict[str, float]:
    """
    Perform detailed statistical analysis of ball query results.
    
    Args:
        results: Ball query results dictionary
        
    Returns:
        Dictionary with statistical metrics
    """
    if not results:
        return {}
    
    # Collect all distances and match counts
    all_distances = []
    match_counts = []
    
    for query_idx, matches in results.items():
        match_counts.append(len(matches))
        all_distances.extend([distance for _, distance in matches])
    
    # Calculate statistics
    stats = {
        'num_queries': len(results),
        'total_matches': sum(match_counts),
        'avg_matches_per_query': sum(match_counts) / len(match_counts) if match_counts else 0,
        'min_matches_per_query': min(match_counts) if match_counts else 0,
        'max_matches_per_query': max(match_counts) if match_counts else 0,
    }
    
    if all_distances:
        stats.update({
            'min_distance': min(all_distances),
            'max_distance': max(all_distances),
            'avg_distance': sum(all_distances) / len(all_distances),
            'total_distance_entries': len(all_distances)
        })
        
        # Calculate distance percentiles
        sorted_distances = sorted(all_distances)
        n = len(sorted_distances)
        stats.update({
            'distance_25th_percentile': sorted_distances[int(0.25 * n)],
            'distance_50th_percentile': sorted_distances[int(0.50 * n)],
            'distance_75th_percentile': sorted_distances[int(0.75 * n)],
            'distance_90th_percentile': sorted_distances[int(0.90 * n)],
        })
    
    return stats


def export_ball_query_results_to_csv(results: Dict[int, List[Tuple[int, float]]], 
                                     filename: str) -> None:
    """
    Export ball query results to CSV format for analysis.
    
    Args:
        results: Ball query results dictionary
        filename: Output CSV filename
    """
    try:
        with open(filename, 'w') as f:
            # Write header
            f.write("query_idx,input_idx,distance\n")
            
            # Write data
            for query_idx, matches in sorted(results.items()):
                for input_idx, distance in matches:
                    f.write(f"{query_idx},{input_idx},{distance:.6f}\n")
        
        total_rows = sum(len(matches) for matches in results.values())
        print(f"Exported {total_rows} ball query results to {filename}")
        
    except Exception as e:
        print(f"Error exporting to CSV: {e}")


def compare_ball_query_results(results1: Dict[int, List[Tuple[int, float]]], 
                              results2: Dict[int, List[Tuple[int, float]]],
                              tolerance: float = 1e-6) -> Dict[str, int]:
    """
    Compare two ball query result sets for differences.
    
    Args:
        results1: First result set
        results2: Second result set
        tolerance: Tolerance for floating point distance comparison
        
    Returns:
        Dictionary with comparison statistics
    """
    comparison = {
        'queries_only_in_first': 0,
        'queries_only_in_second': 0,
        'queries_in_both': 0,
        'matching_queries': 0,
        'queries_with_different_matches': 0,
        'total_distance_differences': 0
    }
    
    all_query_indices = set(results1.keys()) | set(results2.keys())
    
    for query_idx in all_query_indices:
        if query_idx in results1 and query_idx in results2:
            comparison['queries_in_both'] += 1
            
            matches1 = {input_idx: distance for input_idx, distance in results1[query_idx]}
            matches2 = {input_idx: distance for input_idx, distance in results2[query_idx]}
            
            if matches1.keys() == matches2.keys():
                # Same input indices, check distances
                distance_match = True
                for input_idx in matches1.keys():
                    if abs(matches1[input_idx] - matches2[input_idx]) > tolerance:
                        distance_match = False
                        comparison['total_distance_differences'] += 1
                
                if distance_match:
                    comparison['matching_queries'] += 1
                else:
                    comparison['queries_with_different_matches'] += 1
            else:
                comparison['queries_with_different_matches'] += 1
                
        elif query_idx in results1:
            comparison['queries_only_in_first'] += 1
        else:
            comparison['queries_only_in_second'] += 1
    
    return comparison


def main():
    """
    Main function demonstrating the ball query reader functionality.
    """
    parser = argparse.ArgumentParser(description="Ball Query KMap Reader")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input KMap file")
    args = parser.parse_args()

    # Read and analyze results
    results = kernel_map_reader_ballquery(args.input_file, verbose=True)

    if results:
        # Validate results
        print("\nValidating results...")
        is_valid = validate_ball_query_results(results)
        
        # Detailed statistics
        print("\nDetailed statistical analysis:")
        stats = analyze_ball_query_statistics(results)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        

if __name__ == "__main__":
    main()
