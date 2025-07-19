# Ball Query Dense Algorithm - Pseudocode Documentation

## Overview

The Ball Query Dense Algorithm is a spatial search algorithm that efficiently finds all points within a specified radius of query points in 3D space. This implementation uses an octree data structure for acceleration and includes comprehensive memory access tracking for performance analysis.

## Algorithm Structure

### High-Level Algorithm

```
ALGORITHM: Dense Ball Query with Octree Acceleration

INPUT:
- input_points: List of 3D coordinates to search in
- query_points: List of 3D query coordinates  
- radius: Search radius
- max_points: Maximum results per query (default 64)

OUTPUT:
- results: Dictionary mapping query_idx -> [(point_idx, distance), ...]
```

### Main Algorithm Flow

```
MAIN ALGORITHM:
1. PREPROCESSING:
   a. Convert input_points to IndexedCoord format with original indices
   b. Build octree from input_points:
      - max_depth = 8, max_points_per_node = 10
      - Track memory accesses for octree construction
   
2. FOR each query_point in query_points:
   a. Initialize empty results list
   b. Call octree_ball_query(octree.root, query_point, radius, results)
   c. Sort results by distance
   d. Limit results to max_points
   e. Store in results dictionary
```

## Core Components

### 1. Octree Construction

```
OCTREE BUILD (Recursive):
INPUT: points, center, half_size, depth, max_depth, max_points_per_node

1. Record memory access for node creation
2. Record memory access for reading point data

3. IF len(points) <= max_points_per_node OR depth >= max_depth:
   a. Create leaf node with points
   b. Record memory access for storing point indices
   c. RETURN leaf_node

4. ELSE:
   a. Create internal node
   b. Partition points into 8 octants based on center:
      - octant_idx = 0
      - FOR each dimension i in [0,1,2]:
        IF point[i] > center[i]: octant_idx |= (1 << i)
   c. FOR each non-empty octant:
      - Calculate child_center
      - new_half_size = half_size / 2
      - Recursively build child octree
      - Add child to node.children
   d. RETURN internal_node
```

### 2. Ball Query Search

```
OCTREE BALL QUERY (Recursive):
INPUT: node, query_point, radius, results, max_points

1. IF len(results) >= max_points:
   RETURN (early termination)

2. Record memory access for reading node

3. IF NOT ball_intersects_box(query_point, radius, node.bounding_box):
   RETURN (prune subtree)

4. IF node.is_leaf:
   FOR each point in node.points:
      a. IF len(results) >= max_points: BREAK
      b. Record memory access for reading point
      c. distance = euclidean_distance(query_point, point)
      d. IF distance <= radius:
         - Add (point, distance) to results
         - Record memory access for writing result
   
5. ELSE (internal node):
   FOR each child in node.children:
      a. IF len(results) >= max_points: BREAK
      b. Recursively call octree_ball_query(child, query_point, radius, results)
```

### 3. Ball-Box Intersection Test

```
BALL-BOX INTERSECTION TEST:
INPUT: ball_center, radius, box_center, box_half_size

1. distance_squared = 0
2. FOR each dimension i in [0, 1, 2]:
   a. diff = |ball_center[i] - box_center[i]|
   b. IF diff > box_half_size:
      edge_distance = diff - box_half_size
      distance_squared += edge_distance²
3. RETURN distance_squared <= radius²
```

### 4. Euclidean Distance Calculation

```
EUCLIDEAN DISTANCE:
INPUT: point1, point2

1. distance = sqrt(sum((point1[i] - point2[i])² for i in [0,1,2]))
2. RETURN distance
```

## Performance Optimizations

### Early Termination
- Stop searching when `max_points` results are found
- Prevents unnecessary computation for large result sets

### Spatial Pruning
- Use octree bounding box tests to eliminate entire subtrees
- Only traverse nodes that can potentially contain valid results

### Memory Access Tracking
- Record all memory accesses for performance analysis
- Separate tracking for:
  - Octree node accesses (read/write)
  - Point index accesses (read/write)
  - Result storage accesses (write)

### Memory Base Addresses
```
Memory Layout:
- OCTREE_NODES: WV_BASE + (1<<16)
- OCTREE_INDICES: WV_BASE + (1<<17)
- BALL_RESULTS: WV_BASE + (1<<18)
```

## Fallback Algorithm (Brute Force)

```
BRUTE FORCE FALLBACK (if octree disabled):
1. FOR each query_point:
   a. Initialize results = []
   b. FOR each input_point:
      - distance = euclidean_distance(query_point, input_point)
      - IF distance <= radius:
        Add (input_point, distance) to results
   c. Sort results by distance
   d. Limit to max_points
```

## Data Structures

### IndexedCoord
```
STRUCTURE IndexedCoord:
- coord: Coord3D (x, y, z coordinates)
- orig_idx: int (original index in input array)
```

### OctreeNode
```
STRUCTURE OctreeNode:
- center: (float, float, float)
- half_size: float
- points: List[IndexedCoord]
- children: List[OctreeNode] (8 children for octree)
- is_leaf: bool
- node_id: int
```

### Memory Tracker
```
STRUCTURE OctreeMemoryTracker:
- base_addr_nodes: Memory base address for node data
- base_addr_indices: Memory base address for point indices
- base_addr_results: Memory base address for results
- Counters for tracking memory usage
```

### Kernel Map Structure (Sparse Convolution)
```
STRUCTURE KernelMap (SortedByValueLengthDict):
- Key: offset_idx (int) - Index of the convolution kernel offset
- Value: List[(target_idx, source_idx)] - List of coordinate pairs
  - target_idx: Original index of target coordinate found within radius
  - source_idx: Original index of source coordinate that generated the query

KERNEL MAP BUILDING:
1. FOR each source_coordinate in input_coordinates:
   FOR each offset in convolution_kernel_offsets:
     a. query_point = source_coordinate + offset * stride
     b. matches = ball_query(query_point, radius)
     c. FOR each match in matches:
        kernel_map[offset_idx].append((match.orig_idx, source.orig_idx))

KERNEL MAP STORAGE FORMAT (Binary):
- num_total_entries (uint32_t)
- FOR each entry:
  - offset_key (uint32_t) - Packed 32-bit offset coordinates
  - input_idx (uint32_t) - Original index of target coordinate
  - query_src_orig_idx (uint32_t) - Original index of source coordinate
```

## Algorithm Complexity

### Time Complexity
- **Best Case**: O(log n) per query (balanced octree, sparse results)
- **Average Case**: O(log n + k) per query (where k is number of results)
- **Worst Case**: O(n) per query (degenerate tree or dense results)

### Space Complexity
- **Octree Storage**: O(n) for storing all points
- **Results Storage**: O(k) where k ≤ max_points per query
- **Memory Trace**: O(m) where m is total memory accesses

## Implementation Features

### Multi-threading Support
- Thread-safe memory access tracking
- Distributed query processing across threads
- Thread ID recorded for each memory access

### Configurable Parameters
- `max_depth`: Maximum octree depth (default: 8)
- `max_points_per_node`: Leaf node capacity (default: 10)
- `max_points`: Maximum results per query (default: 64)
- `radius`: Search radius (configurable per query)

### Memory Efficiency
- Bounded memory usage with configurable limits
- Overflow protection for memory counters
- Efficient memory layout for cache performance

## Usage Examples

### Basic Ball Query
```python
# Convert input coordinates to IndexedCoord format
indexed_coords = [IndexedCoord(Coord3D(x, y, z), idx) 
                 for idx, (x, y, z) in enumerate(input_coords)]

# Perform dense ball query mapping
results = dense_ball_query_mapping(
    uniq_coords=indexed_coords,
    query_coords=query_points,
    radius=1.5,
    max_points=64,
    use_octree=True
)
```

### With Kernel Map (Sparse Convolution)
```python
# Define convolution kernel offsets
kernel_offsets = [
    (0, 0, 0),   # Center
    (1, 0, 0),   # Right
    (-1, 0, 0),  # Left
    (0, 1, 0),   # Up
    (0, -1, 0),  # Down
    (0, 0, 1),   # Forward
    (0, 0, -1)   # Back
]

# Perform sparse convolution mapping
kmap = sparse_convolution_mapping(
    uniq_coords=indexed_coords,
    off_coords=kernel_offsets,
    stride=1.0,
    radius=1.5,
    use_octree=True
)

# Write kernel map to file
write_kernel_map_to_gz(kmap, "kernel_map.gz", kernel_offsets)
```

### With Memory Tracing
```python
# Enable memory tracing
mem_trace.clear()

# Perform query
results = dense_ball_query_mapping(indexed_coords, query_coords, radius)

# Write memory trace
checksum = write_gmem_trace("memory_trace.gz", 8)
```

### Reading Kernel Map from File
```python
# Read kernel map from compressed file
import gzip
import struct

def read_kernel_map_from_gz(filename: str) -> List[Tuple[int, int, int]]:
    """Read kernel map from gzipped binary file"""
    with gzip.open(filename, 'rb') as f:
        # Read header
        num_entries = struct.unpack('I', f.read(4))[0]
        
        # Read entries
        entries = []
        for _ in range(num_entries):
            offset_key, input_idx, query_src_orig_idx = struct.unpack('III', f.read(12))
            entries.append((offset_key, input_idx, query_src_orig_idx))
    
    return entries
```

## Comparison with Alternative Methods

### Traditional Ball Query vs Inverse Ball Query
The implementation includes both traditional and inverse query strategies:

- **Traditional**: Find points within radius of query point
- **Inverse**: Find points whose spheres contain the query point

Both approaches produce equivalent results but have different memory access patterns.

## Dense vs Sparse Operations

### Dense Ball Query
```
DENSE BALL QUERY MAPPING:
INPUT: input_coordinates, query_coordinates, radius, max_points
OUTPUT: Dict[query_idx -> List[(input_idx, distance)]]

ALGORITHM:
1. Build octree from input_coordinates
2. FOR each query_point:
   a. results = octree.ball_query(query_point, radius, max_points)
   b. Sort results by distance
   c. Store as query_results[query_idx] = results

PURPOSE: Find nearest neighbors within radius for arbitrary query points
USE CASES: Point cloud processing, nearest neighbor search, spatial queries
```

### Sparse Convolution Mapping
```
SPARSE CONVOLUTION MAPPING:
INPUT: input_coordinates, kernel_offsets, stride, radius
OUTPUT: SortedByValueLengthDict[offset_idx -> List[(target_idx, source_idx)]]

ALGORITHM:
1. Build octree from input_coordinates
2. Initialize kernel_map for each offset
3. FOR each source_coordinate:
   FOR each kernel_offset:
     a. query_point = source_coordinate + offset * stride
     b. matches = octree.ball_query(query_point, radius)
     c. FOR each match:
        kernel_map[offset_idx].append((match.orig_idx, source.orig_idx))

PURPOSE: Build convolution kernel mappings for sparse CNN operations
USE CASES: 3D sparse convolution, voxel-based neural networks, discrete kernel operations

KERNEL MAP STRUCTURE:
- Organized by offset index (kernel position)
- Each offset maps to list of (target, source) coordinate pairs
- Sorted by value length for optimization
- Used for efficient sparse matrix operations
```

### Key Differences

| Aspect | Dense Ball Query | Sparse Convolution |
|--------|------------------|-------------------|
| **Query Pattern** | Arbitrary continuous points | Discrete offset-based grid |
| **Output Structure** | Dict[query_idx -> matches] | Dict[offset_idx -> pairs] |
| **Sorting** | By distance from query | By value length (optimization) |
| **Use Case** | General spatial search | CNN kernel operations |
| **Memory Pattern** | Query-centric | Kernel-centric |
| **Result Limit** | max_points per query | Unlimited matches per offset |

## Performance Characteristics

### Strengths
- Logarithmic average-case performance
- Efficient spatial pruning
- Bounded result sets
- Comprehensive memory tracking
- Thread-safe operation

### Limitations
- Performance degrades with very dense point clouds
- Memory overhead for octree construction
- Not optimal for very small datasets (< 100 points)

## References

This implementation is part of the Minuet-Tracer project for analyzing memory access patterns in spatial algorithms. The octree-based approach provides significant performance improvements over brute-force methods for medium to large datasets.
