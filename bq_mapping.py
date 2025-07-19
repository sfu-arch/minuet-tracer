import threading
from dataclasses import dataclass
import struct
import gzip
import bisect
from typing import List, Tuple, Dict, Set

from torch import addr
import bq_config
from coord import pack32f, unpack32, unpack32s, Coord3Df
from tqdm import tqdm
import matplotlib.pyplot as plt
from minuet_utils import file_checksum
from sorted_dict import SortedByValueLengthDict
from sorted_dict import bidict
import bq_config
from bq_config import *
from typing import Dict, List, Tuple, Optional
import numpy as np
import math

# Global variables for ball query operations

def file_checksum(filename: str) -> str:
    """Simple file checksum"""
    import hashlib
    with open(filename, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def record_access(thread_id, op, addr):
    """Record a memory access: virtual thread ID, operation (R/W), tensor, and address."""
    tensor = addr_to_tensor(addr)
    entry = (curr_phase, thread_id, op, tensor, addr)
    mem_trace.append(entry)


def write_gmem_trace(filename, sizeof_addr = 4):
    """Write memory trace to a file in compressed integer format."""
    # Create mappings for strings to integers
    # Write to binary file for maximum compression
    with gzip.open(filename, 'wb') as f:
        # Write header with entry count
        f.write(struct.pack('I', len(mem_trace)))
        # Write each entry as packed integers (BBBBI format)
        for entry in mem_trace:
            # print(entry)
            if sizeof_addr == 4:
                # entry is (phase, thread_id, op, tensor, addr)
                f.write(struct.pack('<BBBBI', *entry))
            elif sizeof_addr == 8:
                f.write(struct.pack('<BBBBQ', *entry))

    print(f"Memory trace written to {filename}")
    print(f"Compressed {len(mem_trace)} entries")
    mem_trace.clear()
    checksum = file_checksum(filename)
    return checksum


PHASES = bidict({
    'RDX': 0,
    'QRY': 1,
    'SRT': 2,
    'PVT': 3,
    'LKP': 4,
    'GTH': 5,
    'SCT': 6,
    'OCTREE_BUILD': 7,    # New phase for octree construction
    'BALL_QUERY': 8       # New phase for ball query operations
})

TENSORS = bidict({
    'I': 0,
    'QK': 1,
    'QI': 2,
    'QO': 3,
    'PIV': 4,
    'KM': 5,
    'WC': 6,
    'TILE': 7,
    'IV': 8,
    'GM': 9,
    'WV': 10,
    'OCTREE_NODES': 11,   # New tensor for octree node data
    'OCTREE_INDICES': 12, # New tensor for octree point indices
    'BALL_RESULTS': 13    # New tensor for ball query results

})

OPS = bidict({
    'R': 0,
    'W': 1
})






@dataclass
class IndexedCoord:
    """Coordinate with associated index"""
    coord: Coord3Df
    orig_idx: int
    
    def to_key(self) -> int:
        """Convert coordinate to packed 32-bit key"""
        return self.coord.to_key()
    
    @classmethod
    def from_key_and_index(cls, key: int, idx: int) -> 'IndexedCoord':
        """Create from key and index"""
        return cls(Coord3Df.from_key(key), idx)





@dataclass
class OctreeNode:
    """Octree node for spatial indexing"""
    center: Tuple[float, float, float]
    half_size: float
    points: List[IndexedCoord]
    children: Optional[List['OctreeNode']]
    is_leaf: bool
    node_id: int
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class OctreeMemoryTracker:
    """Tracks memory accesses during octree operations"""
    
    def __init__(self):
        self.base_addr_nodes = OCTREE_NODES_BASE
        self.base_addr_indices = OCTREE_INDICES_BASE
        self.base_addr_results = BALL_RESULTS_BASE
        self.node_counter = 0
        self.indices_counter = 0
        self.results_counter = 0
        
        # Use limits from configuration
        self.max_nodes = MAX_OCTREE_NODES
        self.max_indices = MAX_OCTREE_INDICES
        self.max_results = MAX_BALL_RESULTS
    
    def record_node_access(self, thread_id: int, op: str, node_id: int, num_bytes: int = 64):
        """Record access to specific octree node data"""
        if node_id >= self.max_nodes:
            return  # Skip recording to prevent overflow
        addr = self.base_addr_nodes + node_id * num_bytes
        record_access(thread_id, OPS[op], addr)
        if op == 'W':
            self.node_counter = max(self.node_counter, node_id + 1)
    
    def record_indices_access(self, thread_id: int, op: str, point_idx: int = None, num_indices: int = 1):
        """Record access to point indices"""
        if point_idx is not None:
            # Access to specific point index
            if point_idx >= self.max_indices:
                return  # Skip recording to prevent overflow
            addr = self.base_addr_indices + point_idx * SIZE_INT
            record_access(thread_id, OPS[op], addr)
        else:
            # Sequential access pattern (legacy)
            if self.indices_counter >= self.max_indices:
                return  # Skip recording to prevent overflow
            addr = self.base_addr_indices + self.indices_counter * SIZE_INT
            record_access(thread_id, OPS[op], addr)
            if op == 'W':
                self.indices_counter += num_indices
    
    def record_results_access(self, thread_id: int, op: str, result_idx: int = None, num_results: int = 1):
        """Record access to ball query results"""
        if result_idx is not None:
            # Access to specific result index
            if result_idx >= self.max_results:
                return  # Skip recording to prevent overflow
            addr = self.base_addr_results + result_idx * (SIZE_INT + 4)  # 4 bytes for int + 4 bytes for float
            record_access(thread_id, OPS[op], addr)
        else:
            # Sequential access pattern (legacy)
            if self.results_counter >= self.max_results:
                return  # Skip recording to prevent overflow
            addr = self.base_addr_results + self.results_counter * (SIZE_INT + 4)
            record_access(thread_id, OPS[op], addr)
            if op == 'W':
                self.results_counter += num_results





class Octree:
    """Simple octree implementation for 3D point cloud spatial indexing"""
    
    def __init__(self, max_depth: int = None, max_points_per_node: int = None):
        self.max_depth = max_depth if max_depth is not None else OCTREE_DEPTH
        self.max_points_per_node = max_points_per_node if max_points_per_node is not None else OCTREE_PTS_PER_NODE
        self.root: Optional[OctreeNode] = None
        self.memory_tracker = OctreeMemoryTracker()
        self.node_id_counter = 0
    
    def _get_next_node_id(self) -> int:
        """Get next unique node ID"""
        self.node_id_counter += 1
        return self.node_id_counter - 1
    
    def _calculate_bounds(self, points: List[IndexedCoord]) -> Tuple[Tuple[float, float, float], float]:
        """Calculate bounding box for points"""
        if not points:
            return (0.0, 0.0, 0.0), 1.0
        
        min_coords = [float('inf')] * 3
        max_coords = [float('-inf')] * 3
        
        for point in points:
            coords = [point.coord.x, point.coord.y, point.coord.z]
            for i in range(3):
                min_coords[i] = min(min_coords[i], coords[i])
                max_coords[i] = max(max_coords[i], coords[i])
        
        center = tuple((min_coords[i] + max_coords[i]) / 2.0 for i in range(3))
        half_size = max((max_coords[i] - min_coords[i]) / 2.0 for i in range(3))
        
        return center, half_size
    
    def build(self, points: List[IndexedCoord], thread_id: int = 0) -> None:
        """Build octree from list of indexed coordinates with memory tracking"""
        global curr_phase
        curr_phase = PHASES['OCTREE_BUILD']
        
        if not points:
            return
        
        center, half_size = self._calculate_bounds(points)
        self.root = self._build_recursive(points, center, half_size, 0, thread_id)
    
    def _build_recursive(self, points: List[IndexedCoord], center: Tuple[float, float, float], 
                        half_size: float, depth: int, thread_id: int) -> OctreeNode:
        """Recursively build octree with memory access tracking"""
        node_id = self._get_next_node_id()
        
        # Track memory access for node creation
        self.memory_tracker.record_node_access(thread_id, 'W', node_id)
        
        # Track memory access for reading point data
        for i, point in enumerate(points):
            self.memory_tracker.record_indices_access(thread_id, 'R', point.orig_idx)
        
        if len(points) <= self.max_points_per_node or depth >= self.max_depth:
            # Create leaf node
            node = OctreeNode(
                center=center,
                half_size=half_size,
                points=points,
                children=None,
                is_leaf=True,
                node_id=node_id
            )
            
            # Track memory access for storing point indices in leaf
            for i, point in enumerate(points):
                self.memory_tracker.record_indices_access(thread_id, 'W', point.orig_idx)
            return node
        
        # Create internal node
        node = OctreeNode(
            center=center,
            half_size=half_size,
            points=[],
            children=[],
            is_leaf=False,
            node_id=node_id
        )
        
        # Partition points into 8 octants
        octants = [[] for _ in range(8)]
        
        for point in points:
            octant_idx = 0
            coords = [point.coord.x, point.coord.y, point.coord.z]
            
            for i in range(3):
                if coords[i] > center[i]:
                    octant_idx |= (1 << i)
            
            octants[octant_idx].append(point)
            
            # Track memory access for reading and partitioning
            self.memory_tracker.record_indices_access(thread_id, 'R', point.orig_idx)
        
        # Recursively build children for non-empty octants
        new_half_size = half_size / 2.0
        
        for i, octant_points in enumerate(octants):
            if octant_points:
                # Calculate child center
                child_center = (
                    center[0] + (new_half_size if i & 1 else -new_half_size),
                    center[1] + (new_half_size if i & 2 else -new_half_size),
                    center[2] + (new_half_size if i & 4 else -new_half_size)
                )
                
                child = self._build_recursive(octant_points, child_center, new_half_size, 
                                            depth + 1, thread_id)
                node.children.append(child)
        
        return node
    
    def ball_query(self, query_point: Tuple[float, float, float], radius: float, 
                  thread_id: int = 0, max_points: int = None) -> List[Tuple[IndexedCoord, float]]:
        """
        Perform ball query to find all points within radius of query point.
        
        Args:
            query_point: The point to query around
            radius: Search radius
            thread_id: Thread ID for memory tracking
            max_points: Maximum number of points to return (uses MAX_NEIGHBORHOOD if None)
        """
        global curr_phase
        curr_phase = PHASES['BALL_QUERY']
        
        if max_points is None:
            max_points = MAX_NEIGHBORHOOD
        
        if self.root is None:
            return []
        
        results = []
        self._ball_query_recursive(self.root, query_point, radius, results, thread_id, max_points)
        
        # Sort results by distance and limit to max_points
        results.sort(key=lambda x: x[1])
        results = results[:max_points]
        
        # Track memory access for writing results
        for i in range(len(results)):
            self.memory_tracker.record_results_access(thread_id, 'W', i)
        
        return results
    
    def _ball_query_recursive(self, node: OctreeNode, query_point: Tuple[float, float, float],
                             radius: float, results: List[Tuple[IndexedCoord, float]], 
                             thread_id: int, max_points: int) -> None:
        """Recursively search octree for points within ball radius"""
        # Early termination if we already have enough points
        if len(results) >= max_points:
            return
            
        # Track memory access for reading node
        self.memory_tracker.record_node_access(thread_id, 'R', node.node_id)
        
        # Check if ball intersects with node's bounding box
        if not self._ball_intersects_box(query_point, radius, node.center, node.half_size):
            return
        
        if node.is_leaf:
            # Check all points in leaf node
            for point in node.points:
                if len(results) >= max_points:
                    break
                    
                # Track memory access for reading point data
                self.memory_tracker.record_indices_access(thread_id, 'R', point.orig_idx)
                
                distance = self._calculate_distance(query_point, 
                                                  (point.coord.x, point.coord.y, point.coord.z))
                
                if distance <= radius:
                    results.append((point, distance))
                    # Track memory access for temporary result storage
                    result_idx = len(results) - 1
                    self.memory_tracker.record_results_access(thread_id, 'W', result_idx)
        else:
            # Recursively search children
            for child in node.children:
                if len(results) >= max_points:
                    break
                self._ball_query_recursive(child, query_point, radius, results, thread_id, max_points)
    
    def _ball_intersects_box(self, ball_center: Tuple[float, float, float], radius: float,
                           box_center: Tuple[float, float, float], box_half_size: float) -> bool:
        """Check if ball intersects with axis-aligned bounding box"""
        distance_sq = 0.0
        
        for i in range(3):
            diff = abs(ball_center[i] - box_center[i])
            if diff > box_half_size:
                edge_distance = diff - box_half_size
                distance_sq += edge_distance * edge_distance
        
        return distance_sq <= radius * radius
    
    def _calculate_distance(self, p1: Tuple[float, float, float], 
                          p2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(3)))
    
    def get_stats(self) -> Dict[str, int]:
        """Get octree statistics"""
        if self.root is None:
            return {"total_nodes": 0, "leaf_nodes": 0, "total_points": 0, "max_depth": 0}
        
        stats = {"total_nodes": 0, "leaf_nodes": 0, "total_points": 0, "max_depth": 0}
        self._collect_stats_recursive(self.root, 0, stats)
        return stats
    
    def _collect_stats_recursive(self, node: OctreeNode, depth: int, stats: Dict[str, int]) -> None:
        """Recursively collect octree statistics"""
        stats["total_nodes"] += 1
        stats["max_depth"] = max(stats["max_depth"], depth)
        
        if node.is_leaf:
            stats["leaf_nodes"] += 1
            stats["total_points"] += len(node.points)
        else:
            for child in node.children:
                self._collect_stats_recursive(child, depth + 1, stats)


def addr_to_tensor(addr):
    """Convert address to tensor name for ball query operations."""
    global curr_phase
    # print(curr_phase)

    if addr >= bq_config.I_BASE and addr < bq_config.QK_BASE:
        return TENSORS['I']
    elif addr >= bq_config.QK_BASE and addr < bq_config.QI_BASE:
        return TENSORS['QK']
    elif addr >= bq_config.QI_BASE and addr < bq_config.QO_BASE:
        return TENSORS['QI']
    elif addr >= bq_config.QO_BASE and addr < bq_config.PIV_BASE:
        return TENSORS['QO']
    elif addr >= bq_config.PIV_BASE and addr < bq_config.KM_BASE:
        return TENSORS['PIV']
    elif addr >= bq_config.KM_BASE and addr < bq_config.WO_BASE:
        return TENSORS['KM']
    elif addr >= bq_config.WO_BASE and addr < bq_config.IV_BASE:
        return TENSORS['WC']    
    elif addr >= bq_config.IV_BASE and addr < bq_config.GM_BASE:
        return TENSORS['IV']
    elif addr >= bq_config.GM_BASE and addr < bq_config.WV_BASE:
        return TENSORS['GM']
    elif addr >= bq_config.WV_BASE and addr < bq_config.OCTREE_NODES_BASE:
        return TENSORS['WV']
    elif addr >= bq_config.OCTREE_NODES_BASE and addr < bq_config.OCTREE_INDICES_BASE:
        return TENSORS['OCTREE_NODES']
    elif addr >= bq_config.OCTREE_INDICES_BASE and addr < bq_config.BALL_RESULTS_BASE:
        return TENSORS['OCTREE_INDICES']
    elif addr >= bq_config.BALL_RESULTS_BASE:
        return TENSORS['BALL_RESULTS']
    else:
        # For addresses that fall outside expected ranges, map to a default tensor
        # This prevents crashes during testing
        if debug:
            print(f"Debug: Address {addr:x} outside expected ranges, mapping to TILE")
        return TENSORS['TILE']
    







def write_kernel_map_to_gz(kmap_data, filename: str, off_list: List[Tuple[int,int,int]]):
    """
    Write the kernel map to a gzipped binary file.
    This function is for sparse convolution mapping only.
    Ball query results should use write_ball_query_results_to_gz() instead.
    
    Format:
    - num_total_entries (uint32_t)
    For each entry:
        - offset_idx (uint32_t)
        - input_idx (uint32_t)  (original index of target coordinate)
        - query_src_orig_idx (uint32_t) (original index of source coordinate)
    """
    # Updated to work with both regular dict and SortedByValueLengthDict
    total_entries = sum(len(entries) for entries in kmap_data.values())
    
    packed_entries = []
    for off_idx, entries in kmap_data.items():
        for entry in entries:
            # entry is (input_idx, query_src_orig_idx)
            input_idx, query_src_orig_idx = entry
            
            off_key = pack32f(off_list[off_idx][0], off_list[off_idx][1], off_list[off_idx][2])
            
            # Format: I (offset_key), I (input_idx), I (query_src_orig_idx)
            packed_entry = struct.pack('III', off_key, input_idx, query_src_orig_idx)
            packed_entries.append(packed_entry)

    # Verify total_entries matches len(packed_entries) in case of filtering
    actual_entries = len(packed_entries)

    with gzip.open(filename, 'wb') as f:
        f.write(struct.pack('I', actual_entries)) # Header: total number of entries
        for entry_data in packed_entries:
            f.write(entry_data)
            
    print(f"Kernel map written to {filename}")
    print(f"Wrote {actual_entries} entries (expected {total_entries} before any filtering).")


# ── Helper Functions ──

def compute_unique_sorted_coords(in_coords: List[Tuple[float, float, float]], 
                               stride: float) -> List[IndexedCoord]:
    """Convert input coordinates to unique sorted IndexedCoord objects"""
    unique_coords = []
    seen_keys = set()
    
    for orig_idx, (x, y, z) in enumerate(in_coords):
        # Quantize coordinates
        qx = round(x / stride) * stride
        qy = round(y / stride) * stride
        qz = round(z / stride) * stride
        
        coord = Coord3Df(qx, qy, qz)
        key = coord.to_key()
        
        if key not in seen_keys:
            unique_coords.append(IndexedCoord(coord, orig_idx))
            seen_keys.add(key)
    
    return unique_coords


def ball_query_mapping(query_coords: List[Tuple[float, float, float]],
                      input_coords: List[Tuple[float, float, float]],
                      radius: float = None,
                      max_points: int = None,
                      max_depth: int = None,
                      max_points_per_node: int = None) -> Dict[int, List[Tuple[int, float]]]:
    """
    Perform ball query mapping - find all points within radius of each query point.
    
    Args:
        query_coords: Query points to search around [(x,y,z), ...]
        input_coords: Input coordinates to search in [(x,y,z), ...]
        radius: Search radius (uses DEFAULT_RADIUS if None)
        max_points: Maximum number of points to return per query (uses MAX_NEIGHBORHOOD if None)
        max_depth: Maximum octree depth (uses OCTREE_DEPTH if None)
        max_points_per_node: Maximum points per leaf node (uses OCTREE_PTS_PER_NODE if None)
        
    Returns:
        Dictionary mapping query indices to lists of (input_index, distance) tuples
    """
    global curr_phase
    curr_phase = PHASES['BALL_QUERY']
    
    # Use configuration defaults if not specified
    if radius is None:
        radius = DEFAULT_RADIUS
    if max_points is None:
        max_points = MAX_NEIGHBORHOOD
    if max_depth is None:
        max_depth = OCTREE_DEPTH
    if max_points_per_node is None:
        max_points_per_node = OCTREE_PTS_PER_NODE
    
    # Convert to IndexedCoord format
    uniq_coords = []
    for idx, (x, y, z) in enumerate(input_coords):
        coord = Coord3Df(x, y, z)
        uniq_coords.append(IndexedCoord(coord, idx))
    
    return dense_ball_query_mapping(uniq_coords, query_coords, radius, max_points=max_points, use_octree=True)


# ── CORRECTED: Two Different Approaches ──

def kernel_mapping(NUM_THREADS: int, uniq_coords: List[IndexedCoord], 
                             off_coords: List[Tuple[int, int, int]],
                             stride: float = None,
                             radius: float = None,
                             use_octree: bool = True) -> SortedByValueLengthDict:
    """
    Perform SPARSE convolution mapping using discrete offsets.
    This is what the original code was trying to do.
    
    Args:
        uniq_coords: Unique input coordinates with original indices
        off_coords: Discrete offset coordinates for convolution kernel [(dx,dy,dz), ...]
        stride: Quantization stride (uses DEFAULT_STRIDE if None)
        radius: Search radius for finding matches around each offset query (uses DEFAULT_RADIUS if None)
        use_octree: Whether to use octree for accelerated search
        
    Returns:
        Kernel map with matches for each offset
    """

    bq_config.curr_phase = PHASES['BALL_QUERY']
    
    # Use configuration defaults if not specified
    if stride is None:
        stride = DEFAULT_STRIDE
    if radius is None:
        radius = DEFAULT_RADIUS
    
    print(f"Sparse convolution mapping with {len(uniq_coords)} points and {len(off_coords)} offsets")
    
    if use_octree:
        # Build octree from input coordinates
        octree = Octree()  # Use default parameters from config
        octree.build(uniq_coords, thread_id=0)
        
        stats = octree.get_stats()
        print(f"Octree built: {stats['total_nodes']} nodes, {stats['leaf_nodes']} leaves")
    
    # Initialize kernel map
    kmap = SortedByValueLengthDict(ascending=False)
    for offset_idx in range(len(off_coords)):
        kmap[offset_idx] = []
    
    # For each source coordinate and each offset
    total_queries = len(uniq_coords) * len(off_coords)
    
    with tqdm(total=total_queries, desc="Sparse convolution mapping") as pbar:
        for src_idx, src_coord in enumerate(uniq_coords):
            thread_id = src_idx % NUM_THREADS
            
            for offset_idx, (dx, dy, dz) in enumerate(off_coords):
                # Calculate discrete query point
                query_point = (
                    float(src_coord.coord.x + dx * stride),
                    float(src_coord.coord.y + dy * stride), 
                    float(src_coord.coord.z + dz * stride)
                )
                
                # Find all points within radius of this query point
                if use_octree:
                    matches = octree.ball_query(query_point, radius, thread_id=thread_id)
                else:
                    # Brute force search
                    matches = []
                    for target_coord in uniq_coords:
                        target_point = (target_coord.coord.x, target_coord.coord.y, target_coord.coord.z)
                        distance = math.sqrt(sum((query_point[i] - target_point[i]) ** 2 for i in range(3)))
                        if distance <= radius:
                            matches.append((target_coord, distance))
                
                # Add matches to kernel map
                for match_coord, distance in matches:
                    target_orig_idx = match_coord.orig_idx
                    src_orig_idx = src_coord.orig_idx
                    kmap[offset_idx].append((target_orig_idx, src_orig_idx))
                
                pbar.update(1)
    
    return kmap


def dense_ball_query_mapping(NUM_THREADS: int, uniq_coords: List[IndexedCoord], 
                           query_coords: List[Tuple[float, float, float]],
                           radius: float = None,
                           max_points: int = None,
                           use_octree: bool = True) -> Dict[int, List[Tuple[int, float]]]:
    """
    Perform DENSE ball query mapping - find all points within radius of each query point.
    This is what ball query should actually be used for.
    
    Args:
        uniq_coords: Input coordinates to search in
        query_coords: Query points to search around [(x,y,z), ...]
        radius: Search radius (uses DEFAULT_RADIUS if None)
        max_points: Maximum number of points to return per query (uses MAX_NEIGHBORHOOD if None)
        use_octree: Whether to use octree for acceleration
        
    Returns:
        Dictionary mapping query indices to lists of (input_index, distance) tuples
    """
    global curr_phase
    
    # Use configuration defaults if not specified
    if radius is None:
        radius = DEFAULT_RADIUS
    if max_points is None:
        max_points = MAX_NEIGHBORHOOD
    
    print(f"Dense ball query mapping with {len(uniq_coords)} input points and {len(query_coords)} queries")
    
    if use_octree:
        # Build octree from input coordinates
        octree = Octree()  # Use default parameters from config
        curr_phase = PHASES['OCTREE_BUILD']
        octree.build(uniq_coords, thread_id=0)
        
        stats = octree.get_stats()
        print(f"Octree built: {stats['total_nodes']} nodes, {stats['leaf_nodes']} leaves")
    
    results = {}
    
    # For each query point, find all points within radius
    with tqdm(total=len(query_coords), desc="Dense ball queries") as pbar:
        for query_idx, query_point in enumerate(query_coords):
            thread_id = query_idx % NUM_THREADS
            
            if use_octree:
                matches = octree.ball_query(query_point, radius, thread_id=thread_id, max_points=max_points)
            else:
                # Brute force search
                matches = []
                for target_coord in uniq_coords:
                    target_point = (target_coord.coord.x, target_coord.coord.y, target_coord.coord.z)
                    distance = math.sqrt(sum((query_point[i] - target_point[i]) ** 2 for i in range(3)))
                    if distance <= radius:
                        matches.append((target_coord, distance))
                
                # Sort by distance and limit to max_points
                matches.sort(key=lambda x: x[1])
                matches = matches[:max_points]
            
            # Convert to expected format
            if matches:
                results[query_idx] = [(match[0].orig_idx, match[1]) for match in matches]
            else:
                results[query_idx] = []
            
            pbar.update(1)
    
    return results




def write_ball_query_results_to_gz(results: Dict[int, List[Tuple[int, float]]], 
                                  filename: str) -> None:
    """
    Write ball query results to a gzipped binary file.
    Format:
    - num_queries (uint32_t)
    For each query:
        - query_idx (uint32_t)
        - num_matches (uint32_t)
        For each match:
            - input_idx (uint32_t)
            - distance (float)
    """
    total_queries = len(results)
    
    with gzip.open(filename, 'wb') as f:
        # Write header
        f.write(struct.pack('I', total_queries))
        
        # Write each query's results
        for query_idx in sorted(results.keys()):
            matches = results[query_idx]
            f.write(struct.pack('II', query_idx, len(matches)))
            
            # Write matches
            for input_idx, distance in matches:
                f.write(struct.pack('If', input_idx, distance))
    
    total_matches = sum(len(matches) for matches in results.values())
    print(f"Ball query results written to {filename}")
    print(f"Wrote {total_queries} queries with {total_matches} total matches")


# Ball query reader functions moved to ballquery_reader.py
# Import them when needed:
# from ballquery_reader import kernel_map_reader_ballquery, validate_ball_query_results, read_ball_query_results_from_gz


def demonstrate_ball_query_octree():
    """Demonstration function showing octree-based ball query usage"""
    
    # Example input coordinates (replace with actual data)
    in_coords = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0), 
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 2.0),
        (-1.0, -1.0, -1.0)
    ]
    
    # Example query coordinates
    query_coords = [
        (0.5, 0.5, 0.5),
        (1.5, 1.5, 1.5),
        (0.0, 0.0, 0.0)
    ]
    
    # Perform ball query mapping
    radius = 1.5
    
    # Convert to IndexedCoord format
    indexed_coords = []
    for idx, (x, y, z) in enumerate(in_coords):
        coord = Coord3Df(x, y, z)
        indexed_coords.append(IndexedCoord(coord, idx))
    
    results = dense_ball_query_mapping(bq_config.NUM_THREADS, indexed_coords, query_coords, radius, max_points=64)
    
    # Print results
    print("\nBall Query Results:")
    for query_idx, matches in results.items():
        print(f"Query {query_idx} at {query_coords[query_idx]}:")
        for input_idx, distance in matches:
            print(f"  Input {input_idx} at {in_coords[input_idx]}, distance: {distance:.3f}")
    
    # Write results to file
    write_ball_query_results_to_gz(results, "ball_query_results.gz")
    
    return results


# ── Main Integration Function ──

def run_comparison_study(in_coords: List[Tuple[float, float, float]], 
                        stride: float = None,
                        radius: float = None,
                        max_points: int = None,
                        tile_size: int = 1024) -> Dict[int, List[Tuple[int, float]]]:
    """
    Run ball query analysis using traditional octree approach.
    
    Args:
        in_coords: Input coordinates
        stride: Quantization stride (uses DEFAULT_STRIDE if None)
        radius: Search radius (uses DEFAULT_RADIUS if None)
        max_points: Maximum number of points to return per query (uses MAX_NEIGHBORHOOD if None)
        tile_size: Tile size (unused, kept for compatibility)
    
    Returns:
        Traditional ball query results
    """
    print("=" * 80)
    print("Ball Query Analysis using Traditional Octree Approach")
    print("=" * 80)
    
    # Use configuration defaults if not specified
    if stride is None:
        stride = DEFAULT_STRIDE
    if radius is None:
        radius = DEFAULT_RADIUS
    if max_points is None:
        max_points = MAX_NEIGHBORHOOD
    
    # Convert to IndexedCoord format
    uniq_coords = []
    for idx, (x, y, z) in enumerate(in_coords):
        coord = Coord3Df(x, y, z)
        uniq_coords.append(IndexedCoord(coord, idx))
    
    # Use input coordinates as query points (each point queries around itself)
    query_coords = [(coord.coord.x, coord.coord.y, coord.coord.z) for coord in uniq_coords]
    
    print(f"\nProcessing {len(uniq_coords)} coordinates with {len(query_coords)} queries")
    print(f"Using radius={radius}, max_points={max_points}, stride={stride}")
    
    # ── Method 1: Traditional ball query ──  
    print("\n--- Traditional Ball Query Method ---")
    global mem_trace
    mem_trace.clear()
    
    bq_kmap = dense_ball_query_mapping(uniq_coords, query_coords, radius, max_points=max_points, use_octree=True)
    
    # Write memory trace using proper size for addresses
    traditional_checksum = write_gmem_trace("bq_map_trace.gz", sizeof_addr=SIZE_ADDR)
    print(f"Traditional ball query method memory trace checksum: {traditional_checksum}")
    
    # Write results
    write_ball_query_results_to_gz(bq_kmap, "bq_kmap.gz")
    
    return bq_kmap


if __name__ == "__main__":
    # Load configuration first
    try:
        get_config("bq_config.json")
        print("Loaded ball query configuration from bq_config.json")
    except:
        print("Using default ball query configuration")
    
    # Example usage and demonstration
    print("Ball Query Mapping with Octree - Demonstration")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    num_points = 1000
    
    # Create some clustered 3D points
    in_coords = []
    for i in range(num_points):
        # Create clusters
        cluster_center = np.random.choice([0, 10, 20], 3)
        noise = np.random.normal(0, 2, 3)
        point = cluster_center + noise
        in_coords.append(tuple(point))
    
    print(f"Generated {len(in_coords)} input points")
    
    # Run analysis using configuration defaults
    bq_kmap = run_comparison_study(
        in_coords=in_coords,
        # Use configuration defaults for stride, radius, max_points
        tile_size=512
    )
    
    print("\nAnalysis complete!")
    print("Generated files:")
    print("- traditional_method_trace.gz")
    print("- bq_kmap.gz")
    
    # Test individual query methods
    print("\n" + "="*50)
    print("Testing Individual Query Methods")
    print("="*50)
    
    # Test direct ball queries
    test_queries = in_coords[:5]  # Test with first 5 points
    print(f"\nTesting ball queries with {len(test_queries)} query points:")
    
    # Convert to IndexedCoord format
    indexed_coords = []
    for idx, (x, y, z) in enumerate(in_coords):
        coord = Coord3Df(x, y, z)
        indexed_coords.append(IndexedCoord(coord, idx))
    
    # Traditional approach using configuration defaults
    bq_kmap = dense_ball_query_mapping(indexed_coords, test_queries, use_octree=True)
    
    print(f"Traditional method found {sum(len(v) for v in bq_kmap.values())} total matches")
    
    # Print some sample results
    print("\nSample results for first query:")
    if 0 in bq_kmap and bq_kmap[0]:
        for i, (input_idx, distance) in enumerate(bq_kmap[0][:5]):
            print(f"  Input {input_idx}: distance {distance:.3f}")
    
    # Demonstrate reading the results back
    print("\n" + "="*50)
    print("Testing Ball Query Reader")
    print("="*50)
    
    try:
        from ballquery_reader import kernel_map_reader_ballquery, validate_ball_query_results
        
        # Read back the results we just wrote
        print("\nReading back ball query results...")
        read_results = kernel_map_reader_ballquery("bq_kmap.gz", verbose=True, max_queries_to_show=3)
        
        # Validate the results
        print("\nValidating results...")
        is_valid = validate_ball_query_results(read_results, max_radius=DEFAULT_RADIUS * 2)
        
        if is_valid:
            print("✓ Ball query results are valid!")
        else:
            print("✗ Ball query results validation failed!")
            
    except ImportError:
        print("Note: ballquery_reader.py not found. Run 'python ballquery_reader.py bq_kmap.gz' to test the reader.")
       