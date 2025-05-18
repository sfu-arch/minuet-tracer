import threading
from dataclasses import dataclass
import struct
import gzip
import bisect
from typing import List, Tuple, Dict, Set
from coord import pack32, unpack32, unpack32s, Coord3D
from tqdm import tqdm
# ── Global Memory Trace Setup ──
gmem_trace = []
current_phase = None
debug = False

# Number of virtual threads (parameterizable)
NUM_THREADS = 4  # example: 3 parallel virtual threads
phases = {
    'Radix-Sort': 0,
    'Build-Queries': 1,
    'Sort-QKeys': 2,
    'Tile-Pivots': 3,
    'Lookup': 4,
    "Lookup-Backward": 5,
    "Lookup-Forward": 6,
}
tensors = {'I': 0, 'QK': 1, 'QI': 2, 'QO': 3, 'PIV': 4, 'KM': 5, 'WO': 6, 'TILE': 7}
ops = {'R': 0, 'W': 1}

# Number of input tiles and pivots for speeding up backward search
I_TILES = 2

SIZE_KEY    = 4   # 32-bit keys
SIZE_INT    = 4
SIZE_WEIGHT = 4

## Tensor Regions
I_BASE    = 0x10000000 # Base address for input point coordinates
TILE_BASE = I_BASE     # Tile data reads (alias for I_BASE)
QK_BASE   = 0x20000000 # Query keys
QI_BASE   = 0x30000000 # Query input-index array
QO_BASE   = 0x40000000 # Query offset-index array
PIV_BASE  = 0x50000000 # Tile pivot keys
KM_BASE   = 0x60000000 # Kernel-map writes
WO_BASE   = 0x80000000 # Weight-offset keys

# Feature vectors (64-bit address space)
IV_BASE = 0x100000000 # Input feature vectors
WV_BASE = 0x800000000 # Weight values



@dataclass
class IndexedCoord:
    """Coordinate with associated index"""
    coord: Coord3D
    orig_idx: int
    
    def to_key(self) -> int:
        """Convert coordinate to packed 32-bit key"""
        return self.coord.to_key()
    
    @classmethod
    def from_key_and_index(cls, key: int, idx: int) -> 'IndexedCoord':
        """Create from key and index"""
        return cls(Coord3D.from_key(key), idx)


def address_to_tensor(addr):
    """Convert address to tensor name."""
    if addr >= I_BASE and addr < QK_BASE:
        return 'I'
    elif addr >= QK_BASE and addr < QI_BASE:
        return 'QK'
    elif addr >= QI_BASE and addr < QO_BASE:
        return 'QI'
    elif addr >= QO_BASE and addr < PIV_BASE:
        return 'QO'
    elif addr >= PIV_BASE and addr < KM_BASE:
        return 'PIV'
    elif addr >= KM_BASE and addr < WO_BASE:
        return 'KM'
    elif addr >= WO_BASE and addr < WV_BASE:
        return 'WO'
    else:
        return 'Unknown'


def write_gmem_trace(filename):
    """Write memory trace to a file in compressed integer format."""
    # Create mappings for strings to integers
    phase_map = {}
    
    # Create compressed trace
    compressed_trace = []
    for entry in gmem_trace:
        phase, thread_id, op, tensor, addr = entry
        
        # Map phase to integer (assign new ID if not seen before)
        if phase not in phase_map:
            phase_map[phase] = len(phase_map)
        phase_id = phase_map[phase]
        
        # Convert address from hex string to integer
        addr_int = int(addr, 16)
        
        # Create compressed entry (all integers)
        compressed_entry = (phase_id, thread_id, ops[op], tensors[tensor], addr_int)
        compressed_trace.append(compressed_entry)
    
    # Write to binary file for maximum compression
    with gzip.open(filename, 'wb') as f:
        # Write header with entry count
        f.write(struct.pack('I', len(compressed_trace)))
        
        # Write each entry as packed integers (BBBBI format)
        for entry in compressed_trace:
            f.write(struct.pack('BBBBI', *entry))
    
    print(f"Memory trace written to {filename}")
    print(f"Compressed {len(gmem_trace)} entries")
    print(f"Phase mapping: {phase_map}")


def record_access(thread_id, op, addr):
    """Record a memory access: virtual thread ID, operation (R/W), tensor, and address."""
    tensor = address_to_tensor(addr)
    entry = (current_phase, thread_id, op, tensor, hex(addr))
    gmem_trace.append(entry)



# ── Radix Sort with Fixed Virtual Threads ──
def radix_sort_with_memtrace(arr, base):
    B, mask = 256, 0xFF
    passes = 4  # 32-bit keys
    N = len(arr)
    aux = [0] * N
    for p in range(passes):
        # Phase: count
        for i in range(N):
            # cycle thread IDs among NUM_THREADS
            t_id = (i % NUM_THREADS) 
            record_access(t_id, 'R', base + i*SIZE_KEY)
            _ = (arr[i] >> (p*8)) & mask
        # Phase: scatter
        for i in range(N):
            t_id = (i % NUM_THREADS) 
            record_access(t_id, 'R', base + i*SIZE_KEY)
            pos = i  # for illustration, stable mapping
            aux[pos] = arr[i]
            record_access(t_id, 'W', base + pos*SIZE_KEY)
        # Swap buffers
        arr, aux = aux, arr
    return arr

# ── Unique-Sort Inputs with Fixed Virtual Threads ──
def compute_unique_sorted_coords(input_coords, stride):
    # Pack & write keys along with original indices
    indexed_keys = []
    for idx, (x, y, z) in enumerate(input_coords):
        coord = Coord3D(x, y, z)
        quantized = coord.quantized(stride)
        key = quantized.to_key()
        indexed_keys.append((key, idx)) 

    # Extract raw keys for memory trace simulation of radix sort
    raw_keys = [item[0] for item in indexed_keys]
    radix_sort_with_memtrace(raw_keys, I_BASE)

    # Sort indexed_keys by key, preserving original index
    sorted_items = sorted(indexed_keys, key=lambda item: item[0])

    # Deduplication. Store (unique_key, original_index)
    unique_indexed_coords = []
    last_key = None
    for item_idx, (key, original_idx) in enumerate(sorted_items):
        if key != last_key:
            unique_indexed_coords.append((key, original_idx))
            last_key = key

    # Print results
    if debug:    
        print("Sorted+Unique Keys with Original Indices:")
        for idx, (key, original_idx_val) in enumerate(unique_indexed_coords):
            coord = Coord3D.from_key(key)
            print(f"key={hex(key)}, coords=({coord.x}, {coord.y}, {coord.z}), original_input_index={original_idx_val}")
    
    return unique_indexed_coords

# Build queries for the kernel map
def build_coordinate_queries(unique_indexed_coords, stride, offset_coords):
    num_inputs = len(unique_indexed_coords)
    num_offsets = len(offset_coords)
    total_queries = num_inputs * num_offsets
    
    query_keys = [None] * total_queries
    query_input_indices = [0] * total_queries
    query_offset_indices = [0] * total_queries
    weight_offsets = [0] * total_queries
    
    for offset_idx, (dx, dy, dz) in enumerate(offset_coords):
        offset = Coord3D(dx, dy, dz)
        quantized_offset = offset.quantized(stride)
        offset_key = pack32(0, quantized_offset.x, quantized_offset.y, quantized_offset.z)
        
        for input_idx, (input_key, original_input_idx) in enumerate(unique_indexed_coords):
            query_idx = offset_idx * num_inputs + input_idx
            
            # Unpack input coordinates
            input_coord = Coord3D.from_key(input_key)
            
            # Create new coordinate by adding offset
            query_coord = Coord3D(
                input_coord.x + quantized_offset.x,
                input_coord.y + quantized_offset.y,
                input_coord.z + quantized_offset.z
            )
            
            # Pack coordinates into query key
            query_keys[query_idx] = IndexedCoord(query_coord.to_key(), original_input_idx)
            query_input_indices[query_idx] = input_idx
            query_offset_indices[query_idx] = offset_idx
            weight_offsets[query_idx] = offset_key
    
    return query_keys, query_input_indices, query_offset_indices, weight_offsets

# Generate tiles and pivot keys for accelerating lookups
def create_tiles_and_pivots(unique_indexed_coords, tile_size):
    tiles, pivots = [], []
    for start in range(0, len(unique_indexed_coords), tile_size):
        tile_items = unique_indexed_coords[start:start+tile_size]
        # Store only keys in tiles
        tiles.append([item for item in tile_items])
        # Pivot is the key from the first tuple
        pivots.append(tile_items[0][0])
        record_access(0, 'W', PIV_BASE + (len(pivots)-1)*SIZE_KEY)
    
    return tiles, pivots

# Lookup phase - main query processing
def perform_coordinate_lookup(unique_indexed_coords, query_keys, query_input_indices, 
                             query_offset_indices, weight_offsets, tiles, pivots, tile_size):
    # Initialize the kernel map for each offset
    offset_count = len(set(query_offset_indices))
    kernel_map = {k: [] for k in range(offset_count)}
    query_count = len(query_keys)
    
    # Thread synchronization
    kernel_map_lock = threading.Lock()
    thread_local = threading.local()
    
    # Define batch size for processing
    BATCH_SIZE = 128  # Adjust this based on your workload characteristics
    num_batches = (query_count + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    
    def record_access_local(thread_id, op, addr):
        """Record a memory access in thread-local storage"""
        if not hasattr(thread_local, 'local_trace'):
            thread_local.local_trace = []
            
        tensor = address_to_tensor(addr)
        entry = (current_phase, thread_id, op, tensor, hex(addr))
        thread_local.local_trace.append(entry)
    
    # Worker function for parallel execution of a portion of a batch
    def process_batch_portion(batch_start, thread_id, thread_start, thread_end):
        thread_local.local_trace = []
        for query_offset in range(thread_start, thread_end):
            query_idx = batch_start + query_offset
            if query_idx >= query_count:
                break  # Prevent going beyond array bounds
                
            # Read query key
            record_access_local(thread_id, 'R', QK_BASE + query_idx*SIZE_KEY)            
            query_key = query_keys[query_idx].coord
            
            # Binary search on pivots
            low, high = 0, len(pivots)-1
            while low <= high:
                mid = (low + high) // 2
                record_access_local(thread_id, 'R', PIV_BASE + mid*SIZE_KEY)
                if pivots[mid] <= query_key:
                    low = mid + 1
                else:
                    high = mid - 1
            
            # Handle boundary case
            if high < 0:
                high = 0
                
            # Search within the identified tile
            tile_idx = high
            base_offset = tile_idx * tile_size
            
            # Linear scan through the tile
            for j, (key,i_idx) in enumerate(tiles[tile_idx]):
                record_access_local(thread_id, 'R', TILE_BASE + (base_offset + j)*SIZE_KEY)
                if key == query_key:
                    # Match found
                    input_idx = query_input_indices[query_idx]
                    offset_idx = query_offset_indices[query_idx]
                    
                    # Extract the original unique key and its index
                    unique_key, original_idx = unique_indexed_coords[input_idx]

                    # Add to kernel map with thread safety
                    with kernel_map_lock:
                        input_coord = unpack32(key)
                        output_coord = unpack32(unique_key)
                        offset_coord = unpack32s(weight_offsets[query_idx])

                        kernel_map[offset_idx].append(( (input_coord,i_idx), (output_coord,query_keys[query_idx].orig_idx)))
                        record_access_local(thread_id, 'W', KM_BASE + offset_idx*SIZE_KEY)
                    
                    break
        
        # Transfer thread-local trace to global trace
        with kernel_map_lock:
            if hasattr(thread_local, 'local_trace') and thread_local.local_trace:
                gmem_trace.extend(thread_local.local_trace)
                thread_local.local_trace = []
    
    # Process queries in batches with tqdm progress bar
    batch_iterator = tqdm(
        range(num_batches), 
        desc="Processing batches", 
        unit="batch", 
        disable=False
    )
    
    for batch in batch_iterator:
        batch_start = batch * BATCH_SIZE
        batch_size = min(BATCH_SIZE, query_count - batch_start)
        
        if batch_size <= 0:
            break
        
        # Show additional info in the progress bar description
        batch_iterator.set_postfix(
            queries=f"{min(batch_start + batch_size, query_count)}/{query_count}",
            matches=sum(len(matches) for matches in kernel_map.values())
        )
        
        # Divide this batch among threads
        portion_size = (batch_size + NUM_THREADS - 1) // NUM_THREADS
        threads = []
        
        for t in range(NUM_THREADS):
            thread_start = t * portion_size
            thread_end = min(thread_start + portion_size, batch_size)
            
            if thread_start < batch_size:
                thread = threading.Thread(
                    target=process_batch_portion,
                    args=(batch_start, t, thread_start, thread_end)
                )
                threads.append(thread)
                thread.start()
        
        # Wait for all threads to complete processing this batch
        for thread in threads:
            thread.join()
    return kernel_map


def write_kernel_map_to_gz(kernel_map_data: Dict[int, List[Tuple[Tuple[Coord3D, int], Tuple[Coord3D, int]]]], filename: str, offset_coords_list: List[Tuple[int,int,int]]):
    """
    Write the kernel map to a gzipped binary file.
    Format:
    - num_total_entries (uint32_t)
    For each entry:
        - offset_idx (uint8_t)
        - input_coord_key (uint32_t)  (packed target coordinate)
        - input_coord_pos (uint32_t)  (original index of target coordinate)
        - output_coord_key (uint32_t) (packed source coordinate)
        - output_coord_pos (uint32_t) (original index of source coordinate)
    """
    total_entries = sum(len(matches) for matches in kernel_map_data.values())
    
    packed_entries = []
    for offset_idx_val, matches in kernel_map_data.items():
        if not (0 <= offset_idx_val < 256):
            print(f"Warning: offset_idx {offset_idx_val} is out of range for uint8_t. Skipping entries for this offset.")
            continue # Or handle with a larger type if necessary

        for match in matches:
            # match is ((input_coord_tuple, input_original_idx), (output_coord_tuple, output_original_idx))
            # input_coord_tuple is the target coordinate found
            # output_coord_tuple is the source coordinate that, when offset, matched the target
            
            input_coord_data, output_coord_data = match
            
            input_xyz_tuple, input_pos = input_coord_data
            output_xyz_tuple, output_pos = output_coord_data
            
            # Ensure input_xyz_tuple and output_xyz_tuple are indeed tuples (x,y,z)
            # The kernel_map stores them as Coord3D objects if created from unpack32 which returns tuples,
            # or if they were passed as Coord3D. Let's assume they are (x,y,z) tuples.
            # If they are Coord3D objects, we'd use .to_key() or access .x, .y, .z
            # Based on current kernel_map population:
            # input_coord = unpack32(key) -> tuple
            # output_coord = unpack32(unique_key) -> tuple
            # So, input_xyz_tuple and output_xyz_tuple are (x,y,z) tuples.

            input_key = pack32(input_xyz_tuple[0], input_xyz_tuple[1], input_xyz_tuple[2])
            output_key = pack32(output_xyz_tuple[0], output_xyz_tuple[1], output_xyz_tuple[2])
            offset_key = pack32(offset_coords_list[offset_idx_val][0], offset_coords_list[offset_idx_val][1], offset_coords_list[offset_idx_val][2])
            
            # Format: B (offset_idx), I (input_key), I (input_pos), I (output_key), I (output_pos)
            packed_entry_data = struct.pack('IIIII', offset_key, input_key, input_pos, output_key, output_pos)
            packed_entries.append(packed_entry_data)

    # Verify total_entries matches len(packed_entries) in case of filtering
    actual_entries_to_write = len(packed_entries)

    with gzip.open(filename, 'wb') as f:
        f.write(struct.pack('I', actual_entries_to_write)) # Header: total number of entries
        for entry_data in packed_entries:
            f.write(entry_data)
            
    print(f"Kernel map written to {filename}")
    print(f"Wrote {actual_entries_to_write} entries (expected {total_entries} before any filtering).")


# ── Example Test with Phases ──
if __name__ == '__main__':
    # Input data
    input_coords = [(1,5,0), (0,0,2), (0,1,1), (0,0,3)]
    stride = 1
   # offset_coords = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)]
    offset_coords = [(0,1,-1)]
    # Phase 1: Sort and deduplicate input coordinates
    current_phase = 'Radix-Sort'
    print(f"\n--- Phase: {current_phase} with {NUM_THREADS} threads ---")
    unique_indexed_coords = compute_unique_sorted_coords(input_coords, stride)

    # Phase 2: Build query data structures
    current_phase = 'Build-Queries'
    print('--- Phase: Build Queries ---')
    query_keys, query_input_indices, query_offset_indices, weight_offsets = build_coordinate_queries(
        unique_indexed_coords, stride, offset_coords
    )

    # Phase 3: Sort query keys (using existing sorted keys)
    current_phase = 'Sort-QKeys'
    print('--- Phase: Sort QKeys ---')
    # No sorting needed in this implementation

    # Phase 4: Create tiles and pivots for lookup optimization
    current_phase = 'Tile-Pivots'
    print('--- Phase: Make Tiles & Pivots ---')
    coordinate_tiles, tile_pivots = create_tiles_and_pivots(unique_indexed_coords, I_TILES)
    
    # Phase 5: Perform coordinate lookup
    current_phase = 'Lookup'
    print('--- Phase: Lookup ---')
    kernel_map = perform_coordinate_lookup(
        unique_indexed_coords, query_keys, query_input_indices,
        query_offset_indices, weight_offsets, coordinate_tiles, tile_pivots, I_TILES
    )
    
    current_phase = None
    
    # Print debug information
    if debug:
        print('\nSorted Source Array (Coordinate, Original Index):')
        for key, orig_idx in unique_indexed_coords:
            coord = Coord3D.from_key(key)
            print(f"  ({coord.x}, {coord.y}, {coord.z}), index={orig_idx}")
            
        print('\nQuery Segments:')
        for offset_idx in range(len(offset_coords)):
            segment = [query_keys[i] for i in range(len(query_keys)) 
                       if query_offset_indices[i] == offset_idx]
            coords = [Coord3D.from_key(k) for k in segment]
            print(f"  Offset {offset_coords[offset_idx]}: {[(c.x, c.y, c.z) for c in coords]}")
        
    print('\nKernel Map:')
    for offset_idx, matches in kernel_map.items():
        if matches:
            print(f"  Offset {offset_coords[offset_idx]}: {matches}")
    
    # Write memory trace to file
    print('\nMemory Trace Entries:')
    for e in gmem_trace[:10]:  # Show first 10 entries only
        print(e)
    print(f"... and {len(gmem_trace)-10} more entries")
    
    write_gmem_trace('map_trace.bin.gz')
    write_kernel_map_to_gz(kernel_map, 'kernel_map.bin.gz', offset_coords)
    