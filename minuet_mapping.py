import threading
from dataclasses import dataclass
import struct
import gzip
import bisect
from typing import List, Tuple, Dict, Set
from coord import pack32, unpack32, unpack32s, Coord3D
from tqdm import tqdm
# ── Global Memory Trace Setup ──
mem_trace = []
curr_phase = None
debug = True

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


def addr_to_tensor(addr):
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
    comp_trace = []
    for entry in mem_trace:
        phase, thread_id, op, tensor, addr = entry
        
        # Map phase to integer (assign new ID if not seen before)
        if phase not in phase_map:
            phase_map[phase] = len(phase_map)
        phase_id = phase_map[phase]
        
        # Convert address from hex string to integer
        addr_int = int(addr, 16)
        
        # Create compressed entry (all integers)
        comp_entry = (phase_id, thread_id, ops[op], tensors[tensor], addr_int)
        comp_trace.append(comp_entry)
    
    # Write to binary file for maximum compression
    with gzip.open(filename, 'wb') as f:
        # Write header with entry count
        f.write(struct.pack('I', len(comp_trace)))
        
        # Write each entry as packed integers (BBBBI format)
        for entry in comp_trace:
            f.write(struct.pack('BBBBI', *entry))
    
    print(f"Memory trace written to {filename}")
    print(f"Compressed {len(mem_trace)} entries")
    print(f"Phase mapping: {phase_map}")


def record_access(thread_id, op, addr):
    """Record a memory access: virtual thread ID, operation (R/W), tensor, and address."""
    tensor = addr_to_tensor(addr)
    entry = (curr_phase, thread_id, op, tensor, hex(addr))
    mem_trace.append(entry)



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
def compute_unique_sorted_coords(in_coords, stride):
    # Pack & write keys along with original indices
    idx_keys = []
    for idx, (x, y, z) in enumerate(in_coords):
        coord = Coord3D(x, y, z)
        qtz = coord.quantized(stride)
        key = qtz.to_key()
        idx_keys.append((key, idx)) 

    # Extract raw keys for memory trace simulation of radix sort
    raw_keys = [item[0] for item in idx_keys]
    radix_sort_with_memtrace(raw_keys, I_BASE)

    # Sort idx_keys by key, preserving original index
    sorted_items = sorted(idx_keys, key=lambda item: item[0])

    # Deduplication. Store IndexedCoord(Coord3D_obj, original_index_from_input)
    uniq_coords = []
    last_key = None
    for item_idx, (key, orig_idx_from_input) in enumerate(sorted_items):
        if key != last_key:
            # Store IndexedCoord with Coord3D and original input index
            uniq_coords.append(IndexedCoord.from_key_and_index(key, orig_idx_from_input))
            last_key = key

    # Print results
    if debug:    
        print("Sorted+Unique Keys with Original Indices:")
        for idx, idxc_item in enumerate(uniq_coords):
            c = idxc_item.coord
            print(f"key={hex(c.to_key())}, coords=({c.x}, {c.y}, {c.z}), original_input_index={idxc_item.orig_idx}")
    
    return uniq_coords

# Build queries for the kernel map
def build_coordinate_queries(uniq_coords: List[IndexedCoord], stride, off_coords):
    num_inputs = len(uniq_coords)
    num_offsets = len(off_coords)
    total_queries = num_inputs * num_offsets
    
    qry_keys = [None] * total_queries
    qry_in_idx = [0] * total_queries
    qry_off_idx = [0] * total_queries
    wt_offsets = [0] * total_queries
    
    for off_idx, (dx, dy, dz) in enumerate(off_coords):
        offset = Coord3D(dx, dy, dz)
        qoff = offset.quantized(stride)
        # off_key = pack32(0, qoff.x, qoff.y, qoff.z) # This seems to be for weight offsets
        
        for in_idx, idxc_item in enumerate(uniq_coords):
            qry_idx = off_idx * num_inputs + in_idx
            # Get Coord3D from IndexedCoord
            in_coord = idxc_item.coord
            original_input_idx = idxc_item.orig_idx # Original index from initial input

            # Create new coordinate by adding offset
            qry_coord_obj = Coord3D(
                in_coord.x + qoff.x,
                in_coord.y + qoff.y,
                in_coord.z + qoff.z
            )
            
            # Store the Coord3D object itself in IndexedCoord for qry_keys
            # The orig_idx here is the original_input_idx of the *source* coordinate that generated this query
            qry_keys[qry_idx] = IndexedCoord(qry_coord_obj, original_input_idx)
            qry_in_idx[qry_idx] = in_idx # Index within the unique_coords list
            qry_off_idx[qry_idx] = off_idx
            
            # wt_offsets stores the packed key of the offset vector itself.
            # Let's ensure off_key is calculated correctly for wt_offsets.
            # The original code for off_key was: pack32(0, qoff.x, qoff.y, qoff.z)
            # This seems correct for representing the offset.
            current_offset_key = pack32(0, qoff.x, qoff.y, qoff.z) # Re-calculate or use from outer loop
            wt_offsets[qry_idx] = current_offset_key
    
    return qry_keys, qry_in_idx, qry_off_idx, wt_offsets

# Generate tiles and pivot keys for accelerating lookups
def create_tiles_and_pivots(uniq_coords: List[IndexedCoord], tile_size):
    tiles, pivs = [], []
    for start in range(0, len(uniq_coords), tile_size):
        tile_items = uniq_coords[start:start+tile_size] # List[IndexedCoord]
        # Store IndexedCoord objects in tiles
        tiles.append([item for item in tile_items])
        # Pivot is the key from the first IndexedCoord's Coord3D object
        pivs.append(tile_items[0].coord.to_key())
        record_access(0, 'W', PIV_BASE + (len(pivs)-1)*SIZE_KEY)
    
    return tiles, pivs

# Lookup phase - main query processing
def perform_coordinate_lookup(uniq_coords: List[IndexedCoord], qry_keys: List[IndexedCoord], qry_in_idx, 
                             qry_off_idx, wt_offsets, tiles: List[List[IndexedCoord]], pivs, tile_size):
    # Initialize the kernel map for each offset
    off_count = len(set(qry_off_idx))
    kmap = {k: [] for k in range(off_count)}
    qry_count = len(qry_keys)
    
    # Thread synchronization
    kmap_lock = threading.Lock()
    t_local = threading.local()
    
    # Define batch size for processing
    BATCH_SIZE = 128  # Adjust this based on your workload characteristics
    num_batches = (qry_count + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    
    def record_local(thread_id, op, addr):
        """Record a memory access in thread-local storage"""
        if not hasattr(t_local, 'local_trace'):
            t_local.local_trace = []
            
        tensor = addr_to_tensor(addr)
        entry = (curr_phase, thread_id, op, tensor, hex(addr))
        t_local.local_trace.append(entry)
    
    # Worker function for parallel execution of a portion of a batch
    def process_batch_portion(batch_start, thread_id, thread_start, thread_end):
        t_local.local_trace = []
        for qry_offset in range(thread_start, thread_end):
            qry_idx = batch_start + qry_offset
            if qry_idx >= qry_count:
                break  # Prevent going beyond array bounds
                
            # Read query key
            record_local(thread_id, 'R', QK_BASE + qry_idx*SIZE_KEY)            
            # qry_keys[qry_idx].coord is now a Coord3D object, convert to key for comparison
            current_qry_key_int = qry_keys[qry_idx].coord.to_key()
            
            # Binary search on pivots
            low, high = 0, len(pivs)-1
            while low <= high:
                mid = (low + high) // 2
                record_local(thread_id, 'R', PIV_BASE + mid*SIZE_KEY)
                if pivs[mid] <= current_qry_key_int:
                    low = mid + 1
                else:
                    high = mid - 1
            
            # Handle boundary case
            if high < 0:
                high = 0
                
            # Search within the identified tile
            tile_idx = high
            base_offset = tile_idx * tile_size # For memory trace, not direct list indexing
            
            # Linear scan through the tile (which contains IndexedCoord objects)
            for j, tile_indexed_coord in enumerate(tiles[tile_idx]):
                record_local(thread_id, 'R', TILE_BASE + (base_offset + j)*SIZE_KEY)
                
                key_from_tile_int = tile_indexed_coord.coord.to_key()
                
                if key_from_tile_int == current_qry_key_int:
                    # Match found
                    # in_idx is the index into uniq_coords for the *source* of the query
                    source_in_idx = qry_in_idx[qry_idx] 
                    current_off_idx = qry_off_idx[qry_idx]
                    
                    # Get the Coord3D and original_input_idx of the *source* coordinate that formed the query
                    # This is from uniq_coords, using source_in_idx
                    source_indexed_coord = uniq_coords[source_in_idx]
                    # out_coord_obj = source_indexed_coord.coord # This is the source Coord3D
                    out_coord_key_int = source_indexed_coord.coord.to_key() # Key of the source coord
                    # The original_idx for the output side of the kernel map entry is the one from qry_keys
                    # which corresponds to the original input index of the source coordinate.
                    output_original_idx = qry_keys[qry_idx].orig_idx


                    # Add to kernel map with thread safety
                    with kmap_lock:
                        # in_coord is the (x,y,z) of the matched coordinate from the input set (found in tile)
                        # key_from_tile_int is its key.
                        in_coord_tuple = unpack32(key_from_tile_int)
                        # i_idx for the input side of kernel map is the original input index of the matched tile coordinate
                        input_original_idx = tile_indexed_coord.orig_idx

                        # out_coord is the (x,y,z) of the source coordinate (from uniq_coords)
                        out_coord_tuple = unpack32(out_coord_key_int)
                        # off_coord_tuple = unpack32s(wt_offsets[qry_idx]) # Not directly used in kmap value

                        kmap[current_off_idx].append(((in_coord_tuple, input_original_idx), 
                                                      (out_coord_tuple, output_original_idx)))
                        record_local(thread_id, 'W', KM_BASE + current_off_idx*SIZE_KEY)
                    
                    break
        
        # Transfer thread-local trace to global trace
        with kmap_lock:
            if hasattr(t_local, 'local_trace') and t_local.local_trace:
                mem_trace.extend(t_local.local_trace)
                t_local.local_trace = []
    
    # Process queries in batches with tqdm progress bar
    batch_iter = tqdm(
        range(num_batches), 
        desc="Processing batches", 
        unit="batch", 
        disable=False
    )
    
    for batch in batch_iter:
        batch_start = batch * BATCH_SIZE
        batch_size = min(BATCH_SIZE, qry_count - batch_start)
        
        if batch_size <= 0:
            break
        
        # Show additional info in the progress bar description
        batch_iter.set_postfix(
            queries=f"{min(batch_start + batch_size, qry_count)}/{qry_count}",
            matches=sum(len(matches) for matches in kmap.values())
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
    return kmap


def write_kernel_map_to_gz(kmap_data: Dict[int, List[Tuple[Tuple[Coord3D, int], Tuple[Coord3D, int]]]], filename: str, off_list: List[Tuple[int,int,int]]):
    """
    Write the kernel map to a gzipped binary file.
    Format:
    - num_total_entries (uint32_t)
    For each entry:
        - offset_idx (uint32_t)
        - input_coord_key (uint32_t)  (packed target coordinate)
        - input_coord_pos (uint32_t)  (original index of target coordinate)
        - output_coord_key (uint32_t) (packed source coordinate)
        - output_coord_pos (uint32_t) (original index of source coordinate)
    """
    total_entries = sum(len(matches) for matches in kmap_data.values())
    
    packed_entries = []
    for off_idx, matches in kmap_data.items():
        for match in matches:
            # match is ((input_coord_tuple, input_original_idx), (output_coord_tuple, output_original_idx))
            # input_coord_tuple is the target coordinate found
            # output_coord_tuple is the source coordinate that, when offset, matched the target
            
            in_coord_data, out_coord_data = match
            
            in_xyz, in_pos = in_coord_data
            out_xyz, out_pos = out_coord_data
            
            # Ensure input_xyz_tuple and output_xyz_tuple are indeed tuples (x,y,z)
            # The kernel_map stores them as Coord3D objects if created from unpack32 which returns tuples,
            # or if they were passed as Coord3D. Let's assume they are (x,y,z) tuples.
            # If they are Coord3D objects, we'd use .to_key() or access .x, .y, .z
            # Based on current kernel_map population:
            # input_coord = unpack32(key) -> tuple
            # output_coord = unpack32(unique_key) -> tuple
            # So, input_xyz_tuple and output_xyz_tuple are (x,y,z) tuples.

            in_key = pack32(in_xyz[0], in_xyz[1], in_xyz[2])
            out_key = pack32(out_xyz[0], out_xyz[1], out_xyz[2])
            off_key = pack32(off_list[off_idx][0], off_list[off_idx][1], off_list[off_idx][2])
            
            # Format: B (offset_idx), I (input_key), I (input_pos), I (output_key), I (output_pos)
            packed_entry = struct.pack('IIIII', off_key, in_key, in_pos, out_key, out_pos)
            packed_entries.append(packed_entry)

    # Verify total_entries matches len(packed_entries) in case of filtering
    actual_entries = len(packed_entries)

    with gzip.open(filename, 'wb') as f:
        f.write(struct.pack('I', actual_entries)) # Header: total number of entries
        for entry_data in packed_entries:
            f.write(entry_data)
            
    print(f"Kernel map written to {filename}")
    print(f"Wrote {actual_entries} entries (expected {total_entries} before any filtering).")


# ── Example Test with Phases ──
if __name__ == '__main__':
    # Input data
    in_coords = [(1,5,0), (0,0,2), (0,1,1), (0,0,3)]
    stride = 1
    off_coords = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)]
    # off_coords = [(0,1,-1)]
    # Phase 1: Sort and deduplicate input coordinates
    curr_phase = 'Radix-Sort'
    print(f"\n--- Phase: {curr_phase} with {NUM_THREADS} threads ---")
    uniq_coords = compute_unique_sorted_coords(in_coords, stride)

    # Phase 2: Build query data structures
    curr_phase = 'Build-Queries'
    print('--- Phase: Build Queries ---')
    qry_keys, qry_in_idx, qry_off_idx, wt_offsets = build_coordinate_queries(
        uniq_coords, stride, off_coords
    )

    # Phase 3: Sort query keys (using existing sorted keys)
    curr_phase = 'Sort-QKeys'
    print('--- Phase: Sort QKeys ---')
    # No sorting needed in this implementation

    # Phase 4: Create tiles and pivots for lookup optimization
    curr_phase = 'Tile-Pivots'
    print('--- Phase: Make Tiles & Pivots ---')
    c_tiles, pivs = create_tiles_and_pivots(uniq_coords, I_TILES)
    
    # Phase 5: Perform coordinate lookup
    curr_phase = 'Lookup'
    print('--- Phase: Lookup ---')
    kmap = perform_coordinate_lookup(
        uniq_coords, qry_keys, qry_in_idx,
        qry_off_idx, wt_offsets, c_tiles, pivs, I_TILES
    )
    
    curr_phase = None
    
    # Print debug information
    if debug:
        print('\nSorted Source Array (Coordinate, Original Index):')
        for idxc_item in uniq_coords: # uniq_coords is List[IndexedCoord]
            coord_obj = idxc_item.coord
            print(f"  key={hex(coord_obj.to_key())}, coords=({coord_obj.x}, {coord_obj.y}, {coord_obj.z}), index={idxc_item.orig_idx}")
            
        print('\nQuery Segments:')
        for off_idx in range(len(off_coords)):
            segment = [qry_keys[i] for i in range(len(qry_keys)) 
                       if qry_off_idx[i] == off_idx]
            # segment contains IndexedCoord objects where .coord is now a Coord3D object
            # To print the (x,y,z) of these query coordinates:
            print(f"  Offset {off_coords[off_idx]}: {[(ic.coord.x, ic.coord.y, ic.coord.z) for ic in segment]}")

    print('\nKernel Map:')
    for off_idx, matches in kmap.items():
        if matches:
            print(f"  Offset {off_coords[off_idx]}: {matches}")
    
    # Write memory trace to file
    print('\nMemory Trace Entries:')
    for e in mem_trace[:10]:  # Show first 10 entries only
        print(e)
    print(f"... and {len(mem_trace)-10} more entries")
    
    write_gmem_trace('map_trace.bin.gz')
    write_kernel_map_to_gz(kmap, 'kernel_map.bin.gz', off_coords)
