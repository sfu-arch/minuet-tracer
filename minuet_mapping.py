import threading
from dataclasses import dataclass
import struct
import gzip
import bisect
from typing import List, Tuple, Dict, Set
from coord import pack32, unpack32, unpack32s, Coord3D
from tqdm import tqdm
import matplotlib.pyplot as plt

# ── Global Memory Trace Setup ──
mem_trace = []
curr_phase = None
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
        phase_id = phases[phase]
        # print(phase_id)
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
    # print(f"Phase mapping: {phase_map}")


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
        for idx, idxc_item in enumerate(uniq_coords): # Renamed idxc_item to idxc_item
            coord = idxc_item.coord # Renamed c to coord
            print(f"key={hex(coord.to_key())}, coords=({coord.x}, {coord.y}, {coord.z}), original_input_index={idxc_item.orig_idx}")
    
    return uniq_coords

# Build queries for the kernel map
def build_coordinate_queries(uniq_coords: List[IndexedCoord], stride, off_coords):
    num_inputs = len(uniq_coords)
    num_offsets = len(off_coords)
    total_queries = num_inputs * num_offsets
    
    qry_keys = [None] * total_queries
    qry_in_idx = [0] * total_queries
    qry_off_idx = [0] * total_queries
    wt_offsets = [None] * total_queries
    
    for off_idx, (dx, dy, dz) in enumerate(off_coords):
        offset = Coord3D(dx, dy, dz)
        q_offset = offset.quantized(stride) # Renamed qoff to q_offset
        
        for in_idx, src_idxcoord in enumerate(uniq_coords): # Renamed idxc_item to src_idxcoord
            qry_idx = off_idx * num_inputs + in_idx
            # Get Coord3D from IndexedCoord
            src_coord = src_idxcoord.coord # Renamed in_coord to src_coord
            original_input_idx = src_idxcoord.orig_idx

            # Create new coordinate by adding offset
            query_coord = Coord3D( # Renamed qry_coord_obj to query_coord
                src_coord.x + q_offset.x,
                src_coord.y + q_offset.y,
                src_coord.z + q_offset.z
            )
            
            # Store the Coord3D object itself in IndexedCoord for qry_keys
            # The orig_idx here is the original_input_idx of the *source* coordinate that generated this query
            qry_keys[qry_idx] = IndexedCoord(query_coord, original_input_idx)
            qry_in_idx[qry_idx] = in_idx 
            qry_off_idx[qry_idx] = off_idx
            
            offset_key = pack32(q_offset.x, q_offset.y, q_offset.z) # Renamed current_offset_key to offset_key
            wt_offsets[qry_idx] = offset_key
    
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
def lookup(uniq_coords: List[IndexedCoord], qry_keys: List[IndexedCoord], qry_in_idx, 
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
            qry_key_val = qry_keys[qry_idx].coord.to_key() # Renamed current_qry_key_int to qry_key_val
            
            # Binary search on pivots
            low, high = 0, len(pivs)-1
            while low <= high:
                mid = (low + high) // 2
                record_local(thread_id, 'R', PIV_BASE + mid*SIZE_KEY)
                if pivs[mid] <= qry_key_val:
                    low = mid + 1
                else:
                    high = mid - 1
            
            # Handle boundary case
            if high < 0:
                high = 0
                
            # Search within the identified tile
            tile_idx = high
            base_offset = tile_idx * tile_size 
            
            # Linear scan through the tile (which contains IndexedCoord objects)
            for j, tile_idxcoord in enumerate(tiles[tile_idx]): # Renamed tile_indexed_coord to tile_idxcoord
                record_local(thread_id, 'R', TILE_BASE + (base_offset + j)*SIZE_KEY)
                
                tile_key_val = tile_idxcoord.coord.to_key() # Renamed key_from_tile_int to tile_key_val
                
                if tile_key_val == qry_key_val:
                    # Match found
                    src_uniq_idx = qry_in_idx[qry_idx] # Renamed source_in_idx to src_uniq_idx
                    curr_off_idx = qry_off_idx[qry_idx] # Renamed current_off_idx to curr_off_idx
                    
                    src_idxc = uniq_coords[src_uniq_idx] # Renamed source_indexed_coord to src_idxc
                    src_coord_key = src_idxc.coord.to_key() # Renamed out_coord_key_int to src_coord_key
                    query_src_orig_idx = qry_keys[qry_idx].orig_idx # Renamed output_original_idx to query_src_orig_idx


                    # Add to kernel map with thread safety
                    with kmap_lock:
                        input_tpl = unpack32(tile_key_val) # Renamed in_coord_tuple to input_tpl
                        input_idx = tile_idxcoord.orig_idx # Renamed input_original_idx to input_idx

                        out_coord_tpl = unpack32(src_coord_key) # Renamed out_coord_tuple to out_coord_tpl

                        kmap[curr_off_idx].append(((input_tpl, input_idx), 
                                                      (out_coord_tpl, query_src_orig_idx)))
                        record_local(thread_id, 'W', KM_BASE + curr_off_idx*SIZE_KEY)
                    
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
    coord_tiles, pivs = create_tiles_and_pivots(uniq_coords, I_TILES) # Renamed c_tiles to coord_tiles
    
    # Phase 5: Perform coordinate lookup
    curr_phase = 'Lookup'
    print('--- Phase: Lookup ---')
    kmap = lookup(
        uniq_coords, qry_keys, qry_in_idx,
        qry_off_idx, wt_offsets, coord_tiles, pivs, I_TILES
    )
    
    curr_phase = None
    
    # Print debug information
    if debug:
        print('\nSorted Source Array (Coordinate, Original Index):')
        for idxc_item in uniq_coords: # Renamed idxc_item to idxc_item
            coord = idxc_item.coord # Renamed coord_obj to coord
            print(f"  key={hex(coord.to_key())}, coords=({coord.x}, {coord.y}, {coord.z}), index={idxc_item.orig_idx}")
            
        print('\nQuery Segments:')
        for off_idx in range(len(off_coords)):
            segment = [qry_keys[i] for i in range(len(qry_keys)) 
                       if qry_off_idx[i] == off_idx]
            # segment contains IndexedCoord objects where .coord is now a Coord3D object
            # To print the (x,y,z) of these query coordinates:
            print(f"  Offset {off_coords[off_idx]}: {[(idxc.coord.x, idxc.coord.y, idxc.coord.z) for idxc in segment]}") # Renamed ic to idxc

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

### End of minuet_mapping.py
### Create Metadatas for kernel map
#### - First sort the list of offsets based on the length of matches
#### - Create an in, out mask for gather scatter [#Total slots * #Total points]
#### - mask[offset_idx][point_idx] = -1 (if not matched) otherwise the position of input in original input array. The points in slot array are listed based on sorted order of coordinates.
#### - offsets_active is a sparse list of offsets that have atleast one match
#### - slot array is number of slots for each offset.

    sorted_kmap = sorted(kmap.items(), key=lambda item: len(item[1]), reverse=True)

    print(sorted_kmap)

    # Count slot_array for allocation
    slot_array = []
    for off_idx, matches in sorted_kmap:
        if len(matches) > 0:
            slot_array.append(len(matches))

    # Create kernel map like this: (offset_idx, source_original_idx, target_original_idx)
    # Example kmap entry for Offset 1: [(((target_coord_X, target_coord_Y, target_coord_Z), target_orig_idx=1), ((source_coord_X, source_coord_Y, source_coord_Z), source_orig_idx=2))]
    # dict[offset, List[(src_idx, dst_idx)]]
    idx_kmap = {}
    for off_idx, matches in sorted_kmap:
        for in_coord, out_coord in matches:
            src_idx = in_coord[1]
            dst_idx = out_coord[1]
            if off_idx not in idx_kmap:
                idx_kmap[off_idx] = []
            idx_kmap[off_idx].append((src_idx, dst_idx))
    
    from minuet_gather import create_in_out_masks
    out_mask, in_mask, offsets_active, slot_addr = create_in_out_masks(idx_kmap, len(in_coords), len(uniq_coords))

    print(slot_addr)
    print(offsets_active)

    from minuet_gather import greedy_group
    
    pos_indices, groups, membership = greedy_group(
        slot_array,
        alignment=4,
        max_group=2,
        max_slots=4
    )

    print("Groups metadata (start, end, base, req, alloc):")
    for g in groups:
        print(g)
    
    # Print total space allocated by groups
    total_alloc = sum(g[3] for g in groups)
    print(f"\nTotal allocated space: {total_alloc} slots")

    print("\nPer-position slot indices:")
    print(pos_indices)

    print("\nGroup membership lists:")
    print(membership)

    from minuet_gather import compact_bar_chart
    compact_bar_chart(groups)

    # print("\nSorted Kernel Map by Length of Matches:")
    # for off_idx, matches in sorted_kmap:
    #     if matches:
    #         print(f"  Offset {off_coords[off_idx]}: {matches}")
    # print(f"Total entries in sorted kernel map: {len(sorted_kmap)}")

    if debug:
        print("Out Mask:")
        for i in range(len(out_mask)):
            print(off_coords[i // len(in_coords)], out_mask[i])
        print("In Mask:")
        for i in range(len(in_mask)):
            print(off_coords[i // len(uniq_coords)], in_mask[i])
