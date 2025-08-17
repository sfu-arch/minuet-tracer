import threading
from dataclasses import dataclass
import struct
import gzip
import bisect
from typing import List, Tuple, Dict, Set
from coord import pack32, unpack32, unpack32s, Coord3D
from tqdm import tqdm
import matplotlib.pyplot as plt
import minuet_config
from minuet_config import *
from minuet_utils import file_checksum
from sorted_dict import SortedByValueLengthDict
from sorted_dict import bidict
curr_phase = None

PHASES = bidict({
    'RDX': 0,
    'QRY': 1,
    'SRT': 2,
    'PVT': 3,
    'LKP': 4,
    'GTH': 5,
    'SCT': 6
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
    'WV': 10
})

OPS = bidict({
    'R': 0,
    'W': 1
})

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
    global curr_phase
    # print(curr_phase)
    """Convert address to tensor name."""
    if addr >= I_BASE and addr < QK_BASE:
        return TENSORS['I']
    elif addr >= QK_BASE and addr < QI_BASE:
        return TENSORS['QK']
    elif addr >= QI_BASE and addr < QO_BASE:
        return TENSORS['QI']
    elif addr >= QO_BASE and addr < PIV_BASE:
        return TENSORS['QO']
    elif addr >= PIV_BASE and addr < KM_BASE:
        return TENSORS['PIV']
    elif addr >= KM_BASE and addr < WO_BASE:
        return TENSORS['KM']
    elif addr >= WO_BASE and addr < IV_BASE:
        return TENSORS['WC']    
    elif addr >= IV_BASE and addr < GM_BASE:
        return TENSORS['IV']
    elif addr >= GM_BASE and addr < WV_BASE:
        return TENSORS['GM']
    elif addr >= WV_BASE and addr < WV_BASE + 2<<32:
        return TENSORS['WV']
    else:
        assert(False), f"Unknown address: {addr}"
        return 'Unknown'
    

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
    # print(f"Phase mapping: {phase_map}")


def record_access(thread_id, op, addr):
    """Record a memory access: virtual thread ID, operation (R/W), tensor, and address."""
    # Always increment the global memory access counter (this is single-threaded for now)
    minuet_config.increment_mem_access_counter()
    
    # Only record trace if tracing is enabled
    if minuet_config.ENABLE_MEM_TRACE:
        tensor = addr_to_tensor(addr)
        entry = (curr_phase, thread_id, op, tensor, addr)
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
            record_access(t_id, OPS['R'], base + i*SIZE_KEY)
            _ = (arr[i] >> (p*8)) & mask
        # Phase: scatter
        for i in range(N):
            t_id = (i % NUM_THREADS) 
            record_access(t_id, OPS['R'], base + i*SIZE_KEY)
            pos = i  # for illustration, stable mapping
            aux[pos] = arr[i]
            record_access(t_id, OPS['W'], base + pos*SIZE_KEY)
        # Swap buffers
        arr, aux = aux, arr
    return arr

# ── Unique-Sort Inputs with Fixed Virtual Threads ──
def compute_unique_sorted_coords(in_coords, stride):
    global curr_phase
    curr_phase = PHASES['RDX']

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
            print(f"key={coord.to_key()}, coords=({coord.x}, {coord.y}, {coord.z}), original_input_index={idxc_item.orig_idx}")

    return uniq_coords

# Build queries for the kernel map
def build_coordinate_queries(uniq_coords: List[IndexedCoord], stride, off_coords):
    global curr_phase
    curr_phase = PHASES['QRY']
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

# Generate tiles and pivot keys for accelerating LKPs
def create_tiles_and_pivots(uniq_coords: List[IndexedCoord], tile_size: int = None):
    global curr_phase
    curr_phase = PHASES['PVT']

    tiles, pivs = [], []
    if tile_size is None:
        tile_size = 16384  # Default to the entire list if no tile size is specified

    for start in range(0, len(uniq_coords), tile_size):
        tile_items = uniq_coords[start:start+tile_size] # List[IndexedCoord]
        # Store IndexedCoord objects in tiles
        tiles.append([item for item in tile_items])
        # Pivot is the key from the first IndexedCoord's Coord3D object
        pivs.append(tile_items[0].coord.to_key())
        record_access(0, OPS['W'], PIV_BASE + (len(pivs)-1)*SIZE_KEY)
    
    return tiles, pivs

# LKP phase - main query processing
def lookup(uniq_coords: List[IndexedCoord], qry_keys: List[IndexedCoord], qry_in_idx, 
                             qry_off_idx, wt_offsets, tiles: List[List[IndexedCoord]], pivs, tile_size):
    global curr_phase
    curr_phase = PHASES['LKP']
    # Initialize the kernel map for each offset as a SortedByValueLengthDict
    # This will automatically sort by length of values in descending order
    off_count = len(set(qry_off_idx))
    kmap = SortedByValueLengthDict(ascending=False)
    
    # Initialize empty lists for each offset to avoid KeyError on first append
    for k in range(off_count):
        kmap[k] = []
        
    qry_count = len(qry_keys)
    
    # Thread synchronization
    kmap_lock = threading.Lock()
    t_local = threading.local()
    
    # Define batch size for processing
    BATCH_SIZE = 128  # Adjust this based on your workload characteristics
    num_batches = (qry_count + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    
    def record_local(thread_id, op, addr):
        """Record a memory access in thread-local storage"""
        # Increment thread-local counter
        if not hasattr(t_local, 'local_counter'):
            t_local.local_counter = 0
        t_local.local_counter += 1
        
        # Only record trace if tracing is enabled
        if minuet_config.ENABLE_MEM_TRACE:
            if not hasattr(t_local, 'local_trace'):
                t_local.local_trace = []
                
            tensor = addr_to_tensor(addr)
            entry = (curr_phase, thread_id, op, tensor, addr)
            t_local.local_trace.append(entry)
    
    # Worker function for parallel execution of a portion of a batch
    def process_batch_portion(batch_start, thread_id, thread_start, thread_end):
        t_local.local_trace = []
        for qry_offset in range(thread_start, thread_end):
            qry_idx = batch_start + qry_offset
            if qry_idx >= qry_count:
                break  # Prevent going beyond array bounds
                
            # Read query key
            record_local(thread_id, OPS['R'], QK_BASE + qry_idx*SIZE_KEY)            
            qry_key_val = qry_keys[qry_idx].coord.to_key()
            
            # Binary search on pivots
            low, high = 0, len(pivs)-1
            while low <= high:
                mid = (low + high) // 2
                record_local(thread_id, OPS['R'], PIV_BASE + mid*SIZE_KEY)
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
            for j, tile_idxcoord in enumerate(tiles[tile_idx]):
                record_local(thread_id, OPS['R'], TILE_BASE + (base_offset + j)*SIZE_KEY)
                
                tile_key_val = tile_idxcoord.coord.to_key()
                
                if tile_key_val == qry_key_val:
                    # Match found
                    src_uniq_idx = qry_in_idx[qry_idx]
                    curr_off_idx = qry_off_idx[qry_idx]
                    
                    query_src_orig_idx = qry_keys[qry_idx].orig_idx
                    input_idx = tile_idxcoord.orig_idx

                    # Add to kernel map with thread safety - MODIFIED: only store (input_idx, query_src_orig_idx)
                    with kmap_lock:
                            
                        kmap[curr_off_idx].append((input_idx, query_src_orig_idx))
                        record_local(thread_id, OPS['W'], KM_BASE + curr_off_idx*SIZE_KEY)

                    break
        
        # Transfer thread-local trace to global trace
        with kmap_lock:
            # Add thread-local counter to global counter
            if hasattr(t_local, 'local_counter'):
                minuet_config.mem_access_counter += t_local.local_counter
                t_local.local_counter = 0
            
            if minuet_config.ENABLE_MEM_TRACE and hasattr(t_local, 'local_trace') and t_local.local_trace:
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
            
    # Remove empty entries from kernel map
    for off_idx in list(kmap.keys()):
        if len(kmap[off_idx]) == 0:
            del kmap[off_idx]
    return kmap


def write_kernel_map_to_gz(kmap_data, filename: str, off_list: List[Tuple[int,int,int]]):
    """
    Write the kernel map to a gzipped binary file.
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
            
            off_key = pack32(off_list[off_idx][0], off_list[off_idx][1], off_list[off_idx][2])
            
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

