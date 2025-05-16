# ── Virtual-Threaded Memory Tracing Minuet Kernel Map ──
# Simulate a fixed pool of virtual threads of size NUM_THREADS
# Basic usage - read and analyze a trace file
import threading
#from pathlib import Path
#from read_pcl import read_point_cloud

# ── Global Memory Trace Setup ──
gmem_trace = []
current_phase = None
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
I_BASE    = 0x10000000 # Base address for input point coordinates. Input feature vectors get separate region
TILE_BASE = I_BASE # TILE: Tile data reads (alias U)
QK_BASE   = 0x20000000 # Query keys. We construct an explicit query keys by combining offset with input keys. In real minuet this would be done on-the-fly. There should be no acceses recorded to this region
QI_BASE   = 0x30000000 # QI: Query input-index array
QO_BASE   = 0x40000000 # QO: Query offset-index array
PIV_BASE  = 0x50000000 # PIV: Tile pivot keys
KM_BASE   = 0x60000000 # KM: Kernel-map writes
WO_BASE   = 0x80000000 # WO: Weight-offset keys


# The feature vectors. Moving into 64 bit space now to avoid collisions between 
# small metadata tensors and larger feature vectors.
IV_BASE = 0x100000000 # Base address for input feature vectors. This is where the input feature vectors are stored.
WV_BASE = 0x800000000


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
    phases = {}
    
    # Create compressed trace
    compressed_trace = []
    for entry in gmem_trace:
        phase, thread_id, op, tensor, addr = entry
        
        # Map phase to integer (assign new ID if not seen before)
        if phase not in phases:
            phases[phase] = len(phases)
        phase_id = phases[phase]
        
        # Convert address from hex string to integer
        addr_int = int(addr, 16)
        
        # Create compressed entry (all integers)
        compressed_entry = (phase_id, thread_id, ops[op], tensors[tensor], addr_int)
        compressed_trace.append(compressed_entry)
    
    # Write to binary file for maximum compression
    import struct
    import gzip
    
    with gzip.open(filename, 'wb') as f:
        # Write header with mapping information
        # f.write(struct.pack('I', len(phases)))
        f.write(struct.pack('I', len(compressed_trace)))
        
        # Write each entry as packed integers
        # Phase id, thread id, ops, tensor_id  map to bytes
        # Format should be BBBBI 
        for entry in compressed_trace:
            # print(entry)
            f.write(struct.pack('BBBBI', *entry))
    
    print(f"Memory trace written to {filename}")
    print(f"Compressed {len(gmem_trace)} entries")
    print(f"Phase mapping: {phases}")




def record_access(thread_id, op, addr):
    """Record a memory access: virtual thread ID, operation (R/W), tensor, and address."""
    tensor = address_to_tensor(addr)
    entry = (current_phase, thread_id, op, tensor, hex(addr))
    gmem_trace.append(entry)
    # print(f"[{current_phase}] Thread{thread_id} {op} {tensor}@{hex(addr)}")



# ── Helper: pack/unpack 32-bit keys ──
def pack32(*coords):
    key = 0
    for c in coords:
        key = (key << 12) | (c & 0xFFF)
    return key

def unpack32(key):
    z = key & 0xFFF; key >>= 12
    y = key & 0xFFF; key >>= 12
    x = key & 0xFFF; key >>= 12 
    return (x, y, z)
# Unpack signed 8 bit integers
# Unpack signed 8 bit integers

def unpack32s(key):
    # Treat as 8 bit signed integers
    z = key & 0xFFF
    z = z if z < 2048 else z - 4096
    key >>= 12

    y = key & 0xFFF
    y = y if y < 2048 else y - 4096
    key >>= 12

    x = key & 0xFFF
    x = x if x < 2048 else x - 4096

    return (x, y, z)

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
        # # Phase: prefix (no memory ops)
        # Assume counters are in register.
        # current_phase = f"Radix-P{p:02d}-Prefix"
        # Phase: scatter
        # current_phase = f"Radix-P{p:02d}-Scatter"
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
def compute_unique_sorted(coords, stride):
    # Pack & write keys
    keys = []
    for idx, (x, y, z) in enumerate(coords):
        t_id = (idx % NUM_THREADS) + 1
        qx, qy, qz = x//stride, y//stride, z//stride
        key = pack32(qx, qy, qz)
        keys.append(key)
        # record_access("G",t_id, 'W', I_BASE + idx*SIZE_KEY)

    # Radix sort
    radix_sort_with_memtrace(keys, I_BASE)
    sorted_keys = sorted(keys)
    # Deduplication. Cleaning up to ensure each point occurs only once.
    current_phase = 'Dedup'
    unique = []
    last = None
    for idx, k in enumerate(sorted_keys):
        t_id = (idx % NUM_THREADS) + 1
        # record_access("G",t_id, 'R', I_BASE + idx*SIZE_KEY)
        if k != last:
            unique.append(k)
            # record_access("G",t_id, 'W', I_BASE + (len(unique)-1)*SIZE_KEY)
            last = k

    # Print results
   
    if debug:    
        print("Sorted+Unique Keys:")
        for idx, k in enumerate(unique):
            print(f"key={hex(k)}, coords={unpack32(k)}")
    return unique



# We are explicitly building here the queries for the kernel map.
# However, in a real implementation, this would be done on-the-fly
# and thus no need to store the query keys in memory.
def build_queries(uniq_in, stride, offsets):
    M, K = len(uniq_in), len(offsets)
    total = M*K
    qkeys, qii, qki, woffs = [0]*total, [0]*total, [0]*total, [0]*total
    for k_idx, off_xyz in enumerate(offsets):
        dx, dy, dz = off_xyz
        odx, ody, odz = dx//stride, dy//stride, dz//stride
        off_key = pack32(0, odx, ody, odz)
        for i_idx, in_key in enumerate(uniq_in):
            idx = k_idx*M + i_idx
                        # Unpack both keys
            x_in, y_in, z_in = unpack32(in_key)
            
            x_new = x_in + odx
            y_new = y_in + ody
            z_new = z_in + odz

            # Repack with non-negative values
            qkeys[idx] = pack32(x_new, y_new, z_new)

            # record_access("G",thread_id, 'W', QK_BASE + idx*SIZE_KEY)
            qii[idx]   = i_idx
            # record_access("G",thread_id, 'W', QI_BASE + idx*SIZE_INT)
            qki[idx]   = k_idx
            # record_access("G",thread_id, 'W', QO_BASE + idx*SIZE_INT)
            woffs[idx] = off_key
            # record_access("G",thread_id, 'W', WO_BASE + idx*SIZE_KEY)
    return qkeys, qii, qki, woffs

# ── Tile and Pivot Generation ──. Assuming tiling is free. 
# Memory only required for pivots.
def make_tiles_and_pivots(uniq_in, tile_size):
    tiles, pivots = [], []
    for start in range(0, len(uniq_in), tile_size):
        tile = uniq_in[start:start+tile_size]
        tiles.append(tile)
        pivots.append(tile[0])
        record_access(0, 'W', PIV_BASE + (len(pivots)-1)*SIZE_KEY)
    return tiles, pivots


def lookup_phase(uniq, qkeys, qii, qki, woffs, tiles, pivots, tile_size):
    kernel_map = {k: [] for k in range(len(set(qki)))}
    Nq = len(qkeys)
    # Thread synchronization
    kernel_map_lock = threading.Lock()
    
    # Create thread-local storage
    thread_local = threading.local()
    
    def record_access_local(thread_id, op, addr):
        """Record a memory access in thread-local storage"""
        # Initialize thread-local trace list if needed
        if not hasattr(thread_local, 'lmem_trace'):
            thread_local.lmem_trace = []
            
        tensor = address_to_tensor(addr)
        entry = (current_phase, thread_id, op, tensor, hex(addr))
        thread_local.lmem_trace.append(entry)
    
    


    # Worker function for parallel execution
    def process_query(q_start, q_end, thread_id):
        thread_local.lmem_trace = []
        for q in range(q_start, q_end):
            record_access_local(thread_id, 'R', QK_BASE + q*SIZE_KEY)
            target = qkeys[q]
            
            # backward search on pivots
            lo, hi = 0, len(pivots)-1
            while lo<=hi:
                mid=(lo+hi)//2
                record_access_local(thread_id, 'R', PIV_BASE+mid*SIZE_KEY)
                if pivots[mid]<=target: lo=mid+1
                else: hi=mid-1
            if hi<0: hi = 0
            tile_idx=hi; base_off=tile_idx*tile_size
            
            # forward scan
            for j,val in enumerate(tiles[tile_idx]):
                record_access_local(thread_id, 'R', TILE_BASE+(base_off+j)*SIZE_KEY)
                if val==target:
                    i_idx, k_idx = qii[q], qki[q]
                    with kernel_map_lock:
                        kernel_map[k_idx].append((unpack32(val), unpack32(uniq[i_idx]), unpack32s(woffs[q])))
                        record_access_local(thread_id, 'W', KM_BASE + k_idx*SIZE_KEY)
                    addr_i=KM_BASE+q*2*SIZE_INT
                    addr_j=addr_i+SIZE_INT
                    # record_access("G",thread_id, 'W', addr_i)
                    # record_access("G",thread_id, 'W', addr_j)
                    # Transfer local trace to global trace and reset
                    break
            with kernel_map_lock:
                if hasattr(thread_local, 'lmem_trace') and thread_local.lmem_trace:
                    # print(f"Thread {thread_id} trace entries: {len(thread_local.lmem_trace)}")
                    gmem_trace.extend(thread_local.lmem_trace)
                    thread_local.lmem_trace = []  # Clear after transfer
            
    # Create threads to process data in parallel
    threads = []
    chunk_size = (Nq + NUM_THREADS - 1) // NUM_THREADS  # Ceiling division
    
    for t in range(NUM_THREADS):
        q_start = t * chunk_size
        q_end = min(q_start + chunk_size, Nq)
        if q_start < Nq:  # Only create thread if there's work to do
            thread = threading.Thread(
                target=process_query,
                args=(q_start, q_end, t)  # thread_id starts from 1
            )
            threads.append(thread)
            thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
        
    return kernel_map

# ── Example Test with Phases ──
if __name__ == '__main__':
    
    coords = [(1,5,0),(0,1,1),(0,0,2),(0,0,3)]
    stride = 1
    offsets = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)]
    # offsets = [(0,1,-1),(0,1,0)]

    # Phase 1
    current_phase = 'Radix-Sort'
    print(f"\n--- Phase: {current_phase} with {NUM_THREADS} threads ---")
    uniq = compute_unique_sorted(coords,stride)

    # Phase 2
    current_phase = 'Build-Queries'
    print('--- Phase: Build Queries ---')
    qkeys_pre, qii, qki, woffs = build_queries(uniq,stride,offsets)

    # Phase 3
    import bisect  # no radix: use built-in
    current_phase = 'Sort-QKeys'
    print('--- Phase: Sort QKeys ---')
    qkeys = qkeys_pre

    # Phase 4
    current_phase = 'Tile-Pivots'
    print('--- Phase: Make Tiles & Pivots ---')
    tiles, pivots = make_tiles_and_pivots(uniq,I_TILES)
    
    # for w in woffs:
    #    print(unpack32s(w))

    # Phase 5
    current_phase = 'Lookup'
    print('--- Phase: Lookup ---')
    km = lookup_phase(uniq,qkeys,qii,qki,woffs,tiles,pivots,I_TILES)
    current_phase = None
    if debug:
        print('\nSorted Source Array:', [unpack32s(k) for k in uniq])
        print('Segmented Query Arrays:', [[unpack32s(k) for k in qkeys_pre[k*len(uniq):(k+1)*len(uniq)] ] for k in range(len(offsets))])
    if debug:
        print('Kernel Map:', km)
    
    print('\nMemory Trace Entries:')
    for e in gmem_trace: print(e)
    write_gmem_trace('memory_trace.bin.gz')




