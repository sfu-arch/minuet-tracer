# ── Global Memory Trace Setup ──
mem_trace = []
debug = False
output_dir = 'out/'

# Number of virtual threads (parameterizable)
NUM_THREADS = 4  # example: 3 parallel virtual threads
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

## GEMM Parameters
GEMM_ALIGNMENT = 4
GEMM_WT_GROUP = 2
GEMM_SIZE = 4


# GATHER PARAMETERS
NUM_TILES = 4 
TILE_FEATS = 16 
BULK_FEATS = 4  
N_THREADS = 1  
TOTAL_FEATS_PT = NUM_TILES * TILE_FEATS


# Feature vectors (64-bit address space)
IV_BASE = 0x100000000 # Input feature vectors
WV_BASE = 0xF00000000 # Weight values

