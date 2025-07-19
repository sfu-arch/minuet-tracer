import json
import os

# ── Ball Query Configuration ──
mem_trace = []
debug = False
output_dir = 'out/'
curr_phase = None


# Ball Query Threading and Memory
NUM_THREADS = 8
SIZE_KEY = 4
SIZE_INT = 4
SIZE_FEAT = 2
SIZE_ADDR = 8  # 8-byte addresses for memory trace

# Ball Query Memory Regions
I_BASE =  0x00000000        # Input coordinates
QK_BASE = 0x10000000       # Query coordinates  
QI_BASE = 0x20000000       # Query indices
QO_BASE = 0x30000000       # Query outputs
PIV_BASE =0x40000000      # Pivot data
KM_BASE = 0x50000000       # Kernel map
WO_BASE = 0x60000000       # Write outputs
IV_BASE = 0x70000000       # Intermediate values
GM_BASE = 0x80000000       # General memory
WV_BASE = 0x90000000       # Working values
TILE_BASE = 0xA0000000     # Tile data

# Octree-specific memory regions (sub-regions of WV_BASE)
OCTREE_NODES_BASE = 0xB0000000    # Node data
OCTREE_INDICES_BASE = 0xC0000000  # Point indices
BALL_RESULTS_BASE = 0xD0000000    # Ball query results

# Ball Query Algorithm Parameters
DEFAULT_RADIUS = 1.5
MAX_NEIGHBORHOOD = 64
OCTREE_DEPTH = 8
OCTREE_PTS_PER_NODE = 64
DEFAULT_STRIDE = 1.0

# Ball Query Gather Parameters
NUM_TILES_GATHER = 4
TILE_FEATS_GATHER = 16
BULK_FEATS_GATHER = 4
N_THREADS_GATHER = 128
TOTAL_FEATS_PT = NUM_TILES_GATHER * TILE_FEATS_GATHER

# Memory Access Limits (to prevent overflow)
MAX_OCTREE_NODES = (1<<15)
MAX_OCTREE_INDICES = (1<<15)
MAX_BALL_RESULTS = (1<<15)

def get_config(_config_path):
    """Load ball query configuration from JSON file."""
    global debug, output_dir, NUM_THREADS, SIZE_KEY, SIZE_INT, SIZE_ADDR
    global I_BASE, QK_BASE, QI_BASE, QO_BASE, PIV_BASE, KM_BASE, WO_BASE, IV_BASE, GM_BASE, WV_BASE, TILE_BASE
    global OCTREE_NODES_BASE, OCTREE_INDICES_BASE, BALL_RESULTS_BASE
    global DEFAULT_RADIUS, MAX_NEIGHBORHOOD, OCTREE_DEPTH, OCTREE_PTS_PER_NODE, DEFAULT_STRIDE
    global NUM_TILES_GATHER, TILE_FEATS_GATHER, BULK_FEATS_GATHER, N_THREADS_GATHER, TOTAL_FEATS_PT
    global MAX_OCTREE_NODES, MAX_OCTREE_INDICES, MAX_BALL_RESULTS
    global mem_trace
    
    try:
        with open(_config_path, 'r') as f:
            _config_data = json.load(f)
        print(f"Successfully loaded ball query configuration from {_config_path}")
    except FileNotFoundError:
        print(f"Warning: Configuration file '{_config_path}' not found. Using default ball query values.")
        _config_data = {}
    except json.JSONDecodeError:
        print(f"Warning: Error decoding JSON from '{_config_path}'. Using default ball query values.")
        _config_data = {}
    except Exception as e:
        print(f"Warning: An unexpected error occurred while loading configuration: {e}. Using default ball query values.")
        _config_data = {}

    # Helper for hex string to int conversion
    def _hex_to_int(value, default_val):
        if isinstance(value, str) and value.startswith("0x"):
            try:
                return int(value, 16)
            except ValueError:
                print(f"Warning: Could not parse hex string '{value}'. Using default {hex(default_val)}.")
                return default_val
        elif isinstance(value, int):
            return value
        print(f"Warning: Unexpected type for hex value '{value}'. Using default {hex(default_val)}.")
        return default_val
    
    print(_config_data)
    
    # Override defaults with values from JSON
    debug = _config_data.get("debug", debug)
    output_dir = _config_data.get("output_dir", output_dir)
    NUM_THREADS = _config_data.get("NUM_THREADS", NUM_THREADS)
    SIZE_KEY = _config_data.get("SIZE_KEY", SIZE_KEY)
    SIZE_INT = _config_data.get("SIZE_INT", SIZE_INT)
    SIZE_ADDR = _config_data.get("SIZE_ADDR", SIZE_ADDR)

    # Memory base addresses
    I_BASE = _hex_to_int(_config_data.get("I_BASE"), I_BASE)
    QK_BASE = _hex_to_int(_config_data.get("QK_BASE"), QK_BASE)
    QI_BASE = _hex_to_int(_config_data.get("QI_BASE"), QI_BASE)
    QO_BASE = _hex_to_int(_config_data.get("QO_BASE"), QO_BASE)
    PIV_BASE = _hex_to_int(_config_data.get("PIV_BASE"), PIV_BASE)
    KM_BASE = _hex_to_int(_config_data.get("KM_BASE"), KM_BASE)
    WO_BASE = _hex_to_int(_config_data.get("WO_BASE"), WO_BASE)
    IV_BASE = _hex_to_int(_config_data.get("IV_BASE"), IV_BASE)
    GM_BASE = _hex_to_int(_config_data.get("GM_BASE"), GM_BASE)
    WV_BASE = _hex_to_int(_config_data.get("WV_BASE"), WV_BASE)
    TILE_BASE = _hex_to_int(_config_data.get("TILE_BASE"), TILE_BASE)

    # Update octree memory regions based on WV_BASE
    OCTREE_NODES_BASE = WV_BASE + (1<<16)
    OCTREE_INDICES_BASE = WV_BASE + (1<<17)
    BALL_RESULTS_BASE = WV_BASE + (1<<18)

    # Ball query algorithm parameters
    DEFAULT_RADIUS = _config_data.get("DEFAULT_RADIUS", DEFAULT_RADIUS)
    MAX_NEIGHBORHOOD = _config_data.get("MAX_NEIGHBORHOOD", MAX_NEIGHBORHOOD)
    OCTREE_DEPTH = _config_data.get("OCTREE_DEPTH", OCTREE_DEPTH)
    OCTREE_PTS_PER_NODE = _config_data.get("OCTREE_PTS_PER_NODE", OCTREE_PTS_PER_NODE)
    DEFAULT_STRIDE = _config_data.get("DEFAULT_STRIDE", DEFAULT_STRIDE)
    
    # Ball query gather parameters
    NUM_TILES_GATHER = _config_data.get("NUM_TILES_GATHER", NUM_TILES_GATHER)
    TILE_FEATS_GATHER = _config_data.get("TILE_FEATS_GATHER", TILE_FEATS_GATHER)
    BULK_FEATS_GATHER = _config_data.get("BULK_FEATS_GATHER", BULK_FEATS_GATHER)
    N_THREADS_GATHER = _config_data.get("N_THREADS_GATHER", N_THREADS_GATHER)
    TOTAL_FEATS_PT = _config_data.get("TOTAL_FEATS_PT", NUM_TILES_GATHER * TILE_FEATS_GATHER)
    
    # Memory limits
    MAX_OCTREE_NODES = _config_data.get("MAX_OCTREE_NODES", MAX_OCTREE_NODES)
    MAX_OCTREE_INDICES = _config_data.get("MAX_OCTREE_INDICES", MAX_OCTREE_INDICES)
    MAX_BALL_RESULTS = _config_data.get("MAX_BALL_RESULTS", MAX_BALL_RESULTS)
    
    print(f"Ball query configuration updated - NUM_THREADS: {NUM_THREADS}, DEFAULT_RADIUS: {DEFAULT_RADIUS}, Output Dir: {output_dir}")
    print(f"Octree memory regions: NODES={hex(OCTREE_NODES_BASE)}, INDICES={hex(OCTREE_INDICES_BASE)}, RESULTS={hex(BALL_RESULTS_BASE)}")

def reload_config(config_path=None):
    """
    Reload ball query configuration from file and update global variables.
    
    Args:
        config_path (str, optional): Path to configuration file. 
                                   Defaults to 'config.json' if None.
    """
    if config_path is None:
        config_path = 'config.json'
    
    print(f"\n=== Reloading Ball Query Configuration from {config_path} ===")
    
    # Store old values for comparison
    old_num_threads = NUM_THREADS
    old_radius = DEFAULT_RADIUS
    old_output_dir = output_dir
    
    # Reload configuration
    get_config(config_path)
    
    # Show what changed
    if old_num_threads != NUM_THREADS:
        print(f"NUM_THREADS: {old_num_threads} -> {NUM_THREADS}")
    if old_radius != DEFAULT_RADIUS:
        print(f"DEFAULT_RADIUS: {old_radius} -> {DEFAULT_RADIUS}")
    if old_output_dir != output_dir:
        print(f"Output Directory: {old_output_dir} -> {output_dir}")
    
    print("=== Ball Query Configuration Reload Complete ===\n")

def print_current_config():
    """Print current ball query configuration values for debugging."""
    print(f"\n=== Current Ball Query Configuration Values ===")
    print(f"NUM_THREADS: {NUM_THREADS}")
    print(f"DEFAULT_RADIUS: {DEFAULT_RADIUS}")
    print(f"MAX_NEIGHBORHOOD: {MAX_NEIGHBORHOOD}")
    print(f"OCTREE_DEPTH: {OCTREE_DEPTH}")
    print(f"OCTREE_PTS_PER_NODE: {OCTREE_PTS_PER_NODE}")
    print(f"DEFAULT_STRIDE: {DEFAULT_STRIDE}")
    print(f"Output Directory: {output_dir}")
    print(f"Debug Mode: {debug}")
    print(f"NUM_TILES_GATHER: {NUM_TILES_GATHER}")
    print(f"TILE_FEATS_GATHER: {TILE_FEATS_GATHER}")
    print(f"BULK_FEATS_GATHER: {BULK_FEATS_GATHER}")
    print(f"N_THREADS_GATHER: {N_THREADS_GATHER}")
    print(f"TOTAL_FEATS_PT: {TOTAL_FEATS_PT}")
    print(f"\nMemory Base Addresses:")
    print(f"  I_BASE: {hex(I_BASE)}")
    print(f"  QK_BASE: {hex(QK_BASE)}")
    print(f"  WV_BASE: {hex(WV_BASE)}")
    print(f"  OCTREE_NODES_BASE: {hex(OCTREE_NODES_BASE)}")
    print(f"  OCTREE_INDICES_BASE: {hex(OCTREE_INDICES_BASE)}")
    print(f"  BALL_RESULTS_BASE: {hex(BALL_RESULTS_BASE)}")
    print(f"\nMemory Limits:")
    print(f"  MAX_OCTREE_NODES: {MAX_OCTREE_NODES}")
    print(f"  MAX_OCTREE_INDICES: {MAX_OCTREE_INDICES}")
    print(f"  MAX_BALL_RESULTS: {MAX_BALL_RESULTS}")
    print("=============================================\n")

# Initialize configuration on import
if os.path.exists('config.json'):
    get_config('config.json')
else:
    print("No config.json found, using default ball query configuration values")

