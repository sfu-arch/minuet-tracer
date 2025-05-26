#ifndef MINUET_TRACE_HPP
#define MINUET_TRACE_HPP

#include <vector>
#include <string>
#include <map>
#include <tuple>
#include <cstdint>
#include <thread>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <functional>
#include <type_traits> // For std::enable_if
#include <cstring>
#include "minuet_config.hpp" // Include the new config header
// --- Forward declaration for tuple printing ---


template<typename... Args>
std::ostream& operator<<(std::ostream& os, const std::tuple<Args...>& t);

// --- Bidirectional map helper ---
template <typename K, typename V>
class bidict {
public:
    std::map<K, V> forward;
    std::map<V, K> inverse;

    bidict(const std::initializer_list<std::pair<const K, V>>& init_list) {
        for (const auto& pair : init_list) {
            forward[pair.first] = pair.second;
            inverse[pair.second] = pair.first;
        }
    }

    const V& at_key(const K& key) const {
        return forward.at(key);
    }

    const K& at_val(const V& val) const {
        return inverse.at(val);
    }

    V& operator[](const K& key) {
        // This will insert if key doesn't exist, then we need to update inverse.
        // For simplicity, assume keys are pre-populated or use .at_key for lookup.
        // Proper handling of [] for insertion would require more logic.
        return forward[key];
    }
};


// --- Global Constants (from minuet_config.py and minuet_mapping.py) ---
// These are now part of the g_config object and loaded from JSON
// const int NUM_THREADS = 4;
// const int SIZE_KEY    = 4;
// const int SIZE_INT    = 4;
// const int SIZE_WEIGHT = 4;

// // Tensor Regions
// const uint64_t I_BASE    = 0x10000000;
// const uint64_t TILE_BASE = I_BASE; // Alias
// const uint64_t QK_BASE   = 0x20000000;
// const uint64_t QI_BASE   = 0x30000000;
// const uint64_t QO_BASE   = 0x40000000;
// const uint64_t PIV_BASE  = 0x50000000;
// const uint64_t KM_BASE   = 0x60000000;
// const uint64_t WO_BASE   = 0x80000000;
// const uint64_t IV_BASE   = 0x100000000; // Feature vectors (64-bit)
// const uint64_t WV_BASE   = 0xF00000000; // Weight values (64-bit)


// // GEMM Parameters (used in gather, but defined in config)
// const int GEMM_ALIGNMENT = 4;
// const int GEMM_WT_GROUP = 2;
// const int GEMM_SIZE = 4;

// // GATHER PARAMETERS (used in gather, but defined in config)
// const int NUM_TILES = 2;
// const int TILE_FEATS = 16;
// const int BULK_FEATS = 4;
// const int N_THREADS_GATHER = 1; // Renamed from N_THREADS to avoid conflict
// const int TOTAL_FEATS_PT = NUM_TILES * TILE_FEATS;


// PHASES, TENSORS, OPS (from minuet_mapping.py)
// These will be extern bidict<std::string, int> defined in minuet_trace.cpp
extern bidict<std::string, int> PHASES;
extern bidict<std::string, int> TENSORS;
extern bidict<std::string, int> OPS;

// Helper function to convert value to hex string
std::string to_hex_string(uint64_t val);


// --- Data Structures ---

/**
 * @brief Represents a 3D coordinate.
 */
struct Coord3D {
    int x, y, z;

    Coord3D(int x_ = 0, int y_ = 0, int z_ = 0) : x(x_), y(y_), z(z_) {}

    /**
     * @brief Returns quantized coordinates based on stride.
     */
    Coord3D quantized(int stride) const {
        if (stride == 0) return *this; // Avoid division by zero
        return Coord3D(x / stride, y / stride, z / stride);
    }

    /**
     * @brief Converts coordinate to packed 30-bit key.
     * Uses the global pack32 function.
     */
    uint32_t to_key() const; // Definition in .cpp due to pack32 dependency

    /**
     * @brief Creates coordinate from packed key using signed unpacking.
     */
    static Coord3D from_key(uint32_t key); // Definition in .cpp
    static Coord3D from_signed_key(uint32_t key); // Added

    /**
     * @brief Adds two coordinates.
     */
    Coord3D operator+(const Coord3D& other) const {
        return Coord3D(x + other.x, y + other.y, z + other.z);
    }

    // For printing Coord3D objects
    friend std::ostream& operator<<(std::ostream& os, const Coord3D& c) {
        os << "(" << c.x << ", " << c.y << ", " << c.z << ")";
        return os;
    }
};

/**
 * @brief Represents a coordinate with an associated original index.
 */
struct IndexedCoord {
    Coord3D coord;
    int orig_idx; // Corresponds to original_index_from_input in Python
    uint32_t key_val; // Store the packed key for direct use

    IndexedCoord(Coord3D c = Coord3D(), int idx = -1) : coord(c), orig_idx(idx) {
        key_val = coord.to_key();
    }
    IndexedCoord(uint32_t k, int idx = -1) : key_val(k), orig_idx(idx) {
        coord = Coord3D::from_key(k); // Or from_signed_key if appropriate
    }


    /**
     * @brief Converts the internal coordinate to a packed 30-bit key.
     */
    uint32_t to_key() const {
        return key_val; // Use stored key
    }

    /**
     * @brief Creates an IndexedCoord from a packed key and an index.
     * Uses Coord3D::from_key for coordinate creation.
     */
    static IndexedCoord from_key_and_index(uint32_t key, int idx) {
        return IndexedCoord(Coord3D::from_key(key), idx);
    }
};


// --- Global Memory Trace Setup ---
struct MemoryAccessEntry {
    std::string phase;
    int thread_id;
    std::string op;
    std::string tensor;
    uint64_t addr;
};

extern std::vector<MemoryAccessEntry> mem_trace; // Updated name
extern std::string curr_phase; // Added
extern bool debug;             // Added
extern const std::string output_dir; // Added

// --- Getter/Setter for global state and mem_trace management ---
std::vector<MemoryAccessEntry> get_mem_trace();
void clear_mem_trace();
void set_curr_phase(const std::string& phase_name);
std::string get_curr_phase();
void set_debug_flag(bool debug_val);
bool get_debug_flag();

// --- Function Declarations (matching Python functions) ---

// Helper: pack/unpack (already in .cpp, ensure signatures match if used directly)
uint32_t pack32(int c1, int c2, int c3);
std::tuple<int, int, int> unpack32(uint32_t key);
std::tuple<int, int, int> unpack32s(uint32_t key); // For signed unpacking

// Memory tracing
std::string addr_to_tensor(uint64_t addr);
void record_access(int thread_id, const std::string& op, uint64_t addr); // op is "R" or "W"
void write_gmem_trace(const std::string& filename);

// Algorithm phases
std::vector<uint32_t> radix_sort_with_memtrace(std::vector<uint32_t>& arr, uint64_t base_addr);

std::vector<IndexedCoord> compute_unique_sorted_coords(
    const std::vector<Coord3D>& in_coords, // Changed from tuples
    int stride
);

// Structure to hold results from build_coordinate_queries
struct BuildQueriesResult {
    std::vector<IndexedCoord> qry_keys; // Stores IndexedCoord(Coord3D_obj, original_index_from_input)
    std::vector<int> qry_in_idx;
    std::vector<int> qry_off_idx;
    std::vector<Coord3D> wt_offsets; // Stores Coord3D for weight offsets
};

BuildQueriesResult build_coordinate_queries(
    const std::vector<IndexedCoord>& uniq_coords,
    int stride, // stride is not used in python version, but kept for signature
    const std::vector<Coord3D>& off_coords // Changed from tuples
);

// Structure to hold results from create_tiles_and_pivots
struct TilesPivotsResult {
    std::vector<std::vector<IndexedCoord>> tiles;
    std::vector<IndexedCoord> pivots; // Python uses 'pivs'
};

TilesPivotsResult create_tiles_and_pivots(
    const std::vector<IndexedCoord>& uniq_coords,
    int tile_size
);

// Kernel Map type (matches Python's kmap structure)
// Key: offset_key (packed Coord3D of the offset)
// Value: list of pairs (input_idx from uniq_coords, query_src_orig_idx from qry_keys)
using KernelMap = std::map<uint32_t, std::vector<std::pair<int, int>>>;

KernelMap perform_coordinate_lookup( // Renamed from lookup
    const std::vector<IndexedCoord>& uniq_coords,
    const std::vector<IndexedCoord>& qry_keys,
    const std::vector<int>& qry_in_idx,
    const std::vector<int>& qry_off_idx,
    const std::vector<Coord3D>& wt_offsets, // Python uses this name
    const std::vector<std::vector<IndexedCoord>>& tiles,
    const std::vector<IndexedCoord>& pivs, // Python uses 'pivs'
    int tile_size
);

void write_kernel_map_to_gz(
    const KernelMap& kmap_data,
    const std::string& filename,
    const std::vector<Coord3D>& off_list // List of offset coordinates
);


// For printing tuples (common use in debug)
template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
print_tuple_elements(std::ostream&, const std::tuple<Tp...>&) {} // Base case

template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type
print_tuple_elements(std::ostream& os, const std::tuple<Tp...>& t) {
    if (I > 0) os << ", ";
    os << std::get<I>(t);
    print_tuple_elements<I + 1, Tp...>(os, t); // Recursive call
}

template<typename... Args>
std::ostream& operator<<(std::ostream& os, const std::tuple<Args...>& t) {
    os << "(";
    print_tuple_elements(os, t);
    os << ")";
    return os;
}


#endif // MINUET_TRACE_HPP
