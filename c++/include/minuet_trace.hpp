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

// --- Forward declaration for tuple printing ---
template<typename... Args>
std::ostream& operator<<(std::ostream& os, const std::tuple<Args...>& t);

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
    int orig_idx;

    IndexedCoord(Coord3D c = Coord3D(), int idx = -1) : coord(c), orig_idx(idx) {}

    /**
     * @brief Converts the internal coordinate to a packed 30-bit key.
     */
    uint32_t to_key() const {
        return coord.to_key();
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
extern std::string curr_phase;                   // Updated name
extern bool debug;

const int NUM_THREADS = 4;

extern const std::map<std::string, int> phases_map_global; // Renamed from phases
extern const std::map<std::string, int> tensors_map_global; // Renamed from tensors
extern const std::map<std::string, int> ops_map_global;     // Renamed from ops

const int I_TILES = 2;

// --- Tensor Regions ---
const uint64_t I_BASE    = 0x10000000;
const uint64_t TILE_BASE = I_BASE;
const uint64_t QK_BASE   = 0x20000000;
const uint64_t QI_BASE   = 0x30000000;
const uint64_t QO_BASE   = 0x40000000;
const uint64_t PIV_BASE  = 0x50000000;
const uint64_t KM_BASE   = 0x60000000;
const uint64_t WO_BASE   = 0x80000000;
const uint64_t IV_BASE   = 0x100000000;
const uint64_t WV_BASE   = 0x800000000;

// --- Data Sizes ---
const int SIZE_KEY    = 4;
const int SIZE_INT    = 4;
const int SIZE_WEIGHT = 4;

// --- Function Declarations ---

// Coordinate packing/unpacking (10-bit fields)
uint32_t pack32(int c1, int c2, int c3);
std::tuple<int, int, int> unpack32(uint32_t key);
std::tuple<int, int, int> unpack32s(uint32_t key);

std::string addr_to_tensor(uint64_t addr); // Renamed
void write_gmem_trace(const std::string& filename);
void record_access(int thread_id, const std::string& op, uint64_t addr);

std::vector<uint32_t> radix_sort_with_memtrace(std::vector<uint32_t>& arr, uint64_t base);

// Updated function signatures
std::vector<IndexedCoord> compute_unique_sorted_coords(
    const std::vector<std::tuple<int, int, int>>& in_coords, // Changed from Coord3D for initial input
    int stride);

struct BuildQueriesResult {
    std::vector<IndexedCoord> qry_keys;
    std::vector<int> qry_in_idx;
    std::vector<int> qry_off_idx;
    std::vector<uint32_t> wt_offsets;
};
BuildQueriesResult build_coordinate_queries(
    const std::vector<IndexedCoord>& uniq_coords,
    int stride,
    const std::vector<std::tuple<int, int, int>>& off_coords); // Changed from Coord3D for offsets

struct TilesPivotsResult {
    std::vector<std::vector<IndexedCoord>> tiles;
    std::vector<uint32_t> pivots;
};
TilesPivotsResult create_tiles_and_pivots(
    const std::vector<IndexedCoord>& uniq_coords,
    int tile_size);

// KernelMap stores: offset_idx -> list of [ ((target_coord_tuple, target_orig_idx), (source_coord_tuple, source_orig_idx)) ]
using KernelMapMatch = std::pair<std::pair<std::tuple<int, int, int>, int>, std::pair<std::tuple<int, int, int>, int>>;
using KernelMap = std::map<int, std::vector<KernelMapMatch>>;

KernelMap perform_coordinate_lookup(
    const std::vector<IndexedCoord>& uniq_coords,
    const std::vector<IndexedCoord>& qry_keys,
    const std::vector<int>& qry_in_idx,
    const std::vector<int>& qry_off_idx,
    const std::vector<uint32_t>& wt_offsets,
    const std::vector<std::vector<IndexedCoord>>& tiles,
    const std::vector<uint32_t>& pivs,
    int tile_size);

void write_kernel_map_to_gz(
    const KernelMap& kmap_data,
    const std::string& filename,
    const std::vector<std::tuple<int, int, int>>& off_list);


std::string to_hex_string(uint64_t val);

// --- Template Function Definitions for Tuple Printing ---
template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
print_tuple_elements(std::ostream& /*os*/, const std::tuple<Tp...>& /*t*/) {}

template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if < I < sizeof...(Tp), void>::type
print_tuple_elements(std::ostream& os, const std::tuple<Tp...>& t) {
    if (I > 0) {
        os << ", ";
    }
    os << std::get<I>(t);
    print_tuple_elements<I + 1, Tp...>(os, t);
}

template<typename... Args>
inline std::ostream& operator<<(std::ostream& os, const std::tuple<Args...>& t) {
    os << "(";
    print_tuple_elements(os, t);
    os << ")";
    return os;
}

// Operator to print KernelMapMatch for debugging
inline std::ostream& operator<<(std::ostream& os, const KernelMapMatch& match) {
    os << "((" << match.first.first << ", " << match.first.second << "), ("
       << match.second.first << ", " << match.second.second << "))";
    return os;
}


#endif // MINUET_TRACE_HPP
