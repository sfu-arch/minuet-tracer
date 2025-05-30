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
#include "coord.hpp"         // Include the new coord header
#include "sorted_map.hpp"    // Include the new sorted_map header
#include "trace.hpp"

// PHASES, TENSORS, OPS (from minuet_mapping.py)
// These will be extern bidict<std::string, int> defined in minuet_trace.cpp
extern bidict<std::string, int> PHASES;
extern bidict<std::string, int> TENSORS;
extern bidict<std::string, int> OPS;


struct BuildQueriesResult {
    std::vector<IndexedCoord> qry_keys; // Vector of IndexedCoord (coord, original_source_idx)
    std::vector<int> qry_in_idx;        // Index into unique_coords
    std::vector<int> qry_off_idx;       // Index into offset_coords
    std::vector<Coord3D> wt_offsets;    // The actual Coord3D offset used for the query
};

struct TilesPivotsResult {
    std::vector<std::vector<IndexedCoord>> tiles;
    std::vector<IndexedCoord> pivots;
};

// Using KernelMapType from sorted_map.hpp
// using KernelMapType = SortedByValueSizeMap<uint32_t, std::vector<std::pair<int, int>>>;
// This is defined in sorted_map.hpp which is included.

struct PerformLookupResult {
    // This is essentially the KernelMapType itself, but let's be explicit if Python side expects a struct
    // For now, let's assume perform_coordinate_lookup directly returns KernelMapType
    // If it needs to be wrapped:
    // KernelMapType kernel_map;
    // std::vector<uint32_t> offsets_active; // If C++ side also determines this
    // std::vector<int> slot_array;          // If C++ side also determines this
};


// --- Global Memory Trace Setup ---
// extern std::vector<MemoryAccessEntry> mem_trace; // Defined in .cpp
// extern std::string curr_phase; // Defined in .cpp
// extern bool debug; // Part of g_config
// extern const std::string output_dir; // Part of g_config

// --- Getter/Setter for global state and mem_trace management ---
std::vector<MemoryAccessEntry> get_mem_trace();
void clear_mem_trace();
void set_curr_phase(const std::string& phase_name);
std::string get_curr_phase();
void set_debug_flag(bool debug_val);
bool get_debug_flag();
void clear_global_mem_trace(); // Added to clear the global memory trace

// --- Function Declarations (matching Python functions) ---

// Helper: pack/unpack (already in .cpp, ensure signatures match if used directly)
// uint32_t pack32(int c1, int c2, int c3);
// std::tuple<int, int, int> unpack32(uint32_t key);
// std::tuple<int, int, int> unpack32s(uint32_t key); // For signed unpacking

// Memory tracing
uint8_t addr_to_tensor(uint64_t addr);
std::string addr_to_tensor_str(uint64_t addr);

// Function to write the global memory trace to a gzipped file
// Returns a CRC32 checksum of the written data.
uint32_t write_gmem_trace(const std::string &filename, int sizeof_addr = 4); // Added sizeof_addr parameter with default 4

void record_access(int thread_id, const std::string &op_str, uint64_t addr);

// --- Algorithm Phases ---
std::vector<uint32_t> radix_sort_with_memtrace(std::vector<uint32_t>& arr, uint64_t base_addr);

std::vector<IndexedCoord> compute_unique_sorted_coords(
    const std::vector<Coord3D>& in_coords, // Changed from tuples
    int stride
);

BuildQueriesResult build_coordinate_queries(
    const std::vector<IndexedCoord>& uniq_coords,
    int stride, 
    const std::vector<Coord3D>& off_coords 
);

// Structure to hold results from create_tiles_and_pivots
TilesPivotsResult create_tiles_and_pivots(
    const std::vector<IndexedCoord>& uniq_coords,
    int tile_size_param // Renamed to avoid conflict with config
);

// Kernel Map type (matches Python's kmap structure)
// Key: offset_key (packed Coord3D of the offset)
// Value: list of pairs (input_idx from uniq_coords, query_src_orig_idx from qry_keys)
// using KernelMap = std::map<uint32_t, std::vector<std::pair<int, int>>>;
// Replace std::map with SortedByValueSizeMap for KernelMap
// The key is the offset (uint32_t), and the value is a vector of pairs (matches).
// We want to sort by the size of this vector (number of matches).
// True for ascending (shortest first), False for descending (longest first).
// Python's SortedByValueLengthDict defaults to ascending=True, but for kmap processing
// in minuet_trace.py, it seems to imply a descending sort for offsets_active (longest match list first).
// Let's assume descending for now, as it's common for processing more significant items first.
using KernelMapType = SortedByValueSizeMap<uint32_t, std::vector<std::pair<int, int>>>;


KernelMapType perform_coordinate_lookup( // Renamed from lookup
    const std::vector<IndexedCoord>& uniq_coords,
    const std::vector<IndexedCoord>& qry_keys,
    const std::vector<int>& qry_in_idx,
    const std::vector<int>& qry_off_idx,
    const std::vector<Coord3D>& wt_offsets, // Python uses this name
    const std::vector<std::vector<IndexedCoord>>& tiles,
    const std::vector<IndexedCoord>& pivs, // Python uses 'pivs'
    int tile_size
);

uint32_t write_kernel_map_to_gz(
    const KernelMapType& kernel_map,
    const std::string& filename,
    const std::vector<Coord3D>& offset_coords
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
