#ifndef MINUET_TRACE_HPP
#define MINUET_TRACE_HPP

#include <vector>
#include <string>     // For std::string
#include <map>
#include <tuple>
#include <cstdint>    // For uint32_t, uint64_t
#include <thread>     // For std::thread
#include <mutex>      // For std::mutex
#include <iostream>   // For std::cout, std::ostream, std::hex in helpers
#include <iomanip>    // For std::setw, std::setfill in helpers
#include <sstream>    // For std::stringstream (used in to_hex_string definition and potentially useful for ostream ops)
#include <algorithm>  // For std::min, std::sort etc.
#include <functional> // For std::ref, std::cref
#include <type_traits> // For std::enable_if

// --- Global Memory Trace Setup ---

// Structure to hold a single memory access trace entry
struct MemoryAccessEntry {
    std::string phase;    // Current computational phase
    int thread_id;        // Virtual thread ID
    std::string op;       // Operation type ("R" or "W")
    std::string tensor;   // Tensor being accessed
    uint64_t addr;        // Memory address (stored as raw integer)
};

// Global vector to store all memory traces
extern std::vector<MemoryAccessEntry> gmem_trace;
// Global string to store the current phase of computation
extern std::string current_phase;
// Global debug flag
extern bool debug;

// Number of virtual threads (parameterizable)
const int NUM_THREADS = 4;

// Predefined mappings for phases, tensors, and operations (primarily for reference or potential future use)
extern const std::map<std::string, int> phases_map_global;
extern const std::map<std::string, int> tensors_map_global;
extern const std::map<std::string, int> ops_map_global;

// Number of input tiles and pivots for speeding up backward search in lookup
const int I_TILES = 2;

// --- Tensor Regions: Base addresses ---
const uint64_t I_BASE    = 0x10000000;
const uint64_t QK_BASE   = 0x20000000;
const uint64_t QI_BASE   = 0x30000000;
const uint64_t QO_BASE   = 0x40000000;
const uint64_t PIV_BASE  = 0x50000000;
const uint64_t TILE_BASE = I_BASE;
const uint64_t KM_BASE   = 0x60000000;
const uint64_t WO_BASE   = 0x80000000;
const uint64_t IV_BASE   = 0x100000000;
const uint64_t WV_BASE   = 0x800000000;

// --- Data Sizes (in bytes) ---
const int SIZE_KEY    = 4;
const int SIZE_INT    = 4;
const int SIZE_WEIGHT = 4;

// --- Function Declarations (Non-template) ---

std::string address_to_tensor(uint64_t addr);
void write_gmem_trace(const std::string& filename);
void record_access(int thread_id, const std::string& op, uint64_t addr);
uint64_t pack32(int c1, int c2, int c3);
std::tuple<int, int, int> unpack32(uint64_t key);
std::tuple<int, int, int> unpack32s(uint64_t key);
std::vector<uint64_t> radix_sort_with_memtrace(std::vector<uint64_t>& arr, uint64_t base);
std::vector<uint64_t> compute_unique_sorted(const std::vector<std::tuple<int, int, int>>& coords, int stride);

struct QueryData {
    std::vector<uint64_t> qkeys;
    std::vector<uint32_t> qii;
    std::vector<uint32_t> qki;
    std::vector<uint64_t> woffs;
};
QueryData build_queries(const std::vector<uint64_t>& uniq_in, int stride, const std::vector<std::tuple<int, int, int>>& offsets);

struct TilesPivots {
    std::vector<std::vector<uint64_t>> tiles;
    std::vector<uint64_t> pivots;
};
TilesPivots make_tiles_and_pivots(const std::vector<uint64_t>& uniq_in, int tile_size);

using KernelMapEntry = std::tuple<std::tuple<int, int, int>, std::tuple<int, int, int>, std::tuple<int, int, int>>;
using KernelMap = std::map<int, std::vector<KernelMapEntry>>;

KernelMap lookup_phase(
    const std::vector<uint64_t>& uniq,
    const std::vector<uint64_t>& qkeys,
    const std::vector<uint32_t>& qii,
    const std::vector<uint32_t>& qki,
    const std::vector<uint64_t>& woffs,
    const std::vector<std::vector<uint64_t>>& tiles,
    const std::vector<uint64_t>& pivots,
    int tile_size
);

std::string to_hex_string(uint64_t val);

// --- Template Function Definitions for Tuple Printing ---

// Forward declaration for the templated operator<< for std::tuple.
// This is crucial for allowing print_tuple_elements to call it for nested tuples.
template<typename... Args>
std::ostream& operator<<(std::ostream& os, const std::tuple<Args...>& t);

// Helper to print individual elements of a tuple
// Base case for recursion: when I (current index) reaches the size of the tuple
template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
print_tuple_elements(std::ostream& /*os*/, const std::tuple<Tp...>& /*t*/) {
    // End of recursion, do nothing
}

// Recursive step: print the Ith element and recurse for I+1
template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if < I < sizeof...(Tp), void>::type
print_tuple_elements(std::ostream& os, const std::tuple<Tp...>& t) {
    if (I > 0) {
        os << ", "; // Add separator for elements after the first
    }
    // This call will now correctly find the forward-declared operator<<
    // when std::get<I>(t) is itself a std::tuple.
    os << std::get<I>(t);  // Print current element
    print_tuple_elements<I + 1, Tp...>(os, t); // Recursive call for the next element
}

// Definition of operator<< for std::tuple
template<typename... Args>
inline std::ostream& operator<<(std::ostream& os, const std::tuple<Args...>& t) {
    os << "(";
    print_tuple_elements(os, t); // Use helper to print all elements
    os << ")";
    return os;
}

#endif // MINUET_TRACE_HPP
