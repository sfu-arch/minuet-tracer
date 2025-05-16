#include "minuet_trace.hpp" // Header file with declarations
#include <fstream>           // For file operations (though zlib handles file directly)
#include <vector>
#include <string>
#include <map>
#include <algorithm>         // For std::sort, std::unique, std::min
#include <stdexcept>         // For std::runtime_error
#include <zlib.h>            // For gzip compression (gzFile, gzopen, gzwrite, gzclose)
#include <cstring>           // For memcpy in write_gmem_trace (not strictly needed with direct writes)
#include <iostream>          // For std::cout, std::cerr
#include <iomanip>           // For std::hex, std::setw, std::setfill

// --- Global Variable Definitions ---
std::vector<MemoryAccessEntry> gmem_trace; // Stores all memory access records
std::string current_phase = "";            // Current computational phase, e.g., "Radix-Sort"
bool debug = true;                         // Flag to enable/disable debug prints

// Definition of global constant maps (can be used for encoding strings to integers if needed elsewhere)
const std::map<std::string, int> phases_map_global = {
    {"Radix-Sort", 0}, {"Build-Queries", 1}, {"Sort-QKeys", 2},
    {"Tile-Pivots", 3}, {"Lookup", 4}, {"Lookup-Backward", 5},
    {"Lookup-Forward", 6}, {"Dedup", 7} // "Dedup" was used as a phase in Python
};

const std::map<std::string, int> tensors_map_global = {
    {"I", 0}, {"QK", 1}, {"QI", 2}, {"QO", 3}, {"PIV", 4},
    {"KM", 5}, {"WO", 6}, {"TILE", 7}, {"Unknown", 8}, // Added "Unknown" for addresses not matching known regions
    {"IV", 9}, {"WV", 10} // Added IV and WV from base addresses
};

const std::map<std::string, int> ops_map_global = {
    {"R", 0}, {"W", 1}
};

// Helper to convert uint64_t to hex string for printing (similar to Python's hex())
std::string to_hex_string(uint64_t val) {
    std::stringstream ss;
    ss << "0x" << std::hex << val;
    return ss.str();
}

// --- Function Implementations ---

std::string address_to_tensor(uint64_t addr) {
    // This function determines the tensor name based on the memory address.
    // The logic follows the Python script's cascading if/elif statements.
    // Note: The Python version had a potentially problematic condition for 'PIV'
    // (addr >= PIV_BASE and addr < TILE_BASE) because TILE_BASE (0x1...) is less than PIV_BASE (0x5...).
    // This means that specific PIV condition `addr < TILE_BASE` would make the PIV range empty.
    // The C++ code below replicates the Python's effective logic flow.

    if (addr >= I_BASE && addr < QK_BASE) {         // Range for Input tensor
        return "I";
    } else if (addr >= QK_BASE && addr < QI_BASE) { // Range for Query Keys tensor
        return "QK";
    } else if (addr >= QI_BASE && addr < QO_BASE) { // Range for Query Input-index tensor
        return "QI";
    } else if (addr >= QO_BASE && addr < PIV_BASE) { // Range for Query Offset-index tensor
        return "QO";
    } else if (addr >= PIV_BASE && addr < KM_BASE) { // Range for Tile tensor
        return "PIV";
    } else if (addr >= KM_BASE && addr < WO_BASE) {   // Range for Kernel Map tensor
        return "KM";
    } else if (addr >= WO_BASE && addr < IV_BASE) {   // Range for Weight Offsets tensor (Python used WV_BASE as upper bound)
        return "WO";
    } else if (addr >= IV_BASE && addr < WV_BASE) {   // Range for Input Feature Vectors
        return "IV";
    } else if (addr >= WV_BASE) {                     // Range for Weight Values
        return "WV";
    } else {
        return "Unknown"; // Address does not fall into any known tensor region
    }
}

void write_gmem_trace(const std::string& filename) {
    // This function writes the recorded memory traces to a gzipped binary file.
    // It first creates a local mapping of phase names to integer IDs for compression.
    std::map<std::string, uint8_t> local_phases_map; // Map phase string to a compact ID
    // Vector to store the trace data in a compressed numerical format
    std::vector<std::tuple<uint8_t, uint8_t, uint8_t, uint8_t, uint32_t>> compressed_trace_data;

    for (const auto& entry : gmem_trace) {
        uint8_t phase_id;
        // Assign a new integer ID to phases encountered for the first time
        if (local_phases_map.find(entry.phase) == local_phases_map.end()) {
            local_phases_map[entry.phase] = static_cast<uint8_t>(local_phases_map.size());
        }
        phase_id = local_phases_map.at(entry.phase);

        // Map operation string ("R"/"W") to integer ID using global map
        uint8_t op_id = ops_map_global.at(entry.op);
        // Map tensor name string to integer ID using global map
        uint8_t tensor_id = tensors_map_global.at(entry.tensor);
        
        // Convert address to uint32_t for packing (Python's 'I' format for struct.pack)
        // This implies potential truncation if addresses exceed 32-bit range, matching Python script.
        uint32_t addr_int = static_cast<uint32_t>(entry.addr);

        // Add the compressed entry (all integers) to the temporary vector
        compressed_trace_data.emplace_back(phase_id, static_cast<uint8_t>(entry.thread_id), op_id, tensor_id, addr_int);
    }

    // Open the output file in binary write mode using zlib for gzip compression
    gzFile outFile = gzopen(filename.c_str(), "wb");
    if (!outFile) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    // Write the total number of compressed entries as a header
    uint32_t num_entries = static_cast<uint32_t>(compressed_trace_data.size());
    if (gzwrite(outFile, &num_entries, sizeof(num_entries)) != sizeof(num_entries)) {
        gzclose(outFile);
        throw std::runtime_error("Failed to write number of entries to gzip file.");
    }
    

    // Write each compressed entry to the file
    // Format is BBBBI (byte, byte, byte, byte, unsigned int)
    for (const auto& centry : compressed_trace_data) {
        uint8_t phase_val = std::get<0>(centry);
        uint8_t tid_val = std::get<1>(centry);
        uint8_t op_val = std::get<2>(centry);
        uint8_t tensor_val = std::get<3>(centry);
        uint32_t addr_val = std::get<4>(centry);

        // Write each field of the compressed entry
        if (gzwrite(outFile, &phase_val, sizeof(phase_val)) != sizeof(phase_val) ||
            gzwrite(outFile, &tid_val, sizeof(tid_val)) != sizeof(tid_val) ||
            gzwrite(outFile, &op_val, sizeof(op_val)) != sizeof(op_val) ||
            gzwrite(outFile, &tensor_val, sizeof(tensor_val)) != sizeof(tensor_val) ||
            gzwrite(outFile, &addr_val, sizeof(addr_val)) != sizeof(addr_val)) {
            gzclose(outFile);
            throw std::runtime_error("Failed to write entry to gzip file.");
        }
    }

    // Close the gzipped file
    gzclose(outFile);

    // Print summary information
    std::cout << "Memory trace written to " << filename << std::endl;
    std::cout << "Compressed " << gmem_trace.size() << " entries" << std::endl;
    std::cout << "Phase mapping (local to this write): {";
    bool first_phase = true;
    for (const auto& pair : local_phases_map) {
        if (!first_phase) std::cout << ", ";
        std::cout << "\"" << pair.first << "\": " << static_cast<int>(pair.second);
        first_phase = false;
    }
    std::cout << "}" << std::endl;
}

void record_access(int thread_id, const std::string& op, uint64_t addr) {
    // Determines tensor name from address
    std::string tensor_name = address_to_tensor(addr);
    // Creates a trace entry and adds it to the global trace vector
    gmem_trace.push_back({current_phase, thread_id, op, tensor_name, addr});

    // Optional debug print for each recorded access (matches Python's commented-out print)
    // if (debug) {
    //     std::cout << "[" << current_phase << "] Thread" << thread_id
    //               << " " << op << " " << tensor_name << "@" << to_hex_string(addr) << std::endl;
    // }
}

uint64_t pack32(int c1, int c2, int c3) {
    // Packs three integer coordinates into a single 64-bit unsigned integer.
    // Python's pack32: key = 0; for c in coords: key = (key << 12) | (c & 0xFFF)
    // For (c1, c2, c3):
    // 1. key = (0 << 12) | (c1 & 0xFFF)  => c1 effectively
    // 2. key = (key << 12) | (c2 & 0xFFF) => (c1 << 12) | c2
    // 3. key = (key << 12) | (c3 & 0xFFF) => ((c1 << 12) | c2) << 12 | c3
    // This means c1 is in the most significant 12 bits (bits 24-35 conceptually),
    // c2 in the middle (bits 12-23), and c3 in the least significant (bits 0-11).
    // This requires a 36-bit number if all 12 bits are used per coord.
    // Since the return type is uint64_t, the result will not be truncated.
    uint64_t key = 0;
    key = (key << 12) | (static_cast<uint64_t>(c1) & 0xFFF); // c1 is now in bits 0-11
    key = (key << 12) | (static_cast<uint64_t>(c2) & 0xFFF); // c1 in bits 12-23, c2 in 0-11
    key = (key << 12) | (static_cast<uint64_t>(c3) & 0xFFF); // c1 in bits 24-31 (upper 8 bits of its 12), c2 in 12-23, c3 in 0-11
    return key; // The result is implicitly truncated to 32 bits.
}

std::tuple<int, int, int> unpack32(uint64_t key) {
    // Unpacks a 32-bit key into three coordinates (x, y, z).
    // This is the reverse of the pack32 logic assuming truncation.
    // c3 was packed last, so it's in the LSBs.
    // c1 was packed first, so it's in the MSBs of the 36-bit conceptual value.
    // Python: z = key & 0xFFF; key >>= 12; y = key & 0xFFF; key >>= 12; x = key & 0xFFF;
    // This implies z is LSB.
    int z_coord = static_cast<int>(key & 0xFFF);
    key >>= 12;
    int y_coord = static_cast<int>(key & 0xFFF);
    key >>= 12;
    int x_coord = static_cast<int>(key & 0xFFF); // This will get bits 24-31 of original key, masked by 0xFFF.
                                               // If original key was 0xABCDEF, x gets 0x0AB.
    return std::make_tuple(x_coord, y_coord, z_coord);
}

std::tuple<int, int, int> unpack32s(uint64_t key) {
    // Unpacks a 32-bit key into three signed coordinates, performing sign extension from 12 bits.
    // A 12-bit number is negative if its 11th bit (0-indexed) is 1.
    // 0xFFF is 12 bits. Max positive is 2047 (0x7FF). Min negative is -2048 (0x800).
    // If value is >= 2048 (0x800), it's negative in 12-bit 2's complement. Subtract 4096 (0x1000).

    uint64_t temp_key = key; // Use a temporary variable for modification

    int z_val = static_cast<int>(temp_key & 0xFFF);
    z_val = (z_val < 2048) ? z_val : z_val - 4096;
    temp_key >>= 12;

    int y_val = static_cast<int>(temp_key & 0xFFF);
    y_val = (y_val < 2048) ? y_val : y_val - 4096;
    temp_key >>= 12;

    int x_val = static_cast<int>(temp_key & 0xFFF);
    x_val = (x_val < 2048) ? x_val : x_val - 4096;

    return std::make_tuple(x_val, y_val, z_val);
}

std::vector<uint64_t> radix_sort_with_memtrace(std::vector<uint64_t>& arr, uint64_t base) {
    // Simulates memory accesses of a radix sort. It does not fully sort the array but mimics
    // the read/write patterns of a simplified radix sort pass.
    const int mask = 0xFF; // Mask for extracting a byte
    const int passes = 4;  // For 32-bit keys (4 bytes)
    size_t N = arr.size();
    if (N == 0) return arr; // Handle empty array
    std::vector<uint64_t> aux(N); // Auxiliary array for simulated data movement

    for (int p = 0; p < passes; ++p) {
        // Phase: count (simulated reads)
        for (size_t i = 0; i < N; ++i) {
            int t_id = static_cast<int>(i % NUM_THREADS); // Cycle thread IDs
            record_access(t_id, "R", base + i * SIZE_KEY); // Record read access
            // The actual value extraction for sorting logic (not used further in this simulation)
            [[maybe_unused]] uint64_t byte_val = (arr[i] >> (p * 8)) & mask;
        }

        // Phase: scatter (simulated reads and writes)
        for (size_t i = 0; i < N; ++i) {
            int t_id = static_cast<int>(i % NUM_THREADS);
            record_access(t_id, "R", base + i * SIZE_KEY); // Read from original position
            size_t pos = i;  // Simplified: assume stable mapping (element goes to same relative position)
            aux[pos] = arr[i]; // Simulate moving data to auxiliary array
            record_access(t_id, "W", base + pos * SIZE_KEY); // Write to new position (in aux, conceptually)
        }
        arr.swap(aux); // Swap buffers: aux becomes arr for the next pass (or final result)
    }
    return arr; // Return the array after simulated passes
}

std::vector<uint64_t> compute_unique_sorted(const std::vector<std::tuple<int, int, int>>& coords, int stride) {
    std::vector<uint64_t> keys;
    keys.reserve(coords.size());

    // Pack coordinates into keys
    for (size_t idx = 0; idx < coords.size(); ++idx) {
        // int t_id = (idx % NUM_THREADS) + 1; // Python used 1-based, C++ often 0-based for array indices
        auto [x, y, z] = coords[idx];
        // Quantize coordinates using the stride
        int qx = x / stride;
        int qy = y / stride;
        int qz = z / stride;
        keys.push_back(pack32(qx, qy, qz)); // Pack quantized coordinates
        // Python's record_access here was commented out:
        // record_access("G",t_id, 'W', I_BASE + idx*SIZE_KEY) -> "G" is not a valid phase.
    }

    // Simulate radix sort on the packed keys (records memory accesses)
    // The Python version of radix_sort_with_memtrace doesn't actually sort perfectly,
    // it just simulates accesses.
    radix_sort_with_memtrace(keys, I_BASE);

    // Python then explicitly sorts: `sorted_keys = sorted(keys)`
    // To match this behavior, we sort the keys after the simulated radix sort.
    std::vector<uint64_t> sorted_keys = keys; // Copy keys (potentially modified by radix_sort_with_memtrace simulation)
    std::sort(sorted_keys.begin(), sorted_keys.end()); // Perform an actual sort

    // Deduplication
    current_phase = "Dedup"; // Set phase for any (commented out) accesses during deduplication
    std::vector<uint64_t> unique_keys;
    if (!sorted_keys.empty()) {
        unique_keys.push_back(sorted_keys[0]);
        // Python's record_access for first unique key write was commented out.
        // Example: record_access(0 % NUM_THREADS, "W", I_BASE + 0*SIZE_KEY);

        for (size_t idx = 1; idx < sorted_keys.size(); ++idx) {
            // int t_id = (idx % NUM_THREADS) + 1; // Python used 1-based
            // Python's record_access for reading from sorted_keys was commented out.
            // Example: record_access(t_id, "R", I_BASE + idx*SIZE_KEY);
            if (sorted_keys[idx] != unique_keys.back()) {
                unique_keys.push_back(sorted_keys[idx]);
                // Python's record_access for writing unique key was commented out.
                // Example: record_access(t_id, "W", I_BASE + (unique_keys.size()-1)*SIZE_KEY);
            }
        }
    }
    
    if (debug) {
        std::cout << "Sorted+Unique Keys:" << std::endl;
        for (const auto& k : unique_keys) {
            auto [px, py, pz] = unpack32s(k); // Unpack for printing
            std::cout << "key=" << to_hex_string(k) << ", coords=(" << px << "," << py << "," << pz << ")" << std::endl;
        }
    }
    return unique_keys;
}

QueryData build_queries(const std::vector<uint64_t>& uniq_in, int stride, const std::vector<std::tuple<int, int, int>>& offsets) {
    size_t M = uniq_in.size(); // Number of unique input keys
    size_t K = offsets.size(); // Number of offsets
    size_t total_queries = M * K;

    QueryData qd;
    qd.qkeys.resize(total_queries);
    qd.qii.resize(total_queries);
    qd.qki.resize(total_queries);
    qd.woffs.resize(total_queries);

    for (size_t k_idx = 0; k_idx < K; ++k_idx) { // Iterate over each offset
        auto [dx, dy, dz] = offsets[k_idx];
        // Quantize offsets
        int odx = dx / stride;
        int ody = dy / stride;
        int odz = dz / stride;
        // Pack the offset (Python used pack32(0, odx, ody, odz) - 4 args, pack32 takes 3. Assuming 0 is placeholder for a coord)
        // Let's assume it means pack32(odx, ody, odz) or similar. Python's *coords takes variable length.
        // If pack32(0, odx, ody, odz) was intended to use the first 3 of these 4, it's pack32(0, odx, ody).
        // The unpack32s for woffs implies it's a 3-coord key.
        uint64_t off_key = pack32(odx, ody, odz); // Assuming offset itself is packed like a coordinate.

        for (size_t i_idx = 0; i_idx < M; ++i_idx) { // Iterate over each unique input key
            size_t idx = k_idx * M + i_idx; // Linear index for the query
            // int thread_id = idx % NUM_THREADS; // Example if accesses were recorded

            uint64_t in_key = uniq_in[i_idx];
            auto [x_in, y_in, z_in] = unpack32s(in_key); // Unpack input key

            // Calculate new coordinates by applying offset
            int x_new = x_in + odx;
            int y_new = y_in + ody;
            int z_new = z_in + odz;
            
            qd.qkeys[idx] = pack32(x_new, y_new, z_new); // Pack new coordinates as query key
            // Python's record_access for QK_BASE was commented out.

            qd.qii[idx] = static_cast<uint32_t>(i_idx); // Store original input index
            // Python's record_access for QI_BASE was commented out.

            qd.qki[idx] = static_cast<uint32_t>(k_idx); // Store offset index (kernel index)
            // Python's record_access for QO_BASE was commented out.

            qd.woffs[idx] = off_key; // Store packed offset key
            // Python's record_access for WO_BASE was commented out.
        }
    }
    return qd;
}

TilesPivots make_tiles_and_pivots(const std::vector<uint64_t>& uniq_in, int tile_size) {
    TilesPivots result;
    if (tile_size <= 0) { // Basic validation for tile_size
        if (debug) std::cerr << "Warning: tile_size is non-positive in make_tiles_and_pivots." << std::endl;
        if (!uniq_in.empty()) { // If uniq_in is not empty, treat it as a single tile
             result.tiles.push_back(uniq_in);
             result.pivots.push_back(uniq_in[0]);
             record_access(0, "W", PIV_BASE + (result.pivots.size() - 1) * SIZE_KEY);
        }
        return result;
    }

    for (size_t start = 0; start < uniq_in.size(); start += tile_size) {
        std::vector<uint64_t> tile;
        size_t end = std::min(start + static_cast<size_t>(tile_size), uniq_in.size());
        for(size_t i = start; i < end; ++i) {
            tile.push_back(uniq_in[i]);
        }
        if (!tile.empty()) {
            result.tiles.push_back(tile);
            result.pivots.push_back(tile[0]); // First element of the tile is its pivot
            // Record write access for the pivot key
            record_access(0, "W", PIV_BASE + (result.pivots.size() - 1) * SIZE_KEY);
        }
    }
    return result;
}

// Thread-local storage for memory traces. Each thread in lookup_phase logs to its own vector.
thread_local std::vector<MemoryAccessEntry> thread_lmem_trace;

// Records a memory access into the current thread's local trace vector.
void record_access_local(int thread_id, const std::string& op, uint64_t addr) {
    std::string tensor = address_to_tensor(addr);
    // Note: current_phase is global, shared by all threads within lookup_phase.
    thread_lmem_trace.push_back({current_phase, thread_id, op, tensor, addr});
}

// Worker function executed by each thread in the lookup_phase.
void process_query_range(
    int q_start, int q_end, int thread_id, // Range of queries for this thread and its ID
    const std::vector<uint64_t>& uniq_keys_c, // Const ref to unique input keys
    const std::vector<uint64_t>& qkeys_c,     // Const ref to query keys
    const std::vector<uint32_t>& qii_c,       // Const ref to query input-indices
    const std::vector<uint32_t>& qki_c,       // Const ref to query kernel-indices
    const std::vector<uint64_t>& woffs_c,     // Const ref to weight-offset keys
    const std::vector<std::vector<uint64_t>>& tiles_c, // Const ref to data tiles
    const std::vector<uint64_t>& pivots_c,    // Const ref to pivot keys
    int tile_size_p,                          // Tile size parameter
    KernelMap& kernel_map_ref,                // Reference to the shared kernel map
    std::mutex& kernel_map_lock,              // Mutex for kernel_map_ref
    std::mutex& gmem_trace_lock) {            // Mutex for global gmem_trace

    thread_lmem_trace.clear(); // Clear any previous entries in thread-local storage

    for (int q = q_start; q < q_end; ++q) { // Process assigned range of queries
        record_access_local(thread_id, "R", QK_BASE + static_cast<uint64_t>(q) * SIZE_KEY); // Read query key
        uint64_t target = qkeys_c[q]; // The query key to search for

        // Backward search on pivots to find the relevant tile
        int lo = 0, hi = static_cast<int>(pivots_c.size()) - 1;
        int potential_tile_idx = 0; // Default to first tile if target is smaller than all pivots

        if (!pivots_c.empty()) { // Ensure pivots exist
            int found_pivot_idx = -1; 
            while(lo <= hi) {
                int mid = lo + (hi - lo) / 2; // Avoid overflow with (lo+hi)/2
                record_access_local(thread_id, "R", PIV_BASE + static_cast<uint64_t>(mid) * SIZE_KEY); // Read pivot
                if (pivots_c[mid] <= target) {
                    found_pivot_idx = mid; // This pivot (or an earlier one) could be the one
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            // Python logic: if hi_py < 0: hi_py = 0; tile_idx = hi_py;
            // found_pivot_idx is the index of the rightmost pivot <= target.
            // If all pivots > target, found_pivot_idx = -1. Tile should be 0.
            // If target >= all pivots, found_pivot_idx = pivots_c.size()-1. Tile should be pivots_c.size()-1.
            potential_tile_idx = (found_pivot_idx == -1) ? 0 : found_pivot_idx;
        }
        
        size_t tile_idx_final = static_cast<size_t>(potential_tile_idx);
        size_t base_off = tile_idx_final * static_cast<size_t>(tile_size_p); // Base offset for accessing tile elements

        // Forward scan within the identified tile
        if (tile_idx_final < tiles_c.size()) { // Check tile_idx is valid
            const auto& current_tile = tiles_c[tile_idx_final];
            for (size_t j = 0; j < current_tile.size(); ++j) {
                record_access_local(thread_id, "R", TILE_BASE + (base_off + j) * SIZE_KEY); // Read tile element
                if (current_tile[j] == target) { // If query key is found in tile
                    uint32_t i_idx_val = qii_c[q]; // Original input index for this query
                    uint32_t k_idx_val = qki_c[q]; // Kernel index (offset index) for this query

                    if (i_idx_val < uniq_keys_c.size()) { // Boundary check for uniq_keys_c
                        KernelMapEntry entry = std::make_tuple(
                            unpack32(target),                // Coords of the found key (target)
                            unpack32(uniq_keys_c[i_idx_val]), // Coords of the original input key
                            unpack32(woffs_c[q])            // Signed coords of the offset
                        );
                        { // Lock kernel_map for writing
                            std::lock_guard<std::mutex> guard(kernel_map_lock);
                            kernel_map_ref[k_idx_val].push_back(entry);
                            // Python recorded: KM_BASE + k_idx*SIZE_KEY. This is a simplified representation
                            // of writing to the kernel map structure for this k_idx.
                            record_access_local(thread_id, "W", KM_BASE + static_cast<uint64_t>(k_idx_val) * SIZE_KEY);
                        }
                    }
                    // Python's commented out accesses for addr_i, addr_j are omitted.
                    break; // Found target in tile, move to next query
                }
            }
        }

    // After processing all assigned queries, transfer local trace to global trace
    if (!thread_lmem_trace.empty()) {
        std::lock_guard<std::mutex> guard(gmem_trace_lock); // Lock global gmem_trace
        gmem_trace.insert(gmem_trace.end(), thread_lmem_trace.begin(), thread_lmem_trace.end());
        thread_lmem_trace.clear(); // Cleared at the start of next call or explicitly if needed
    }
    // Sleep randombly to avoid contention on the global trace
    std::this_thread::sleep_for(std::chrono::nanoseconds(rand() % 1000)); // Simulate random sleep
    }
}

KernelMap lookup_phase(
    const std::vector<uint64_t>& uniq,
    const std::vector<uint64_t>& qkeys,
    const std::vector<uint32_t>& qii,
    const std::vector<uint32_t>& qki,
    const std::vector<uint64_t>& woffs,
    const std::vector<std::vector<uint64_t>>& tiles,
    const std::vector<uint64_t>& pivots,
    int tile_size) {

    KernelMap kernel_map_result;         // The final kernel map to be populated
    std::mutex kernel_map_result_lock;   // Mutex to protect concurrent writes to kernel_map_result
    std::mutex gmem_trace_global_lock;   // Mutex to protect concurrent writes to global gmem_trace

    size_t Nq = qkeys.size(); // Total number of queries
    if (Nq == 0) return kernel_map_result; // No queries to process

    std::vector<std::thread> worker_threads;
    worker_threads.reserve(NUM_THREADS);

    // Calculate chunk size for distributing queries among threads
    size_t chunk_size = (Nq + NUM_THREADS - 1) / NUM_THREADS; // Ceiling division

    for (int t = 0; t < NUM_THREADS; ++t) {
        size_t q_start = t * chunk_size;
        size_t q_end = std::min(q_start + chunk_size, Nq);

        if (q_start < q_end) { // Only create thread if there's work to do
            worker_threads.emplace_back(
                process_query_range,
                static_cast<int>(q_start), static_cast<int>(q_end), t, // query range and thread ID
                std::cref(uniq), std::cref(qkeys), std::cref(qii), std::cref(qki), std::cref(woffs), // const refs to data
                std::cref(tiles), std::cref(pivots), tile_size,
                std::ref(kernel_map_result),       // ref to shared kernel map
                std::ref(kernel_map_result_lock),  // ref to its mutex
                std::ref(gmem_trace_global_lock)   // ref to global trace mutex
            );
        }
    }

    // Wait for all worker threads to complete
    for (auto& th : worker_threads) {
        if (th.joinable()) {
            th.join();
        }
    }
    return kernel_map_result;
}
