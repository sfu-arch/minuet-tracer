#include "minuet_trace.hpp" // Main header for the trace simulation
#include <iostream>         // For std::cout, std::cerr
#include <vector>
#include <string>
#include <algorithm>        // For std::sort, std::min etc. (though sort not used on qkeys_pre here)

int main() {
    // --- Initial Data Setup (matches Python script's example) ---
    std::vector<std::tuple<int, int, int>> coords = {
        {1, 5, 0}, {0, 1, 1}, {0, 0, 2}, {0, 0, 3}
    };
    int stride = 1;
    std::vector<std::tuple<int, int, int>> offsets;
    // Generate 3x3x3 offsets, from (-1,-1,-1) to (1,1,1)
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                offsets.emplace_back(dx, dy, dz);
            }
        }
    }
    // Example for testing with a single offset:
    // offsets = {{0, 1, -1}};

    // --- Phase 1: Radix Sort (Unique Sorted Input Keys) ---
    current_phase = "Radix-Sort";
    std::cout << "\n--- Phase: " << current_phase << " with " << NUM_THREADS << " threads ---" << std::endl;
    std::vector<uint32_t> uniq = compute_unique_sorted(coords, stride);

    // --- Phase 2: Build Queries ---
    current_phase = "Build-Queries";
    std::cout << "--- Phase: " << current_phase << " ---" << std::endl;
    QueryData query_data = build_queries(uniq, stride, offsets);
    // In Python: qkeys_pre, qii, qki, woffs = build_queries(...)
    // query_data.qkeys corresponds to qkeys_pre

    // --- Phase 3: Sort Query Keys ---
    // Python script: current_phase = 'Sort-QKeys'; qkeys = qkeys_pre
    // This implies no actual sorting operation in this phase, just an assignment.
    // The qkeys from build_queries are used directly.
    current_phase = "Sort-QKeys";
    std::cout << "--- Phase: " << current_phase << " ---" << std::endl;
    std::vector<uint32_t> qkeys_for_lookup = query_data.qkeys; // Use the qkeys as generated

    // --- Phase 4: Tile and Pivot Generation ---
    current_phase = "Tile-Pivots";
    std::cout << "--- Phase: " << current_phase << " ---" << std::endl;
    TilesPivots tiles_pivots_data = make_tiles_and_pivots(uniq, I_TILES);

    // --- Phase 5: Lookup ---
    current_phase = "Lookup";
    std::cout << "--- Phase: " << current_phase << " ---" << std::endl;
    KernelMap kernel_map_result = lookup_phase(
        uniq, qkeys_for_lookup, query_data.qii, query_data.qki, query_data.woffs,
        tiles_pivots_data.tiles, tiles_pivots_data.pivots, I_TILES
    );
    current_phase = ""; // Clear phase after completion

    // --- Print Debug Information (if enabled) ---
    if (debug) {
        std::cout << "\nSorted Source Array (unique keys from compute_unique_sorted):" << std::endl;
        for (const auto& k_val : uniq) {
            // Python example uses unpack32s for this print
            std::cout << unpack32s(k_val) << " ";
        }
        std::cout << std::endl;

        std::cout << "Segmented Query Arrays (from build_queries, Python's qkeys_pre):" << std::endl;
        size_t uniq_len = uniq.size();
        if (uniq_len > 0) { // Avoid division by zero if uniq is empty
            for (size_t k_offset_idx = 0; k_offset_idx < offsets.size(); ++k_offset_idx) {
                std::cout << "Offset " << k_offset_idx << " (key " << unpack32s(query_data.woffs[k_offset_idx * uniq_len]) << "): ";
                for (size_t i = 0; i < uniq_len; ++i) {
                    size_t q_idx = k_offset_idx * uniq_len + i;
                    if (q_idx < query_data.qkeys.size()) {
                         // Python example uses unpack32s for this print
                        std::cout << unpack32s(query_data.qkeys[q_idx]) << " ";
                    }
                }
                std::cout << std::endl;
            }
        }
        
        std::cout << "\nKernel Map:" << std::endl;
        for (const auto& pair : kernel_map_result) {
            std::cout << pair.first << ": [";
            bool first_entry = true;
            for (const auto& entry : pair.second) {
                if (!first_entry) std::cout << ", ";
                // The overloaded operator<< for std::tuple will handle printing KernelMapEntry
                std::cout << entry;
                first_entry = false;
            }
            std::cout << "]" << std::endl;
        }
    }

    // --- Print Memory Trace and Write to File ---
    std::cout << "\nMemory Trace Entries (" << gmem_trace.size() << " total):" << std::endl;
    for (const auto& e : gmem_trace) {
        // Mimic Python's tuple print format for trace entries
        std::cout << "(\"" << e.phase << "\", " << e.thread_id << ", \"" << e.op
                  << "\", \"" << e.tensor << "\", \"" << to_hex_string(e.addr) << "\")" << std::endl;
    }

    try {
        write_gmem_trace("memory_trace_cpp.bin.gz"); // Write trace to compressed file
    } catch (const std::exception& e) {
        std::cerr << "Error writing trace file: " << e.what() << std::endl;
        return 1; // Indicate failure
    }

    return 0; // Indicate successful execution
}
