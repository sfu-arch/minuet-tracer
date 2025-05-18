#include "minuet_trace.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> 
#include <numeric> // For std::iota if needed, not directly here

int main() {
    // --- Initial Data Setup (matches Python script's example) ---
    std::vector<std::tuple<int, int, int>> initial_coords_tuples = { // Renamed from coords
        {1, 5, 0}, {0, 0, 2}, {0, 1, 1}, {0, 0, 3}
    };
    int stride = 1;
    std::vector<std::tuple<int, int, int>> offset_coords_tuples; // Renamed from offsets
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                offset_coords_tuples.emplace_back(dx, dy, dz);
            }
        }
    }

    // --- Phase 1: Radix Sort (Unique Sorted Input Coords with Original Indices) ---
    curr_phase = "Radix-Sort"; // Updated global variable name
    std::cout << "\n--- Phase: " << curr_phase << " with " << NUM_THREADS << " threads ---" << std::endl;
    std::vector<IndexedCoord> unique_indexed_coords = compute_unique_sorted_coords(initial_coords_tuples, stride);

    // --- Phase 2: Build Queries ---
    curr_phase = "Build-Queries";
    std::cout << "--- Phase: " << curr_phase << " ---" << std::endl;
    BuildQueriesResult query_data = build_coordinate_queries(unique_indexed_coords, stride, offset_coords_tuples);

    // --- Phase 3: Sort Query Keys ---
    curr_phase = "Sort-QKeys";
    std::cout << "--- Phase: " << curr_phase << " ---" << std::endl;
    // No actual sorting of qry_keys in this phase as per Python logic.
    // qry_keys from query_data is used directly.

    // --- Phase 4: Tile and Pivot Generation ---
    curr_phase = "Tile-Pivots";
    std::cout << "--- Phase: " << curr_phase << " ---" << std::endl;
    TilesPivotsResult tiles_pivots_data = create_tiles_and_pivots(unique_indexed_coords, I_TILES);

    // --- Phase 5: Lookup ---
    curr_phase = "Lookup";
    std::cout << "--- Phase: " << curr_phase << " ---" << std::endl;
    KernelMap kernel_map_result = perform_coordinate_lookup(
        unique_indexed_coords, query_data.qry_keys, query_data.qry_in_idx, 
        query_data.qry_off_idx, query_data.wt_offsets,
        tiles_pivots_data.tiles, tiles_pivots_data.pivots, I_TILES
    );
    curr_phase = ""; // Clear phase

    // --- Print Debug Information (if enabled) ---
    if (debug) {
        std::cout << "\nSorted Source Array (Coordinate, Original Index):" << std::endl;
        for (const auto& idxc_item : unique_indexed_coords) {
            std::cout << "  key=" << to_hex_string(idxc_item.coord.to_key())
                      << ", coords=" << idxc_item.coord // Uses Coord3D's operator<<
                      << ", index=" << idxc_item.orig_idx << std::endl;
        }
        
        std::cout << "\nQuery Segments:" << std::endl;
        if (!query_data.qry_keys.empty() && !offset_coords_tuples.empty()) {
            size_t num_unique_inputs = unique_indexed_coords.size();
             if (num_unique_inputs == 0 && query_data.qry_keys.empty()) { // Handle case with no inputs
                std::cout << "  No unique inputs, so no query segments generated." << std::endl;
            } else if (num_unique_inputs == 0 && !query_data.qry_keys.empty()){
                 std::cout << "  Warning: No unique inputs, but query keys exist. Printing all query keys as one segment." << std::endl;
                 std::cout << "  All Queries: ";
                 for(const auto& q_idxc : query_data.qry_keys){
                     std::cout << q_idxc.coord << " (orig_src_idx: " << q_idxc.orig_idx << ") ";
                 }
                 std::cout << std::endl;
            }
            else {
                for (size_t off_idx = 0; off_idx < offset_coords_tuples.size(); ++off_idx) {
                    std::cout << "  Offset " << offset_coords_tuples[off_idx] << ": ";
                    bool first_in_segment = true;
                    for (size_t i = 0; i < num_unique_inputs; ++i) {
                        size_t q_glob_idx = off_idx * num_unique_inputs + i;
                        if (q_glob_idx < query_data.qry_keys.size()) {
                             if (!first_in_segment) std::cout << ", ";
                             std::cout << query_data.qry_keys[q_glob_idx].coord 
                                       << "(src_orig:" << query_data.qry_keys[q_glob_idx].orig_idx << ")";
                             first_in_segment = false;
                        }
                    }
                    std::cout << std::endl;
                }
            }
        } else {
            std::cout << "  No queries to display." << std::endl;
        }

        std::cout << "\nKernel Map:" << std::endl;
        if (kernel_map_result.empty()) {
            std::cout << "  Kernel map is empty." << std::endl;
        }
        for (const auto& pair : kernel_map_result) {
            int off_idx = pair.first;
            const auto& matches = pair.second;
            if (!matches.empty()) {
                 if (static_cast<size_t>(off_idx) < offset_coords_tuples.size()){
                    std::cout << "  Offset " << offset_coords_tuples[off_idx] << " (idx " << off_idx << "):" << std::endl;
                 } else {
                    std::cout << "  Offset Index " << off_idx << " (original offset tuple unavailable):" << std::endl;
                 }
                for (const auto& match : matches) {
                    // KernelMapMatch is std::pair<std::pair<std::tuple<int,int,int>, int>, std::pair<std::tuple<int,int,int>, int>>
                    // ((target_coord_tuple, target_orig_idx), (source_coord_tuple, source_orig_idx))
                    std::cout << "    Match: (Output: " << match.first.first << " [orig_idx:" << match.first.second << "])"
                              << " -> (Input: " << match.second.first << " [orig_idx:" << match.second.second << "])"
                              << std::endl;
                }
            }
        }
    }

    // --- Print Memory Trace and Write to File ---
    std::cout << "\nMemory Trace Entries (" << mem_trace.size() << " total):" << std::endl; // Use mem_trace
    size_t entries_to_show = std::min(static_cast<size_t>(10), mem_trace.size());
    for (size_t i = 0; i < entries_to_show; ++i) {
        const auto& e = mem_trace[i];
        std::cout << "(\"" << e.phase << "\", " << e.thread_id << ", \"" << e.op
                  << "\", \"" << e.tensor << "\", \"" << to_hex_string(e.addr) << "\")" << std::endl;
    }
    if (mem_trace.size() > entries_to_show) {
        std::cout << "... and " << mem_trace.size() - entries_to_show << " more entries." << std::endl;
    }

    try {
        write_gmem_trace("map_trace.bin.gz"); // Updated filename
        write_kernel_map_to_gz(kernel_map_result, "kernel_map.bin.gz", offset_coords_tuples); // New function call
    } catch (const std::exception& e) {
        std::cerr << "Error during file writing: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

