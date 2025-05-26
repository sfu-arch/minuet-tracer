#include "minuet_config.hpp"
#include "minuet_trace.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> 
#include <numeric> // For std::iota if needed, not directly here
#include <filesystem> // For create_directory

// Helper to convert vector of tuples to vector of Coord3D
std::vector<Coord3D> tuples_to_coords(const std::vector<std::tuple<int, int, int>>& tuples) {
    std::vector<Coord3D> coords_vec;
    coords_vec.reserve(tuples.size());
    for (const auto& t : tuples) {
        coords_vec.emplace_back(std::get<0>(t), std::get<1>(t), std::get<2>(t));
    }
    return coords_vec;
}

// Forward declare to_hex_string if it's defined in minuet_trace.cpp and used here
// If it's a local helper, it should be defined or declared before use in main.cpp as well.
// Assuming it's in minuet_trace.cpp and accessible globally or via hpp.
// If not, it needs to be added to minuet_trace.hpp or defined statically in main.cpp.
// For now, let's assume it's available through includes.
std::string to_hex_string(uint64_t val); // Declaration if defined elsewhere like minuet_trace.cpp

int main(int argc, char *argv[]) {
    std::string config_filepath = "config.json"; // Default config file path
    if (argc > 1) {
        config_filepath = argv[1]; // Use path from command line argument if provided
    }

    // Load configuration
    if (!g_config.loadFromFile(config_filepath)) {
        std::cerr << "Failed to load configuration from " << config_filepath << ". Exiting." << std::endl;
        return 1;
    }

    // --- Initial Data Setup (matches Python script's example) ---
    std::vector<std::tuple<int, int, int>> initial_coords_tuples_raw = { // Renamed from coords
        {1, 5, 0}, {0, 0, 2}, {0, 1, 1}, {0, 0, 3}
    };
    std::vector<Coord3D> initial_coords = tuples_to_coords(initial_coords_tuples_raw);

    int stride = 1;
    std::vector<std::tuple<int, int, int>> offset_coords_tuples_raw; // Renamed from offsets
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                offset_coords_tuples_raw.emplace_back(dx, dy, dz);
            }
        }
    }
    std::vector<Coord3D> offset_coords = tuples_to_coords(offset_coords_tuples_raw);

    // --- Phase 1: Radix Sort (Unique Sorted Input Coords with Original Indices) ---
    // Python: curr_phase = PHASES['RDX']
    // C++: curr_phase is set within compute_unique_sorted_coords
    std::cout << "\n--- Phase: " << PHASES.inverse.at(0) << " with " << g_config.NUM_THREADS << " threads ---" << std::endl;
    std::vector<IndexedCoord> unique_indexed_coords = compute_unique_sorted_coords(initial_coords, stride);

    // --- Phase 2: Build Queries ---
    // Python: curr_phase = PHASES['QRY']
    // C++: curr_phase is set within build_coordinate_queries
    std::cout << "--- Phase: " << PHASES.inverse.at(1) << " ---" << std::endl;
    BuildQueriesResult query_data = build_coordinate_queries(unique_indexed_coords, stride, offset_coords);

    // --- Phase 3: Sort Query Keys ---
    // curr_phase = PHASES.inverse.at(2); // "SRT"
    set_curr_phase(PHASES.inverse.at(2)); // "SRT"
    // std::cout << "--- Phase: " << curr_phase << " ---" << std::endl;
    std::cout << "--- Phase: " << get_curr_phase() << " ---" << std::endl;
    // No actual sorting of qry_keys in this phase as per Python logic.
    // qry_keys from query_data is used directly.

    // --- Phase 4: Tile and Pivot Generation ---
    // Python: curr_phase = PHASES['PVT']
    // C++: curr_phase is set within create_tiles_and_pivots
    std::cout << "--- Phase: " << PHASES.inverse.at(3) << " ---" << std::endl;
    TilesPivotsResult tiles_pivots_data = create_tiles_and_pivots(unique_indexed_coords, g_config.NUM_PIVOTS); // Python example uses tile_size = 2

    // --- Phase 5: Lookup ---
    // Python: curr_phase = PHASES['LKP']
    // C++: curr_phase is set within perform_coordinate_lookup
    std::cout << "--- Phase: " << PHASES.inverse.at(4) << " ---" << std::endl;
    KernelMapType kernel_map_result = perform_coordinate_lookup(
        unique_indexed_coords, query_data.qry_keys, query_data.qry_in_idx, 
        query_data.qry_off_idx, query_data.wt_offsets, // wt_offsets from query_data
        tiles_pivots_data.tiles, tiles_pivots_data.pivots, g_config.NUM_PIVOTS // tile_size = 2
    );
    // curr_phase = ""; // Clear phase
    set_curr_phase(""); // Clear phase

    // --- Print Debug Information (if enabled) ---
    if (get_debug_flag()) { // Use get_debug_flag() which uses g_config.debug
        std::cout << "\\nSorted Source Array (Coordinate, Original Index):" << std::endl;
        for (const auto& idxc_item : unique_indexed_coords) {
            std::cout << "  key=" << to_hex_string(idxc_item.to_key()) // Changed from idxc_item.coord.to_key()
                      << ", coords=" << idxc_item.coord // Uses Coord3D's operator<<
                      << ", index=" << idxc_item.orig_idx << std::endl;
        }
        
        std::cout << "\nQuery Segments:" << std::endl;
        if (!query_data.qry_keys.empty() && !offset_coords.empty()) { // Changed offset_coords_tuples to offset_coords
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
                for (size_t off_idx_print = 0; off_idx_print < offset_coords.size(); ++off_idx_print) {
                    std::cout << "  Offset " << offset_coords[off_idx_print] << ": ";
                    bool first_in_segment = true;
                    for (size_t i = 0; i < num_unique_inputs; ++i) {
                        size_t q_glob_idx = off_idx_print * num_unique_inputs + i;
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
        for (const auto& kmap_pair : kernel_map_result) { // Renamed pair to kmap_pair
            Coord3D offset_as_coord = Coord3D::from_signed_key(kmap_pair.first);
            std::cout << "  Offset " << offset_as_coord << " (Key: " << to_hex_string(kmap_pair.first) << "):";
            if (kmap_pair.second.empty()) {
                std::cout << " No matches" << std::endl;
            } else {
                std::cout << std::endl;
                size_t entries_to_show = kmap_pair.second.size(); // Define entries_to_show
                for (size_t match_idx = 0; match_idx < entries_to_show; ++match_idx) {
                    const auto& match_pair = kmap_pair.second[match_idx];
                    std::cout << "    Match " << match_idx + 1
                              << ": Input original_idx: " << match_pair.first
                              << " -> Query source_original_idx: " << match_pair.second << std::endl;
                }
            }
        }
    }

    // --- Write Memory Trace and Kernel Map ---
    // std::cout << "\\nMemory Trace Entries (" << mem_trace.size() << " total):" << std::endl;
    // for (size_t i = 0; i < std::min(mem_trace.size(), static_cast<size_t>(10)); ++i) {
    //     const auto& e = mem_trace[i];
    //     std::cout << "  Phase: " << e.phase << ", TID: " << e.thread_id 
    //               << ", Op: " << e.op << ", Tensor: " << e.tensor 
    //               << ", Addr: " << to_hex_string(e.addr) << std::endl;
    // }
    // if (mem_trace.size() > 10) {
    //     std::cout << "... and " << mem_trace.size() - 10 << " more entries" << std::endl;
    // }

    // Retrieve mem_trace using the getter
    const auto& current_mem_trace = get_mem_trace();
    std::cout << "\\nMemory Trace Entries (" << current_mem_trace.size() << " total):" << std::endl;
    for (size_t i = 0; i < std::min(current_mem_trace.size(), static_cast<size_t>(10)); ++i) {
        const auto& e = current_mem_trace[i];
        std::cout << "  Phase: " << e.phase << ", TID: " << e.thread_id 
                  << ", Op: " << e.op << ", Tensor: " << e.tensor 
                  << ", Addr: " << to_hex_string(e.addr) << std::endl;
    }
    if (current_mem_trace.size() > 10) {
        std::cout << "... and " << current_mem_trace.size() - 10 << " more entries" << std::endl;
    }

    // Create output directory if it doesn't exist
    if (!std::filesystem::exists(g_config.output_dir)) { // Use g_config.output_dir
        std::filesystem::create_directories(g_config.output_dir);
        std::cout << "Created output directory: " << g_config.output_dir << std::endl;
    }

    try {
        write_gmem_trace(g_config.output_dir + "map_trace.bin.gz"); // Use g_config.output_dir
        write_kernel_map_to_gz(kernel_map_result, g_config.output_dir + "kernel_map.bin.gz", offset_coords); // Use g_config.output_dir
    } catch (const std::exception& e) {
        std::cerr << "Error during file writing: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nC++ Minuet mapping trace generation complete." << std::endl;

    // --- Start of Gather Logic Integration ---
    std::cout << "\n--- Minuet Gather (C++) ---" << std::endl;

    // 1. Prepare offsets_active and slots (mimicking Python's minuet_gather.py logic)
    std::vector<std::pair<int, size_t>> offset_counts; // pair: {offset_idx, count}
    // Iterate using get_sorted_items() to respect the SortedByValueSizeMap order if needed here
    // or iterate the underlying map if key order is preferred for this step.
    // For preparing offsets_active_cpp and slots_cpp, Python sorts kmap by len(matches) descending.
    // Our KernelMapType is already sorted this way (if initialized with ascending=false).
    auto sorted_kmap_items = kernel_map_result.get_sorted_items();
    for(const auto& entry : sorted_kmap_items) {
        offset_counts.push_back({entry.first, entry.second.size()});
    }
    // The sorting done in Python: offsets_active = list(kmap._get_sorted_keys())
    // slot_array = [len(kmap[off_idx]) for off_idx in offsets_active]
    // Our KernelMapType(false) should already provide keys in descending order of value size.

    std::vector<int> offsets_active_cpp;
    std::vector<int> slots_cpp; // slot counts for offsets_active_cpp
    // We iterate through sorted_kmap_items which is already sorted by match count (descending)
    for(const auto& oc_pair : sorted_kmap_items) { // oc_pair is std::pair<const Key, ValueContainer>
        if (!oc_pair.second.empty()) { // Only consider offsets with actual matches
            offsets_active_cpp.push_back(oc_pair.first);
            slots_cpp.push_back(oc_pair.second.size());
        }
    }

    // ... existing gather logic continues ...

    return 0;
}

