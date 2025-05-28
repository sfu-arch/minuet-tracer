#include "minuet_config.hpp"
#include "minuet_trace.hpp"
#include "minuet_gather.hpp" // Added for GreedyGroupResult and greedy_group_cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> 
#include <numeric> // For std::iota if needed
#include <filesystem> // For create_directory
#include <map>      // For std::map
#include <fstream>  // For std::ofstream
#include <iomanip>  // For std::setw
// nlohmann/json.hpp is included via minuet_config.hpp -> ext/json.hpp

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
    // KernelMapType kernel_map_result = perform_coordinate_lookup( // This line is modified below
    KernelMapType kernel_map_result_obj(false); // Create kmap, false for sorting order (desc by value size)
    kernel_map_result_obj = perform_coordinate_lookup( // Pass by reference (this is the kmap)
        unique_indexed_coords, 
        query_data.qry_keys, 
        query_data.qry_in_idx, 
        query_data.qry_off_idx, 
        query_data.wt_offsets,
        tiles_pivots_data.tiles, 
        tiles_pivots_data.pivots, 
        g_config.NUM_TILES // num_tiles_config parameter
    );
    set_curr_phase(""); // Clear phase

    // --- Print Debug Information (if enabled) ---
    if (get_debug_flag()) {
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
        if (kernel_map_result_obj.empty()) {
            std::cout << "  Kernel map is empty." << std::endl;
        }
        // Iterate using get_sorted_items to respect the map's internal sort order (desc by value size)
        auto sorted_kmap_debug_items = kernel_map_result_obj.get_sorted_items();
        for (const auto& kmap_pair : sorted_kmap_debug_items) { 
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

    uint32_t map_trace_checksum = 0;
    uint32_t kernel_map_checksum = 0;

    try {
        map_trace_checksum = write_gmem_trace(g_config.output_dir + "map_trace.bin.gz"); 
        kernel_map_checksum = write_kernel_map_to_gz(kernel_map_result_obj, g_config.output_dir + "kernel_map.bin.gz", offset_coords);
    } catch (const std::exception& e) {
        std::cerr << "Error during file writing: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\\nC++ Minuet mapping trace generation complete." << std::endl;

    // --- Start of Gather/Scatter Metadata (mirroring Python minuet_trace.py) ---
    std::cout << "\\n--- Phase: Gather/Scatter Metadata (C++) ---" << std::endl;

    std::vector<uint32_t> offsets_active_cpp;
    std::vector<int> slot_array_cpp; 

    // kernel_map_result_obj is already sorted by value size (descending) due to KernelMapType(false)
    auto sorted_kmap_items_for_active = kernel_map_result_obj.get_sorted_items(); 
    for (const auto& item : sorted_kmap_items_for_active) {
        if (!item.second.empty()) {
            offsets_active_cpp.push_back(item.first); 
            slot_array_cpp.push_back(static_cast<int>(item.second.size())); 
        }
    }

    if (g_config.debug) {
        std::cout << "Offsets active (derived from C++ KernelMapType iteration order, desc by match count):" << std::endl;
        for (size_t o_idx = 0; o_idx < offsets_active_cpp.size(); ++o_idx) {
            std::cout << "  [" << o_idx << "] " << to_hex_string(offsets_active_cpp[o_idx])
                      << " (Matches: " << slot_array_cpp[o_idx] << ")" << std::endl;
        }
        std::cout << "Slot array (ordered by match count desc): [";
        for(size_t i=0; i<slot_array_cpp.size(); ++i) {
            std::cout << slot_array_cpp[i] << (i == slot_array_cpp.size()-1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;
    }

    std::vector<int> cpp_slots_arg_for_greedy_group = slot_array_cpp; 

    std::cout << "Calling C++ greedy_group_cpp with " << cpp_slots_arg_for_greedy_group.size() << " slot sizes." << std::endl;
    if (g_config.debug && !cpp_slots_arg_for_greedy_group.empty()) {
        std::cout << "  Slot sizes for C++ greedy_group_cpp (sorted desc): [";
        for(size_t i=0; i < std::min(cpp_slots_arg_for_greedy_group.size(), static_cast<size_t>(10)); ++i) {
             std::cout << cpp_slots_arg_for_greedy_group[i] << (i == std::min(cpp_slots_arg_for_greedy_group.size(), static_cast<size_t>(10))-1 || i == cpp_slots_arg_for_greedy_group.size()-1 ? "" : ", ");
        }
        if (cpp_slots_arg_for_greedy_group.size() > 10) std::cout << "...";
        std::cout << "]" << std::endl;
    }

    GreedyGroupResult greedy_group_result_cpp = greedy_group_cpp(
        cpp_slots_arg_for_greedy_group,
        g_config.GEMM_ALIGNMENT,
        g_config.GEMM_WT_GROUP,
        g_config.GEMM_SIZE
    );

    std::vector<uint64_t> slot_indices_cpp = greedy_group_result_cpp.pos_indices;
    int total_slots_cpp = greedy_group_result_cpp.total_slots_allocated;
    uint32_t gemm_checksum_cpp = greedy_group_result_cpp.checksum;

    std::map<uint32_t, int> slot_dict_cpp;
    for (size_t i = 0; i < offsets_active_cpp.size(); ++i) {
        slot_dict_cpp[offsets_active_cpp[i]] = slot_indices_cpp[i];
    }

    uint32_t num_total_system_offsets_cpp = static_cast<uint32_t>(offset_coords.size());
    uint32_t num_total_system_sources_cpp = static_cast<uint32_t>(unique_indexed_coords.size());

    std::vector<int32_t> out_mask_cpp(static_cast<size_t>(num_total_system_offsets_cpp) * num_total_system_sources_cpp, -1);
    std::vector<int32_t> in_mask_cpp(static_cast<size_t>(num_total_system_offsets_cpp) * num_total_system_sources_cpp, -1);

    std::map<uint32_t, uint32_t> offset_key_to_dense_idx_map;
    for (uint32_t i = 0; i < offset_coords.size(); ++i) {
        offset_key_to_dense_idx_map[offset_coords[i].to_key()] = i;
    }

    for (const auto& k_item : sorted_kmap_items_for_active) { 
        uint32_t offset_key = k_item.first;
        const auto& matches_list = k_item.second;

        if (matches_list.empty()) continue;

        auto it_slot_dict = slot_dict_cpp.find(offset_key);
        if (it_slot_dict == slot_dict_cpp.end()) {
            std::cerr << "Warning: Offset key " << to_hex_string(offset_key) << " from kernel_map not found in slot_dict_cpp." << std::endl;
            continue;
        }
        int base_slot_for_offset = it_slot_dict->second;

        auto it_dense_idx = offset_key_to_dense_idx_map.find(offset_key);
        if (it_dense_idx == offset_key_to_dense_idx_map.end()) {
            std::cerr << "Warning: Offset key " << to_hex_string(offset_key) << " from kernel_map not found in offset_key_to_dense_idx_map (original offset_coords)." << std::endl;
            continue;
        }
        uint32_t dense_offset_idx = it_dense_idx->second;

        for (size_t slot_in_offset = 0; slot_in_offset < matches_list.size(); ++slot_in_offset) {
            int in_idx = matches_list[slot_in_offset].first;
            int q_src_idx = matches_list[slot_in_offset].second;

            if (dense_offset_idx < num_total_system_offsets_cpp && static_cast<uint32_t>(in_idx) < num_total_system_sources_cpp && in_idx >=0) {
                out_mask_cpp[dense_offset_idx * num_total_system_sources_cpp + static_cast<uint32_t>(in_idx)] = static_cast<int32_t>(slot_in_offset);
            }

            int32_t global_slot_idx = static_cast<int32_t>(base_slot_for_offset + slot_in_offset);
            if (dense_offset_idx < num_total_system_offsets_cpp && static_cast<uint32_t>(q_src_idx) < num_total_system_sources_cpp && q_src_idx >=0) {
                in_mask_cpp[dense_offset_idx * num_total_system_sources_cpp + static_cast<uint32_t>(q_src_idx)] = global_slot_idx;
            }
        }
    }
    
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> active_offset_data_for_cpp;
    for (size_t i = 0; i < offsets_active_cpp.size(); ++i) {
        uint32_t offset_key = offsets_active_cpp[i];
        uint32_t base_addr = static_cast<uint32_t>(slot_dict_cpp.at(offset_key)); 
        uint32_t num_matches = static_cast<uint32_t>(cpp_slots_arg_for_greedy_group[i]); 
        active_offset_data_for_cpp.emplace_back(offset_key, base_addr, num_matches);
    }

    std::string metadata_filename_cpp = g_config.output_dir + "/metadata.bin.gz"; // Added path separator
    std::cout << "Writing metadata to " << metadata_filename_cpp << " using C++ implementation." << std::endl;

    uint32_t metadata_checksum_cpp = write_metadata_cpp(
        out_mask_cpp,
        in_mask_cpp,
        active_offset_data_for_cpp,
        num_total_system_offsets_cpp,
        num_total_system_sources_cpp,
        static_cast<uint32_t>(total_slots_cpp),
        metadata_filename_cpp
    );
    std::cout << "C++ calculated CRC32 for metadata: " << to_hex_string(metadata_checksum_cpp) << std::endl;

    nlohmann::json checksums_json;
    checksums_json["map_trace_cpp.bin.gz"] = to_hex_string(map_trace_checksum);
    checksums_json["kernel_map_cpp.bin.gz"] = to_hex_string(kernel_map_checksum);
    checksums_json["gemms.bin.gz"] = to_hex_string(gemm_checksum_cpp); // This is from greedy_group_result_cpp.checksum
    checksums_json["metadata.bin.gz"] = to_hex_string(metadata_checksum_cpp);

    std::string checksum_filename_cpp = g_config.output_dir + "/checksums_cpp.json"; // Added path separator
    std::ofstream checksum_file(checksum_filename_cpp);
    if (checksum_file.is_open()) {
        checksum_file << std::setw(2) << checksums_json << std::endl;
        checksum_file.close();
        std::cout << "Checksums written to " << checksum_filename_cpp << std::endl;
    } else {
        std::cerr << "Error: Unable to open " << checksum_filename_cpp << " for writing." << std::endl;
    }

    if (g_config.debug) {
        std::cout << "In_mask_cpp (first few elements): [";
        for(size_t i=0; i < std::min(in_mask_cpp.size(), static_cast<size_t>(20)); ++i) {
            std::cout << in_mask_cpp[i] << (i == std::min(in_mask_cpp.size(), static_cast<size_t>(20))-1 || i == in_mask_cpp.size()-1 ? "" : ", ");
        }
        if (in_mask_cpp.size() > 20) std::cout << "...";
        std::cout << "]" << std::endl;

        std::cout << "Groups metadata (from C++ GreedyGroupResult):" << std::endl;
        for(const auto& g_info : greedy_group_result_cpp.groups) {
            std::cout << "  Members: [";
            for(size_t i=0; i<g_info.members.size(); ++i) std::cout << g_info.members[i] << (i==g_info.members.size()-1 ? "" : ", ");
            std::cout << "], BaseAddr: " << g_info.base_addr
                      << ", ReqSlots: " << g_info.required_slots
                      << ", AllocSlots: " << g_info.allocated_slots << std::endl;
        }
        std::cout << "GEMM List (from C++ GreedyGroupResult):" << std::endl;
        for(const auto& gemm_info_item : greedy_group_result_cpp.gemm_list) {
            std::cout << "  NumOffsets: " << gemm_info_item.num_offsets
                      << ", Gemm_M: " << gemm_info_item.gemm_M
                      << ", Slots: " << gemm_info_item.slots
                      << ", Padding: " << gemm_info_item.padding << std::endl;
        }
        std::cout << "\\nTotal allocated space (from C++): " << total_slots_cpp << " slots" << std::endl;

        std::cout << "\\nPer-position slot indices (from C++ for sorted slot sizes): [";
        for(size_t i=0; i<slot_indices_cpp.size(); ++i) {
            std::cout << slot_indices_cpp[i] << (i==slot_indices_cpp.size()-1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;
    }

    // --- Start of Gather Logic Integration ---
    // This comment and the code below it seems to be a leftover from a previous state or a duplicate thought process.
    // The gather logic has been implemented above.
    // std::cout << "\\n--- Minuet Gather (C++) ---" << std::endl;
    // ... remove duplicated or obsolete gather logic here ...

    return 0;
}

