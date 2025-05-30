#include "minuet_config.hpp"
#include "minuet_map.hpp"
#include "trace.hpp"
#include "minuet_gather.hpp" // Added for GreedyGroupResult and greedy_group
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

std::string to_hex_string(uint64_t val) {
    std::stringstream ss;
    ss << "0x" << std::hex << val;
    return ss.str();
}

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
    std::vector<std::tuple<int, int, int>> raw_inputs = { // Renamed from coords
        {1, 5, 0}, {0, 0, 2}, {0, 1, 1}, {0, 0, 3}
    };
    std::vector<Coord3D> inputs = tuples_to_coords(raw_inputs);

    int stride = 1;
    std::vector<std::tuple<int, int, int>> offsets_raw; // Renamed from offsets
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                offsets_raw.emplace_back(dx, dy, dz);
            }
        }
    }
    std::vector<Coord3D> offset_coords = tuples_to_coords(offsets_raw);

    // --- Phase 1: Radix Sort (Unique Sorted Input Coords with Original Indices) ---
    std::cout << "\n--- Phase: " << PHASES.inverse.at(0) << " with " << g_config.NUM_THREADS << " threads ---" << std::endl;
    std::vector<IndexedCoord> unique_indexed_coords = compute_unique_sorted_coords(inputs, stride);

    // --- Phase 2: Build Queries ---
    std::cout << "--- Phase: " << PHASES.inverse.at(1) << " ---" << std::endl;
    BuildQueriesResult query_data = build_coordinate_queries(unique_indexed_coords, stride, offset_coords);

    // --- Phase 3: Tile and Pivot Generation ---
    std::cout << "--- Phase: " << PHASES.inverse.at(3) << " ---" << std::endl;
    TilesPivotsResult tiles_pivots_data = create_tiles_and_pivots(unique_indexed_coords, g_config.NUM_PIVOTS); 

    // --- Phase 4: Lookup ---
    std::cout << "--- Phase: " << PHASES.inverse.at(4) << " ---" << std::endl;
    KernelMapType kmap(false); // Create kmap, false for sorting order (descending order by value size)
    kmap = perform_coordinate_lookup( // Pass by reference (this is the kmap)
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

    std::cout << "... and " << get_mem_trace().size() - 10 << " more entries" << std::endl;

    // Create output directory if it doesn't exist
    if (!std::filesystem::exists(g_config.output_dir)) { // Use g_config.output_dir
        std::filesystem::create_directories(g_config.output_dir);
        std::cout << "Created output directory: " << g_config.output_dir << std::endl;
    }

    uint32_t map_trace_checksum = 0;
    uint32_t kernel_map_checksum = 0;

    try {
        map_trace_checksum = write_gmem_trace(g_config.output_dir + "map_trace.bin.gz"); 
        kernel_map_checksum = write_kernel_map_to_gz(kmap, g_config.output_dir + "kernel_map.bin.gz", offset_coords);
    } catch (const std::exception& e) {
        std::cerr << "Error during file writing: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\\nC++ Minuet mapping trace generation complete." << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::cout << "\n--- Phase: Metadata  ---" << std::endl;

    // Create some metadata for gemm grouping. 
    std::vector<uint32_t> offsets_active;
    std::vector<int> slot_array; 
    auto sorted_kmap_items_for_active = kmap.get_sorted_items(); 
    for (const auto& item : sorted_kmap_items_for_active) {
        if (!item.second.empty()) {
            offsets_active.push_back(item.first); 
            slot_array.push_back(static_cast<int>(item.second.size())); 
        }
    }


    GreedyGroupResult greedy_group_result = greedy_group_cpp(
        slot_array,
        g_config.GEMM_ALIGNMENT,
        g_config.GEMM_WT_GROUP,
        g_config.GEMM_SIZE
    );

    std::vector<uint64_t> slot_indices = greedy_group_result.pos_indices;
    int total_slots = greedy_group_result.total_slots_allocated;
    uint32_t gemm_checksum = greedy_group_result.checksum;

    std::map<uint32_t, int> slot_dict;
    for (size_t i = 0; i < offsets_active.size(); ++i) {
        slot_dict[offsets_active[i]] = slot_indices[i];
    }

    uint32_t num_points = static_cast<uint32_t>(unique_indexed_coords.size());
    MasksResult masks = create_in_out_masks_cpp(
        kmap, 
        slot_dict, 
        static_cast<uint32_t>(offset_coords.size()), // Pass the size of the original offset_coords vector
        num_points
    );

    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> active_offset_data_for;
    for (size_t i = 0; i < offsets_active.size(); ++i) {
        uint32_t offset_key = offsets_active[i];
        uint32_t base_addr = static_cast<uint32_t>(slot_dict.at(offset_key)); 
        uint32_t num_matches = static_cast<uint32_t>(slot_array[i]); 
        active_offset_data_for.emplace_back(offset_key, base_addr, num_matches);
    }

    std::string metadata_filename = g_config.output_dir + "/metadata.bin.gz"; // Added path separator
    std::cout << "Writing metadata to " << metadata_filename << " using C++ implementation." << std::endl;

    uint32_t metadata_checksum = write_metadata_cpp(
        masks.out_mask, // Use out_mask from MasksResult
        masks.in_mask,  // Use in_mask from MasksResult
        active_offset_data_for,
        offset_coords.size(),
        num_points,
        total_slots,
        metadata_filename
    );
    std::cout << "C++ calculated CRC32 for metadata: " << to_hex_string(metadata_checksum) << std::endl;

    nlohmann::json checksums_json;
    checksums_json["map_trace.bin.gz"] = to_hex_string(map_trace_checksum);
    checksums_json["kernel_map.bin.gz"] = to_hex_string(kernel_map_checksum);
    checksums_json["gemms.bin.gz"] = to_hex_string(gemm_checksum); // This is from greedy_group_result.checksum
    checksums_json["metadata.bin.gz"] = to_hex_string(metadata_checksum);

    // --- Phase: Gather (C++) ---
    std::cout << "\n--- Minuet Gather (C++) ---" << std::endl;
    
    // Clear previous memory trace before gather operation
    clear_global_mem_trace(); // Clear mem_trace for gather-specific trace

    // Parameters for mt_gather_cpp
    // These should align with how they are used/derived in the Python script
    // and available data in the C++ context.
    int gather_num_threads = g_config.N_THREADS_GATHER; // From config
    int gather_num_points = static_cast<int>(unique_indexed_coords.size());
    int gather_num_offsets = static_cast<int>(offset_coords.size());
    int gather_num_tiles_per_pt = g_config.NUM_TILES; // NUM_TILES_GATHER in Python, using NUM_TILES from C++ config
    int gather_tile_feat_size = g_config.TILE_FEATS;  // TILE_FEATS_GATHER in Python, using TILE_FEATS from C++ config
    int gather_bulk_feat_size = g_config.BULK_FEATS;  // BULK_FEATS_GATHER in Python, using BULK_FEATS from C++ config
    
    // source_masks is masks.in_mask
    const std::vector<int32_t>& gather_source_masks = masks.in_mask;
    
    // sources is an empty vector (float type) as per requirement
    std::vector<float> gather_sources; 
    
    size_t gemm_buffer_size = static_cast<size_t>(total_slots) * g_config.TOTAL_FEATS_PT;
    std::vector<float> gather_gemm_buffers(0, 0.0f); // Initialize with zeros. None initialization. Only need traces. 

    std::cout << "Gather parameters:" << std::endl;
    std::cout << "  num_threads: " << gather_num_threads << std::endl;
    std::cout << "  num_points: " << gather_num_points << std::endl;
    std::cout << "  num_offsets: " << gather_num_offsets << std::endl;
    std::cout << "  num_tiles_per_pt: " << gather_num_tiles_per_pt << std::endl;
    std::cout << "  tile_feat_size: " << gather_tile_feat_size << std::endl;
    std::cout << "  bulk_feat_size: " << gather_bulk_feat_size << std::endl;
    std::cout << "  in_mask size: " << gather_source_masks.size() << std::endl;
    std::cout << "  sources size: " << gather_sources.size() << " (empty as intended)" << std::endl;
    std::cout << "  gemm_buffers size: " << gather_gemm_buffers.size() << std::endl;


    mt_gather_cpp(
        gather_num_threads,
        gather_num_points,
        gather_num_offsets,
        gather_num_tiles_per_pt,
        gather_tile_feat_size,
        gather_bulk_feat_size,
        gather_source_masks,
        gather_sources,       // Empty sources
        gather_gemm_buffers
    );

    uint32_t gather_trace_checksum = 0;
    try {
        // Assuming write_gmem_trace uses the global mem_trace which now contains gather accesses
        gather_trace_checksum = write_gmem_trace(g_config.output_dir + "gather_trace.bin.gz",8);
        std::cout << "C++ calculated CRC32 for gather trace: " << to_hex_string(gather_trace_checksum) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during gather trace writing: " << e.what() << std::endl;
        // Decide if to return 1 or continue for scatter
    }

    // --- End of Gather Phase ---
    // Write checksums to checksums.json
    checksums_json["gather_trace.bin.gz"] = to_hex_string(gather_trace_checksum);

    // --- Phase: Scatter (C++) ---
    std::cout << "\\n--- Minuet Scatter (C++) ---" << std::endl;
    clear_global_mem_trace(); // Clear mem_trace for scatter-specific trace

    // Parameters for mt_scatter_cpp
    int scatter_num_threads = g_config.N_THREADS_GATHER; 
    int scatter_num_points = static_cast<int>(unique_indexed_coords.size());
    int scatter_num_offsets = static_cast<int>(offset_coords.size());
    int scatter_num_tiles_per_pt = g_config.NUM_TILES; 
    int scatter_tile_feat_size = g_config.TILE_FEATS;
    int scatter_bulk_feat_size = g_config.BULK_FEATS;
    const std::vector<int32_t>& scatter_out_mask = masks.out_mask;
    
    std::vector<float> scatter_gemm_buffers;
    std::vector<float> scatter_outputs; 

    std::cout << "Scatter parameters:" << std::endl;
    std::cout << "  num_threads: " << scatter_num_threads << std::endl;
    std::cout << "  num_points: " << scatter_num_points << std::endl;
    std::cout << "  num_offsets: " << scatter_num_offsets << std::endl;
    std::cout << "  num_tiles_per_pt: " << scatter_num_tiles_per_pt << std::endl;
    std::cout << "  tile_feat_size: " << scatter_tile_feat_size << std::endl;
    std::cout << "  bulk_feat_size: " << scatter_bulk_feat_size << std::endl;
    std::cout << "  out_mask size: " << scatter_out_mask.size() << std::endl;
    std::cout << "  gemm_buffers size: " << scatter_gemm_buffers.size() << " (empty as per Python None)" << std::endl;
    std::cout << "  outputs size: " << scatter_outputs.size() << " (empty as per Python None)" << std::endl;

    mt_scatter_cpp(
        scatter_num_threads,
        scatter_num_points,
        scatter_num_offsets,
        scatter_num_tiles_per_pt,
        scatter_tile_feat_size,
        scatter_bulk_feat_size,
        scatter_out_mask,
        scatter_gemm_buffers, // Empty, as per Python's None
        scatter_outputs       // Empty, as per Python's None
    );

    uint32_t scatter_trace_checksum = 0;
    try {
        scatter_trace_checksum = write_gmem_trace(g_config.output_dir + "scatter_trace.bin.gz", 8); // sizeof_addr = 8
        std::cout << "C++ calculated CRC32 for scatter trace: " << to_hex_string(scatter_trace_checksum) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during scatter trace writing: " << e.what() << std::endl;
    }

    // --- End of Scatter Phase ---
    checksums_json["scatter_trace.bin.gz"] = to_hex_string(scatter_trace_checksum);

    // Final write of all checksums
    std::string checksum_filename = g_config.output_dir + "/checksums.json";
    std::ofstream checksum_file(checksum_filename);
    if (checksum_file.is_open()) {
        checksum_file << std::setw(2) << checksums_json << std::endl;
        checksum_file.close();
        std::cout << "Checksums written to " << checksum_filename << std::endl;
    } else {
        std::cerr << "Error: Unable to open " << checksum_filename << " for writing." << std::endl;
    }

    // ... existing return 0 ...
    return 0;
}

