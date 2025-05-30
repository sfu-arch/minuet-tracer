#ifndef MINUET_GATHER_HPP
#define MINUET_GATHER_HPP

#include <vector>
#include <string>
#include <cstdint> // For uint32_t, uint64_t
#include <map>     // For std::map
#include <utility> // For std::pair
#include "coord.hpp"        // For Coord3D
#include "sorted_map.hpp"   // For SortedByValueSizeMap
#include "trace.hpp"

// Define KernelMapType consistently
// This is the type for kernel_map, storing matches for each offset key,
// sorted by the number of matches.
using KernelMapType = SortedByValueSizeMap<uint32_t, std::vector<std::pair<int, int>>>;

// --- Structs and Functions for Greedy Grouping ---
struct GemmInfo {
    uint32_t num_offsets;
    uint32_t gemm_M;
    uint32_t slots;
    uint32_t padding;
};

struct GroupInfo {
    std::vector<int> members;
    uint64_t base_addr;
    uint32_t required_slots;
    uint32_t allocated_slots;
};

struct GreedyGroupResult {
    std::vector<uint64_t> pos_indices;
    std::vector<GroupInfo> groups;
    std::vector<std::vector<int>> membership;
    std::vector<GemmInfo> gemm_list;
    uint64_t total_slots_allocated;
    uint32_t checksum;
};

uint32_t write_gemm_list_cpp(const std::vector<GemmInfo>& gemm_data_list, const std::string& filename);

GreedyGroupResult greedy_group_cpp(
    const std::vector<int>& slots,
    int alignment = 4,
    int max_group_items = 6,
    int max_raw_slots = -1
);


// --- Structs for Metadata Reading ---
struct ActiveOffsetInfo {
    uint32_t offset_key;
    uint32_t base_address;
    uint32_t num_matches; // Number of matches for this offset
};

struct MetadataContents {
    uint32_t version;
    uint32_t num_total_system_offsets; // Total number of offsets in the system (e.g., 27 for 3x3x3 kernel)
    uint32_t num_total_system_sources; // Total number of unique source points
    uint32_t total_slots_in_gemm_buffer;
    uint32_t num_active_offsets_in_map;  // Number of offsets that actually have matches
    std::vector<ActiveOffsetInfo> active_offsets_details;
    std::vector<int32_t> out_mask;
    std::vector<int32_t> in_mask;
};

// --- Mask Generation ---
struct MasksResult {
    std::vector<int32_t> out_mask;
    std::vector<int32_t> in_mask;
};

MasksResult create_in_out_masks_cpp(
    const KernelMapType& kernel_map,
    const std::map<uint32_t, int>& slot_dict,
    uint32_t num_total_system_offsets,
    uint32_t num_total_system_sources
);

// --- Function Declarations ---
MetadataContents read_metadata_cpp(const std::string& filename);

uint32_t write_metadata_cpp(
    const std::vector<int32_t>& out_mask,
    const std::vector<int32_t>& in_mask,
    const std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>& active_offset_data, // (offset_key, base_addr, num_matches)
    uint32_t num_total_system_offsets,
    uint32_t num_total_system_sources,
    uint32_t total_slots_in_gemm_buffer,
    const std::string& filename
);

// --- Gather and Scatter Operations ---
void mt_gather_cpp(
    uint32_t num_threads,
    uint32_t num_points,
    uint32_t num_offsets,
    uint32_t num_tiles_per_pt,
    uint32_t tile_feat_size,
    uint32_t bulk_feat_size,
    const std::vector<int32_t>& source_masks, // Assuming int32_t for masks
    const std::vector<float>& sources,       // Assuming float for feature data
    std::vector<float>& gemm_buffers         // Assuming float for feature data
);

void mt_scatter_cpp(
    uint32_t num_threads,
    uint32_t num_points,          // Number of *output* points
    uint32_t num_offsets,
    uint32_t num_tiles_per_pt,
    uint32_t tile_feat_size,
    uint32_t bulk_feat_size,
    const std::vector<int32_t>& out_mask,    // Assuming int32_t for masks
    const std::vector<float>& gemm_buffers,  // Assuming float for feature data
    std::vector<float>& outputs              // Assuming float for feature data
);


#endif // MINUET_GATHER_HPP