// minuet_gather.cpp
#include "minuet_gather.hpp"
#include "minuet_config.hpp" // For g_config
#include <zlib.h>
#include <stdexcept>
#include <iostream>
#include <numeric>   // For std::accumulate if used by greedy_group_cpp (not directly used here)
#include <algorithm> // For std::min if used (not directly used here)

// Helper to convert host uint32_t to big-endian network byte order
// uint32_t hton_u32(uint32_t val) { // REMOVED
//     return htonl(val);
// }

uint32_t write_gemm_list_cpp(const std::vector<GemmInfo>& gemm_data_list, const std::string& filename) {
    gzFile outFile = gzopen(filename.c_str(), "wb");
    if (!outFile) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    uLong crc = crc32(0L, Z_NULL, 0);
    auto write_and_crc = [&](const void* data, unsigned int len) {
        if (gzwrite(outFile, data, len) != static_cast<int>(len)) {
            gzclose(outFile);
            throw std::runtime_error("Failed to write data to gzip file during GEMM list writing.");
        }
        crc = crc32(crc, static_cast<const Bytef*>(data), len);
    };

    for (const auto& gemm : gemm_data_list) {
        uint32_t num_offsets = gemm.num_offsets;
        uint32_t gemm_M = gemm.gemm_M;
        uint32_t padding = gemm.padding;
        
        write_and_crc(&num_offsets, sizeof(num_offsets));
        write_and_crc(&gemm_M, sizeof(gemm_M));
        write_and_crc(&padding, sizeof(padding));
    }

    gzclose(outFile);
    std::cout << "GEMM list successfully written to " << filename << " with "
              << gemm_data_list.size() << " entries." << std::endl;
    return static_cast<uint32_t>(crc);
}


GreedyGroupResult greedy_group_cpp(
    const std::vector<int>& slots,
    int alignment,
    int max_group_items,
    int max_raw_slots
) {
    GreedyGroupResult result; // result is of type GreedyGroupResult
    int n = slots.size();

    uint64_t current_addr = 0;
    uint32_t current_sum_slots = 0;
    int current_item_count = 0;
    std::vector<int> current_members_indices;

    result.pos_indices.resize(n);

    auto flush_group_lambda = [&]() {
        if (current_members_indices.empty()) {
            return;
        }
        uint32_t req = current_sum_slots;
        uint32_t alloc = ((req + alignment - 1) / alignment) * alignment;
        
        result.groups.push_back({current_members_indices, current_addr, req, alloc});
        
        current_addr += alloc;
        current_sum_slots = 0;
        current_item_count = 0;
        current_members_indices.clear();
    }; // Semicolon for lambda assignment

    for (int i = 0; i < n; ++i) {
        int slot_size = slots[i];
        if ((current_item_count >= max_group_items) ||
            (max_raw_slots != -1 && current_sum_slots + slot_size > static_cast<uint32_t>(max_raw_slots))) {
            flush_group_lambda();
        }
        result.pos_indices[i] = current_addr + current_sum_slots;
        current_members_indices.push_back(i);
        current_sum_slots += slot_size;
        current_item_count += 1;
    } // End of for loop

    flush_group_lambda();

    result.total_slots_allocated = 0;
    for (const auto& group : result.groups) { 
        result.membership.push_back(group.members);
        result.total_slots_allocated += group.allocated_slots;

        GemmInfo gemm_item; 
        
        if (group.members.empty()) {
            throw std::runtime_error("Group members list is empty unexpectedly for GemmInfo calculation.");
        }
        
        if (group.members.size() == 1) {
            gemm_item.num_offsets = 1;
        } else {
            gemm_item.num_offsets = static_cast<uint32_t>(group.members.size());
        }
        
        gemm_item.gemm_M = group.allocated_slots;
        gemm_item.slots = group.required_slots;
        gemm_item.padding = group.allocated_slots - group.required_slots; 
        result.gemm_list.push_back(gemm_item); 
    } 
    
    std::string gemm_filename = "gemms.bin.gz"; 
    if (g_config.output_dir.empty()) { 
         std::cerr << "Warning: g_config.output_dir is not set. Writing gemms.bin.gz to current directory." << std::endl;
    } else {
        gemm_filename = g_config.output_dir + "/gemms.bin.gz"; 
    }

    result.checksum = write_gemm_list_cpp(result.gemm_list, gemm_filename);
    return result;
}
