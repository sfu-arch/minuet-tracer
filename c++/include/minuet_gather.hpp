#ifndef MINUET_GATHER_HPP
#define MINUET_GATHER_HPP

#include <vector>
#include <string>
#include <cstdint> // For uint32_t, uint64_t

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

#endif // MINUET_GATHER_HPP