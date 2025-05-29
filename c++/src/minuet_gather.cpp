// minuet_gather.cpp
#include "minuet_gather.hpp"
#include "minuet_config.hpp" // For g_config
#include <algorithm>         // For std::min if used (not directly used here)
#include <iomanip>           // Required for std::hex
#include <iostream>
#include <numeric> // For std::accumulate if used by greedy_group_cpp (not directly used here)
#include <sstream> // Required for std::stringstream
#include <stdexcept>
#include <zlib.h>

// Helper function to convert uint32_t to hex string for debugging
// This is similar to the one in main.cpp or minuet_trace.cpp
// Ensure it's available or consistently defined where used.
static std::string to_hex_string(uint32_t val) {
  std::stringstream ss;
  ss << "0x" << std::hex << val;
  return ss.str();
}

uint32_t write_gemm_list_cpp(const std::vector<GemmInfo> &gemm_data_list,
                             const std::string &filename) {
  gzFile outFile = gzopen(filename.c_str(), "wb");
  if (!outFile) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }

  uLong crc = crc32(0L, Z_NULL, 0);
  auto write_and_crc = [&](const void *data, unsigned int len) {
    if (gzwrite(outFile, data, len) != static_cast<int>(len)) {
      gzclose(outFile);
      throw std::runtime_error(
          "Failed to write data to gzip file during GEMM list writing.");
    }
    crc = crc32(crc, static_cast<const Bytef *>(data), len);
  };

  for (const auto &gemm : gemm_data_list) {
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

GreedyGroupResult greedy_group_cpp(const std::vector<int> &slots, int alignment,
                                   int max_group_items, int max_raw_slots) {
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

    result.groups.push_back(
        {current_members_indices, current_addr, req, alloc});

    current_addr += alloc;
    current_sum_slots = 0;
    current_item_count = 0;
    current_members_indices.clear();
  }; // Semicolon for lambda assignment

  for (int i = 0; i < n; ++i) {
    int slot_size = slots[i];
    if ((current_item_count >= max_group_items) ||
        (max_raw_slots != -1 && current_sum_slots + slot_size >
                                    static_cast<uint32_t>(max_raw_slots))) {
      flush_group_lambda();
    }
    result.pos_indices[i] = current_addr + current_sum_slots;
    current_members_indices.push_back(i);
    current_sum_slots += slot_size;
    current_item_count += 1;
  } // End of for loop

  flush_group_lambda();

  result.total_slots_allocated = 0;
  for (const auto &group : result.groups) {
    result.membership.push_back(group.members);
    result.total_slots_allocated += group.allocated_slots;

    GemmInfo gemm_item;

    if (group.members.empty()) {
      throw std::runtime_error(
          "Group members list is empty unexpectedly for GemmInfo calculation.");
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
    std::cerr << "Warning: g_config.output_dir is not set. Writing "
                 "gemms.bin.gz to current directory."
              << std::endl;
  } else {
    gemm_filename = g_config.output_dir + "/gemms.bin.gz";
  }

  result.checksum = write_gemm_list_cpp(result.gemm_list, gemm_filename);
  return result;
}

MetadataContents read_metadata_cpp(const std::string &filename) {
  gzFile inFile = gzopen(filename.c_str(), "rb");
  if (!inFile) {
    throw std::runtime_error("Failed to open metadata file for reading: " +
                             filename);
  }

  MetadataContents contents;
  char magic[4];

  // Helper lambda for reading from gzFile
  auto read_data = [&](void *buffer, unsigned int size) {
    if (gzread(inFile, buffer, size) != static_cast<int>(size)) {
      gzclose(inFile);
      throw std::runtime_error(
          "Failed to read data from metadata gzip file or unexpected EOF.");
    }
  };

  // Read and verify magic number ("MINU")
  read_data(magic, sizeof(magic));
  if (magic[0] != 'M' || magic[1] != 'I' || magic[2] != 'N' ||
      magic[3] != 'U') {
    gzclose(inFile);
    throw std::runtime_error(
        "Invalid metadata file format: magic number mismatch.");
  }

  // Read version
  read_data(&contents.version, sizeof(contents.version));
  if (contents.version != 1) { // Assuming version 1 for now
    gzclose(inFile);
    throw std::runtime_error("Unsupported metadata file version: " +
                             std::to_string(contents.version));
  }

  // Read num_total_system_offsets, num_total_system_sources,
  // total_slots_in_gemm_buffer
  read_data(&contents.num_total_system_offsets,
            sizeof(contents.num_total_system_offsets));
  read_data(&contents.num_total_system_sources,
            sizeof(contents.num_total_system_sources));
  read_data(&contents.total_slots_in_gemm_buffer,
            sizeof(contents.total_slots_in_gemm_buffer));

  // Read number of active offsets
  read_data(&contents.num_active_offsets_in_map,
            sizeof(contents.num_active_offsets_in_map));

  // Read each active offset's details
  contents.active_offsets_details.resize(contents.num_active_offsets_in_map);
  for (uint32_t i = 0; i < contents.num_active_offsets_in_map; ++i) {
    read_data(&contents.active_offsets_details[i].offset_key, sizeof(uint32_t));
    read_data(&contents.active_offsets_details[i].base_address,
              sizeof(uint32_t));
    read_data(&contents.active_offsets_details[i].num_matches,
              sizeof(uint32_t));
  }

  // Calculate mask size and read masks
  if (contents.num_total_system_offsets > 0 &&
      contents.num_total_system_sources > 0) {
    size_t mask_elements =
        static_cast<size_t>(contents.num_total_system_offsets) *
        contents.num_total_system_sources;
    if (mask_elements > 0) {
      contents.out_mask.resize(mask_elements);
      read_data(contents.out_mask.data(),
                static_cast<unsigned int>(mask_elements * sizeof(int32_t)));

      contents.in_mask.resize(mask_elements);
      read_data(contents.in_mask.data(),
                static_cast<unsigned int>(mask_elements * sizeof(int32_t)));
    }
  } else {
    // Handle cases where masks might be empty if num_total_system_offsets or
    // num_total_system_sources is 0 contents.out_mask and contents.in_mask will
    // remain empty (as default initialized or after resize(0))
  }

  gzclose(inFile);
  return contents;
}

uint32_t write_metadata_cpp(
    const std::vector<int32_t> &out_mask, const std::vector<int32_t> &in_mask,
    const std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>
        &active_offset_data,
    uint32_t num_total_system_offsets, uint32_t num_total_system_sources,
    uint32_t total_slots_in_gemm_buffer, const std::string &filename) {
  gzFile outFile = gzopen(filename.c_str(), "wb");
  if (!outFile) {
    throw std::runtime_error("Failed to open metadata file for writing: " +
                             filename);
  }

  uLong crc = crc32(0L, Z_NULL, 0);
  auto write_data_and_crc = [&](const void *data, unsigned int size) {
    if (gzwrite(outFile, data, size) != static_cast<int>(size)) {
      gzclose(outFile);
      throw std::runtime_error("Failed to write data to metadata gzip file.");
    }
    crc = crc32(crc, static_cast<const Bytef *>(data), size);
  };

  // Magic number "MINU" and version (1) - Little-endian
  char magic[4] = {'M', 'I', 'N', 'U'};
  uint32_t version = 1;
  write_data_and_crc(magic, sizeof(magic));
  write_data_and_crc(&version, sizeof(version));

  // Number of total system offsets and sources - Little-endian
  write_data_and_crc(&num_total_system_offsets,
                     sizeof(num_total_system_offsets));
  write_data_and_crc(&num_total_system_sources,
                     sizeof(num_total_system_sources));

  // Total slots allocated for gemm buffer - Little-endian
  write_data_and_crc(&total_slots_in_gemm_buffer,
                     sizeof(total_slots_in_gemm_buffer));

  // Number of active offsets in map - Little-endian
  uint32_t num_active_offsets =
      static_cast<uint32_t>(active_offset_data.size());
  write_data_and_crc(&num_active_offsets, sizeof(num_active_offsets));

  // For each active offset: Offset key, Base address, Number of matches -
  // Little-endian
  for (const auto &offset_tuple : active_offset_data) {
    uint32_t offset_key = std::get<0>(offset_tuple);
    uint32_t base_address = std::get<1>(offset_tuple);
    uint32_t num_matches = std::get<2>(offset_tuple);
    write_data_and_crc(&offset_key, sizeof(offset_key));
    write_data_and_crc(&base_address, sizeof(base_address));
    write_data_and_crc(&num_matches, sizeof(num_matches));
  }
  std::cout << out_mask.size() << " out_mask elements, "
            << in_mask.size() << " in_mask elements." << std::endl;
  // Masks: Output mask, Input mask (bytes from int32 vector)
  if (!out_mask.empty()) {
    write_data_and_crc(out_mask.data(), static_cast<unsigned int>(
                                            out_mask.size() * sizeof(int32_t)));
  }
  if (!in_mask.empty()) {
    write_data_and_crc(in_mask.data(), static_cast<unsigned int>(
                                           in_mask.size() * sizeof(int32_t)));
  }

  gzclose(outFile);
  return static_cast<uint32_t>(crc);
}

MasksResult create_in_out_masks_cpp(const KernelMapType &kernel_map,
                                    const std::map<uint32_t, int> &slot_dict,
                                    uint32_t num_total_system_offsets,
                                    uint32_t num_total_system_sources) {
  MasksResult result;

  result.out_mask.assign(static_cast<size_t>(num_total_system_offsets) *
                             num_total_system_sources,
                         -1);
  result.in_mask.assign(static_cast<size_t>(num_total_system_offsets) *
                            num_total_system_sources,
                        -1);

  auto sorted_kmap_items =
      kernel_map.get_sorted_items(); // Respects internal sort order

  for (const auto &k_item : sorted_kmap_items) {
    uint32_t off_idx = k_item.first;
    const auto &matches_list =
        k_item.second; // std::vector<std::pair<int, int>>: list of (in_idx,
                       // q_src_idx)
    if (matches_list.empty())
      continue;

    for (size_t match = 0; match < matches_list.size(); ++match) {
      int in_idx = matches_list[match].first;
      result.in_mask[off_idx * num_total_system_sources + in_idx] =
          slot_dict.find(off_idx)->second + match;
      int out_idx = matches_list[match].second;
      result.out_mask[off_idx * num_total_system_sources + out_idx] =
          slot_dict.find(off_idx)->second + match;
    }
  }
  return result;
}
