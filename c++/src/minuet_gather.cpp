// minuet_gather.cpp
#include "minuet_gather.hpp"
#include "minuet_map.hpp"
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

// --- Gather and Scatter Thread Worker Functions ---

// Thread-local storage for memory traces
static thread_local std::vector<MemoryAccessEntry> local_mem_trace;

// Mutex for protecting access to the global memory trace
static std::mutex global_mem_trace_access_mutex;


// Helper function to record memory access locally
void record_local_access_cpp(uint8_t thread_id, const std::string& op_str, uint64_t addr) {
    uint8_t phase_id = PHASES.forward.at(get_curr_phase()); // Assumes get_curr_phase is accessible
    uint8_t op_id = OPS.forward.at(op_str); // Assumes OPS is accessible
    uint8_t tensor_id = addr_to_tensor(addr); // Assumes addr_to_tensor is accessible
    
    local_mem_trace.push_back({phase_id, thread_id, op_id, tensor_id, addr});
}

// Helper function to flush local trace to global trace
void flush_local_trace_cpp() {
    if (!local_mem_trace.empty()) {
        std::lock_guard<std::mutex> lock(global_mem_trace_access_mutex); // Corrected mutex name
        // Assumes mem_trace is the global trace vector, accessible here
        // and that MemoryAccessEntry is the correct type.
        // This might require including "minuet_trace.hpp" for MemoryAccessEntry and mem_trace declaration
        // For now, assuming mem_trace is a global std::vector<MemoryAccessEntry>
        // If mem_trace is not directly accessible, a function like append_to_global_trace might be needed.
         extern std::vector<MemoryAccessEntry> mem_trace; // Make sure this is declared (e.g. in minuet_trace.hpp)
         mem_trace.insert(mem_trace.end(), local_mem_trace.begin(), local_mem_trace.end());
        local_mem_trace.clear();
    }
}


void gather_thread_worker_cpp(
    uint32_t thread_id,
    uint32_t num_threads,
    uint32_t num_points,
    uint32_t num_offsets,
    uint32_t num_tiles_per_pt,
    uint32_t tile_feat_size,
    uint32_t bulk_feat_size,
    const std::vector<int32_t>& source_masks,
    const std::vector<float>& sources,
    std::vector<float>& gemm_buffers) {

    if (tile_feat_size % bulk_feat_size != 0) {
        throw std::invalid_argument("tile_feat_size must be divisible by bulk_feat_size");
    }
    uint32_t num_bulks = tile_feat_size / bulk_feat_size;
    uint64_t total_feats_per_pt = static_cast<uint64_t>(num_tiles_per_pt) * tile_feat_size;

    for (uint32_t pt_idx = thread_id; pt_idx < num_points; pt_idx += num_threads) {
        uint64_t pt_base = static_cast<uint64_t>(pt_idx) * total_feats_per_pt;
        for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_pt; ++tile_idx) {
            uint64_t tile_start_in_source = pt_base + static_cast<uint64_t>(tile_idx) * tile_feat_size;

            for (uint32_t b = 0; b < num_bulks; ++b) {
                uint64_t bulk_start_in_source = tile_start_in_source + b * bulk_feat_size;
                // Record read from IV_BASE
                record_local_access_cpp(static_cast<uint8_t>(thread_id), OPS.inverse.at(0), g_config.IV_BASE + bulk_start_in_source * g_config.SIZE_FEAT);
            }

            for (uint32_t off_idx = 0; off_idx < num_offsets; ++off_idx) {
                uint32_t mask_idx = off_idx * num_points + pt_idx;
                int32_t dest_slot = source_masks[mask_idx];
                if (dest_slot < 0) {
                    continue;
                }

                uint64_t dest_tile_base_in_gemm = static_cast<uint64_t>(dest_slot) * total_feats_per_pt + static_cast<uint64_t>(tile_idx) * tile_feat_size;
                for (uint32_t b = 0; b < num_bulks; ++b) {
                    uint64_t bulk_start_in_source = tile_start_in_source + static_cast<uint64_t>(b) * bulk_feat_size;
                    uint64_t dest_bulk_start_in_gemm = dest_tile_base_in_gemm + static_cast<uint64_t>(b) * bulk_feat_size;

                    if (!sources.empty()) { // Check if sources has data
                        for(uint32_t i = 0; i < bulk_feat_size; ++i) {
                            if ((bulk_start_in_source + i < sources.size()) && (dest_bulk_start_in_gemm + i < gemm_buffers.size())) {
                                gemm_buffers[dest_bulk_start_in_gemm + i] = sources[bulk_start_in_source + i];
                            }
                        }
                    }
                    // Record write to GM_BASE
                    record_local_access_cpp(static_cast<uint8_t>(thread_id), OPS.inverse.at(1), g_config.GM_BASE + dest_bulk_start_in_gemm * g_config.SIZE_FEAT);
                }
            }
        }
    }
    flush_local_trace_cpp();
}

void scatter_thread_worker_cpp(
    uint32_t thread_id,
    uint32_t num_threads,
    uint32_t num_points, // Number of *output* points
    uint32_t num_offsets,
    uint32_t num_tiles_per_pt,
    uint32_t tile_feat_size,
    uint32_t bulk_feat_size,
    const std::vector<int32_t>& out_mask,
    const std::vector<float>& gemm_buffers,
    std::vector<float>& outputs) {

    if (tile_feat_size % bulk_feat_size != 0) {
        throw std::invalid_argument("tile_feat_size must be divisible by bulk_feat_size");
    }
    int num_bulks = tile_feat_size / bulk_feat_size;
    uint64_t total_feats_per_pt = static_cast<uint64_t>(num_tiles_per_pt) * tile_feat_size;
    std::vector<float> tile_data_temp(tile_feat_size); // Temporary buffer for one tile

    for (uint32_t pt_idx = thread_id; pt_idx < num_points; pt_idx += num_threads) { // pt_idx is output point index
        uint64_t dest_pt_base = static_cast<uint64_t>(pt_idx) * total_feats_per_pt;

        for (int off_idx = 0; off_idx < num_offsets; ++off_idx) {
            int mask_idx = off_idx * num_points + pt_idx;
            int32_t source_slot = out_mask[mask_idx];
            if (source_slot == -1) {
                continue;
            }

            for (int tile_idx = 0; tile_idx < num_tiles_per_pt; ++tile_idx) {
                uint64_t source_tile_base = static_cast<uint64_t>(source_slot) * total_feats_per_pt + static_cast<uint64_t>(tile_idx) * tile_feat_size;
                
                // Phase 1: Read entire source tile from gemm_buffers into tile_data_temp
                for (int b = 0; b < num_bulks; ++b) {
                    uint64_t bulk_offset_in_tile = static_cast<uint64_t>(b) * bulk_feat_size;
                    uint64_t source_bulk_addr_in_gemm = source_tile_base + bulk_offset_in_tile;
                    record_local_access_cpp(static_cast<uint8_t>(thread_id), OPS.inverse.at(0), g_config.GM_BASE + source_bulk_addr_in_gemm * g_config.SIZE_FEAT);
                    if (!gemm_buffers.empty()) {
                         for(int i = 0; i < bulk_feat_size; ++i) {
                            if ((source_bulk_addr_in_gemm + i < gemm_buffers.size()) && (bulk_offset_in_tile + i < tile_data_temp.size())) {
                                tile_data_temp[bulk_offset_in_tile + i] = gemm_buffers[source_bulk_addr_in_gemm + i];
                            }
                        }
                    }
                }

                // Phase 2: Write from tile_data_temp to outputs array
                uint64_t dest_tile_base_in_output = dest_pt_base + static_cast<uint64_t>(tile_idx) * tile_feat_size;
                for (int b = 0; b < num_bulks; ++b) {
                    uint64_t bulk_offset_in_tile = static_cast<uint64_t>(b) * bulk_feat_size;
                    uint64_t dest_bulk_addr_in_output = dest_tile_base_in_output + bulk_offset_in_tile;
                    record_local_access_cpp(static_cast<uint8_t>(thread_id), OPS.inverse.at(1), g_config.IV_BASE + dest_bulk_addr_in_output * g_config.SIZE_FEAT);
                     if (!outputs.empty() && !gemm_buffers.empty()) { // Check if outputs and gemm_buffers (implies tile_data_temp is valid) have data
                        for(int i = 0; i < bulk_feat_size; ++i) {
                             if ((dest_bulk_addr_in_output + i < outputs.size()) && (bulk_offset_in_tile + i < tile_data_temp.size())) {
                                outputs[dest_bulk_addr_in_output + i] += tile_data_temp[bulk_offset_in_tile + i]; // Accumulate
                            }
                        }
                    }
                }
            }
        }
    }
    flush_local_trace_cpp();
}


// --- Main Gather and Scatter Functions ---
void mt_gather_cpp(
    uint32_t num_threads,
    uint32_t num_points,
    uint32_t num_offsets,
    uint32_t num_tiles_per_pt,
    uint32_t tile_feat_size,
    uint32_t bulk_feat_size,
    const std::vector<int32_t>& source_masks,
    const std::vector<float>& sources,
    std::vector<float>& gemm_buffers) {

    set_curr_phase(PHASES.inverse.at(5)); // Assuming GTH is 5, needs to be added to PHASES map if not present
                                          // Or use a new phase string e.g. "GTH_CPP" and add to PHASES

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(
            gather_thread_worker_cpp,
            i, num_threads, num_points, num_offsets, num_tiles_per_pt,
            tile_feat_size, bulk_feat_size, std::ref(source_masks),
            std::ref(sources), std::ref(gemm_buffers));
    }

    for (auto& t : threads) {
        t.join();
    }
    set_curr_phase(""); // Clear phase
}

void mt_scatter_cpp(
    uint32_t num_threads,
    uint32_t num_points, // Number of *output* points
    uint32_t num_offsets,
    uint32_t num_tiles_per_pt,
    uint32_t tile_feat_size,
    uint32_t bulk_feat_size,
    const std::vector<int32_t>& out_mask,
    const std::vector<float>& gemm_buffers,
    std::vector<float>& outputs) {

    set_curr_phase(PHASES.inverse.at(6)); // Assuming SCT is 6, needs to be added to PHASES map
                                          // Or use a new phase string e.g. "SCT_CPP" and add to PHASES

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(
            scatter_thread_worker_cpp,
            i, num_threads, num_points, num_offsets, num_tiles_per_pt,
            tile_feat_size, bulk_feat_size, std::ref(out_mask),
            std::ref(gemm_buffers), std::ref(outputs));
    }

    for (auto& t : threads) {
        t.join();
    }
    set_curr_phase(""); // Clear phase
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
