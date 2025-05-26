#include "minuet_trace.hpp"
#include <algorithm>
#include <cmath> // For std::ceil in progress reporting
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>

// --- Global Variable Definitions ---
std::vector<MemoryAccessEntry> mem_trace; // Updated name
std::string curr_phase = "";              // Updated name

// --- Getter/Setter for global state and mem_trace management ---
std::vector<MemoryAccessEntry> get_mem_trace() {
    return mem_trace;
}

void clear_mem_trace() {
    mem_trace.clear();
}

void set_curr_phase(const std::string& phase_name) {
    curr_phase = phase_name;
}

std::string get_curr_phase() {
    return curr_phase;
}

void set_debug_flag(bool debug_val) {
    g_config.debug = debug_val; // Set in global config
}

bool get_debug_flag() {
    return g_config.debug; // Get from global config
}

// Global constant maps (matching Python names for clarity)
// Using the bidict class defined in the header
bidict<std::string, int>
    PHASES({{"RDX", 0}, {"QRY", 1}, {"SRT", 2}, {"PVT", 3}, {"LKP", 4}});

bidict<std::string, int> TENSORS({
    {"I", 0},
    {"QK", 1},
    {"QI", 2},
    {"QO", 3},
    {"PIV", 4},
    {"KM", 5},
    {"WO", 6},
    {"TILE", 7} // TILE is I_BASE, handled in addr_to_tensor
});

bidict<std::string, int> OPS({{"R", 0}, {"W", 1}});

std::string to_hex_string(uint64_t val) {
    std::stringstream ss;
    ss << "0x" << std::hex << val;
    return ss.str();
}


// --- Memory Tracing Functions ---
std::string addr_to_tensor(uint64_t addr) { // Renamed
  // Order of checks matters, more specific ranges first.
  if (addr >= g_config.WO_BASE && addr < g_config.IV_BASE)
    return TENSORS.inverse.at(TENSORS.forward.at("WO"));
  if (addr >= g_config.KM_BASE && addr < g_config.WO_BASE)
    return TENSORS.inverse.at(TENSORS.forward.at("KM"));
  if (addr >= g_config.PIV_BASE && addr < g_config.KM_BASE)
    return TENSORS.inverse.at(TENSORS.forward.at("PIV"));
  if (addr >= g_config.QO_BASE && addr < g_config.PIV_BASE)
    return TENSORS.inverse.at(TENSORS.forward.at("QO"));
  if (addr >= g_config.QI_BASE && addr < g_config.QO_BASE)
    return TENSORS.inverse.at(TENSORS.forward.at("QI"));
  if (addr >= g_config.QK_BASE && addr < g_config.QI_BASE)
    return TENSORS.inverse.at(TENSORS.forward.at("QK"));
  // I_BASE and TILE_BASE are the same. If it's in this range and not caught
  // above, it's I or TILE. Python logic implies TILE is an alias for I, so we
  // can map to "I" or "TILE". Let's be consistent with Python's trace which
  // might show TILE for reads from I_BASE during tiling. The Python
  // addr_to_tensor has a TILE_BASE check that could map to "TILE". Given
  // TILE_BASE == I_BASE, this check needs to be specific if TILE operations are
  // distinct. For now, let's assume if it falls into I_BASE range, it's "I". If
  // specific tile operations need to be logged as "TILE", the logic might need
  // adjustment based on when those operations occur, possibly by setting
  // curr_phase or another indicator.
  if (addr >= g_config.I_BASE && addr < g_config.QK_BASE) { // QK_BASE is the next boundary after
                                          // I_BASE for distinct tensors
    // If we are in a phase that specifically reads tiles, we might label it
    // TILE. For now, stick to the direct address range mapping like Python's
    // intial checks.
    return TENSORS.inverse.at(TENSORS.forward.at("I"));
  }
  // IV_BASE and WV_BASE are for features, not typically in the mapping phase
  // trace but included for completeness if other operations use them.
  if (addr >= g_config.IV_BASE && addr < g_config.WV_BASE)
    return "IV"; // Not in TENSORS map, handle as string
  if (addr >= g_config.WV_BASE)
    return "WV"; // Not in TENSORS map, handle as string

  return "Unknown"; // Default if no range matches
}

void write_gmem_trace(const std::string &filename) {
  // Create mappings for strings to integers (already done by bidict)
  std::vector<std::tuple<uint8_t, uint8_t, uint8_t, uint8_t, uint32_t>>
      compressed_trace_data;

  for (const auto &entry : mem_trace) {
    uint8_t phase_id = PHASES.at_key(entry.phase);
    uint8_t op_id = OPS.at_key(entry.op);
    uint8_t tensor_id;
    try {
      tensor_id = TENSORS.at_key(entry.tensor);
    } catch (const std::out_of_range &oor) {
      // Handle tensors like "IV", "WV", "Unknown" that are not in the TENSORS
      // bidict For simplicity, assign a placeholder or extend TENSORS if these
      // are common
      if (entry.tensor == "IV")
        tensor_id = 99; // Example placeholder
      else if (entry.tensor == "WV")
        tensor_id = 98;
      else
        tensor_id = 255; // Placeholder for "Unknown"
    }
    uint32_t addr_int = static_cast<uint32_t>(
        entry.addr); // Python converts hex string from trace

    compressed_trace_data.emplace_back(phase_id,
                                       static_cast<uint8_t>(entry.thread_id),
                                       op_id, tensor_id, addr_int);
  }

  gzFile outFile = gzopen(filename.c_str(), "wb");
  if (!outFile) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }

  uint32_t num_entries = static_cast<uint32_t>(compressed_trace_data.size());
  if (gzwrite(outFile, &num_entries, sizeof(num_entries)) !=
      sizeof(num_entries)) {
    gzclose(outFile);
    throw std::runtime_error("Failed to write number of entries to gzip file.");
  }

  for (const auto &centry : compressed_trace_data) {
    uint8_t phase_val = std::get<0>(centry);
    uint8_t tid_val = std::get<1>(centry);
    uint8_t op_val = std::get<2>(centry);
    uint8_t tensor_val = std::get<3>(centry);
    uint32_t addr_val = std::get<4>(centry);
    if (gzwrite(outFile, &phase_val, sizeof(phase_val)) != sizeof(phase_val) ||
        gzwrite(outFile, &tid_val, sizeof(tid_val)) != sizeof(tid_val) ||
        gzwrite(outFile, &op_val, sizeof(op_val)) != sizeof(op_val) ||
        gzwrite(outFile, &tensor_val, sizeof(tensor_val)) !=
            sizeof(tensor_val) ||
        gzwrite(outFile, &addr_val, sizeof(addr_val)) != sizeof(addr_val)) {
      gzclose(outFile);
      throw std::runtime_error("Failed to write entry to gzip file.");
    }
  }
  gzclose(outFile);

  std::cout << "Memory trace written to " << filename << std::endl;
  std::cout << "Compressed " << mem_trace.size() << " entries"
            << std::endl; // Use mem_trace
  // The phase_map printout in Python is for the local map created in that
  // function. Since we use a global PHASES bidict, we don't need to print a
  // dynamic mapping here. If a dynamic mapping was truly needed (e.g. for new
  // phases discovered at runtime), the logic would be different.
}

void record_access(int thread_id, const std::string &op, uint64_t addr) {
  std::string tensor_str = addr_to_tensor(addr);
  mem_trace.push_back({curr_phase, thread_id, op, tensor_str, addr});
}

// --- Algorithm Phases ---
// Radix Sort (Simplified: only records accesses, actual sort not fully
// implemented for brevity) Matches Python's radix_sort_with_memtrace which
// focuses on memory access patterns.
std::vector<uint32_t> radix_sort_with_memtrace(std::vector<uint32_t> &arr,
                                               uint64_t base_addr) {
  const int passes = 4;
  size_t N = arr.size();
  if (N == 0)
    return arr;

  // std::vector<uint32_t> aux(N); // Conceptually for sorting

  for (int p = 0; p < passes; ++p) {
    // Simulate the part of radix sort that builds the auxiliary array.
    // Python's version:
    // for i in range(N - 1, -1, -1):
    //     record_access(t_id, 'R', base + i*SIZE_KEY) // First read
    //     val = arr[i]
    //     # ... logic to get byte_val and target aux_idx using counts ...
    //     record_access(t_id, 'R', base + i*SIZE_KEY) // Second read (of arr[i]
    //     / val) aux[aux_idx] = val record_access(t_id, 'W', base +
    //     aux_idx*SIZE_KEY) // Write to aux

    // We simulate N elements being processed.
    for (size_t i = 0; i < N; ++i) {
      int t_id = static_cast<int>(i % g_config.NUM_THREADS);
      // First read of an element from arr
      record_access(t_id, OPS.inverse.at(0), base_addr + i * g_config.SIZE_KEY); // "R"
    }
    for (size_t i = 0; i < N; ++i) {
      int t_id = static_cast<int>(i % g_config.NUM_THREADS);
      // Second read of the same element from arr
      record_access(t_id, OPS.inverse.at(0), base_addr + i * g_config.SIZE_KEY); // "R"

      // Write to auxiliary array (simulated position)
      // In a real sort, this write would be to aux[calculated_pos]
      // For simulation, we record a write to a conceptual 'aux' region,
      // using 'i' as a proxy for calculated_pos to ensure N writes.
      // The base address for aux is assumed to be the same as arr for this
      // trace.
      record_access(t_id, OPS.inverse.at(1), base_addr + i * g_config.SIZE_KEY); // "W"
    }
    // arr, aux = aux, arr // Conceptually, data is swapped or copied back
    // If arr is swapped with aux, the next pass reads from what was aux.
    // The base address remains 'base_addr' for tracing purposes of this logical
    // array.
  }
  return arr;
}

std::vector<IndexedCoord>
compute_unique_sorted_coords(const std::vector<Coord3D> &in_coords,
                             int stride) {
  curr_phase = PHASES.inverse.at(0); // "RDX"

  std::vector<std::pair<uint32_t, int>>
      idx_keys_pairs; // Stores (key, original_index)
  idx_keys_pairs.reserve(in_coords.size());

  for (size_t idx = 0; idx < in_coords.size(); ++idx) {
    const auto &coord = in_coords[idx];
    // Python: record_access(idx % NUM_THREADS, 'W', I_BASE + idx * SIZE_KEY)
    // This write is for the initial list of idx_keys before sorting.
    // Let's assume I_BASE is the start of a conceptual array for these packed
    // keys. record_access(static_cast<int>(idx % NUM_THREADS),
    // OPS.inverse.at(1), g_config.I_BASE + idx * g_config.SIZE_KEY); // "W"
    Coord3D qtz_coord = coord.quantized(stride);
    idx_keys_pairs.emplace_back(qtz_coord.to_key(), static_cast<int>(idx));
  }

  // Extract raw keys for memory trace simulation of radix sort
  std::vector<uint32_t> raw_keys;
  raw_keys.reserve(idx_keys_pairs.size());
  for (const auto &pair : idx_keys_pairs) {
    raw_keys.push_back(pair.first);
  }
  // The base address for radix sort in Python is I_BASE.
  radix_sort_with_memtrace(raw_keys, g_config.I_BASE);

  // Sort idx_keys_pairs by key, preserving original index for tie-breaking
  // (std::stable_sort if needed) Python's `sorted(idx_keys, key=lambda item:
  // item[0])` is stable.
  std::stable_sort(
      idx_keys_pairs.begin(), idx_keys_pairs.end(),
      [](const auto &a, const auto &b) { return a.first < b.first; });

  std::vector<IndexedCoord> uniq_coords_vec; // Renamed from uniq_coords
  if (idx_keys_pairs.empty()) {
    return uniq_coords_vec;
  }

  uint32_t last_key = idx_keys_pairs[0].first;
  int orig_idx_from_input = idx_keys_pairs[0].second;
  // Python: record_access(0 % NUM_THREADS, 'R', I_BASE + 0 * SIZE_KEY)
  // This read is for accessing the sorted key during deduplication.
  record_access(0 % g_config.NUM_THREADS, OPS.inverse.at(0),
                g_config.I_BASE + 0 * g_config.SIZE_KEY); // "R"
  uniq_coords_vec.emplace_back(Coord3D::from_key(last_key),
                               orig_idx_from_input);

  for (size_t i = 1; i < idx_keys_pairs.size(); ++i) {
    // record_access(static_cast<int>(i % NUM_THREADS), OPS.inverse.at(0),
    //               g_config.I_BASE + i * g_config.SIZE_KEY); // "R"
    if (idx_keys_pairs[i].first != last_key) {
      last_key = idx_keys_pairs[i].first;
      orig_idx_from_input = idx_keys_pairs[i].second;
      uniq_coords_vec.emplace_back(Coord3D::from_key(last_key),
                                   orig_idx_from_input);
    }
  }

  if (g_config.debug) {
    std::cout << "Unique sorted coordinates (count: " << uniq_coords_vec.size()
              << ")" << std::endl;
    for (const auto &ic : uniq_coords_vec) {
      std::cout << "  Key: " << to_hex_string(ic.to_key())
                << ", Coord: " << ic.coord << ", Orig Idx: " << ic.orig_idx
                << std::endl;
    }
  }
  return uniq_coords_vec;
}

BuildQueriesResult
build_coordinate_queries(const std::vector<IndexedCoord> &uniq_coords,
                         int stride, // stride is not used in python version
                         const std::vector<Coord3D> &off_coords) {
  curr_phase = PHASES.inverse.at(1); // "QRY"
  size_t num_inputs = uniq_coords.size();
  size_t num_offsets = off_coords.size();
  size_t total_queries = num_inputs * num_offsets;

  BuildQueriesResult result;
  result.qry_keys.resize(total_queries);
  result.qry_in_idx.resize(total_queries);
  result.qry_off_idx.resize(total_queries);
  result.wt_offsets.resize(
      total_queries); // Python uses [None] * total_queries initially

  for (size_t off_idx = 0; off_idx < num_offsets; ++off_idx) {
    const auto &offset_val = off_coords[off_idx]; // dx, dy, dz
    for (size_t in_idx = 0; in_idx < num_inputs; ++in_idx) {
      const auto &indexed_coord_item = uniq_coords[in_idx];
      size_t glob_idx = off_idx * num_inputs + in_idx;

      // Python version does not record accesses in this function.
      // Removing record_access calls.
      // record_access(static_cast<int>(glob_idx % NUM_THREADS),
      // OPS.inverse.at(0), g_config.I_BASE + in_idx * g_config.SIZE_KEY); // "R"

      Coord3D qk_coord = indexed_coord_item.coord + offset_val;
      result.qry_keys[glob_idx] =
          IndexedCoord(qk_coord, indexed_coord_item.orig_idx);
      result.qry_in_idx[glob_idx] = static_cast<int>(in_idx);
      result.qry_off_idx[glob_idx] = static_cast<int>(off_idx);
      result.wt_offsets[glob_idx] = offset_val; // Store the Coord3D offset

      // Removing record_access calls for writes.
      // record_access(static_cast<int>(glob_idx % NUM_THREADS),
      // OPS.inverse.at(1), g_config.QK_BASE + glob_idx * g_config.SIZE_KEY);    // "W" for
      // qry_key record_access(static_cast<int>(glob_idx % NUM_THREADS),
      // OPS.inverse.at(1), g_config.QI_BASE + glob_idx * g_config.SIZE_INT);    // "W" for
      // qry_in_idx record_access(static_cast<int>(glob_idx % NUM_THREADS),
      // OPS.inverse.at(1), g_config.QO_BASE + glob_idx * g_config.SIZE_INT);    // "W" for
      // qry_off_idx record_access(static_cast<int>(glob_idx % NUM_THREADS),
      // OPS.inverse.at(1), g_config.WO_BASE + glob_idx * g_config.SIZE_WEIGHT); // "W" for
      // wt_offset (packed key)
    }
  }
  return result;
}

TilesPivotsResult
create_tiles_and_pivots(const std::vector<IndexedCoord> &uniq_coords,
                        int tile_size_param) // Renamed from tile_size to avoid
                                             // conflict with local var
{
  curr_phase = PHASES.inverse.at(3); // "PVT"
  TilesPivotsResult result;
  int current_tile_size = tile_size_param;

  if (uniq_coords.empty()) {
    if (g_config.debug)
      std::cout << "Skipping tile creation, no unique coordinates."
                << std::endl;
    return result;
  }

  if (current_tile_size <= 0) { // Python: if tile_size is None or <= 0
    current_tile_size = static_cast<int>(uniq_coords.size());
    if (current_tile_size == 0) { // Still could be zero if uniq_coords was
                                  // empty and handled above, but defensive
      if (g_config.debug)
        std::cout << "Tile size is zero, cannot create tiles." << std::endl;
      return result; // No tiles can be made
    }
    if (g_config.debug)
      std::cout << "Tile size not specified or invalid, using full range: "
                << current_tile_size << std::endl;
  }

  for (size_t start = 0; start < uniq_coords.size();
       start += current_tile_size) {
    size_t end = std::min(start + current_tile_size, uniq_coords.size());
    std::vector<IndexedCoord> current_tile;
    current_tile.reserve(end - start);
    for (size_t i = start; i < end; ++i) {
      // Read unique coordinate for tiling
      // Python: record_access(i % NUM_THREADS, 'R', I_BASE + i * SIZE_KEY)
      // record_access(static_cast<int>(i % NUM_THREADS), OPS.inverse.at(0),
      // g_config.I_BASE + i * g_config.SIZE_KEY); // "R"
      current_tile.push_back(uniq_coords[i]);

      // Write to conceptual tile data region (simulated)
      // Python: record_access(i % NUM_THREADS, 'W', TILE_BASE + i * SIZE_KEY)
      // TILE_BASE is an alias for I_BASE. This implies an in-place usage or
      // conceptual copy. record_access(static_cast<int>(i % NUM_THREADS),
      // OPS.inverse.at(1), g_config.TILE_BASE + i * g_config.SIZE_KEY); // "W"
    }
    result.tiles.push_back(current_tile);

    // Add pivot: the first element of the tile
    // Python: record_access(start % NUM_THREADS, 'R', I_BASE + start *
    // SIZE_KEY) record_access(static_cast<int>(start % NUM_THREADS),
    // OPS.inverse.at(0), g_config.I_BASE + start * g_config.SIZE_KEY); // "R" for pivot
    result.pivots.push_back(uniq_coords[start]);
    // Write pivot key
    // Python: record_access( (start // tile_size) % NUM_THREADS, 'W', PIV_BASE
    // + (start // tile_size) * SIZE_KEY)
    record_access(static_cast<int>((start / current_tile_size) % g_config.NUM_THREADS),
                  OPS.inverse.at(1),
                  g_config.PIV_BASE + (start / current_tile_size) * g_config.SIZE_KEY); // "W"
  }
  return result;
}

KernelMap perform_coordinate_lookup( // Renamed from lookup
    const std::vector<IndexedCoord> &uniq_coords,
    const std::vector<IndexedCoord> &qry_keys,
    const std::vector<int> &qry_in_idx, const std::vector<int> &qry_off_idx,
    const std::vector<Coord3D> &wt_offsets,
    const std::vector<std::vector<IndexedCoord>> &tiles,
    const std::vector<IndexedCoord> &pivs, int tile_size_param) {
  curr_phase = PHASES.inverse.at(4); // "LKP"
  KernelMap kmap;
  uint64_t kmap_write_idx = 0; // Local counter for KM writes, like Python

  if (uniq_coords.empty() || qry_keys.empty()) {
    return kmap;
  }

  for (size_t q_glob_idx = 0; q_glob_idx < qry_keys.size(); ++q_glob_idx) {
    const auto &q_key_item = qry_keys[q_glob_idx];
    uint32_t current_query_key = q_key_item.to_key();
    int query_original_src_idx = q_key_item.orig_idx;
    int current_offset_idx =
        qry_off_idx[q_glob_idx]; // Index for off_coords / wt_offsets
    int tid = static_cast<int>(q_glob_idx % g_config.NUM_THREADS);

    // 1. Read query key
    record_access(tid, OPS.inverse.at(0),
                  g_config.QK_BASE + q_glob_idx * g_config.SIZE_KEY); // "R"

    // Simulate Python's find_tile_id (binary search on pivs)
    int target_tile_id = -1;
    if (!pivs.empty()) {
      int low = 0, high = static_cast<int>(pivs.size()) - 1;
      target_tile_id =
          0; // Default to first tile if not found or q_key is smaller

      while (low <= high) {
        int mid = low + (high - low) / 2;
        record_access(tid, OPS.inverse.at(0),
                      g_config.PIV_BASE + mid * g_config.SIZE_KEY); // "R" pivot
        if (pivs[mid].to_key() <= current_query_key) {
          target_tile_id = mid;
          low = mid + 1;
        } else {
          high = mid - 1;
        }
      }
    }

    // Simulate Python's search_in_tile (binary search in tiles[target_tile_id])
    int found_uc_idx = -1; // Index in uniq_coords
    if (target_tile_id != -1 &&
        target_tile_id < static_cast<int>(tiles.size())) {
      const auto &current_tile = tiles[target_tile_id];
      if (!current_tile.empty()) {
        int low = 0, high = static_cast<int>(current_tile.size()) - 1;
        while (low <= high) {
          int mid_local = low + (high - low) / 2;
          // Calculate the conceptual address in the TILE_BASE region
          // The tile_id * tile_size gives the start index in the original
          // uniq_coords Python: TILE_BASE + (tile_id * tile_size + mid_local) *
          // SIZE_KEY Here, current_tile[mid_local] is an IndexedCoord. Its
          // original position in uniq_coords would be roughly target_tile_id *
          // tile_size_param + mid_local if tiles are contiguous slices. For
          // simplicity in tracing, we use an estimated index. The key is to
          // record a read from TILE_BASE.
          size_t approx_tile_element_orig_idx =
              static_cast<size_t>(target_tile_id * tile_size_param + mid_local);
          if (approx_tile_element_orig_idx >=
              uniq_coords.size()) { // Boundary check
            approx_tile_element_orig_idx =
                uniq_coords.size() > 0 ? uniq_coords.size() - 1 : 0;
          }

          record_access(tid, OPS.inverse.at(0),
                        g_config.TILE_BASE + approx_tile_element_orig_idx *
                                        g_config.SIZE_KEY); // "R" from TILE

          if (current_tile[mid_local].to_key() == current_query_key) {
            // To get the index relative to the original uniq_coords array:
            // We need to know how tiles map back to uniq_coords indices.
            // Assuming tiles are contiguous blocks of uniq_coords:
            // The first element of tiles[target_tile_id] corresponds to
            // uniq_coords[target_tile_id * tile_size_param]
            found_uc_idx = static_cast<int>(
                static_cast<size_t>(target_tile_id * tile_size_param) +
                mid_local);
            if (found_uc_idx >= static_cast<int>(uniq_coords.size()) ||
                uniq_coords[found_uc_idx].to_key() != current_query_key) {
              // Fallback if indexing assumption is wrong, search linearly in
              // uniq_coords for the actual object This part is tricky without
              // knowing the exact structure of tiles vs uniq_coords For now, we
              // assume found_uc_idx is correct if key matches. The Python
              // version's search_in_tile returns the element itself, not its
              // global index. Let's assume current_tile[mid_local] is the
              // matched IndexedCoord from uniq_coords. We need its original_idx
              // field.
              for (size_t uc_scan_idx = 0; uc_scan_idx < uniq_coords.size();
                   ++uc_scan_idx) {
                if (uniq_coords[uc_scan_idx].to_key() == current_query_key &&
                    uniq_coords[uc_scan_idx].orig_idx ==
                        current_tile[mid_local].orig_idx) {
                  found_uc_idx = static_cast<int>(uc_scan_idx);
                  break;
                }
              }
            }
            break;
          } else if (current_tile[mid_local].to_key() < current_query_key) {
            low = mid_local + 1;
          } else {
            high = mid_local - 1;
          }
        }
      }
    }

    if (found_uc_idx != -1 &&
        found_uc_idx < static_cast<int>(uniq_coords.size())) {
      // Match found

      kmap[current_offset_idx].emplace_back(uniq_coords[found_uc_idx].orig_idx,
                                          query_original_src_idx);

      // Record write to kernel map (Python: KM_BASE + kmap_write_idx *
      // (SIZE_INT + SIZE_INT))
      record_access(tid, OPS.inverse.at(1),
                    g_config.KM_BASE + kmap_write_idx * (g_config.SIZE_INT + g_config.SIZE_INT)); // "W"
      kmap_write_idx++;
    }
  }
  return kmap;
}

void write_kernel_map_to_gz(
    const KernelMap &kmap_data, const std::string &filename,
    const std::vector<Coord3D>
        &off_list 
) {
  gzFile outFile = gzopen(filename.c_str(), "wb");
  if (!outFile) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }

  // Calculate total number of entries (pairs) across all offsets
  uint32_t num_total_entries = 0;
  for (const auto &pair : kmap_data) {
    num_total_entries += static_cast<uint32_t>(pair.second.size());
  }

  if (gzwrite(outFile, &num_total_entries, sizeof(num_total_entries)) !=
      sizeof(num_total_entries)) {
    gzclose(outFile);
    throw std::runtime_error(
        "Failed to write number of entries to kernel map gzip file.");
  }

  // Iterate through the map (sorted by offset_idx due to std::map)
  for (const auto &pair : kmap_data) {
    uint32_t offset_idx = pair.first; // This is the integer index for off_list

    if (offset_idx >= off_list.size()) {
        std::cerr << "Error in write_kernel_map_to_gz: offset_idx " << offset_idx 
                  << " is out of bounds for off_list (size " << off_list.size() 
                  << "). Skipping this kmap entry." << std::endl;
        continue;
    }
    const Coord3D& actual_offset_coord = off_list[offset_idx];
    uint32_t packed_offset_key_to_write = actual_offset_coord.to_key(); 

    const auto &matches =
        pair.second; // vector of (input_idx, query_src_orig_idx)

    for (const auto &match : matches) {
      uint32_t input_idx = static_cast<uint32_t>(match.first);
      uint32_t query_src_orig_idx = static_cast<uint32_t>(match.second);

      // Write: packed_offset_key_to_write, input_idx, query_src_orig_idx
      if (gzwrite(outFile, &packed_offset_key_to_write, sizeof(packed_offset_key_to_write)) !=
              sizeof(packed_offset_key_to_write) ||
          gzwrite(outFile, &input_idx, sizeof(input_idx)) !=
              sizeof(input_idx) ||
          gzwrite(outFile, &query_src_orig_idx, sizeof(query_src_orig_idx)) !=
              sizeof(query_src_orig_idx)) {
        gzclose(outFile);
        throw std::runtime_error(
            "Failed to write kernel map entry to gzip file.");
      }
    }
  }

  gzclose(outFile);
  std::cout << "Kernel map successfully written to " << filename << " with "
            << num_total_entries << " entries." << std::endl;
}
