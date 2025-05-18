#include "minuet_trace.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <zlib.h>
#include <iostream>
#include <iomanip>
#include <cmath> // For std::ceil in progress reporting

// --- Global Variable Definitions ---
std::vector<MemoryAccessEntry> mem_trace; // Updated name
std::string curr_phase = "";              // Updated name
bool debug = true;

// Global constant maps (matching Python names for clarity)
const std::map<std::string, int> phases_map_global = { // Renamed from phases
    {"Radix-Sort", 0}, {"Build-Queries", 1}, {"Sort-QKeys", 2},
    {"Tile-Pivots", 3}, {"Lookup", 4}, {"Lookup-Backward", 5},
    {"Lookup-Forward", 6}, {"Dedup", 7}
};
const std::map<std::string, int> tensors_map_global = { // Renamed from tensors
    {"I", 0}, {"QK", 1}, {"QI", 2}, {"QO", 3}, {"PIV", 4},
    {"KM", 5}, {"WO", 6}, {"TILE", 7}, {"Unknown", 8},
    {"IV", 9}, {"WV", 10}
};
const std::map<std::string, int> ops_map_global = { // Renamed from ops
    {"R", 0}, {"W", 1}
};

// --- Coord3D method definitions ---
uint32_t Coord3D::to_key() const {
    return pack32(x, y, z);
}

Coord3D Coord3D::from_key(uint32_t key) {
    auto [ux, uy, uz] = unpack32s(key); // Using signed unpack consistently
    return Coord3D(ux, uy, uz);
}

// --- Packing/Unpacking (10-bit fields) ---
uint32_t pack32(int c1, int c2, int c3) {
    // Packs three 10-bit integer coordinates into a single 30-bit key within a uint32_t.
    // c1: bits 20-29, c2: bits 10-19, c3: bits 0-9
    uint32_t key = 0;
    key = (key << 10) | (static_cast<uint32_t>(c1) & 0x3FF);
    key = (key << 10) | (static_cast<uint32_t>(c2) & 0x3FF);
    key = (key << 10) | (static_cast<uint32_t>(c3) & 0x3FF);
    return key;
}

std::tuple<int, int, int> unpack32(uint32_t key) {
    // Unpacks a 30-bit key (stored in uint32_t) into three 10-bit integer coordinates.
    // Assumes c3 is LSB, c1 is MSB of the 30-bit value.
    int c3 = static_cast<int>(key & 0x3FF);
    key >>= 10;
    int c2 = static_cast<int>(key & 0x3FF);
    key >>= 10;
    int c1 = static_cast<int>(key & 0x3FF);
    return std::make_tuple(c1, c2, c3);
}

std::tuple<int, int, int> unpack32s(uint32_t key) {
    // Unpacks a 30-bit key into three 10-bit signed integers.
    // Sign extension for 10-bit numbers: if value >= 512 (0x200), it's negative. Subtract 1024 (0x400).
    uint32_t temp_key = key;

    int c3_val = static_cast<int>(temp_key & 0x3FF);
    c3_val = (c3_val < 512) ? c3_val : c3_val - 1024;
    temp_key >>= 10;

    int c2_val = static_cast<int>(temp_key & 0x3FF);
    c2_val = (c2_val < 512) ? c2_val : c2_val - 1024;
    temp_key >>= 10;

    int c1_val = static_cast<int>(temp_key & 0x3FF);
    c1_val = (c1_val < 512) ? c1_val : c1_val - 1024;

    return std::make_tuple(c1_val, c2_val, c3_val);
}

// --- Memory Tracing Functions ---
std::string to_hex_string(uint64_t val) {
    std::stringstream ss;
    ss << "0x" << std::hex << val;
    return ss.str();
}

std::string addr_to_tensor(uint64_t addr) { // Renamed
    if (addr >= I_BASE && addr < QK_BASE) return "I";
    else if (addr >= QK_BASE && addr < QI_BASE) return "QK";
    else if (addr >= QI_BASE && addr < QO_BASE) return "QI";
    else if (addr >= QO_BASE && addr < PIV_BASE) return "QO";
    // PIV_BASE (0x5) vs TILE_BASE (0x1). Original Python PIV rule was effectively dead.
    // The cascade means TILE covers a broad range if not caught by earlier specific ones.
    else if (addr >= TILE_BASE && addr < KM_BASE) return "TILE"; // TILE_BASE is I_BASE
    else if (addr >= KM_BASE && addr < WO_BASE) return "KM";
    else if (addr >= WO_BASE && addr < IV_BASE) return "WO"; // Python used WV_BASE as upper for WO
    else if (addr >= IV_BASE && addr < WV_BASE) return "IV";
    else if (addr >= WV_BASE) return "WV";
    else return "Unknown";
}

void write_gmem_trace(const std::string& filename) {
    std::map<std::string, uint8_t> local_phase_map; // Renamed from phase_map
    std::vector<std::tuple<uint8_t, uint8_t, uint8_t, uint8_t, uint32_t>> compressed_trace_data; // Renamed

    for (const auto& entry : mem_trace) { // Use mem_trace
        uint8_t phase_id;
        if (local_phase_map.find(entry.phase) == local_phase_map.end()) {
            local_phase_map[entry.phase] = static_cast<uint8_t>(local_phase_map.size());
        }
        phase_id = local_phase_map.at(entry.phase);

        uint8_t op_id = ops_map_global.at(entry.op);
        uint8_t tensor_id = tensors_map_global.at(entry.tensor);
        uint32_t addr_int = static_cast<uint32_t>(entry.addr); // Python converts hex string from trace

        compressed_trace_data.emplace_back(phase_id, static_cast<uint8_t>(entry.thread_id), op_id, tensor_id, addr_int);
    }

    gzFile outFile = gzopen(filename.c_str(), "wb");
    if (!outFile) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    uint32_t num_entries = static_cast<uint32_t>(compressed_trace_data.size());
    if (gzwrite(outFile, &num_entries, sizeof(num_entries)) != sizeof(num_entries)) {
        gzclose(outFile);
        throw std::runtime_error("Failed to write number of entries to gzip file.");
    }

    for (const auto& centry : compressed_trace_data) {
        uint8_t phase_val = std::get<0>(centry);
        uint8_t tid_val = std::get<1>(centry);
        uint8_t op_val = std::get<2>(centry);
        uint8_t tensor_val = std::get<3>(centry);
        uint32_t addr_val = std::get<4>(centry);

        if (gzwrite(outFile, &phase_val, sizeof(phase_val)) != sizeof(phase_val) ||
            gzwrite(outFile, &tid_val, sizeof(tid_val)) != sizeof(tid_val) ||
            gzwrite(outFile, &op_val, sizeof(op_val)) != sizeof(op_val) ||
            gzwrite(outFile, &tensor_val, sizeof(tensor_val)) != sizeof(tensor_val) ||
            gzwrite(outFile, &addr_val, sizeof(addr_val)) != sizeof(addr_val)) {
            gzclose(outFile);
            throw std::runtime_error("Failed to write entry to gzip file.");
        }
    }
    gzclose(outFile);

    std::cout << "Memory trace written to " << filename << std::endl;
    std::cout << "Compressed " << mem_trace.size() << " entries" << std::endl; // Use mem_trace
    std::cout << "Phase mapping: {";
    bool first = true;
    for (const auto& pair : local_phase_map) {
        if (!first) std::cout << ", ";
        std::cout << "\"" << pair.first << "\": " << static_cast<int>(pair.second);
        first = false;
    }
    std::cout << "}" << std::endl;
}

void record_access(int thread_id, const std::string& op, uint64_t addr) {
    std::string tensor = addr_to_tensor(addr);
    // Python script uses hex(addr) for storage in the tuple, but C++ MemoryAccessEntry stores uint64_t.
    // write_gmem_trace handles conversion. The hex string was for Python's internal list before writing.
    mem_trace.push_back({curr_phase, thread_id, op, tensor, addr}); // Use curr_phase, mem_trace
}

// --- Algorithm Phases ---
std::vector<uint32_t> radix_sort_with_memtrace(std::vector<uint32_t>& arr, uint64_t base) {
    const int mask = 0xFF;
    const int passes = 4;
    size_t N = arr.size();
    if (N == 0) return arr;
    std::vector<uint32_t> aux(N);

    for (int p = 0; p < passes; ++p) {
        for (size_t i = 0; i < N; ++i) {
            int t_id = static_cast<int>(i % NUM_THREADS);
            record_access(t_id, "R", base + i * SIZE_KEY);
            [[maybe_unused]] uint32_t byte_val = (arr[i] >> (p * 8)) & mask;
        }
        for (size_t i = 0; i < N; ++i) {
            int t_id = static_cast<int>(i % NUM_THREADS);
            record_access(t_id, "R", base + i * SIZE_KEY);
            size_t pos = i;
            aux[pos] = arr[i];
            record_access(t_id, "W", base + pos * SIZE_KEY);
        }
        arr.swap(aux);
    }
    return arr;
}

std::vector<IndexedCoord> compute_unique_sorted_coords(
    const std::vector<std::tuple<int, int, int>>& in_coords_tuples, // Taking tuples as per main
    int stride) {
    std::vector<std::pair<uint32_t, int>> idx_keys; // Store (key, original_index)
    idx_keys.reserve(in_coords_tuples.size());

    for (size_t idx = 0; idx < in_coords_tuples.size(); ++idx) {
        Coord3D coord(std::get<0>(in_coords_tuples[idx]), std::get<1>(in_coords_tuples[idx]), std::get<2>(in_coords_tuples[idx]));
        Coord3D qtz = coord.quantized(stride);
        uint32_t key = qtz.to_key();
        idx_keys.emplace_back(key, static_cast<int>(idx));
    }

    std::vector<uint32_t> raw_keys;
    raw_keys.reserve(idx_keys.size());
    for (const auto& item : idx_keys) {
        raw_keys.push_back(item.first);
    }
    radix_sort_with_memtrace(raw_keys, I_BASE); // Simulate sort on raw keys for trace

    // Sort idx_keys by key, preserving original index for tie-breaking (std::stable_sort not strictly needed if only key matters for order)
    std::sort(idx_keys.begin(), idx_keys.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    std::vector<IndexedCoord> uniq_coords_result;
    uint32_t last_key = (std::numeric_limits<uint32_t>::max)(); // Initialize with a value guaranteed not to be a key if keys are non-negative

    if (!idx_keys.empty()) { // Ensure idx_keys is not empty before accessing first element
        last_key = idx_keys[0].first + 1; // Ensure first element is different from initial last_key
    }


    for (const auto& item : idx_keys) {
        if (item.first != last_key) {
            uniq_coords_result.push_back(IndexedCoord::from_key_and_index(item.first, item.second));
            last_key = item.first;
        }
    }

    if (debug) {
        std::cout << "Sorted+Unique Keys with Original Indices:" << std::endl;
        for (const auto& idxc_item : uniq_coords_result) {
            std::cout << "  key=" << to_hex_string(idxc_item.coord.to_key())
                      << ", coords=(" << idxc_item.coord.x << ", " << idxc_item.coord.y << ", " << idxc_item.coord.z << ")"
                      << ", original_input_index=" << idxc_item.orig_idx << std::endl;
        }
    }
    return uniq_coords_result;
}

BuildQueriesResult build_coordinate_queries(
    const std::vector<IndexedCoord>& uniq_coords,
    int stride,
    const std::vector<std::tuple<int, int, int>>& off_coords_tuples) { // Taking tuples as per main

    size_t num_inputs = uniq_coords.size();
    size_t num_offsets = off_coords_tuples.size();
    size_t total_queries = num_inputs * num_offsets;

    BuildQueriesResult res;
    res.qry_keys.resize(total_queries);
    res.qry_in_idx.resize(total_queries);
    res.qry_off_idx.resize(total_queries);
    res.wt_offsets.resize(total_queries);

    for (size_t off_idx = 0; off_idx < num_offsets; ++off_idx) {
        Coord3D offset(std::get<0>(off_coords_tuples[off_idx]), std::get<1>(off_coords_tuples[off_idx]), std::get<2>(off_coords_tuples[off_idx]));
        Coord3D q_offset = offset.quantized(stride);

        for (size_t in_idx = 0; in_idx < num_inputs; ++in_idx) {
            size_t qry_idx = off_idx * num_inputs + in_idx;
            const IndexedCoord& src_idxcoord = uniq_coords[in_idx];
            const Coord3D& src_coord = src_idxcoord.coord;

            Coord3D query_coord_obj = src_coord + q_offset;
            
            res.qry_keys[qry_idx] = IndexedCoord(query_coord_obj, src_idxcoord.orig_idx);
            res.qry_in_idx[qry_idx] = static_cast<int>(in_idx);
            res.qry_off_idx[qry_idx] = static_cast<int>(off_idx);
            // Python: offset_key = pack32(0, q_offset.x, q_offset.y, q_offset.z)
            // Assuming 3-component key for offset as per unpack32s usage.
            res.wt_offsets[qry_idx] = pack32(q_offset.x, q_offset.y, q_offset.z);
        }
    }
    return res;
}

TilesPivotsResult create_tiles_and_pivots(
    const std::vector<IndexedCoord>& uniq_coords,
    int tile_size_param) { // Renamed tile_size to avoid conflict
    TilesPivotsResult res;
    if (tile_size_param <= 0) { // Basic validation
         if (debug) std::cerr << "Warning: tile_size is non-positive in create_tiles_and_pivots." << std::endl;
        if (!uniq_coords.empty()) {
             res.tiles.push_back(uniq_coords);
             res.pivots.push_back(uniq_coords[0].coord.to_key());
             record_access(0, "W", PIV_BASE + (res.pivots.size() - 1) * SIZE_KEY);
        }
        return res;
    }

    for (size_t start = 0; start < uniq_coords.size(); start += static_cast<size_t>(tile_size_param)) {
        std::vector<IndexedCoord> tile_items;
        size_t end = std::min(start + static_cast<size_t>(tile_size_param), uniq_coords.size());
        for (size_t i = start; i < end; ++i) {
            tile_items.push_back(uniq_coords[i]);
        }
        if (!tile_items.empty()) {
            res.tiles.push_back(tile_items);
            res.pivots.push_back(tile_items[0].coord.to_key());
            record_access(0, "W", PIV_BASE + (res.pivots.size() - 1) * SIZE_KEY);
        }
    }
    return res;
}

// Thread-local storage for lookup phase
thread_local std::vector<MemoryAccessEntry> thread_local_lookup_trace;

void record_local(int thread_id, const std::string& op, uint64_t addr) { // Renamed
    std::string tensor = addr_to_tensor(addr);
    thread_local_lookup_trace.push_back({curr_phase, thread_id, op, tensor, addr});
}

void process_batch_portion(
    size_t batch_start_idx, int thread_id, size_t portion_start_offset, size_t portion_end_offset,
    size_t qry_count_total, // Total queries for boundary check
    const std::vector<IndexedCoord>& uniq_coords_c,
    const std::vector<IndexedCoord>& qry_keys_c,
    const std::vector<int>& qry_in_idx_c,
    const std::vector<int>& qry_off_idx_c,
    // const std::vector<uint32_t>& wt_offsets_c, // wt_offsets not used in this function body
    const std::vector<std::vector<IndexedCoord>>& tiles_c,
    const std::vector<uint32_t>& pivs_c,
    int tile_size_p,
    KernelMap& kmap_ref,
    std::mutex& kmap_lock_ref,
    std::mutex& mem_trace_lock_ref) {

    thread_local_lookup_trace.clear();

    for (size_t qry_offset_in_batch = portion_start_offset; qry_offset_in_batch < portion_end_offset; ++qry_offset_in_batch) {
        size_t qry_idx = batch_start_idx + qry_offset_in_batch;
        if (qry_idx >= qry_count_total) {
            break; 
        }

        record_local(thread_id, "R", QK_BASE + qry_idx * SIZE_KEY);
        uint32_t qry_key_val = qry_keys_c[qry_idx].coord.to_key();

        int lo = 0, hi = static_cast<int>(pivs_c.size()) - 1;
        int found_pivot_idx = -1; 
        if (!pivs_c.empty()) {
             while(lo <= hi) {
                int mid = lo + (hi - lo) / 2;
                record_local(thread_id, "R", PIV_BASE + static_cast<uint64_t>(mid) * SIZE_KEY);
                if (pivs_c[mid] <= qry_key_val) {
                    found_pivot_idx = mid;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
        }
        size_t tile_idx = (found_pivot_idx == -1) ? 0 : static_cast<size_t>(found_pivot_idx);
        if (tile_idx >= tiles_c.size() && !tiles_c.empty()) tile_idx = tiles_c.size() -1; // Boundary condition

        size_t base_offset = tile_idx * static_cast<size_t>(tile_size_p);

        if (tile_idx < tiles_c.size()) {
            const auto& current_tile = tiles_c[tile_idx];
            for (size_t j = 0; j < current_tile.size(); ++j) {
                record_local(thread_id, "R", TILE_BASE + (base_offset + j) * SIZE_KEY);
                const IndexedCoord& tile_idxcoord = current_tile[j];
                uint32_t tile_key_val = tile_idxcoord.coord.to_key();

                if (tile_key_val == qry_key_val) {
                    int src_uniq_idx = qry_in_idx_c[qry_idx];
                    int curr_off_idx = qry_off_idx_c[qry_idx];
                    
                    const IndexedCoord& src_idxc = uniq_coords_c[src_uniq_idx];
                    // uint32_t src_coord_key = src_idxc.coord.to_key(); // Not needed directly for map if storing tuples

                    std::tuple<int,int,int> target_coord_tpl = unpack32(tile_key_val);
                    int target_orig_idx = tile_idxcoord.orig_idx;

                    std::tuple<int,int,int> src_coord_tpl = unpack32(src_idxc.coord.to_key());
                    int query_src_orig_idx = qry_keys_c[qry_idx].orig_idx;
                    
                    {
                        std::lock_guard<std::mutex> guard(kmap_lock_ref);
                        kmap_ref[curr_off_idx].push_back(
                            std::make_pair(
                                std::make_pair(target_coord_tpl, target_orig_idx),
                                std::make_pair(src_coord_tpl, query_src_orig_idx)
                            )
                        );
                        record_local(thread_id, "W", KM_BASE + static_cast<uint64_t>(curr_off_idx) * SIZE_KEY);
                    }
                    break;
                }
            }
        }
    }

    if (!thread_local_lookup_trace.empty()) {
        std::lock_guard<std::mutex> guard(mem_trace_lock_ref);
        mem_trace.insert(mem_trace.end(), thread_local_lookup_trace.begin(), thread_local_lookup_trace.end());
    }
}


KernelMap perform_coordinate_lookup(
    const std::vector<IndexedCoord>& uniq_coords,
    const std::vector<IndexedCoord>& qry_keys,
    const std::vector<int>& qry_in_idx,
    const std::vector<int>& qry_off_idx,
    const std::vector<uint32_t>& wt_offsets, // wt_offsets is passed but not used in C++ process_batch_portion
    const std::vector<std::vector<IndexedCoord>>& tiles,
    const std::vector<uint32_t>& pivs,
    int tile_size_p) {

    KernelMap kmap_result;
    std::mutex kmap_result_lock;
    std::mutex mem_trace_global_lock;

    size_t qry_count = qry_keys.size();
    if (qry_count == 0) return kmap_result;

    const size_t BATCH_SIZE = 128;
    size_t num_batches = (qry_count + BATCH_SIZE - 1) / BATCH_SIZE;

    std::cout << "Starting lookup phase with " << num_batches << " batches." << std::endl;
    int kmap_total_matches = 0;

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        size_t batch_start = batch_idx * BATCH_SIZE;
        size_t current_batch_actual_size = std::min(BATCH_SIZE, qry_count - batch_start);

        if (current_batch_actual_size == 0) break;

        // Simple progress update
        if ((batch_idx + 1) % std::max(1UL, num_batches / 10) == 0 || batch_idx == num_batches - 1) {
             kmap_total_matches = 0; // Recalculate for progress
             for(const auto& pair : kmap_result) kmap_total_matches += pair.second.size();
             std::cout << "Processing batch " << (batch_idx + 1) << "/" << num_batches
                       << " (Queries processed: " << std::min(batch_start + current_batch_actual_size, qry_count) << "/" << qry_count
                       << ", Current Matches: " << kmap_total_matches << ")" << std::endl;
        }
        
        std::vector<std::thread> worker_threads;
        worker_threads.reserve(NUM_THREADS);
        size_t portion_size = (current_batch_actual_size + NUM_THREADS - 1) / NUM_THREADS;

        for (int t = 0; t < NUM_THREADS; ++t) {
            size_t thread_start_offset_in_batch = t * portion_size;
            size_t thread_end_offset_in_batch = std::min(thread_start_offset_in_batch + portion_size, current_batch_actual_size);

            if (thread_start_offset_in_batch < current_batch_actual_size) {
                worker_threads.emplace_back(
                    process_batch_portion,
                    batch_start, t, thread_start_offset_in_batch, thread_end_offset_in_batch,
                    qry_count, // Pass total query count for boundary checks
                    std::cref(uniq_coords), std::cref(qry_keys), std::cref(qry_in_idx), std::cref(qry_off_idx),
                    /*std::cref(wt_offsets),*/ std::cref(tiles), std::cref(pivs), tile_size_p,
                    std::ref(kmap_result), std::ref(kmap_result_lock), std::ref(mem_trace_global_lock)
                );
            }
        }
        for (auto& th : worker_threads) {
            if (th.joinable()) {
                th.join();
            }
        }
    }
    std::cout << "Lookup phase completed." << std::endl;
    return kmap_result;
}

void write_kernel_map_to_gz(
    const KernelMap& kmap_data,
    const std::string& filename,
    const std::vector<std::tuple<int, int, int>>& off_list) { // off_list provides original offsets

    std::vector<std::vector<uint8_t>> packed_entries_binary;
    size_t total_entries = 0;

    for (const auto& pair : kmap_data) {
        int original_offset_idx = pair.first; // This is the index into off_list
        const auto& matches = pair.second;
        total_entries += matches.size();

        if (static_cast<size_t>(original_offset_idx) >= off_list.size()) {
            std::cerr << "Warning: offset_idx " << original_offset_idx << " is out of bounds for off_list." << std::endl;
            continue;
        }
        std::tuple<int,int,int> current_offset_tuple = off_list[original_offset_idx];
        uint32_t packed_offset_key = pack32(std::get<0>(current_offset_tuple), std::get<1>(current_offset_tuple), std::get<2>(current_offset_tuple));

        for (const auto& match : matches) {
            // match is std::pair<std::pair<std::tuple<int,int,int>, int>, std::pair<std::tuple<int,int,int>, int>>
            // ((target_coord_tuple, target_orig_idx), (source_coord_tuple, source_orig_idx))
            const auto& target_data = match.first;  // pair: (target_coord_tuple, target_orig_idx)
            const auto& source_data = match.second; // pair: (source_coord_tuple, source_orig_idx)

            uint32_t target_coord_key = pack32(std::get<0>(target_data.first), std::get<1>(target_data.first), std::get<2>(target_data.first));
            uint32_t target_pos = static_cast<uint32_t>(target_data.second);

            uint32_t source_coord_key = pack32(std::get<0>(source_data.first), std::get<1>(source_data.first), std::get<2>(source_data.first));
            uint32_t source_pos = static_cast<uint32_t>(source_data.second);
            
            std::vector<uint8_t> current_packed_entry(5 * sizeof(uint32_t));
            uint32_t values[] = {packed_offset_key, target_coord_key, target_pos, source_coord_key, source_pos};
            memcpy(current_packed_entry.data(), values, 5 * sizeof(uint32_t));
            packed_entries_binary.push_back(current_packed_entry);
        }
    }
    
    size_t actual_entries_written = packed_entries_binary.size();

    gzFile outFile = gzopen(filename.c_str(), "wb");
    if (!outFile) {
        throw std::runtime_error("Failed to open kernel map file for writing: " + filename);
    }

    uint32_t num_entries_header = static_cast<uint32_t>(actual_entries_written);
    if (gzwrite(outFile, &num_entries_header, sizeof(num_entries_header)) != sizeof(num_entries_header)) {
        gzclose(outFile);
        throw std::runtime_error("Failed to write kernel map entry count.");
    }

    for (const auto& entry_bytes : packed_entries_binary) {
        if (gzwrite(outFile, entry_bytes.data(), static_cast<unsigned int>(entry_bytes.size())) != static_cast<int>(entry_bytes.size())) {
             gzclose(outFile);
             throw std::runtime_error("Failed to write kernel map entry.");
        }
    }
    gzclose(outFile);

    std::cout << "Kernel map written to " << filename << std::endl;
    std::cout << "Wrote " << actual_entries_written << " entries." << std::endl;
}
