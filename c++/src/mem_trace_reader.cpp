#include <string>
#include <algorithm> // For std::transform
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>
#include <zlib.h>
#include "sorted_map.hpp"
#include "trace.hpp"

// It's generally better to have these global maps (PHASES, OPS, TENSORS)
// accessible if this reader is part of the larger project.
// If it's truly standalone, they need to be defined here or passed in a more
// complex way. For this version, we'll assume minuet_trace.hpp provides them or
// they are linked. The example main() below will use local definitions for true
// standalone compilation.

std::string to_hex_string(uint64_t val) {
    std::stringstream ss;
    ss << "0x" << std::hex << val;
    return ss.str();
}

// Structure to hold a decoded memory trace entry for display
struct DecodedMemoryAccessEntry {
  std::string phase;
  uint8_t thread_id;
  std::string op;
  std::string tensor;
  uint64_t addr;
  uint32_t count; // For aggregated view

  DecodedMemoryAccessEntry(const std::string &p, uint8_t tid,
                           const std::string &o, const std::string &t,
                           uint64_t a, uint32_t c = 1)
      : phase(p), thread_id(tid), op(o), tensor(t), addr(a), count(c) {}

  // For easy printing
  friend std::ostream &operator<<(std::ostream &os,
                                  const DecodedMemoryAccessEntry &entry);
};

class MemTraceReader {
public:
  MemTraceReader(const bidict<std::string, int> &phases,
                 const bidict<std::string, int> &ops,
                 const bidict<std::string, int> &tensors);

  bool load_trace_file(const std::string &filename, int sizeof_addr = 4);
  void print_trace(const std::string &filter_phase = "",
                   const std::string &filter_op = "",
                   const std::string &filter_tensor = "",
                   bool aggregate = false, int max_entries = 0);

  const std::vector<MemoryAccessEntry> &get_raw_trace() const;

private:
  std::vector<MemoryAccessEntry> raw_trace_entries;
  const bidict<std::string, int> &PHASES_MAP;
  const bidict<std::string, int> &OPS_MAP;
  const bidict<std::string, int> &TENSORS_MAP;

  DecodedMemoryAccessEntry
  decode_entry(const MemoryAccessEntry &raw_entry) const;
};

// Implementation of operator<< for DecodedMemoryAccessEntry
std::ostream &operator<<(std::ostream &os,
                         const DecodedMemoryAccessEntry &entry) {
  os << "Phase: " << std::setw(4) << std::left << entry.phase
     << " TID: " << std::setw(3) << static_cast<int>(entry.thread_id)
     << " Op: " << std::setw(2) << std::left << entry.op
     << " Tensor: " << std::setw(8) << std::left << entry.tensor << " Addr: "
     << to_hex_string(
            entry.addr); // Assumes global to_hex_string from minuet_trace.hpp
  if (entry.count > 1) {
    os << " Count: " << entry.count;
  }
  return os;
}

// Implementation of MemTraceReader methods
MemTraceReader::MemTraceReader(const bidict<std::string, int> &phases,
                               const bidict<std::string, int> &ops,
                               const bidict<std::string, int> &tensors)
    : PHASES_MAP(phases), OPS_MAP(ops), TENSORS_MAP(tensors) {}

DecodedMemoryAccessEntry
MemTraceReader::decode_entry(const MemoryAccessEntry &raw_entry) const {
  std::string phase_str, op_str, tensor_str;
  try {
    phase_str = PHASES_MAP.inverse.at(raw_entry.phase);
  } catch (const std::out_of_range &) {
    phase_str = "UNK_PH(" + std::to_string(raw_entry.phase) + ")";
  }
  try {
    op_str = OPS_MAP.inverse.at(raw_entry.op);
  } catch (const std::out_of_range &) {
    op_str = "UNK_OP(" + std::to_string(raw_entry.op) + ")";
  }
  try {
    tensor_str = TENSORS_MAP.inverse.at(raw_entry.tensor);
  } catch (const std::out_of_range &) {
    tensor_str = "UNK_TN(" + std::to_string(raw_entry.tensor) + ")";
  }
  return DecodedMemoryAccessEntry(phase_str, raw_entry.thread_id, op_str,
                                  tensor_str, raw_entry.addr);
}

bool MemTraceReader::load_trace_file(const std::string &filename,
                                     int sizeof_addr /*= 4*/) {
  raw_trace_entries.clear();
  gzFile inFile = gzopen(filename.c_str(), "rb");
  if (!inFile) {
    std::cerr << "Error: Failed to open trace file: " << filename << std::endl;
    return false;
  }

  if (sizeof_addr != 4 && sizeof_addr != 8) {
    std::cerr << "Error: sizeof_addr must be 4 or 8, got: " << sizeof_addr
              << std::endl;
    gzclose(inFile);
    return false;
  }

  uint32_t num_entries;
  if (gzread(inFile, &num_entries, sizeof(num_entries)) !=
      sizeof(num_entries)) {
    std::cerr << "Error: Failed to read number of entries from " << filename
              << std::endl;
    gzclose(inFile);
    return false;
  }

  raw_trace_entries.reserve(num_entries);

  for (uint32_t i = 0; i < num_entries; ++i) {
    MemoryAccessEntry entry;
    if (gzread(inFile, &entry.phase, sizeof(entry.phase)) !=
            sizeof(entry.phase) ||
        gzread(inFile, &entry.thread_id, sizeof(entry.thread_id)) !=
            sizeof(entry.thread_id) ||
        gzread(inFile, &entry.op, sizeof(entry.op)) != sizeof(entry.op) ||
        gzread(inFile, &entry.tensor, sizeof(entry.tensor)) !=
            sizeof(entry.tensor)) {
      std::cerr << "Error: Failed to read entry " << i << " (fields) from "
                << filename << std::endl;
      gzclose(inFile);
      return false;
    }
    if (sizeof_addr == 4) {
      uint32_t addr32;
      if (gzread(inFile, &addr32, sizeof(addr32)) != sizeof(addr32)) {
        std::cerr << "Error: Failed to read 4-byte address for entry " << i
                  << " from " << filename << std::endl;
        gzclose(inFile);
        return false;
      }
      entry.addr = addr32;
    } else { // sizeof_addr == 8
      uint64_t addr64;
      if (gzread(inFile, &addr64, sizeof(addr64)) != sizeof(addr64)) {
        std::cerr << "Error: Failed to read 8-byte address for entry " << i
                  << " from " << filename << std::endl;
        gzclose(inFile);
        return false;
      }
      entry.addr = addr64;
    }
    raw_trace_entries.push_back(entry);
  }

  gzclose(inFile);
  std::cout << "Successfully loaded " << raw_trace_entries.size()
            << " entries from " << filename << std::endl;
  return true;
}

void MemTraceReader::print_trace(const std::string &filter_phase,
                                 const std::string &filter_op,
                                 const std::string &filter_tensor,
                                 bool aggregate, int max_entries) {
  if (raw_trace_entries.empty()) {
    std::cout << "No trace entries loaded." << std::endl;
    return;
  }

  std::string lower_filter_phase = filter_phase;
  std::transform(lower_filter_phase.begin(), lower_filter_phase.end(),
                 lower_filter_phase.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::string lower_filter_op = filter_op;
  std::transform(lower_filter_op.begin(), lower_filter_op.end(),
                 lower_filter_op.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::string lower_filter_tensor = filter_tensor;
  std::transform(lower_filter_tensor.begin(), lower_filter_tensor.end(),
                 lower_filter_tensor.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  std::vector<DecodedMemoryAccessEntry> filtered_decoded_entries;

  for (const auto &raw_entry : raw_trace_entries) {
    DecodedMemoryAccessEntry decoded = decode_entry(raw_entry);

    std::string current_phase_lower = decoded.phase;
    std::transform(current_phase_lower.begin(), current_phase_lower.end(),
                   current_phase_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    std::string current_op_lower = decoded.op;
    std::transform(current_op_lower.begin(), current_op_lower.end(),
                   current_op_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    std::string current_tensor_lower = decoded.tensor;
    std::transform(current_tensor_lower.begin(), current_tensor_lower.end(),
                   current_tensor_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (!filter_phase.empty() &&
        current_phase_lower.find(lower_filter_phase) == std::string::npos)
      continue;
    if (!filter_op.empty() &&
        current_op_lower.find(lower_filter_op) == std::string::npos)
      continue;
    if (!filter_tensor.empty() &&
        current_tensor_lower.find(lower_filter_tensor) == std::string::npos)
      continue;

    filtered_decoded_entries.push_back(decoded);
  }

  if (aggregate) {
    if (filtered_decoded_entries.empty()) {
      std::cout << "No entries match filter criteria for aggregation."
                << std::endl;
      return;
    }
    std::map<
        std::tuple<std::string, uint8_t, std::string, std::string, uint64_t>,
        uint32_t>
        aggregated_counts;
    for (const auto &entry : filtered_decoded_entries) {
      aggregated_counts[{entry.phase, entry.thread_id, entry.op, entry.tensor,
                         entry.addr}]++;
    }

    std::vector<DecodedMemoryAccessEntry> final_aggregated_list;
    for (const auto &pair : aggregated_counts) {
      final_aggregated_list.emplace_back(
          std::get<0>(pair.first), std::get<1>(pair.first),
          std::get<2>(pair.first), std::get<3>(pair.first),
          std::get<4>(pair.first), pair.second);
    }
    std::sort(final_aggregated_list.begin(), final_aggregated_list.end(),
              [](const DecodedMemoryAccessEntry &a,
                 const DecodedMemoryAccessEntry &b) {
                return a.count > b.count; // Sort by count descending
              });

    int count = 0;
    for (const auto &entry : final_aggregated_list) {
      std::cout << entry << std::endl;
      count++;
      if (max_entries > 0 && count >= max_entries) {
        if (final_aggregated_list.size() - count > 0) {
          std::cout << "... and " << (final_aggregated_list.size() - count)
                    << " more aggregated entries." << std::endl;
        }
        break;
      }
    }
    std::cout << "Total unique aggregated entries printed: " << count
              << std::endl;

  } else {
    int count = 0;
    for (const auto &entry : filtered_decoded_entries) {
      std::cout << entry << std::endl;
      count++;
      if (max_entries > 0 && count >= max_entries) {
        if (filtered_decoded_entries.size() - count > 0) {
          std::cout << "... and " << (filtered_decoded_entries.size() - count)
                    << " more entries." << std::endl;
        }
        break;
      }
    }
    std::cout << "Total entries printed: " << count << std::endl;
  }
}

const std::vector<MemoryAccessEntry> &MemTraceReader::get_raw_trace() const {
  return raw_trace_entries;
}

// Example main for testing the reader (compile separately or include in a test
// build)
int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: ./minuet_mem_trace_reader <trace_file.bin.gz> "
                 "[sizeof_addr (4 or 8, default 4)] [filter_phase] [filter_op] "
                 "[filter_tensor] [aggregate (0 or 1)] [max_entries]"
              << std::endl;
    return 1;
  }

  std::string trace_file = argv[1];
  int sizeof_addr = (argc > 2) ? std::stoi(argv[2]) : 4;
  std::string filter_phase = (argc > 3) ? argv[3] : "";
  std::string filter_op = (argc > 4) ? argv[4] : "";
  std::string filter_tensor = (argc > 5) ? argv[5] : "";
  bool aggregate = (argc > 6) ? (std::stoi(argv[6]) != 0) : false;
  int max_entries = (argc > 7) ? std::stoi(argv[7]) : 0;

  // For true standalone compilation, define PHASES, OPS, TENSORS locally.
  // These definitions should match those in minuet_trace.cpp

  bidict<std::string, int> PHASES({{"RDX", 0},
                                   {"QRY", 1},
                                   {"SRT", 2},
                                   {"PVT", 3},
                                   {"LKP", 4},
                                   {"GTH", 5},
                                   {"SCT", 6}});

  bidict<std::string, int> TENSORS({
      {"I", 0},
      {"QK", 1},
      {"QI", 2},
      {"QO", 3},
      {"PIV", 4},
      {"KM", 5},
      {"WC", 6},
      {"TILE", 7},     // TILE is I_BASE, handled in addr_to_tensor
      {"IV", 8},       // IV_BASE is not in TENSORS, handled as string
      {"GM", 9},       // GM_BASE is not in TENSORS, handled as string
      {"WV", 10},      // WV_BASE is not in TENSORS, handled as string
      {"Unknown", 255} // Default case for unknown tensors
  });

  bidict<std::string, int> OPS({{"R", 0}, {"W", 1}});

  // Note: The TENSORS map in minuet_trace.cpp might have slightly different int
  // values if it was modified. Ensure these local versions are consistent if
  // used. It's better to link with the actual definitions from minuet_trace.o
  // if possible.

  // If you are linking this with the rest of your project where PHASES, OPS,
  // TENSORS are global externs: extern bidict<std::string, int> PHASES; // From
  // minuet_trace.cpp extern bidict<std::string, int> OPS;    // From
  // minuet_trace.cpp extern bidict<std::string, int> TENSORS; // From
  // minuet_trace.cpp MemTraceReader reader(PHASES, OPS, TENSORS);

  // Using local maps for this standalone example:
  MemTraceReader reader(PHASES, OPS, TENSORS);

  if (!reader.load_trace_file(trace_file, sizeof_addr)) {
    return 1;
  }

  reader.print_trace(filter_phase, filter_op, filter_tensor, aggregate,
                     max_entries);

  return 0;
}
