#ifndef TRACE_HPP
#define TRACE_HPP

# include <cstdint>
#include <fstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <sstream>
// Helper function to convert value to hex string
std::string to_hex_string(uint64_t val); // Forward declaration

// --- Structs for function results (matching Python for clarity) ---
struct MemoryAccessEntry { // Renamed from mem_trace_entry_t
    uint8_t phase;
    uint8_t thread_id;
    uint8_t op;
    uint8_t tensor;
    uint64_t addr;

    // For pybind11, if you want to print it easily from Python or use __repr__
    std::string toString() const {
        std::ostringstream oss;
        oss << "MemoryAccessEntry(phase=" << phase << ", thread_id=" << thread_id
            << ", op=" << op << ", tensor=" << tensor << ", addr=" << to_hex_string(addr) << ")";
        return oss.str();
    }
};

#endif