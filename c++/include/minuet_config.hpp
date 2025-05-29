#ifndef MINUET_CONFIG_HPP
#define MINUET_CONFIG_HPP

#include <string>
#include <cstdint>
// #include <vector> // Not used in the struct definition itself
#include <fstream>  // For std::ifstream
#include <iostream> // For std::cerr
#include <ext/json.hpp> // Assuming nlohmann/json is used and located here

struct MinuetConfig {
    // Number of virtual threads
    uint32_t NUM_THREADS;
    uint32_t SIZE_KEY;
    uint32_t SIZE_INT;
    uint32_t SIZE_WEIGHT;
    uint32_t SIZE_FEAT;

    // Tensor Regions
    uint64_t I_BASE;
    uint64_t TILE_BASE; // Alias, will be set to I_BASE after loading
    uint64_t QK_BASE;
    uint64_t QI_BASE;
    uint64_t QO_BASE;
    uint64_t PIV_BASE;
    uint64_t KM_BASE;
    uint64_t WO_BASE;
    uint64_t IV_BASE; // Feature vectors (64-bit)
    uint64_t GM_BASE; // GEMM buffers (64-bit)
    uint64_t WV_BASE; // Weight values (64-bit)

    // GEMM Parameters
    uint32_t GEMM_ALIGNMENT;
    uint32_t GEMM_WT_GROUP;
    uint32_t GEMM_SIZE;

    // GATHER PARAMETERS
    uint32_t NUM_TILES;
    uint32_t TILE_FEATS;
    uint32_t BULK_FEATS;
    uint32_t N_THREADS_GATHER;
    uint32_t TOTAL_FEATS_PT; // Calculated: NUM_TILES * TILE_FEATS

    bool debug; // Added for debug flag
    std::string output_dir; // Added for output directory
    uint32_t NUM_PIVOTS; // Added NUM_PIVOTS

    MinuetConfig(); // Constructor for default values

    // Function to load configuration from a JSON file
    bool loadFromFile(const std::string& filepath); // Method declaration
};

// Declaration of the global config object
extern MinuetConfig g_config;

#endif // MINUET_CONFIG_HPP
