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
    int NUM_THREADS;
    int SIZE_KEY;
    int SIZE_INT;
    int SIZE_WEIGHT;

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
    uint64_t WV_BASE; // Weight values (64-bit)

    // GEMM Parameters
    int GEMM_ALIGNMENT;
    int GEMM_WT_GROUP;
    int GEMM_SIZE;

    // GATHER PARAMETERS
    int NUM_TILES;
    int TILE_FEATS;
    int BULK_FEATS;
    int N_THREADS_GATHER;
    int TOTAL_FEATS_PT; // Calculated: NUM_TILES * TILE_FEATS

    bool debug; // Added for debug flag
    std::string output_dir; // Added for output directory
    int NUM_PIVOTS; // Added NUM_PIVOTS

    MinuetConfig(); // Constructor for default values

    // Function to load configuration from a JSON file
    bool loadFromFile(const std::string& filepath); // Method declaration
};

// Declaration of the global config object
extern MinuetConfig g_config;

#endif // MINUET_CONFIG_HPP
