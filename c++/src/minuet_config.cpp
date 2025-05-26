#include "minuet_config.hpp"
#include <fstream>
#include <iostream>
#include <ext/json.hpp> // Assuming nlohmann/json is used

// Definition of the global config object
MinuetConfig g_config;

// Constructor for default values
MinuetConfig::MinuetConfig() :
    NUM_THREADS(4),
    SIZE_KEY(4),
    SIZE_INT(4),
    SIZE_WEIGHT(4),
    I_BASE(0x10000000),
    TILE_BASE(I_BASE), // Alias
    QK_BASE(0x20000000),
    QI_BASE(0x30000000),
    QO_BASE(0x40000000),
    PIV_BASE(0x50000000),
    KM_BASE(0x60000000),
    WO_BASE(0x80000000),
    IV_BASE(0x100000000),
    WV_BASE(0xF00000000),
    GEMM_ALIGNMENT(4),
    GEMM_WT_GROUP(2),
    GEMM_SIZE(4),
    NUM_TILES(2),
    TILE_FEATS(16),
    BULK_FEATS(4),
    N_THREADS_GATHER(1),
    TOTAL_FEATS_PT(NUM_TILES* TILE_FEATS),
    debug(false), // Initialize debug flag
    output_dir("out/") // Initialize output_dir
    {}

// Function to load configuration from a JSON file
bool MinuetConfig::loadFromFile(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "Error: Could not open config file: " << filepath << std::endl;
        return false;
    }

    try {
        nlohmann::json data = nlohmann::json::parse(f);

        NUM_THREADS = data.value("NUM_THREADS", NUM_THREADS);
        SIZE_KEY = data.value("SIZE_KEY", SIZE_KEY);
        SIZE_INT = data.value("SIZE_INT", SIZE_INT);
        SIZE_WEIGHT = data.value("SIZE_WEIGHT", SIZE_WEIGHT);

        // Helper lambda to parse hex string fields for BASE addresses
        auto load_base_address = [&](uint64_t& member, const std::string& key) {
            if (data.contains(key)) {
                if (data[key].is_string()) {
                    std::string val_str = data[key].get<std::string>();
                    try {
                        member = std::stoull(val_str, nullptr, 0);
                    } catch (const std::invalid_argument& ia) {
                        std::cerr << "Warning: Invalid hex string for " << key << ": '" << val_str << "'. Using default value. Error: " << ia.what() << std::endl;
                    } catch (const std::out_of_range& oor) {
                        std::cerr << "Warning: Hex string for " << key << " out of range: '" << val_str << "'. Using default value. Error: " << oor.what() << std::endl;
                    }
                } else if (data[key].is_number()) {
                    try {
                        member = data[key].get<uint64_t>();
                    } catch (const nlohmann::json::type_error& te) {
                         std::cerr << "Warning: " << key << " in JSON is a number but could not be converted to uint64_t: " << data[key].dump() << ". Using default value. Error: " << te.what() << std::endl;
                    }
                } else {
                    std::cerr << "Warning: " << key << " in JSON is not a string or number: " << data[key].dump() << ". Using default value." << std::endl;
                }
            }
        };

        load_base_address(I_BASE, "I_BASE");
        TILE_BASE = I_BASE; // Update alias after I_BASE is loaded
        load_base_address(QK_BASE, "QK_BASE");
        load_base_address(QI_BASE, "QI_BASE");
        load_base_address(QO_BASE, "QO_BASE");
        load_base_address(PIV_BASE, "PIV_BASE");
        load_base_address(KM_BASE, "KM_BASE");
        load_base_address(WO_BASE, "WO_BASE");
        load_base_address(IV_BASE, "IV_BASE");
        load_base_address(WV_BASE, "WV_BASE");

        GEMM_ALIGNMENT = data.value("GEMM_ALIGNMENT", GEMM_ALIGNMENT);
        GEMM_WT_GROUP = data.value("GEMM_WT_GROUP", GEMM_WT_GROUP);
        GEMM_SIZE = data.value("GEMM_SIZE", GEMM_SIZE);

        NUM_TILES = data.value("NUM_TILES", NUM_TILES);
        TILE_FEATS = data.value("TILE_FEATS", TILE_FEATS);
        BULK_FEATS = data.value("BULK_FEATS", BULK_FEATS);
        N_THREADS_GATHER = data.value("N_THREADS_GATHER", N_THREADS_GATHER);
        TOTAL_FEATS_PT = NUM_TILES * TILE_FEATS; // Recalculate
        debug = data.value("debug", debug); // Load debug flag
        output_dir = data.value("output_dir", output_dir); // Load output_dir

    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return false;
    } catch (const nlohmann::json::type_error& e) {
        std::cerr << "JSON type error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "An unexpected error occurred while parsing JSON: " << e.what() << std::endl;
        return false;
    }

    return true;
}
