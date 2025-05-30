#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h> // For operator overloading
#include "minuet_map.hpp"     // Your main header
#include "minuet_config.hpp"    // Include the config header for g_config
#include "minuet_gather.hpp"    // Include the gather header

namespace py = pybind11;

// Helper to bind KernelMapType (SortedByValueSizeMap) to a Python dict-like object
// This allows Python to treat it somewhat like a dictionary, though true
// dict behavior (e.g., direct item assignment from Python) is complex to replicate fully.
// For now, we expose it as an opaque type or provide specific accessors.
// Since pybind11/stl.h handles std::map well, and our SortedByValueSizeMap wraps one,
// we might not need a custom binding if we just expose its contents or specific methods.
// However, if we want Python to see it as a dict sorted by value size, that's more involved.

// For now, let KernelMapType be handled by pybind11's default STL conversions
// where possible, or expose specific methods like get_sorted_items().
// If direct dict-like access is needed from Python, a custom type caster
// or a wrapper class in C++ exposed to Python would be required.

PYBIND11_MODULE(minuet_cpp_module, m) {
    m.doc() = "Pybind11 bindings for Minuet C++ trace and mapping functions";

    // Bind Coord3D
    py::class_<Coord3D>(m, "Coord3D")
        .def(py::init<int, int, int>(), py::arg("x_") = 0, py::arg("y_") = 0, py::arg("z_") = 0)
        .def_readwrite("x", &Coord3D::x)
        .def_readwrite("y", &Coord3D::y)
        .def_readwrite("z", &Coord3D::z)
        .def("quantized", &Coord3D::quantized, py::arg("stride"))
        .def("to_key", &Coord3D::to_key)
        .def_static("from_key", &Coord3D::from_key, py::arg("key"))
        .def_static("from_signed_key", &Coord3D::from_signed_key, py::arg("key"))
        .def(py::self + py::self)
        .def("__repr__", [](const Coord3D &c) {
            return "<Coord3D (" + std::to_string(c.x) + ", " + std::to_string(c.y) + ", " + std::to_string(c.z) + ")>";
        });

    // Bind IndexedCoord
    py::class_<IndexedCoord>(m, "IndexedCoord")
        .def(py::init<Coord3D, int>(), py::arg("c") = Coord3D(), py::arg("idx") = -1)
        .def(py::init<uint32_t, int>(), py::arg("k"), py::arg("idx") = -1)
        .def_readwrite("coord", &IndexedCoord::coord)
        .def_readwrite("orig_idx", &IndexedCoord::orig_idx)
        .def_readwrite("key_val", &IndexedCoord::key_val)
        .def("to_key", &IndexedCoord::to_key)
        .def_static("from_key_and_index", &IndexedCoord::from_key_and_index, py::arg("key"), py::arg("idx"))
        .def("__repr__", [](const IndexedCoord &ic) {
            std::string coord_repr = "<Coord3D (" + std::to_string(ic.coord.x) + ", " +
                                     std::to_string(ic.coord.y) + ", " +
                                     std::to_string(ic.coord.z) + ")>";
            return "<IndexedCoord coord=" + coord_repr +
                   ", orig_idx=" + std::to_string(ic.orig_idx) +
                   ", key_val=" + std::to_string(ic.key_val) + ">";
        });

    // Bind MemoryAccessEntry
    py::class_<MemoryAccessEntry>(m, "MemoryAccessEntry")
        .def(py::init<>())
        .def_readwrite("phase", &MemoryAccessEntry::phase)
        .def_readwrite("thread_id", &MemoryAccessEntry::thread_id)
        .def_readwrite("op", &MemoryAccessEntry::op)
        .def_readwrite("tensor", &MemoryAccessEntry::tensor)
        .def_readwrite("addr", &MemoryAccessEntry::addr)
        .def("__repr__", [](const MemoryAccessEntry &e) {
            std::stringstream ss;
            ss << "<MemoryAccessEntry phase=" << e.phase
               << ", tid=" << e.thread_id
               << ", op=" << e.op
               << ", tensor=" << e.tensor
               << ", addr=0x" << std::hex << e.addr << ">";
            return ss.str();
        });

    // Bind BuildQueriesResult
    py::class_<BuildQueriesResult>(m, "BuildQueriesResult")
        .def(py::init<>())
        .def_readwrite("qry_keys", &BuildQueriesResult::qry_keys)
        .def_readwrite("qry_in_idx", &BuildQueriesResult::qry_in_idx)
        .def_readwrite("qry_off_idx", &BuildQueriesResult::qry_off_idx)
        .def_readwrite("wt_offsets", &BuildQueriesResult::wt_offsets);

    // Bind TilesPivotsResult
    py::class_<TilesPivotsResult>(m, "TilesPivotsResult")
        .def(py::init<>())
        .def_readwrite("tiles", &TilesPivotsResult::tiles)
        .def_readwrite("pivots", &TilesPivotsResult::pivots);

    // Bind KernelMapType (SortedByValueSizeMap<uint32_t, std::vector<std::pair<int, int>>>)
    // Expose it as an opaque type for now, or bind specific methods if needed.
    // py::bind_map<KernelMapType>(m, "KernelMap"); // This won't work directly for custom types
    // Instead, let's expose it as a class and bind methods to access its data if Python needs to inspect it.
    // For functions that return KernelMapType, pybind11 will try to convert if possible.
    // If it's just passed around, it might work as an opaque type.
    // Given its usage, Python side likely needs to iterate it or look up items.
    // A simple way is to provide a method that converts it to a Python dictionary.
    py::class_<KernelMapType>(m, "KernelMap") // Keep Python name "KernelMap" for consistency
        .def(py::init<bool>(), py::arg("ascending") = true)
        .def("get_sorted_items", [](const KernelMapType &kmap) {
            // Convert to a Python list of tuples (key, value_list_of_pairs)
            py::list items;
            auto sorted_cpp_items = kmap.get_sorted_items();
            for (const auto& cpp_item : sorted_cpp_items) {
                // cpp_item is std::pair<const uint32_t, std::vector<std::pair<int, int>>>
                py::list val_list;
                for (const auto& p : cpp_item.second) {
                    val_list.append(py::make_tuple(p.first, p.second));
                }
                items.append(py::make_tuple(cpp_item.first, val_list));
            }
            return items;
        })
        .def("__getitem__", [](const KernelMapType &kmap, uint32_t key) {
            // This provides kmap[key] access from Python (read-only)
            try {
                const auto& val_vec = kmap.at(key);
                py::list val_list;
                for (const auto& p : val_vec) {
                    val_list.append(py::make_tuple(p.first, p.second));
                }
                return val_list;
            } catch (const std::out_of_range& e) {
                throw py::key_error("key not found");
            }
        })
        .def("__contains__", [](const KernelMapType &kmap, uint32_t key) {
            return kmap.count(key) > 0;
        })
        .def("__len__", &KernelMapType::size)
        .def("empty", &KernelMapType::empty)
        .def("items", [](const KernelMapType &kmap) { // Mimics dict.items() based on sorted order
            py::list items_list;
            auto sorted_cpp_items = kmap.get_sorted_items();
            for (const auto& cpp_item : sorted_cpp_items) {
                py::list val_list;
                for (const auto& p : cpp_item.second) {
                    val_list.append(py::make_tuple(p.first, p.second));
                }
                items_list.append(py::make_tuple(cpp_item.first, val_list));
            }
            return items_list;
        });

    // Bind global state accessors
    m.def("get_mem_trace", &get_mem_trace, py::return_value_policy::reference_internal); // Or copy
    m.def("clear_mem_trace", &clear_mem_trace);
    m.def("set_curr_phase", &set_curr_phase, py::arg("phase_name"));
    m.def("get_curr_phase", &get_curr_phase);
    m.def("set_debug_flag", &set_debug_flag, py::arg("debug_val"));
    m.def("get_debug_flag", &get_debug_flag);

    // Add a function to load configuration from a JSON file
    m.def("load_config_from_file", [](const std::string& filepath) {
        return g_config.loadFromFile(filepath);
    }, py::arg("filepath"), "Loads configuration from a JSON file into the global C++ config object.");

    // Access to global config object (read-only for Python side for safety)
    // Expose individual fields if mutable access is needed, or provide a load function.
    // For now, Python can see the values loaded by C++ main().
    // If Python needs to *set* these, a dedicated function in C++ should handle it
    // and update g_config, then Python can re-read.
    py::class_<MinuetConfig>(m, "MinuetConfigReader") // Expose as a read-only view
        .def_property_readonly("NUM_THREADS", [](const MinuetConfig& c){ return c.NUM_THREADS; })
        .def_property_readonly("SIZE_KEY", [](const MinuetConfig& c){ return c.SIZE_KEY; })
        .def_property_readonly("SIZE_INT", [](const MinuetConfig& c){ return c.SIZE_INT; })
        .def_property_readonly("SIZE_WEIGHT", [](const MinuetConfig& c){ return c.SIZE_WEIGHT; })
        .def_property_readonly("I_BASE", [](const MinuetConfig& c){ return c.I_BASE; })
        .def_property_readonly("TILE_BASE", [](const MinuetConfig& c){ return c.TILE_BASE; })
        .def_property_readonly("QK_BASE", [](const MinuetConfig& c){ return c.QK_BASE; })
        .def_property_readonly("QI_BASE", [](const MinuetConfig& c){ return c.QI_BASE; })
        .def_property_readonly("QO_BASE", [](const MinuetConfig& c){ return c.QO_BASE; })
        .def_property_readonly("PIV_BASE", [](const MinuetConfig& c){ return c.PIV_BASE; })
        .def_property_readonly("KM_BASE", [](const MinuetConfig& c){ return c.KM_BASE; })
        .def_property_readonly("WO_BASE", [](const MinuetConfig& c){ return c.WO_BASE; })
        .def_property_readonly("IV_BASE", [](const MinuetConfig& c){ return c.IV_BASE; })
        .def_property_readonly("WV_BASE", [](const MinuetConfig& c){ return c.WV_BASE; })
        .def_property_readonly("GEMM_ALIGNMENT", [](const MinuetConfig& c){ return c.GEMM_ALIGNMENT; })
        .def_property_readonly("GEMM_WT_GROUP", [](const MinuetConfig& c){ return c.GEMM_WT_GROUP; })
        .def_property_readonly("GEMM_SIZE", [](const MinuetConfig& c){ return c.GEMM_SIZE; })
        .def_property_readonly("NUM_TILES", [](const MinuetConfig& c){ return c.NUM_TILES; })
        .def_property_readonly("NUM_PIVOTS", [](const MinuetConfig& c){ return c.NUM_PIVOTS; })
        .def_property_readonly("TILE_FEATS", [](const MinuetConfig& c){ return c.TILE_FEATS; })
        .def_property_readonly("BULK_FEATS", [](const MinuetConfig& c){ return c.BULK_FEATS; })
        .def_property_readonly("N_THREADS_GATHER", [](const MinuetConfig& c){ return c.N_THREADS_GATHER; })
        .def_property_readonly("TOTAL_FEATS_PT", [](const MinuetConfig& c){ return c.TOTAL_FEATS_PT; })
        .def_property_readonly("debug", [](const MinuetConfig& c){ return c.debug; }) // Added
        .def_property_readonly("output_dir", [](const MinuetConfig& c){ return c.output_dir; }); // Added

    // Expose the global g_config instance (as a reader)
    m.def("get_global_config", []() -> const MinuetConfig& {
        return g_config;
    }, py::return_value_policy::reference); // Expose as a const reference

    // Bind functions
    // m.def("addr_to_tensor", &addr_to_tensor, py::arg("addr")); // Internal, not typically bound
    m.def("record_access", &record_access, py::arg("thread_id"), py::arg("op"), py::arg("addr"));
    m.def("write_gmem_trace", &write_gmem_trace, py::arg("filename"),
          py::arg("sizeof_addr") = 4, // Add sizeof_addr argument with default
          "Writes the memory trace to a gzipped file and returns its CRC32 checksum.");
    
    m.def("compute_unique_sorted_coords", &compute_unique_sorted_coords, 
          py::arg("in_coords"), py::arg("stride"));
    
    m.def("build_coordinate_queries", &build_coordinate_queries,
          py::arg("uniq_coords"), py::arg("stride"), py::arg("off_coords"));

    m.def("create_tiles_and_pivots", &create_tiles_and_pivots,
          py::arg("uniq_coords"), py::arg("tile_size"));

    m.def("perform_coordinate_lookup", &perform_coordinate_lookup,
          py::arg("uniq_coords"), py::arg("qry_keys"), py::arg("qry_in_idx"),
          py::arg("qry_off_idx"), py::arg("wt_offsets"), py::arg("tiles"),
          py::arg("pivs"), py::arg("num_tiles_config"), // Corrected arg name to match C++
          py::return_value_policy::move); // KernelMapType is returned by value

    m.def("write_kernel_map_to_gz", &write_kernel_map_to_gz,
          py::arg("kmap_data"), py::arg("filename"), py::arg("off_list"),
          "Writes the kernel map to a gzipped file and returns its CRC32 checksum.");

    // --- Bindings for Greedy Grouping (from minuet_gather.hpp / minuet_trace.cpp) ---

    py::class_<GemmInfo>(m, "GemmInfo")
        .def(py::init<>())
        .def_readwrite("num_offsets", &GemmInfo::num_offsets)
        .def_readwrite("gemm_M", &GemmInfo::gemm_M)
        .def_readwrite("slots", &GemmInfo::slots)
        .def_readwrite("padding", &GemmInfo::padding)
        .def("__repr__", [](const GemmInfo &gi) {
            return "<GemmInfo num_offsets=" + std::to_string(gi.num_offsets) +
                   ", gemm_M=" + std::to_string(gi.gemm_M) +
                   ", slots=" + std::to_string(gi.slots) +
                   ", padding=" + std::to_string(gi.padding) + ">";
        });

    py::class_<GroupInfo>(m, "GroupInfo")
        .def(py::init<>())
        .def_readwrite("members", &GroupInfo::members)
        .def_readwrite("base_addr", &GroupInfo::base_addr)
        .def_readwrite("required_slots", &GroupInfo::required_slots)
        .def_readwrite("allocated_slots", &GroupInfo::allocated_slots)
        .def("__repr__", [](const GroupInfo &gi) {
            std::string members_str = "[";
            for (size_t i = 0; i < gi.members.size(); ++i) {
                members_str += std::to_string(gi.members[i]) + (i < gi.members.size() - 1 ? ", " : "");
            }
            members_str += "]";
            return "<GroupInfo members=" + members_str +
                   ", base_addr=" + std::to_string(gi.base_addr) +
                   ", required_slots=" + std::to_string(gi.required_slots) +
                   ", allocated_slots=" + std::to_string(gi.allocated_slots) + ">";
        });

    py::class_<GreedyGroupResult>(m, "GreedyGroupResult")
        .def(py::init<>())
        .def_readwrite("pos_indices", &GreedyGroupResult::pos_indices)
        .def_readwrite("groups", &GreedyGroupResult::groups)
        .def_readwrite("membership", &GreedyGroupResult::membership)
        .def_readwrite("gemm_list", &GreedyGroupResult::gemm_list)
        .def_readwrite("total_slots_allocated", &GreedyGroupResult::total_slots_allocated)
        .def_readwrite("checksum", &GreedyGroupResult::checksum)
        .def("__repr__", [](const GreedyGroupResult &ggr) {
            return "<GreedyGroupResult groups_count=" + std::to_string(ggr.groups.size()) +
                   ", gemm_list_count=" + std::to_string(ggr.gemm_list.size()) +
                   ", total_slots_allocated=" + std::to_string(ggr.total_slots_allocated) +
                   ", checksum=" + std::to_string(ggr.checksum) + ">";
        });

    m.def("greedy_group_cpp", &greedy_group_cpp,
          py::arg("slots"),
          py::arg("alignment") = 4,
          py::arg("max_group_items") = 6,
          py::arg("max_raw_slots") = -1, // -1 for None
          "Performs greedy grouping of slots, similar to Python's greedy_group.");

    // Bind write_gemm_list_cpp separately if direct access is desired from Python
    // This is already called by greedy_group_cpp, but can be exposed for flexibility/testing.
    m.def("write_gemm_list_cpp", &write_gemm_list_cpp,
        py::arg("gemm_data_list"), py::arg("filename"),
        "Writes a list of GemmInfo to a gzipped file and returns its CRC32 checksum.");


    // --- Bindings for Metadata Reading ---
    py::class_<ActiveOffsetInfo>(m, "ActiveOffsetInfo")
        .def(py::init<>())
        .def_readwrite("offset_key", &ActiveOffsetInfo::offset_key)
        .def_readwrite("base_address", &ActiveOffsetInfo::base_address)
        .def_readwrite("num_matches", &ActiveOffsetInfo::num_matches)
        .def("__repr__", [](const ActiveOffsetInfo&aoi) {
            return "<ActiveOffsetInfo key=" + std::to_string(aoi.offset_key) +
                   ", addr=" + std::to_string(aoi.base_address) +
                   ", matches=" + std::to_string(aoi.num_matches) + ">";
        });

    py::class_<MetadataContents>(m, "MetadataContents")
        .def(py::init<>())
        .def_readwrite("version", &MetadataContents::version)
        .def_readwrite("num_total_system_offsets", &MetadataContents::num_total_system_offsets)
        .def_readwrite("num_total_system_sources", &MetadataContents::num_total_system_sources)
        .def_readwrite("total_slots_in_gemm_buffer", &MetadataContents::total_slots_in_gemm_buffer)
        .def_readwrite("num_active_offsets_in_map", &MetadataContents::num_active_offsets_in_map)
        .def_readwrite("active_offsets_details", &MetadataContents::active_offsets_details)
        .def_readwrite("out_mask", &MetadataContents::out_mask)
        .def_readwrite("in_mask", &MetadataContents::in_mask)
        .def("__repr__", [](const MetadataContents& mc) {
            return "<MetadataContents version=" + std::to_string(mc.version) +
                   ", num_sys_offsets=" + std::to_string(mc.num_total_system_offsets) +
                   ", num_sys_sources=" + std::to_string(mc.num_total_system_sources) +
                   ", total_gemm_slots=" + std::to_string(mc.total_slots_in_gemm_buffer) +
                   ", num_active_offsets=" + std::to_string(mc.num_active_offsets_in_map) +
                   ", active_offsets_count=" + std::to_string(mc.active_offsets_details.size()) +
                   ", out_mask_size=" + std::to_string(mc.out_mask.size()) +
                   ", in_mask_size=" + std::to_string(mc.in_mask.size()) + ">";
        });

    m.def("read_metadata_cpp", &read_metadata_cpp,
          py::arg("filename"),
          "Reads metadata from a gzipped binary file.");

    m.def("write_metadata_cpp", &write_metadata_cpp,
          py::arg("out_mask"),
          py::arg("in_mask"),
          py::arg("active_offset_data"), // std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>
          py::arg("num_total_system_offsets"),
          py::arg("num_total_system_sources"),
          py::arg("total_slots_in_gemm_buffer"),
          py::arg("filename"),
          "Writes metadata to a gzipped binary file and returns its CRC32 checksum.");

    // Bind Gather/Scatter functions
    m.def("mt_gather_cpp", &mt_gather_cpp,
          py::arg("num_threads"),
          py::arg("num_points"),
          py::arg("num_offsets"),
          py::arg("num_tiles_per_pt"),
          py::arg("tile_feat_size"),
          py::arg("bulk_feat_size"),
          py::arg("source_masks"),
          py::arg("sources"),
          py::arg("gemm_buffers"),
          "Performs the gather operation using C++ implementation.");

    m.def("mt_scatter_cpp", &mt_scatter_cpp,
          py::arg("num_threads"),
          py::arg("num_points"),
          py::arg("num_offsets"),
          py::arg("num_tiles_per_pt"),
          py::arg("tile_feat_size"),
          py::arg("bulk_feat_size"),
          py::arg("out_mask"),
          py::arg("gemm_buffers"),
          py::arg("outputs"),
          "Performs the scatter operation using C++ implementation.");

    // Bind MasksResult struct
    py::class_<MasksResult>(m, "MasksResult")
        .def(py::init<>())
        .def_readwrite("out_mask", &MasksResult::out_mask)
        .def_readwrite("in_mask", &MasksResult::in_mask);

    // Bind create_in_out_masks_cpp
    m.def("create_in_out_masks_cpp", &create_in_out_masks_cpp,
          py::arg("kernel_map"),
          py::arg("slot_dict"), // std::map<uint32_t, int>
          py::arg("num_total_system_offsets"),
          py::arg("num_total_system_sources"),
          "Creates input and output masks for gather/scatter operations.");


}
