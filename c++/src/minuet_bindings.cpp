#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h> // For operator overloading
#include "minuet_trace.hpp"     // Your main header

namespace py = pybind11;

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

    // Bind KernelMap (std::map<uint32_t, std::vector<std::pair<int, int>>>)
    // Pybind11 handles stl containers automatically with pybind11/stl.h

    // Bind global state accessors
    m.def("get_mem_trace", &get_mem_trace, py::return_value_policy::reference_internal); // Or copy
    m.def("clear_mem_trace", &clear_mem_trace);
    m.def("set_curr_phase", &set_curr_phase, py::arg("phase_name"));
    m.def("get_curr_phase", &get_curr_phase);
    m.def("set_debug_flag", &set_debug_flag, py::arg("debug_val"));
    m.def("get_debug_flag", &get_debug_flag);
    m.attr("output_dir") = py::str(output_dir);


    // Bind functions
    m.def("addr_to_tensor", &addr_to_tensor, py::arg("addr"));
    m.def("record_access", &record_access, py::arg("thread_id"), py::arg("op"), py::arg("addr"));
    m.def("write_gmem_trace", &write_gmem_trace, py::arg("filename"));
    
    m.def("compute_unique_sorted_coords", &compute_unique_sorted_coords, 
          py::arg("in_coords"), py::arg("stride"));
    
    m.def("build_coordinate_queries", &build_coordinate_queries,
          py::arg("uniq_coords"), py::arg("stride"), py::arg("off_coords"));

    m.def("create_tiles_and_pivots", &create_tiles_and_pivots,
          py::arg("uniq_coords"), py::arg("tile_size"));

    m.def("perform_coordinate_lookup", &perform_coordinate_lookup,
          py::arg("uniq_coords"), py::arg("qry_keys"), py::arg("qry_in_idx"),
          py::arg("qry_off_idx"), py::arg("wt_offsets"), py::arg("tiles"),
          py::arg("pivs"), py::arg("tile_size"));

    m.def("write_kernel_map_to_gz", &write_kernel_map_to_gz,
          py::arg("kmap_data"), py::arg("filename"), py::arg("off_list"));

    // Expose constants
    m.attr("NUM_THREADS") = py::int_(NUM_THREADS);
    m.attr("I_BASE") = py::int_(I_BASE);
    m.attr("TILE_BASE") = py::int_(TILE_BASE);
    m.attr("QK_BASE") = py::int_(QK_BASE);
    m.attr("QI_BASE") = py::int_(QI_BASE);
    m.attr("QO_BASE") = py::int_(QO_BASE);
    m.attr("PIV_BASE") = py::int_(PIV_BASE);
    m.attr("KM_BASE") = py::int_(KM_BASE);
    m.attr("WO_BASE") = py::int_(WO_BASE);
    m.attr("SIZE_KEY") = py::int_(SIZE_KEY);
    m.attr("SIZE_INT") = py::int_(SIZE_INT);

    // Expose PHASES, TENSORS, OPS (forward maps)
    m.attr("PHASES") = PHASES.forward;
    m.attr("TENSORS") = TENSORS.forward;
    m.attr("OPS") = OPS.forward;
    
    // Utility pack/unpack functions
    m.def("pack32", &pack32, py::arg("c1"), py::arg("c2"), py::arg("c3"));
    m.def("unpack32", [](uint32_t key) {
        auto [c1, c2, c3] = unpack32(key);
        return std::make_tuple(c1, c2, c3);
    }, py::arg("key"));
    m.def("unpack32s", [](uint32_t key) {
        auto [c1, c2, c3] = unpack32s(key);
        return std::make_tuple(c1, c2, c3);
    }, py::arg("key"));
}
