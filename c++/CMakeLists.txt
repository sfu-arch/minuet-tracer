cmake_minimum_required(VERSION 3.20)
project(MinuetTraceConverter)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find ZLIB (required for gzip compression)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED) 
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add the pybind11 module
# The first argument is the name of the module
pybind11_add_module(minuet_cpp_module
    src/minuet_map.cpp
    src/minuet_bindings.cpp # Add the new bindings file
    src/minuet_config.cpp   # Add the config source file
    src/coord.cpp # Add the coord source file
    src/minuet_gather.cpp # Add the gather source file
)

# Specify include directories
target_include_directories(minuet_cpp_module PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ZLIB_INCLUDE_DIRS} # Add ZLIB include directory
    ${pybind11_INCLUDE_DIRS} # Add pybind11 include directory
)

# Link necessary libraries
target_link_libraries(minuet_cpp_module PUBLIC
    ${ZLIB_LIBRARIES}   # Link ZLIB
    Threads::Threads    # Link pthreads or equivalent for std::thread
)

# Optional: Enable C++17 filesystem support if needed by the compiler/platform
# (Not strictly necessary for this specific code but good practice for C++17 projects)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.12")
    target_compile_features(minuet_cpp_module PUBLIC cxx_std_17)
endif()

# Remove the old executable target if it exists, or comment it out
add_executable(main_minuet_cpp
    src/main.cpp
    src/minuet_map.cpp
    src/minuet_config.cpp
    src/coord.cpp
    src/minuet_gather.cpp   
)
target_include_directories(main_minuet_cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ZLIB_INCLUDE_DIRS}
)
target_link_libraries(main_minuet_cpp PUBLIC
    ${ZLIB_LIBRARIES}
    Threads::Threads
)

# Remove the old executable target if it exists, or comment it out
add_executable(mem_trace_reader
    src/mem_trace_reader.cpp
)
target_include_directories(mem_trace_reader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${ZLIB_INCLUDE_DIRS}
)
# Trace reader is standalone, so it should not need any includes.
target_link_libraries(mem_trace_reader PUBLIC
    ${ZLIB_LIBRARIES}
    Threads::Threads
)
