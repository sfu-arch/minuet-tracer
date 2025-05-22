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
# Add the executable
add_executable(minuet_trace_cpp
    src/main.cpp
    src/minuet_trace.cpp
)

# Specify include directories
# This allows #include "minuet_trace.hpp" to work from src/*.cpp
target_include_directories(minuet_trace_cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ZLIB_INCLUDE_DIRS} # Add ZLIB include directory
)

# Link necessary libraries
target_link_libraries(minuet_trace_cpp PUBLIC
    ${ZLIB_LIBRARIES}   # Link ZLIB
    Threads::Threads    # Link pthreads or equivalent for std::thread
)

# Optional: Enable C++17 filesystem support if needed by the compiler/platform
# (Not strictly necessary for this specific code but good practice for C++17 projects)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.12")
    target_compile_features(minuet_trace_cpp PUBLIC cxx_std_17)
endif()