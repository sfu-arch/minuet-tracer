#ifndef COORD_HPP
#define COORD_HPP

#include <cstdint>
#include <iostream> // For std::ostream
#include <string>   // Not strictly needed by declarations but often by users
#include <tuple>    // For unpack functions
#include <vector>   // Required for std::vector if used in method signatures or returns

// --- Packing/Unpacking Declarations ---
// These functions are essential for Coord3D::to_key and Coord3D::from_key
uint32_t pack32(int c1, int c2, int c3);
std::tuple<int, int, int> unpack32(uint32_t key);
std::tuple<int, int, int> unpack32s(uint32_t key); // For signed unpacking

/**
 * @brief Represents a 3D coordinate.
 */
struct Coord3D {
    int x, y, z;

    Coord3D(int x_ = 0, int y_ = 0, int z_ = 0);

    Coord3D quantized(int stride) const;
    uint32_t to_key() const; // Uses pack32
    static Coord3D from_key(uint32_t key); // Uses unpack32s
    static Coord3D from_signed_key(uint32_t key); // Uses unpack32s

    Coord3D operator+(const Coord3D& other) const;

    friend std::ostream& operator<<(std::ostream& os, const Coord3D& c);
};

/**
 * @brief Represents a coordinate with an associated original index.
 */
struct IndexedCoord {
    Coord3D coord;
    int orig_idx; 
    uint32_t key_val; // Store the packed key

    IndexedCoord(Coord3D c = Coord3D(), int idx = -1);
    IndexedCoord(uint32_t k, int idx = -1); // Constructor from key

    uint32_t to_key() const;
    static IndexedCoord from_key_and_index(uint32_t key, int idx);
};

#endif // COORD_HPP
