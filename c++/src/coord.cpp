#include "coord.hpp" // Self include
#include <stdexcept> // For potential errors
#include <sstream>   // For ostream in Coord3D operator<<

// --- Coord3D method definitions ---
Coord3D::Coord3D(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}

Coord3D Coord3D::quantized(int stride) const {
    if (stride == 0) return *this; // Avoid division by zero
    return Coord3D(x / stride, y / stride, z / stride);
}

uint32_t Coord3D::to_key() const {
    return pack32(x, y, z); 
}

Coord3D Coord3D::from_key(uint32_t key) {
  auto [ux, uy, uz] = unpack32s(key); // Consistently use signed unpack for from_key
  return Coord3D(ux, uy, uz);
}

Coord3D Coord3D::from_signed_key(uint32_t key) {
  auto [sx, sy, sz] = unpack32s(key);
  return Coord3D(sx, sy, sz);
}

Coord3D Coord3D::operator+(const Coord3D& other) const {
    return Coord3D(x + other.x, y + other.y, z + other.z);
}

std::ostream& operator<<(std::ostream& os, const Coord3D& c) {
    os << "(" << c.x << ", " << c.y << ", " << c.z << ")";
    return os;
}

// --- IndexedCoord method definitions ---
IndexedCoord::IndexedCoord(Coord3D c, int idx) : coord(c), orig_idx(idx) {
    key_val = coord.to_key();
}

IndexedCoord::IndexedCoord(uint32_t k, int idx) : key_val(k), orig_idx(idx) {
    coord = Coord3D::from_key(k); // Create Coord3D from key
}

uint32_t IndexedCoord::to_key() const {
    return key_val;
}

IndexedCoord IndexedCoord::from_key_and_index(uint32_t key, int idx) {
    return IndexedCoord(key, idx); // Use the constructor that takes key and index
}

// --- Packing/Unpacking (10-bit fields) ---

// Helper function to convert value to hex string (moved from main.cpp for broader use)

uint32_t pack32(int c1, int c2, int c3) {
  // Packs three 10-bit integer coordinates into a single 30-bit key within a
  // uint32_t. c1: bits 20-29, c2: bits 10-19, c3: bits 0-9
  uint32_t key = 0;
  key = (key << 10) | (static_cast<uint32_t>(c1) & 0x3FF);
  key = (key << 10) | (static_cast<uint32_t>(c2) & 0x3FF);
  key = (key << 10) | (static_cast<uint32_t>(c3) & 0x3FF);
  return key;
}

std::tuple<int, int, int> unpack32(uint32_t key) {
  // Unpacks a 30-bit key (stored in uint32_t) into three 10-bit integer
  // coordinates. Assumes c3 is LSB, c1 is MSB of the 30-bit value.
  int c3 = static_cast<int>(key & 0x3FF);
  key >>= 10;
  int c2 = static_cast<int>(key & 0x3FF);
  key >>= 10;
  int c1 = static_cast<int>(key & 0x3FF);
  return std::make_tuple(c1, c2, c3);
}

std::tuple<int, int, int> unpack32s(uint32_t key) {
  // Unpacks a 30-bit key into three 10-bit signed integers.
  // Sign extension for 10-bit numbers: if value >= 512 (0x200), it's negative.
  // Subtract 1024 (0x400).
  uint32_t temp_key = key;

  int c3_val = static_cast<int>(temp_key & 0x3FF);
  c3_val = (c3_val < 512) ? c3_val : c3_val - 1024;
  temp_key >>= 10;

  int c2_val = static_cast<int>(temp_key & 0x3FF);
  c2_val = (c2_val < 512) ? c2_val : c2_val - 1024;
  temp_key >>= 10;

  int c1_val = static_cast<int>(temp_key & 0x3FF);
  c1_val = (c1_val < 512) ? c1_val : c1_val - 1024;

  return std::make_tuple(c1_val, c2_val, c3_val);
}
