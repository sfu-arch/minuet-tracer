from dataclasses import dataclass
from typing import Tuple, Union
from typing import List, Tuple, Dict, Set

# ── Helper: pack/unpack 32-bit keys ──
def pack32(*coords):
    key = 0
    for c in coords:
        key = (key << 10) | (c & 0x3FF)
    return key

def unpack32(key):
    z = key & 0x3FF; key >>= 10
    y = key & 0x3FF; key >>= 10
    x = key & 0x3FF; key >>= 10 
    return (x, y, z)

def unpack32s(key):
    """Unpack signed 10 bit integers"""
    z = key & 0x3FF
    z = z if z < 512 else z - 1024
    key >>= 10

    y = key & 0x3FF
    y = y if y < 512 else y - 1024
    key >>= 10

    x = key & 0x3FF
    x = x if x < 512 else x - 1024
    return (x, y, z)

@dataclass
class Coord3D:
    """Three-dimensional coordinate representation"""
    x: int
    y: int
    z: int

    def quantized(self, stride: int) -> 'Coord3D':
        """Return quantized coordinates based on stride"""
        return Coord3D(self.x // stride, self.y // stride, self.z // stride)
    
    def to_key(self) -> int:
        """Convert coordinate to packed 32-bit key"""
        return pack32(self.x, self.y, self.z)
    
    @classmethod
    def from_key(cls, key: int) -> 'Coord3D':
        """Create coordinate from packed key"""
        x, y, z = unpack32s(key)
        return cls(x, y, z)
    
    @classmethod
    def from_signed_key(cls, key: int) -> 'Coord3D':
        """Create coordinate from packed key with signed values"""
        x, y, z = unpack32s(key)
        return cls(x, y, z)
    
    def __add__(self, other: 'Coord3D') -> 'Coord3D':
        """Add two coordinates"""
        return Coord3D(self.x + other.x, self.y + other.y, self.z + other.z)
