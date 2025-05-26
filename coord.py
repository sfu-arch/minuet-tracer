from dataclasses import dataclass
from typing import Tuple, Union
from typing import List, Tuple, Dict, Set
import gzip
import struct
from typing import Sequence, Any, Mapping

# ── Helper: pack/unpack 32-bit keys ──
def pack32(*coords):
    key = 0
    for c in coords:
        key = (key << 10) | (c & 0x3FF)
    return key

def unpack32(key):
    x = key & 0x3FF; key >>= 10
    y = key & 0x3FF; key >>= 10
    z = key & 0x3FF; key >>= 10 
    return (x, y, z)

def unpack32s(key):
    """Unpack signed 10 bit integers"""
    x = key & 0x3FF
    x = x if x < 512 else x - 1024
    key >>= 10

    y = key & 0x3FF
    y = y if y < 512 else y - 1024
    key >>= 10

    z = key & 0x3FF
    z = z if z < 512 else z - 1024
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


# Assumes that write and read phases are completely separate.
# Concurrent mask index.
class mask_index:
    def __init__(self, num_points: int, offsets: Sequence[Any]):
        """
        num_points: total number of points (indexed 0..num_points-1)
        offsets:    fixed list of all possible offsets (len U)
        """
        self.num_points = num_points
        self.offsets = list(offsets)
        self.U = len(self.offsets)

        # maps from offset value → bit index and back
        self._off2bit = {off: i for i, off in enumerate(self.offsets)}
        self._bit2off = {i: off for i, off in enumerate(self.offsets)}
        self.pointlocks = [threading.Lock() for _ in range(num_points)]

        # Choose storage strategy based on universe size
        if self.U <= 32:
            self._mask = np.zeros(self.num_points, dtype=np.uint32)
            self._large = False
        elif self.U <= 64:
            self._mask = np.zeros(self.num_points, dtype=np.uint64)
            self._large = False
        else:
            # For U > 64, use a byte array per point
            self._nbytes = (self.U + 7) // 8
            self._mask = np.zeros((self.num_points, self._nbytes), dtype=np.uint8)
            self._large = True

    def add_point_offsets(self, point_idx: int, valid_offsets: Sequence[Any]) -> None:
        """Set exactly those offsets as valid for this point (overwrites previous)."""
        self.pointlocks[point_idx].acquire()
        if not self._large:
            m = 0
            for off in valid_offsets:
                b = self._off2bit[off]
                m |= (1 << b)
            self._mask[point_idx] = m
        else:
            row = self._mask[point_idx]
            row[:] = 0
            for off in valid_offsets:
                b = self._off2bit[off]
                byte_idx, bit = divmod(b, 8)
                row[byte_idx] |= (1 << bit)
        self.pointlocks[point_idx].release()
        
    def is_valid(self, point_idx: int, offset: Any) -> bool:
        """Test if `offset` is valid for point `point_idx`."""
        b = self._off2bit[offset]
        if not self._large:
            return bool(self._mask[point_idx] & (1 << b))
        else:
            byte_idx, bit = divmod(b, 8)
            return bool(self._mask[point_idx, byte_idx] & (1 << bit))

    def get_valid_offsets(self, point_idx: int) -> list[Any]:
        """Return the list of offsets valid for point `point_idx`."""
        result = []
        if not self._large:
            m = int(self._mask[point_idx])
            for b in range(self.U):
                if (m >> b) & 1:
                    result.append(self._bit2off[b])
        else:
            row = self._mask[point_idx]
            for b in range(self.U):
                byte_idx, bit = divmod(b, 8)
                if row[byte_idx] & (1 << bit):
                    result.append(self._bit2off[b])
        return result

    def bulk_add(self, mapping: Mapping[int, Sequence[Any]]) -> None:
        """Initialize many points at once from a {point_idx: [offset,…]} dict."""
        for idx, offs in mapping.items():
            self.add_point_offsets(idx, offs)

    def gz_write(self, f):
       # Write all points to the file
        if self._large:
            for row in self._mask:
                f.write(row.tobytes())
        else:
            f.write(struct.pack('!I', self.num_points))

    def __repr__(self):
        return (f"<OffsetIndex num_points={self.num_points} "
                f"universe={self.U} offsets={self.offsets[:5]}...>")

