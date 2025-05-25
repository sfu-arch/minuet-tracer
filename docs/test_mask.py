import numpy as np
from typing import Sequence, Any, Mapping

class OffsetIndex:
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

    def __repr__(self):
        return (f"<OffsetIndex num_points={self.num_points} "
                f"universe={self.U} offsets={self.offsets[:5]}...>")

if __name__ == "__main__":
    import time, random

    # 1. parameters
    N = 1_000_000         # points
    U = 27                # offsets
    K = 8                 # valid per point
    Q = 10_000            # number of queries

    # 2. build universe of offsets (here just integers 0..26)
    offs = list(range(U))
    idx = OffsetIndex(N, offs)

    # 3. randomly assign K offsets to each point
    point_map = {}
    for i in range(N):
        point_map[i] = random.sample(offs, K)
    t0 = time.time()
    idx.bulk_add(point_map)
    t1 = time.time()
    μs = (t1 - t0) / N * 1e6
    print(f"Bulk add {N} points → {(t1 - t0):.3f}s total, {μs:.1f}µs/point")

    # 4. benchmark enumeration
    qs = [random.randrange(N) for _ in range(Q)]
    t0 = time.time()
    out = [idx.get_valid_offsets(q) for q in qs]
    t1 = time.time()
    μs = (t1 - t0) / Q * 1e6

    print(f"{Q} queries → {(t1 - t0):.3f}s total, {μs:.1f}µs/query")

    # 5. test membership
    test_i = qs[0]
    test_off = point_map[test_i][0]
    print(f"Point {test_i} valid offsets:", idx.get_valid_offsets(test_i))
    print(f"Check {test_off!r} →", idx.is_valid(test_i, test_off))
    
    # print(point_map)
    u32mask = np.zeros(N*U, dtype=np.uint32)
    t0 = time.time()
    for idx, offs in point_map.items():
        for off in offs:
            # print(idx, off)
            mask_idx = idx*U + off
            u32mask[mask_idx] = 1
    t1 = time.time()
    μs = (t1 - t0) / N * 1e6
    print(f"Bulk add mask {N} points → {(t1 - t0):.3f}s total, {μs:.1f}µs/point")


        
    # 6. test membership using uint32 mask
    t0 = time.time()
    results = []
    for q in qs:
        idx = q*U
        for i in range(idx, idx+U):
            if u32mask[i]:
                results.append(i)
    t1 = time.time()
    t1 = time.time()
    μs = (t1 - t0) / Q * 1e6
    print(f"Membership test {Q} points → {(t1 - t0):.3f}s total, {μs:.1f}µs/point")
