- [Design doc minuet](#design-doc-minuet)
  - [Metadata Arrays Documentation](#metadata-arrays-documentation)
    - [Mask Arrays](#mask-arrays)
    - [ASCII Illustration](#ascii-illustration)
      - [`in_mask` (3 × 2 → 6 entries)](#in_mask-3--2--6-entries)
      - [`offsets_active`](#offsets_active)
    - [Grouping for GEMM : greedy\_group](#grouping-for-gemm--greedy_group)

# Design doc minuet

## Kernel Map

The kernel map is a data structure that defines the relationships between input and output elements in the GEMM operation. It is represented as a dictionary where each key corresponds to an output offset, and the value is a list of tuples. Each tuple contains an input index and the corresponding output index. The offset indexes are a mapping back to the orignal offset list. 

Example:

```python
kernel_map = {
  0: [(in0, out2), (in1, out0)],
  1: [],
  2: [(in1, out3)]
}
```

## Metadata Arrays Documentation

This document describes the format of the metadata arrays returned by the `build_masks` function.

### Mask Arrays

1. **`out_mask`**

   * Type: `List[int]`
   * Length: `num_offsets * num_sources`
   * View as a 2D array with dimensions `[num_offsets][num_sources]`, flattened row-major.
   * Access element `(o, s)` at index:

     ```plaintext
     out_mask[o * num_sources + s]
     ```
   * Value: index of out `s` in `kernel_map[o]`, or `(-1,-1)` if absent. Otherwise its a tuple `out_idx, slot_idx` where `out_idx` is the index in the original unsorted point array for scattering, and `slot_idx` is the index within the allocated gemm buffer.

`out_mask` (3 × 4 → 12 entries)

```python
# Example Values
num_offsets = 3
num_points = 4
kernel_map = {
  0: [(in0, out2), (in1, out0)],
  1: [],
  2: [(in1, out3)]
}
```

```
2D view (o → rows, s → cols):
   s=0  1   2   3
o=0 [ (1,0), (-1,-1), (2,0), (-1,-1)]
o=1 [(-1,-1), (-1,-1), (-1,-1), (-1,-1)]
o=2 [(-1,-1), (-1,-1), (-1,-1),  (0,0)]

Flat array:
[(1,0), (-1,-1), (2,0), (-1,-1), (-1,-1), (-1,-1), (-1,-1), (-1,-1), (-1,-1), (-1,-1), (-1,-1),  (0,0)]
```


2. **`in_mask`**

   * Type: `List[int]`
   * Length: `num_offsets * num_targets`
   * View as a 2D array with dimensions `[num_offsets][num_targets]`, flattened row-major.
   * Access element `(o, t)` at index:

     ```plaintext
     in_mask[o * num_targets + t]
     ```
   * Value: index of in `t` in `kernel_map[o]`, or `-1` if absent. Otherwise each entry is a tuple of the form (`in_idx`, `slot_idx`), where `in_idx` is the index of the input point in the original unsorted point array, and `slot_idx` is the index within the allocated gemm buffer for that offset. 
  
3. **`offsets_active`**

   * Type: `List[int]`
   * Contains all offset indices `o` for which `kernel_map[o]` is non-empty (in descending order). Offset with more matches are listed first.

```
[0, 2]  # only offsets with matches
```


4. **`slot_array`**

   * Type: `List[int]`
   * Length: `num_offsets + 1`
   * Prefix-sum array: cumulative counts of matches per offset.
   * Definition:

     ```plaintext
     slot_array[0] = 0
     slot_array[i] = sum_{k=0}^{i-1} count(kernel_map[k])
     ```

`slot_array` (len=4)

```
counts per offset = [2, 0, 1]
prefix-sums    = [0, 2, 2, 3] 

slot_array = [0, 2, 2, 3]
```


### ASCII Illustration

Given:




#### `in_mask` (3 × 2 → 6 entries)

```
2D view (o → rows, t → cols):
   t=0  1
o=0 [ 0,  1]
o=1 [-1, -1]
o=2 [-1,  0]

Flat array:
[ 0, 1,   -1, -1,   -1, 0 ]
```

#### `offsets_active`


### Grouping for GEMM : greedy_group

The `greedy_group` function returns three arrays describing the grouping of positions after sorting by descending slot requirement:

1. **`pos_indices`**

   * Type: `List[int]`, length = number of positions (n).
   * `pos_indices[i]` is the **absolute slot index** at which the original position `i` begins within its assigned group.
   * Type: `List[Tuple[List[int], int, int, int]]`
     Each element is a tuple:
     * `members`: `List[int]` of original position indices in that group (in processing order).
     * `base_addr`: `int` starting slot index of the group (aligned).
     * `req`: `int` total raw slots required by all positions in `members`.
     * `alloc`: `int` `ceil(req/alignment)*alignment`, the padded allocation size.


2. **`membership`**

   * Type: `List[List[int]]`
   * Each sublist is identical to the `members` list in each corresponding `groups` tuple.
   * Provides a clear, index‐based view of which positions belong to which group.
   * Example:

These outputs can be used to drive visualizations, memory simulations, or further alignment logic.



```bash
Offset_IDX, [List of tuples of (in, out) pairs. For each point we list coordinate and the index of the point in the original unsorted point cloud list]
13: [(((0, 0, 2), 1), ((0, 0, 2), 1)), (((0, 0, 3), 3), ((0, 0, 3), 3)), (((0, 1, 1), 2), ((0, 1, 1), 2)), (((1, 5, 0), 0), ((1, 5, 0), 0))], 

11: [(((0, 0, 2), 1), ((0, 1, 1), 2))], 

12, [(((0, 0, 2), 1), ((0, 0, 3), 3))], 
14, [(((0, 0, 3), 3), ((0, 0, 2), 1))], 
15, [(((0, 1, 1), 2), ((0, 0, 2), 1))],

# After grouping

Groups metadata ([start, end], base, req, alloc):
([0, 0], 0, 4, 4)
([1, 2], 4, 2, 4)
([3, 4], 8, 2, 4)

# The alloc (size of gemm in points). Size of A matrix.  [start, end] the weight offsets. Size of B matrix. \# of columns in B matrix 

Total allocated space: 12 slots

Per-position slot indices:
[0, 4, 5, 8, 9]

List of gemms. Group membership lists:
[[0, 0], [1, 2], [3, 4]]
```