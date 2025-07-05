from minuet_mapping import *
from minuet_gather import *
from minuet_config import *
from read_pcl import *
import os
from pathlib import Path
import numpy as np
import json


def sample_dataset():
    script_dir = Path(__file__).parent.resolve()
    src_path = (script_dir / '../Datasets/Data/dataset/sequences/00/voxels').resolve()
    dest_path = (script_dir / 'examples').resolve()
    sample_point_clouds(src_path, dest_path, 10)
    return dest_path


def load_and_visualize_sample(dest_path):
    sample_path = (dest_path / '000000.simbin').resolve()
    in_coords, features = read_simbin(sample_path)
    visualize_point_cloud(in_coords)
    return in_coords


def run_mapping_phase(in_coords, stride, off_coords):
    print(f"\n--- Phase: Sort Unique Coords with {NUM_THREADS} threads ---")
    uniq_coords = compute_unique_sorted_coords(in_coords, stride)

    print('--- Phase: Build Queries ---')
    qry_keys, qry_in_idx, qry_off_idx, wt_offsets = build_coordinate_queries(
        uniq_coords, stride, off_coords
    )

    print('--- Phase: Sort QKeys ---')
    # No sorting needed

    print('--- Phase: Make Tiles & Pivots ---')
    coord_tiles, pivs = create_tiles_and_pivots(uniq_coords, NUM_PIVOTS)

    print('--- Phase: Lookup ---')
    kmap = lookup(
        uniq_coords, qry_keys, qry_in_idx,
        qry_off_idx, wt_offsets, coord_tiles, pivs, 2
    )

    return uniq_coords, qry_keys, qry_in_idx, qry_off_idx, kmap


def debug_mapping(uniq_coords, qry_keys, qry_off_idx, kmap, off_coords):
    if not debug:
        return

    print('\nSorted Source Array (Coordinate, Original Index):')
    for idxc_item in uniq_coords:
        coord = idxc_item.coord
        print(f"  key={hex(coord.to_key())}, coords=({coord.x}, {coord.y}, {coord.z}), index={idxc_item.orig_idx}")

    print('\nQuery Segments:')
    for off_idx in range(len(off_coords)):
        segment = [qry_keys[i] for i in range(len(qry_keys)) if qry_off_idx[i] == off_idx]
        print(f"  Offset {off_coords[off_idx]}: {[(idxc.coord.x, idxc.coord.y, idxc.coord.z) for idxc in segment]}")

    print('\nKernel Map:')
    for off_idx, matches in kmap.items():
        print(f"  Offset {off_coords[off_idx]}: {matches}")


def write_mapping_results(kmap, off_coords, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    map_trace_checksum = write_gmem_trace(output_dir + 'map_trace.bin.gz')
    write_kernel_map_to_gz(kmap, output_dir + 'kernel_map.bin.gz', off_coords)
    return map_trace_checksum


def generate_gather_metadata(kmap, off_coords, uniq_coords, output_dir):
    kmap._invalidate_cache()
    kmap._get_sorted_keys()
    offsets_active = list(kmap._get_sorted_keys())
    slot_array = [len(kmap[off_idx]) for off_idx in offsets_active]

    if debug:
        print("Offsets sorted by matches count:", offsets_active)
        print("Slot array:", slot_array)

    slot_indices, groups, membership, gemm_list, total_slots, gemm_checksum = greedy_group(
        slot_array,
        alignment=GEMM_ALIGNMENT,
        max_group=GEMM_WT_GROUP,
        max_slots=GEMM_SIZE,
    )

    slot_dict = {offsets_active[i]: slot_indices[i] for i in range(len(slot_indices))}
    out_mask, in_mask = create_in_out_masks(kmap, slot_dict, len(off_coords), len(uniq_coords))

    metadata_checksum = write_metadata(
        out_mask, in_mask, slot_dict, slot_array,
        len(off_coords), len(uniq_coords), total_slots,
        filename=output_dir + 'metadata.bin.gz'
    )

    return out_mask, in_mask, total_slots, gemm_checksum, metadata_checksum, slot_dict, groups, gemm_list


def run_gather_and_scatter(out_mask, in_mask, uniq_coords, off_coords, total_slots, output_dir, groups, gemm_list):
    gemm_buffer = np.zeros(total_slots * TOTAL_FEATS_PT, dtype=np.uint16)

    if debug:
        print("Groups metadata ([start, end], base, req, alloc):")
        for g in groups:
            print(g)
        print("GEMM List:")
        for g in gemm_list:
            print(g)

    mt_gather(
        num_threads=1,
        num_points=len(uniq_coords),
        num_offsets=len(off_coords),
        num_tiles_per_pt=minuet_config.NUM_TILES_GATHER,
        tile_feat_size=minuet_config.TILE_FEATS_GATHER,
        bulk_feat_size=minuet_config.BULK_FEATS_GATHER,
        source_masks=in_mask,
        sources=None,
        gemm_buffers=gemm_buffer
    )
    gather_checksum = write_gmem_trace(output_dir + 'gather_trace.bin.gz', sizeof_addr=8)

    mt_scatter(
        num_threads=2,
        num_points=len(uniq_coords),
        num_offsets=len(off_coords),
        num_tiles_per_pt=minuet_config.NUM_TILES_GATHER,
        tile_feat_size=minuet_config.TILE_FEATS_GATHER,
        bulk_feat_size=minuet_config.BULK_FEATS_GATHER,
        out_mask=out_mask,
        gemm_buffers=None,
        outputs=None
    )
    scatter_checksum = write_gmem_trace(output_dir + 'scatter_trace.bin.gz', sizeof_addr=8)

    return gather_checksum, scatter_checksum


def write_checksums(map_trace_checksum, metadata_checksum, gemm_checksum, gather_checksum, scatter_checksum, output_dir):
    checksums = {
        "map_trace.bin.gz": map_trace_checksum,
        "metadata.bin.gz": metadata_checksum,
        "gemms.bin.gz": gemm_checksum,
        "gather_trace.bin.gz": gather_checksum,
        "scatter_trace.bin.gz": scatter_checksum,
    }
    with open(output_dir + 'checksums.json', 'w') as f:
        json.dump(checksums, f, indent=2)


if __name__ == '__main__':
    global phase
    stride = 1
    off_coords = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

    dest_path = sample_dataset()
    in_coords = load_and_visualize_sample(dest_path)

    uniq_coords, qry_keys, qry_in_idx, qry_off_idx, kmap = run_mapping_phase(in_coords, stride, off_coords)

    debug_mapping(uniq_coords, qry_keys, qry_off_idx, kmap, off_coords)

    map_trace_checksum = write_mapping_results(kmap, off_coords, output_dir)

    out_mask, in_mask, total_slots, gemm_checksum, metadata_checksum, slot_dict, groups, gemm_list = generate_gather_metadata(
        kmap, off_coords, uniq_coords, output_dir
    )

    gather_checksum, scatter_checksum = run_gather_and_scatter(
        out_mask, in_mask, uniq_coords, off_coords, total_slots, output_dir, groups, gemm_list
    )

    write_checksums(map_trace_checksum, metadata_checksum, gemm_checksum, gather_checksum, scatter_checksum, output_dir)
