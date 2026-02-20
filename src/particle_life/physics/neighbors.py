from __future__ import annotations

import numpy as np


_NEIGHBOR_OFFSETS = np.array(
    [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [-1, 1],
    ],
    dtype=np.int32,
)


def build_cell_index(positions: np.ndarray, world_size: float, cell_size: float):
    nx = max(1, int(np.floor(world_size / max(cell_size, 1e-6))))
    ny = nx
    scaled = positions * (nx / world_size)
    cell_xy = np.floor(scaled).astype(np.int32)
    np.clip(cell_xy, 0, nx - 1, out=cell_xy)
    cell_id = cell_xy[:, 0] + cell_xy[:, 1] * nx

    order = np.argsort(cell_id, kind="mergesort")
    sorted_cells = cell_id[order]
    unique_cells, starts, counts = np.unique(sorted_cells, return_index=True, return_counts=True)
    ends = starts + counts

    return {
        "nx": nx,
        "ny": ny,
        "order": order,
        "unique_cells": unique_cells,
        "starts": starts,
        "ends": ends,
    }


def build_pair_chunks(index: dict, max_pairs_per_chunk: int = 250_000):
    nx = int(index["nx"])
    unique_cells = index["unique_cells"]
    starts = index["starts"]
    ends = index["ends"]
    order = index["order"]

    if unique_cells.size == 0:
        return []

    cell_lookup = {int(c): i for i, c in enumerate(unique_cells.tolist())}

    left_chunks: list[np.ndarray] = []
    right_chunks: list[np.ndarray] = []
    buffered_pairs = 0
    chunks: list[tuple[np.ndarray, np.ndarray]] = []

    def flush():
        nonlocal buffered_pairs
        if not left_chunks:
            return
        chunks.append((np.concatenate(left_chunks), np.concatenate(right_chunks)))
        left_chunks.clear()
        right_chunks.clear()
        buffered_pairs = 0

    for idx, cell_id in enumerate(unique_cells.tolist()):
        start_a = int(starts[idx])
        end_a = int(ends[idx])
        if end_a - start_a <= 0:
            continue
        pa = order[start_a:end_a]

        cx = cell_id % nx
        cy = cell_id // nx
        for off in _NEIGHBOR_OFFSETS:
            nx_cell = int((cx + int(off[0])) % nx)
            ny_cell = int((cy + int(off[1])) % nx)
            other_id = nx_cell + ny_cell * nx
            other_idx = cell_lookup.get(other_id)
            if other_idx is None:
                continue

            start_b = int(starts[other_idx])
            end_b = int(ends[other_idx])
            pb = order[start_b:end_b]
            if pb.size == 0:
                continue

            if other_id == cell_id:
                if pa.size < 2:
                    continue
                tri = np.triu_indices(pa.size, k=1)
                left = pa[tri[0]]
                right = pa[tri[1]]
            else:
                left = np.repeat(pa, pb.size)
                right = np.tile(pb, pa.size)

            left_chunks.append(left)
            right_chunks.append(right)
            buffered_pairs += int(left.size)
            if buffered_pairs >= max_pairs_per_chunk:
                flush()

    flush()
    return chunks
