from __future__ import annotations

import numpy as np


EPS = 1e-12


def accumulate_forces(
    positions: np.ndarray,
    species: np.ndarray,
    pair_chunks: list[tuple[np.ndarray, np.ndarray]],
    matrix: np.ndarray,
    interaction_radius: float,
    repel_radius: float,
    force_scale: float,
    world_size: float,
    boundary_mode: str,
) -> np.ndarray:
    forces = np.zeros_like(positions, dtype=np.float32)
    if not pair_chunks:
        return forces

    inv_interaction = 1.0 / max(interaction_radius, 1e-8)
    inv_repel = 1.0 / max(repel_radius, 1e-8)

    for left, right in pair_chunks:
        delta = positions[right] - positions[left]
        if boundary_mode == "wrap":
            delta -= np.round(delta / world_size) * world_size

        dist2 = np.sum(delta * delta, axis=1)
        dist = np.sqrt(np.maximum(dist2, EPS))
        within = dist < interaction_radius
        if not np.any(within):
            continue

        l = left[within]
        r = right[within]
        d = delta[within]
        dd = dist[within]
        unit = d / dd[:, None]

        influence = np.clip(1.0 - dd * inv_interaction, 0.0, 1.0)
        repel = np.clip(1.0 - dd * inv_repel, 0.0, 1.0)
        interaction = matrix[species[l], species[r]]
        strength = (interaction * influence - 1.5 * repel) * force_scale

        f = unit * strength[:, None]
        np.add.at(forces, l, f)
        np.add.at(forces, r, -f)

    return forces
