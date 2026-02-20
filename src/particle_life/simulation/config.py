from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimConfig:
    species_count: int = 6
    particles_per_species: int = 160
    particle_counts: list[int] = field(default_factory=list)
    world_size: float = 1.0
    interaction_radius: float = 0.11
    repel_radius: float = 0.025
    force_scale: float = 0.42
    dt: float = 0.015
    damping: float = 0.975
    max_speed: float = 0.05
    noise_strength: float = 0.0
    steps_per_frame: int = 1
    boundary_mode: str = "wrap"
    point_size: float = 3.0
    point_opacity: float = 0.95
    background_alpha: float = 1.0
    show_hud: bool = True
    pbc_tiling: bool = False
    color_mode: str = "species"
    type_colors: list[str] = field(default_factory=lambda: ["#ff6f5f", "#56c3ff", "#7aff63", "#ffe26a", "#d086ff", "#ffa04d", "#60ffd0", "#c8dbff", "#ff73b8", "#88ff9e", "#6f8bff", "#ffc56f"])
    seed: int = 0


def sanitize_particle_counts(counts, species_count: int, default_count: int) -> list[int]:
    if not isinstance(counts, (list, tuple)):
        counts = []
    sanitized = []
    for idx in range(species_count):
        raw = counts[idx] if idx < len(counts) else default_count
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = int(default_count)
        sanitized.append(int(max(0, min(10_000, value))))
    return sanitized


def sanitize_matrix(matrix, species_count: int):
    import numpy as np

    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape != (species_count, species_count):
        raise ValueError("matrix shape mismatch")
    if not np.all(np.isfinite(arr)):
        raise ValueError("matrix has non-finite values")
    return np.clip(arr, -1.0, 1.0).astype(np.float32)
