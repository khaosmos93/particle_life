from __future__ import annotations

from dataclasses import asdict

import numpy as np

from particle_life.physics.integrator import update_state
from particle_life.physics.kernels import accumulate_forces
from particle_life.physics.neighbors import build_cell_index, build_pair_chunks
from particle_life.simulation.config import SimConfig, sanitize_particle_counts


class ParticleLifeSim:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.paused = False
        self.matrix_version = 0
        self.reset_state(random_matrix=True)

    def reset_state(self, random_matrix: bool = False) -> None:
        cfg = self.cfg
        cfg.particle_counts = sanitize_particle_counts(cfg.particle_counts, cfg.species_count, cfg.particles_per_species)
        self.count = int(sum(cfg.particle_counts))
        self.positions = self.rng.random((self.count, 2), dtype=np.float32) * cfg.world_size
        self.velocities = np.zeros((self.count, 2), dtype=np.float32)
        self.species = np.concatenate(
            [np.full(count, species_idx, dtype=np.int32) for species_idx, count in enumerate(cfg.particle_counts) if count > 0]
        ) if self.count else np.empty((0,), dtype=np.int32)
        self.rng.shuffle(self.species)
        if random_matrix or not hasattr(self, "matrix") or self.matrix.shape[0] != cfg.species_count:
            self.matrix = self.rng.uniform(-1.0, 1.0, (cfg.species_count, cfg.species_count)).astype(np.float32)
            np.fill_diagonal(self.matrix, self.rng.uniform(0.2, 1.0, cfg.species_count))

    def load_state(self, cfg: SimConfig, matrix: np.ndarray, positions: np.ndarray, velocities: np.ndarray, species: np.ndarray) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.count = int(species.shape[0])
        self.positions = positions.astype(np.float32, copy=True)
        self.velocities = velocities.astype(np.float32, copy=True)
        self.species = species.astype(np.int32, copy=True)
        self.matrix = matrix.astype(np.float32, copy=True)
        self.matrix_version += 1

    def set_matrix(self, matrix: np.ndarray) -> None:
        self.matrix = matrix.astype(np.float32, copy=True)
        self.matrix_version += 1

    def matrix_values(self) -> list[list[float]]:
        return self.matrix.astype(float).tolist()

    def step(self) -> None:
        cfg = self.cfg
        index = build_cell_index(self.positions, cfg.world_size, cfg.interaction_radius)
        chunks = build_pair_chunks(index)
        forces = accumulate_forces(
            self.positions,
            self.species,
            chunks,
            self.matrix,
            cfg.interaction_radius,
            cfg.repel_radius,
            cfg.force_scale,
            cfg.world_size,
            cfg.boundary_mode,
        )
        update_state(
            self.positions,
            self.velocities,
            forces,
            cfg.dt,
            cfg.damping,
            cfg.max_speed,
            cfg.world_size,
            cfg.boundary_mode,
            self.rng,
            cfg.noise_strength,
        )

    def step_many(self, steps: int) -> None:
        for _ in range(int(steps)):
            self.step()

    def information_entropy(self, bins_per_dim: int = 16) -> float:
        if self.count <= 0:
            return 0.0
        hist, _ = np.histogramdd(self.positions, bins=[bins_per_dim, bins_per_dim], range=[(0.0, float(self.cfg.world_size))] * 2)
        probs = hist.ravel().astype(np.float64)
        probs /= float(self.count)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    def snapshot(self) -> bytes:
        data = np.empty((self.count, 5), dtype=np.float32)
        data[:, :2] = self.positions / self.cfg.world_size
        data[:, 2] = self.species.astype(np.float32)
        data[:, 3:5] = self.velocities
        return data.tobytes()

    def metadata_dict(self) -> dict:
        values = asdict(self.cfg)
        values["interaction_matrix"] = self.matrix_values()
        values["particle_count"] = int(self.count)
        return values
