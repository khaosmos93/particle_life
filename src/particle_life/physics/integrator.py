from __future__ import annotations

import numpy as np


def update_state(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    dt: float,
    damping: float,
    max_speed: float,
    world_size: float,
    boundary_mode: str,
    rng: np.random.Generator,
    noise_strength: float,
):
    velocities *= damping
    velocities += forces * dt
    if noise_strength > 0:
        velocities += rng.normal(0.0, noise_strength * np.sqrt(dt), velocities.shape).astype(np.float32)

    speed = np.linalg.norm(velocities, axis=1)
    over = speed > max_speed
    if np.any(over):
        velocities[over] *= (max_speed / np.maximum(speed[over], 1e-9))[:, None]

    positions += velocities * dt
    if boundary_mode == "wrap":
        np.mod(positions, world_size, out=positions)
    else:
        for axis in (0, 1):
            low = positions[:, axis] < 0
            high = positions[:, axis] > world_size
            mask = low | high
            if np.any(mask):
                positions[mask, axis] = np.clip(positions[mask, axis], 0, world_size)
                velocities[mask, axis] *= -1
