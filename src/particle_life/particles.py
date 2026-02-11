from dataclasses import dataclass

import numpy as np


@dataclass
class Particle:
    m: float
    pos: np.ndarray  # shape (2,)
    vel: np.ndarray  # shape (2,)
    state: np.ndarray  # shape (D,)

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.vel = np.asarray(self.vel, dtype=np.float64)
        self.state = np.asarray(self.state, dtype=np.float64)

        if self.m <= 0:
            raise ValueError("Mass must be positive")

        if self.pos.shape != (2,):
            raise ValueError("pos must be shape (2,)")

        if self.vel.shape != (2,):
            raise ValueError("vel must be shape (2,)")

        if self.state.ndim != 1 or self.state.size == 0:
            raise ValueError("state must be 1D and non-empty")


def coupling(p: Particle, q: Particle) -> float:
    d = p.state.size
    return float(np.tanh(np.dot(p.state, q.state) / max(d, 1)))


def pair_force(
    p: Particle,
    q: Particle,
    r_min: float = 0.02,
    r0: float = 0.08,
    r_cut: float = 0.25,
    k_rep: float = 1.0,
    k_mid: float = 0.5,
) -> np.ndarray:
    delta = q.pos - p.pos
    r = np.linalg.norm(delta)

    if r == 0:
        return np.zeros(2, dtype=np.float64)

    unit = delta / r

    if r < r_min:
        mag = k_rep * (r_min - r) / r_min
    elif r < r_cut:
        c = coupling(p, q)
        mag = k_mid * c * (1 - r / r_cut) * (r0 - r) / r0
    else:
        mag = 0.0

    # positive mag => repulsive
    return -mag * unit


def total_force(
    p: Particle,
    q: Particle,
    gamma: float = 0.2,
    sigma: float = 0.05,
    rng: np.random.Generator | None = None,
    **force_params,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    f_det = pair_force(p, q, **force_params)

    damping = -gamma * p.vel

    noise = sigma * rng.normal(0.0, 1.0, size=2)

    return f_det + damping + noise
