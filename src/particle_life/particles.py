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


def minimum_image(delta: np.ndarray, box_size: float) -> np.ndarray:
    return delta - box_size * np.round(delta / box_size)


def build_cell_list(
    pos: np.ndarray, box_size: float, r_cut: float
) -> tuple[dict[tuple[int, int], list[int]], int]:
    cell_size = r_cut
    ncell = max(1, int(np.floor(box_size / cell_size)))
    effective_cell_size = box_size / ncell

    cx = np.floor(pos[:, 0] / effective_cell_size).astype(int) % ncell
    cy = np.floor(pos[:, 1] / effective_cell_size).astype(int) % ncell

    cells: dict[tuple[int, int], list[int]] = {}
    for idx, (x, y) in enumerate(zip(cx, cy)):
        key = (int(x), int(y))
        cells.setdefault(key, []).append(idx)

    return cells, ncell


def neighbor_pairs(
    pos: np.ndarray, box_size: float, r_cut: float
) -> tuple[np.ndarray, np.ndarray]:
    cells, ncell = build_cell_list(pos, box_size, r_cut)
    pairs: list[tuple[int, int]] = []

    for (cx, cy), i_list in cells.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx = (cx + dx) % ncell
                ny = (cy + dy) % ncell
                j_list = cells.get((nx, ny), [])
                for i in i_list:
                    for j in j_list:
                        if i != j:
                            pairs.append((i, j))

    if not pairs:
        empty = np.empty(0, dtype=int)
        return empty, empty

    pair_arr = np.asarray(pairs, dtype=int)
    return pair_arr[:, 0], pair_arr[:, 1]


def coupling_state(state_i: np.ndarray, state_j: np.ndarray) -> float:
    # d = max(1, state_i.size)
    # c = float(np.tanh(np.dot(state_i, state_j) / d))
    c = float(0.0)
    if np.array_equal(state_i, state_j):
        c = 0.1
    elif state_i[0] > state_j[0]:
        c = 1.0
    else:
        c = -0.5
    return c

# FIXME
def pair_force_from_delta(
    delta: np.ndarray,
    c: float,
    r_min: float,
    r0: float,
    r_cut: float,
    k_rep: float,
    k_mid: float,
) -> np.ndarray:
    r = np.linalg.norm(delta)
    if r == 0 or r >= r_cut:
        return np.zeros(2, dtype=np.float64)

    unit = delta / r
    if r < r_min:
        mag = k_rep * (r_min - r) / r_min
    else:
        mag = k_mid * c * (1 - r / r_cut) * (r0 - r) / r0

    return -mag * unit

# FIXME: this is obsolute
def compute_net_forces(
    particles: list[Particle],
    box_size: float,
    wrap: bool,
    r_min: float,
    r0: float,
    r_cut: float,
    k_rep: float,
    k_mid: float,
    gamma: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(particles)
    pos = np.asarray([p.pos for p in particles], dtype=np.float64)
    i_idx, j_idx = neighbor_pairs(pos, box_size, r_cut)
    forces = np.zeros((n, 2), dtype=np.float64)

    for i, j in zip(i_idx, j_idx):
        delta = pos[j] - pos[i]
        if wrap:
            delta = minimum_image(delta, box_size)

        r = np.linalg.norm(delta)
        if r == 0 or r >= r_cut:
            continue

        c = coupling_state(particles[i].state, particles[j].state)
        fij = pair_force_from_delta(delta, c, r_min, r0, r_cut, k_rep, k_mid)
        forces[i] += fij
        # forces[j] -= fij

    for i, p in enumerate(particles):
        forces[i] += -gamma * p.vel + sigma * rng.normal(0.0, 1.0, size=2)

    return forces
