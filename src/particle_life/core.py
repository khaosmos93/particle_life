from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SimulationParams:
    n_particles: int = 200
    state_dim: int = 1
    dt: float = 0.001
    seed: int = 0
    box_size: float = 1.0
    wrap: bool = True
    r_min: float = 0.3
    r0: float = 0.3
    r_cut: float = 0.1
    k_rep: float = 1.0
    k_mid: float = 1.0
    gamma: float = 0.0
    sigma: float = 0.0


class ParticleLifeSim:
    def __init__(self, params: SimulationParams) -> None:
        self.params = params
        self.rng = np.random.default_rng(params.seed)
        self.mass = np.ones(params.n_particles, dtype=np.float32)
        self.pos = self.rng.uniform(0.0, params.box_size, (params.n_particles, 2)).astype(np.float32)
        self.vel = np.zeros((params.n_particles, 2), dtype=np.float32)
        self.state = np.zeros((params.n_particles, params.state_dim), dtype=np.float32)
        s = params.n_particles // 2
        self.state[:s, 0] = 1.0
        self.state[s:, 0] = -1.0

    def reseed(self, seed: int) -> None:
        self.params.seed = int(seed)
        self.__init__(self.params)

    def _coupling(self, si0: float, sj0: float) -> float:
        if si0 > 0 and sj0 > 0:
            return 1.0
        if si0 < 0 and sj0 < 0:
            return 1.0
        if si0 > 0 and sj0 < 0:
            return 0.0
        return -1.0

    def _neighbor_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.params.n_particles
        cell_size = max(self.params.r_cut, 1e-6)
        ncell = max(1, int(np.floor(self.params.box_size / cell_size)))
        eff = self.params.box_size / ncell

        cx = np.floor(self.pos[:, 0] / eff).astype(np.int32) % ncell
        cy = np.floor(self.pos[:, 1] / eff).astype(np.int32) % ncell

        cells: dict[tuple[int, int], list[int]] = {}
        for i in range(n):
            key = (int(cx[i]), int(cy[i]))
            cells.setdefault(key, []).append(i)

        ii: list[int] = []
        jj: list[int] = []
        for (x, y), ilist in cells.items():
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    jlist = cells.get(((x + dx) % ncell, (y + dy) % ncell))
                    if not jlist:
                        continue
                    for i in ilist:
                        for j in jlist:
                            if i != j:
                                ii.append(i)
                                jj.append(j)
        return np.asarray(ii, dtype=np.int32), np.asarray(jj, dtype=np.int32)

    def _force_from_delta(self, delta: np.ndarray, c: float) -> np.ndarray:
        r = float(np.linalg.norm(delta))
        if r == 0.0 or r >= self.params.r_cut:
            return np.zeros(2, dtype=np.float32)
        unit = delta / r
        if r < self.params.r_min:
            mag = self.params.k_rep * (r / self.params.r_min - 1.0)
        elif r < 1.0:
            mag = self.params.k_mid * c * (1.0 - abs(2.0 * r - 1.0 - self.params.r_min) / (1.0 - self.params.r_min))
        else:
            mag = 0.0
        return (-mag * unit).astype(np.float32)

    def step(self, dt_eff: float) -> None:
        ii, jj = self._neighbor_pairs()
        forces = np.zeros_like(self.pos)

        for i, j in zip(ii, jj):
            delta = self.pos[j] - self.pos[i]
            if self.params.wrap:
                delta -= self.params.box_size * np.round(delta / self.params.box_size)
            c = self._coupling(float(self.state[i, 0]), float(self.state[j, 0]))
            forces[i] += self._force_from_delta(delta, c)

        if self.params.gamma or self.params.sigma:
            forces += -self.params.gamma * self.vel
            if self.params.sigma:
                forces += self.params.sigma * self.rng.normal(0.0, 1.0, self.vel.shape)

        damping = np.float32(np.power(0.5, dt_eff / 10.0))
        self.vel = damping * self.vel + (forces / self.mass[:, None]) * dt_eff
        self.pos = self.pos + self.vel * dt_eff
        if self.params.wrap:
            self.pos %= self.params.box_size


def pack_binary_frame(pos: np.ndarray, state: np.ndarray, box_size: float) -> bytes:
    p = np.pad(pos, ((0, 0), (0, 1))) if pos.shape[1] == 2 else pos
    s = np.pad(state, ((0, 0), (0, max(0, 3 - state.shape[1])))) if state.shape[1] < 3 else state
    header_i = np.array([p.shape[1], s.shape[1], p.shape[0]], dtype=np.int32)
    header_f = np.array([box_size], dtype=np.float32)
    body = np.concatenate([p, s], axis=1).astype(np.float32, copy=False)
    return header_i.tobytes() + header_f.tobytes() + body.tobytes()
