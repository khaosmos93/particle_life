import argparse
import os
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from particle_life.initializers import build_initial_state, resolve_seed
from particle_life.particles import Particle, compute_net_forces


@dataclass
class SimulationConfig:
    n_particles: int = 200
    state_dim: int = 2
    dt: float = 0.01
    steps: int = 1000
    seed: int = 0
    box_size: float = 1.0
    wrap: bool = True
    r_min: float = 0.01
    r0: float = 0.01
    r_cut: float = 0.25
    k_rep: float = 0.5
    k_mid: float = 100.0
    gamma: float = 0.2
    sigma: float = 0.05
    chunk: int = 1
    out_path: str | None = None


def init_particles(cfg: SimulationConfig) -> list[Particle]:
    _, particles, _ = build_initial_state(cfg, preset_id=None, seed=cfg.seed)
    return particles


def step(particles: list[Particle], cfg: SimulationConfig, rng: np.random.Generator) -> None:
    forces = compute_net_forces(
        particles=particles,
        box_size=cfg.box_size,
        wrap=cfg.wrap,
        r_min=cfg.r_min,
        r0=cfg.r0,
        r_cut=cfg.r_cut,
        k_rep=cfg.k_rep,
        k_mid=cfg.k_mid,
        gamma=cfg.gamma,
        sigma=cfg.sigma,
        rng=rng,
    )

    for i, p in enumerate(particles):
        p.vel = p.vel + (forces[i] / p.m) * cfg.dt
        p.pos = p.pos + p.vel * cfg.dt
        if cfg.wrap:
            p.pos %= cfg.box_size


def make_table_from_buffer(
    buffer: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    state_dim: int,
    schema: pa.Schema,
) -> pa.Table:
    if not buffer:
        raise ValueError("buffer must not be empty")

    n_particles = buffer[0][1].shape[0]
    n_steps = len(buffer)

    steps = np.repeat(np.array([step_idx for step_idx, *_ in buffer], dtype=np.int32), n_particles)
    ids = np.tile(np.arange(n_particles, dtype=np.int32), n_steps)
    masses = np.concatenate([m_step for _, m_step, _, _, _ in buffer]).astype(np.float64, copy=False)
    pos = np.concatenate([pos_step for _, _, pos_step, _, _ in buffer], axis=0).astype(np.float64, copy=False)
    vel = np.concatenate([vel_step for _, _, _, vel_step, _ in buffer], axis=0).astype(np.float64, copy=False)
    state = np.concatenate([state_step for _, _, _, _, state_step in buffer], axis=0).astype(
        np.float64, copy=False
    )

    step_col = pa.array(steps, type=pa.int32())
    id_col = pa.array(ids, type=pa.int32())
    m_col = pa.array(masses, type=pa.float64())
    pos_col = pa.FixedSizeListArray.from_arrays(pa.array(pos.ravel(), type=pa.float64()), 2)
    vel_col = pa.FixedSizeListArray.from_arrays(pa.array(vel.ravel(), type=pa.float64()), 2)
    state_col = pa.FixedSizeListArray.from_arrays(pa.array(state.ravel(), type=pa.float64()), state_dim)

    return pa.Table.from_arrays([step_col, id_col, m_col, pos_col, vel_col, state_col], schema=schema)


def write_chunk(out_dir: str, chunk_id: int, table: pa.Table) -> None:
    partition_dir = os.path.join(out_dir, f"chunk_id={chunk_id}")
    os.makedirs(partition_dir, exist_ok=True)
    out_file = os.path.join(partition_dir, "part-000000.parquet")

    try:
        pq.write_table(table, out_file, compression="zstd")
    except Exception:
        pq.write_table(table, out_file)


class Simulator:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        _, self.rng = resolve_seed(cfg.seed)
        _, self.particles, _ = build_initial_state(cfg, preset_id=None, seed=cfg.seed)

    def run(self, out_path: str) -> None:
        os.makedirs(out_path, exist_ok=True)

        schema = pa.schema(
            [
                pa.field("step", pa.int32()),
                pa.field("id", pa.int32()),
                pa.field("m", pa.float64()),
                pa.field("pos", pa.list_(pa.float64(), list_size=2)),
                pa.field("vel", pa.list_(pa.float64(), list_size=2)),
                pa.field("state", pa.list_(pa.float64(), list_size=self.cfg.state_dim)),
            ]
        )

        buffer: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        chunk_id = 0

        for t in range(self.cfg.steps):
            pos_step = np.stack([p.pos for p in self.particles]).copy()
            vel_step = np.stack([p.vel for p in self.particles]).copy()
            state_step = np.stack([p.state for p in self.particles]).copy()
            m_step = np.array([float(p.m) for p in self.particles], dtype=np.float64)
            buffer.append((t, m_step, pos_step, vel_step, state_step))

            if len(buffer) == self.cfg.chunk:
                table = make_table_from_buffer(buffer, self.cfg.state_dim, schema)
                write_chunk(out_path, chunk_id, table)
                buffer.clear()
                chunk_id += 1

            step(self.particles, self.cfg, self.rng)

        if buffer:
            table = make_table_from_buffer(buffer, self.cfg.state_dim, schema)
            write_chunk(out_path, chunk_id, table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Particle Life simulation")
    parser.add_argument("--out", required=True, help="Output dataset directory in Parquet format.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps")
    parser.add_argument("--chunk", type=int, default=1, help="Number of steps per Parquet file")
    parser.add_argument("--n", type=int, default=200, help="Number of particles")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--box", type=float, default=1.0, help="Box size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk < 1:
        raise ValueError("--chunk must be >= 1")
    cfg = SimulationConfig(
        n_particles=args.n,
        dt=args.dt,
        steps=args.steps,
        chunk=args.chunk,
        seed=0 if args.seed is None else args.seed,
        box_size=args.box,
        out_path=args.out,
    )
    if args.seed is None:
        seed_used, _ = resolve_seed(None)
        cfg.seed = seed_used
    Simulator(cfg).run(args.out)


if __name__ == "__main__":
    main()
