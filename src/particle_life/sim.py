import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np

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
    r_min: float = 0.02
    r0: float = 0.08
    r_cut: float = 0.25
    k_rep: float = 1.0
    k_mid: float = 0.5
    gamma: float = 0.2
    sigma: float = 0.05
    out_path: str | None = None


def init_particles(cfg: SimulationConfig) -> list[Particle]:
    rng = np.random.default_rng(cfg.seed)
    mass = 1.0
    pos = rng.uniform(0.0, cfg.box_size, size=(cfg.n_particles, 2))
    vel = 0.01 * rng.normal(size=(cfg.n_particles, 2))
    state = rng.normal(size=(cfg.n_particles, cfg.state_dim))

    return [Particle(mass, pos[i], vel[i], state[i]) for i in range(cfg.n_particles)]


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


def write_step_csv(writer, step_idx: int, particles: list[Particle]) -> None:
    for i, p in enumerate(particles):
        row = [
            step_idx,
            i,
            float(p.m),
            float(p.pos[0]),
            float(p.pos[1]),
            float(p.vel[0]),
            float(p.vel[1]),
        ]
        row.extend(float(x) for x in p.state)
        writer.writerow(row)


class Simulator:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.particles = init_particles(cfg)

    def run(self, out_path: str) -> None:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["step", "id", "m", "x", "y", "vx", "vy"]
            for d in range(self.cfg.state_dim):
                header.append(f"state_{d}")
            writer.writerow(header)

            for t in range(self.cfg.steps):
                write_step_csv(writer, t, self.particles)
                step(self.particles, self.cfg, self.rng)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Particle Life simulation")
    parser.add_argument("--out", required=True, help="Output is CSV compatible with pandas read_csv.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps")
    parser.add_argument("--n", type=int, default=200, help="Number of particles")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--box", type=float, default=1.0, help="Box size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimulationConfig(
        n_particles=args.n,
        dt=args.dt,
        steps=args.steps,
        seed=args.seed,
        box_size=args.box,
        out_path=args.out,
    )
    Simulator(cfg).run(args.out)


if __name__ == "__main__":
    main()
