from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from particle_life.simulation import ParticleLifeSim, SimConfig
from particle_life.storage import ReplayWriter


def generate_run(args) -> None:
    cfg = SimConfig(
        species_count=args.species_count,
        particles_per_species=args.particles_per_species,
        steps_per_frame=args.steps_per_frame,
        seed=args.seed,
    )
    sim = ParticleLifeSim(cfg)
    out_root = Path(args.output)
    writer = ReplayWriter(out_root, args.name, {**asdict(sim.cfg), "particle_count": sim.count, "interaction_matrix": sim.matrix_values()})
    t0 = time.perf_counter()
    for _ in range(args.frames):
        sim.step_many(args.steps_per_frame)
        writer.write_frame(sim.snapshot())
    writer.close()
    dt = time.perf_counter() - t0
    print(f"generated {args.frames} frames ({sim.count} particles) in {dt:.3f}s -> {args.output}/{args.name}")


def profile_steps(args) -> None:
    cfg = SimConfig(species_count=args.species_count, particles_per_species=max(1, args.particles // args.species_count), seed=args.seed)
    cfg.particle_counts = [args.particles // args.species_count] * args.species_count
    cfg.particle_counts[0] += args.particles - sum(cfg.particle_counts)
    sim = ParticleLifeSim(cfg)
    sim.step_many(args.warmup)
    t0 = time.perf_counter()
    sim.step_many(args.steps)
    dt = time.perf_counter() - t0
    print(f"particles={sim.count} steps={args.steps} total={dt:.4f}s step_ms={(dt / args.steps) * 1000:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Particle Life offline and profiling utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Generate offline replay data")
    gen.add_argument("--name", default="run")
    gen.add_argument("--output", default="data/replays")
    gen.add_argument("--frames", type=int, default=1200)
    gen.add_argument("--steps-per-frame", type=int, default=1)
    gen.add_argument("--species-count", type=int, default=6)
    gen.add_argument("--particles-per-species", type=int, default=1600)
    gen.add_argument("--seed", type=int, default=0)
    gen.set_defaults(func=generate_run)

    prof = sub.add_parser("profile", help="Profile step time")
    prof.add_argument("--particles", type=int, default=10_000)
    prof.add_argument("--species-count", type=int, default=6)
    prof.add_argument("--steps", type=int, default=120)
    prof.add_argument("--warmup", type=int, default=30)
    prof.add_argument("--seed", type=int, default=0)
    prof.set_defaults(func=profile_steps)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
