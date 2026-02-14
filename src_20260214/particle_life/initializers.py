from dataclasses import asdict

import numpy as np

from particle_life.particles import Particle


def resolve_seed(seed: int | None) -> tuple[int, np.random.Generator]:
    if seed is None:
        seed_seq = np.random.SeedSequence()
        seed_used = int(seed_seq.generate_state(1, dtype=np.uint64)[0])
    else:
        seed_used = int(seed)
    return seed_used, np.random.default_rng(seed_used)

# FIXME: hmmm
def _build_random_particles(cfg, rng: np.random.Generator) -> list[Particle]:
    mass = 1.0
    pos = rng.uniform(0.0, cfg.box_size, size=(cfg.n_particles, 2))
    vel = np.zeros((cfg.n_particles, 2), dtype=np.float64)
    # vel = 0.01 * rng.normal(size=(cfg.n_particles, 2))

    # state = rng.normal(size=(cfg.n_particles, cfg.state_dim))
    state = np.zeros((cfg.n_particles, cfg.state_dim), dtype=np.float64)
    s = cfg.n_particles // 2
    state[:s, 0] = 1.0
    state[s:, 0] = -1.0
    
    # invsqrt2 = 1.0 / np.sqrt(2.0)
    # state[s:, 0] = invsqrt2
    # state[s:, 2] = invsqrt2
    # state[s:2*s, 0] = invsqrt2
    # state[s:2*s, 2] = invsqrt2
    # state[2*s:, 1] = 1.0

    return [Particle(mass, pos[i], vel[i], state[i]) for i in range(cfg.n_particles)]


def build_initial_state(
    cfg,
    preset_id: str | None = None,
    seed: int | None = None,
) -> tuple[object, list[Particle], dict]:
    del preset_id
    seed_used, rng = resolve_seed(seed)
    cfg_dict = asdict(cfg)
    cfg_dict["seed"] = seed_used
    seed_cfg = cfg.__class__(**cfg_dict)

    particles = _build_random_particles(seed_cfg, rng)
    return seed_cfg, particles, {
        "speed": 1.0,
        "canonical_states": None,
        "coupling_fn": None,
        "coupling_params": {},
        "interaction": {
            "r_repulse": seed_cfg.r_min,
            "r_cut": seed_cfg.r_cut,
            "strength": seed_cfg.k_mid,
            "noise": seed_cfg.sigma,
            "damping": seed_cfg.gamma,
        },
    }
