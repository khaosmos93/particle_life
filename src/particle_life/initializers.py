from dataclasses import asdict

import numpy as np

from particle_life.particles import Particle
from particle_life.presets import build_preset


def coupling_two_state_asymmetric(
    state_i: np.ndarray,
    state_j: np.ndarray,
    canonical_states: np.ndarray,
    params: dict,
) -> float:
    i_idx = None
    j_idx = None
    for idx in (0, 1):
        if np.array_equal(state_i, canonical_states[idx]):
            i_idx = idx
        if np.array_equal(state_j, canonical_states[idx]):
            j_idx = idx

    if i_idx is None or j_idx is None:
        raise ValueError("Particle state must exactly match one of the two canonical states")

    if i_idx == j_idx:
        return float(params["same"])
    if i_idx == 0 and j_idx == 1:
        return float(params["s0_to_s1"])
    return float(params["s1_to_s0"])


COUPLING_FNS = {
    "two_state_asymmetric": coupling_two_state_asymmetric,
}


def resolve_seed(seed: int | None) -> tuple[int, np.random.Generator]:
    if seed is None:
        seed_seq = np.random.SeedSequence()
        seed_used = int(seed_seq.generate_state(1, dtype=np.uint64)[0])
    else:
        seed_used = int(seed)
    return seed_used, np.random.default_rng(seed_used)


def _build_random_particles(cfg, rng: np.random.Generator) -> list[Particle]:
    mass = 1.0
    pos = rng.uniform(0.0, cfg.box_size, size=(cfg.n_particles, 2))
    vel = 0.01 * rng.normal(size=(cfg.n_particles, 2))
    state = rng.normal(size=(cfg.n_particles, cfg.state_dim))
    return [Particle(mass, pos[i], vel[i], state[i]) for i in range(cfg.n_particles)]


def _parse_preset(data: dict, source_name: str, base_cfg: dict, seed_used: int, cfg_type) -> dict:
    world = data["world"]
    model = data["model"]
    sim = data["sim"]
    particles_json = data["particles"]

    if int(world["dim"]) != 2:
        raise ValueError("Only dim=2 presets are supported")

    canonical = np.asarray(model["canonical_states"], dtype=np.float64)
    if canonical.shape != (2, int(model["state_dim"])):
        raise ValueError("canonical_states must be exactly two vectors with length state_dim")

    coupling = model["coupling"]
    fn_name = coupling["fn"]
    if fn_name not in COUPLING_FNS:
        raise ValueError(f"Unknown coupling function: {fn_name}")

    parsed_particles: list[Particle] = []
    for idx, p in enumerate(particles_json):
        state = np.asarray(p["state"], dtype=np.float64)
        if not (np.array_equal(state, canonical[0]) or np.array_equal(state, canonical[1])):
            raise ValueError(f"Particle {idx} state does not match canonical_states[0] or canonical_states[1]")
        pos = np.asarray(p["pos"], dtype=np.float64)
        vel = np.asarray(p["vel"], dtype=np.float64)
        if pos.shape != (int(world["dim"]),):
            raise ValueError(f"Particle {idx} position must have dim={int(world['dim'])}")
        if vel.shape != (int(world["dim"]),):
            raise ValueError(f"Particle {idx} velocity must have dim={int(world['dim'])}")
        parsed_particles.append(
            Particle(
                m=float(p.get("m", 1.0)),
                pos=pos,
                vel=vel,
                state=state,
            )
        )

    if len(parsed_particles) == 0:
        raise ValueError("Preset has no particles")

    interaction = model["interaction"]
    cfg_dict = dict(base_cfg)
    cfg_dict.update(
        {
            "n_particles": len(parsed_particles),
            "state_dim": int(model["state_dim"]),
            "dt": float(sim["dt"]),
            "seed": seed_used,
            "box_size": float(world["box_size"]),
            "r_min": float(interaction["r_repulse"]),
            "r_cut": float(interaction["r_cut"]),
            "k_mid": float(interaction["strength"]),
            "sigma": float(interaction["noise"]),
            "gamma": float(interaction["damping"]),
        }
    )

    return {
        "name": data.get("name", source_name),
        "description": data.get("description", ""),
        "source": source_name,
        "cfg": cfg_type(**cfg_dict),
        "speed": float(sim["speed"]),
        "particles": parsed_particles,
        "canonical_states": canonical,
        "coupling_fn": fn_name,
        "coupling_params": dict(coupling.get("params", {})),
        "interaction": {
            "r_repulse": float(interaction["r_repulse"]),
            "r_cut": float(interaction["r_cut"]),
            "strength": float(interaction["strength"]),
            "noise": float(interaction["noise"]),
            "damping": float(interaction["damping"]),
        },
    }


def build_initial_state(
    cfg,
    preset_id: str | None = None,
    seed: int | None = None,
) -> tuple[object, list[Particle], dict]:
    seed_used, rng = resolve_seed(seed)
    cfg_dict = asdict(cfg)
    cfg_dict["seed"] = seed_used
    seed_cfg = cfg.__class__(**cfg_dict)

    if preset_id is None:
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

    data = build_preset(preset_id)
    parsed = _parse_preset(data, preset_id, asdict(seed_cfg), seed_used, cfg.__class__)
    return parsed["cfg"], parsed["particles"], {
        "name": parsed["name"],
        "description": parsed["description"],
        "source": parsed["source"],
        "speed": parsed["speed"],
        "canonical_states": parsed["canonical_states"],
        "coupling_fn": parsed["coupling_fn"],
        "coupling_params": parsed["coupling_params"],
        "interaction": parsed["interaction"],
    }
