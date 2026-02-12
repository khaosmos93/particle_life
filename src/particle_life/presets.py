import math
from copy import deepcopy

S0 = [1.0, 0.0, 0.0]
S1 = [0.0, 0.0, 1.0]


BASE_WORLD = {"dim": 2, "box_size": 1.0}
BASE_MODEL = {
    "state_dim": 3,
    "canonical_states": [S0, S1],
    "interaction": {
        "r_repulse": 0.015,
        "r_cut": 0.2,
        "strength": 0.8,
        "noise": 0.08,
        "damping": 0.9,
    },
    "coupling": {
        "fn": "two_state_asymmetric",
        "params": {
            "same": 0.9,
            "s0_to_s1": -0.4,
            "s1_to_s0": 0.35,
        },
    },
}
BASE_SIM = {"dt": 1.0, "speed": 1.0, "seed": 0}


def _particle(pid: int, x: float, y: float, state: list[float], vx: float = 0.0, vy: float = 0.0) -> dict:
    return {
        "id": pid,
        "m": 1.0,
        "pos": [float(x), float(y)],
        "vel": [float(vx), float(vy)],
        "state": list(state),
    }


def _linspace_grid(n_x: int, n_y: int, margin: float = 0.08):
    for iy in range(n_y):
        y = margin + (1.0 - 2 * margin) * (iy + 0.5) / n_y
        for ix in range(n_x):
            x = margin + (1.0 - 2 * margin) * (ix + 0.5) / n_x
            yield ix, iy, x, y


def _make_preset(preset_id: str, name: str, description: str, particles: list[dict], speed: float = 1.0) -> dict:
    return {
        "id": preset_id,
        "name": name,
        "description": description,
        "world": deepcopy(BASE_WORLD),
        "model": deepcopy(BASE_MODEL),
        "sim": {**deepcopy(BASE_SIM), "speed": float(speed)},
        "particles": particles,
    }


def _two_clusters() -> dict:
    particles = []
    pid = 0
    for ix, iy, x, y in _linspace_grid(10, 8, margin=0.05):
        cx = 0.25 + (x - 0.5) * 0.35
        cy = 0.5 + (y - 0.5) * 0.5
        particles.append(_particle(pid, cx, cy, S0))
        pid += 1
    for ix, iy, x, y in _linspace_grid(10, 8, margin=0.05):
        cx = 0.75 + (x - 0.5) * 0.35
        cy = 0.5 + (y - 0.5) * 0.5
        particles.append(_particle(pid, cx, cy, S1))
        pid += 1
    return _make_preset("two_clusters", "Two Clusters", "Two compact groups with opposite states.", particles)


def _checkerboard() -> dict:
    particles = []
    pid = 0
    for ix, iy, x, y in _linspace_grid(16, 12, margin=0.06):
        state = S0 if (ix + iy) % 2 == 0 else S1
        particles.append(_particle(pid, x, y, state))
        pid += 1
    return _make_preset("checkerboard", "Checkerboard", "Alternating states on a regular lattice.", particles)


def _concentric_rings() -> dict:
    particles = []
    pid = 0
    center_x, center_y = 0.5, 0.5
    rings = [(0.12, 28), (0.22, 44), (0.32, 60), (0.42, 76)]
    for ridx, (r, count) in enumerate(rings):
        state = S0 if ridx % 2 == 0 else S1
        for i in range(count):
            theta = 2.0 * math.pi * i / count
            x = center_x + r * math.cos(theta)
            y = center_y + r * math.sin(theta)
            particles.append(_particle(pid, x, y, state))
            pid += 1
    return _make_preset("concentric_rings", "Concentric Rings", "Alternating state rings around center.", particles)


def _stripe_bands() -> dict:
    particles = []
    pid = 0
    for ix, iy, x, y in _linspace_grid(18, 10, margin=0.05):
        band = int(x * 8)
        state = S0 if band % 2 == 0 else S1
        particles.append(_particle(pid, x, y, state))
        pid += 1
    return _make_preset("stripe_bands", "Stripe Bands", "Vertical alternating state bands.", particles)


def _spiral_arms() -> dict:
    particles = []
    pid = 0
    cx, cy = 0.5, 0.5
    n_per_arm = 90
    for arm in range(2):
        state = S0 if arm == 0 else S1
        phase = arm * math.pi
        for i in range(n_per_arm):
            t = i / (n_per_arm - 1)
            theta = 5.2 * math.pi * t + phase
            r = 0.06 + 0.4 * t
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            particles.append(_particle(pid, x, y, state))
            pid += 1
    return _make_preset("spiral_arms", "Spiral Arms", "Two opposite-state logarithmic-like spiral arms.", particles)


def _yin_yang() -> dict:
    particles = []
    pid = 0
    cx, cy = 0.5, 0.5
    for ix, iy, x, y in _linspace_grid(16, 12, margin=0.07):
        dx = x - cx
        dy = y - cy
        r = math.hypot(dx, dy)
        if r > 0.45:
            continue
        state = S0 if (dy >= 0 and r < 0.43) or (math.hypot(dx, dy - 0.18) < 0.13) else S1
        particles.append(_particle(pid, x, y, state))
        pid += 1
    return _make_preset("yin_yang", "Yin Yang", "Yin-yang style split with embedded pockets.", particles)


def _four_quadrants() -> dict:
    particles = []
    pid = 0
    for ix, iy, x, y in _linspace_grid(14, 14, margin=0.05):
        state = S0 if (x < 0.5 and y >= 0.5) or (x >= 0.5 and y < 0.5) else S1
        particles.append(_particle(pid, x, y, state))
        pid += 1
    return _make_preset("four_quadrants", "Four Quadrants", "Alternating-state checker of four regions.", particles)


def _line_vs_cloud() -> dict:
    particles = []
    pid = 0
    for i in range(90):
        x = 0.08 + 0.84 * i / 89
        y = 0.25
        particles.append(_particle(pid, x, y, S0))
        pid += 1
    for ix, iy, x, y in _linspace_grid(12, 10, margin=0.1):
        px = 0.58 + 0.35 * (x - 0.5)
        py = 0.62 + 0.3 * (y - 0.5)
        particles.append(_particle(pid, px, py, S1))
        pid += 1
    return _make_preset("line_vs_cloud", "Line vs Cloud", "One state in a line versus one state in a compact cloud.", particles)


def _two_lanes() -> dict:
    particles = []
    pid = 0
    for i in range(120):
        x = 0.05 + 0.9 * i / 119
        particles.append(_particle(pid, x, 0.35, S0, vx=0.03, vy=0.0))
        pid += 1
    for i in range(120):
        x = 0.95 - 0.9 * i / 119
        particles.append(_particle(pid, x, 0.65, S1, vx=-0.03, vy=0.0))
        pid += 1
    return _make_preset("two_lanes", "Two Lanes", "Counter-flowing lanes with opposite states.", particles, speed=1.2)


def _radial_burst() -> dict:
    particles = []
    pid = 0
    cx, cy = 0.5, 0.5
    rays = 20
    per_ray = 10
    for ray in range(rays):
        angle = 2.0 * math.pi * ray / rays
        state = S0 if ray % 2 == 0 else S1
        for i in range(per_ray):
            r = 0.05 + 0.4 * (i / (per_ray - 1))
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            particles.append(_particle(pid, x, y, state))
            pid += 1
    return _make_preset("radial_burst", "Radial Burst", "Alternating-state rays expanding from center.", particles)


_PRESETS = {
    "two_clusters": _two_clusters,
    "checkerboard": _checkerboard,
    "concentric_rings": _concentric_rings,
    "stripe_bands": _stripe_bands,
    "spiral_arms": _spiral_arms,
    "yin_yang": _yin_yang,
    "four_quadrants": _four_quadrants,
    "line_vs_cloud": _line_vs_cloud,
    "two_lanes": _two_lanes,
    "radial_burst": _radial_burst,
}


def list_presets() -> list[dict]:
    items = []
    for preset_id, builder in _PRESETS.items():
        p = builder()
        items.append({"id": preset_id, "name": p["name"], "description": p["description"]})
    return items


def build_preset(preset_id: str) -> dict:
    if preset_id not in _PRESETS:
        raise ValueError(f"Unknown preset: {preset_id}")
    return _PRESETS[preset_id]()


def load_preset(preset_id: str) -> dict:
    return build_preset(preset_id)
