from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


@dataclass
class SimConfig:
    species_count: int = 6
    particles_per_species: int = 160
    world_size: float = 1.0
    interaction_radius: float = 0.11
    repel_radius: float = 0.025
    force_scale: float = 0.42
    dt: float = 0.015
    damping: float = 0.975
    max_speed: float = 0.05
    steps_per_frame: int = 1
    boundary_mode: str = "wrap"
    point_size: float = 3.0
    point_opacity: float = 0.95
    background_alpha: float = 1.0
    show_hud: bool = True
    pbc_tiling: bool = False
    color_mode: str = "species"
    seed: int = 0


CONFIG_SECTIONS = [
    {
        "key": "simulation",
        "label": "Simulation",
        "controls": [
            {"key": "species_count", "type": "range", "label": "Species", "min": 2, "max": 12, "step": 1, "default": 6, "apply": "reset"},
            {"key": "particles_per_species", "type": "range", "label": "Particles / Species", "min": 20, "max": 400, "step": 10, "default": 160, "apply": "reset"},
            {"key": "world_size", "type": "range", "label": "World Size", "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.0, "apply": "reset"},
            {"key": "interaction_radius", "type": "range", "label": "Interaction Radius", "min": 0.02, "max": 0.35, "step": 0.005, "default": 0.11, "apply": "immediate"},
            {"key": "repel_radius", "type": "range", "label": "Repel Radius", "min": 0.005, "max": 0.08, "step": 0.001, "default": 0.025, "apply": "immediate"},
            {"key": "force_scale", "type": "range", "label": "Force", "min": 0.05, "max": 1.2, "step": 0.01, "default": 0.42, "apply": "immediate"},
            {"key": "dt", "type": "range", "label": "dt", "min": 0.001, "max": 0.06, "step": 0.001, "default": 0.015, "apply": "immediate"},
            {"key": "damping", "type": "range", "label": "Damping", "min": 0.85, "max": 0.999, "step": 0.001, "default": 0.975, "apply": "immediate"},
            {"key": "max_speed", "type": "range", "label": "Max Speed", "min": 0.005, "max": 0.2, "step": 0.001, "default": 0.05, "apply": "immediate"},
            {"key": "steps_per_frame", "type": "range", "label": "Steps / Frame", "min": 1, "max": 8, "step": 1, "default": 1, "apply": "immediate"},
            {"key": "boundary_mode", "type": "select", "label": "Boundary", "options": ["wrap", "bounce"], "default": "wrap", "apply": "immediate"},
        ],
    },
    {
        "key": "render",
        "label": "Render",
        "controls": [
            {"key": "point_size", "type": "range", "label": "Point Size", "min": 1, "max": 8, "step": 0.1, "default": 3.0, "apply": "immediate"},
            {"key": "point_opacity", "type": "range", "label": "Point Opacity", "min": 0.1, "max": 1.0, "step": 0.01, "default": 0.95, "apply": "immediate"},
            {"key": "background_alpha", "type": "range", "label": "Background Alpha", "min": 0.02, "max": 1.0, "step": 0.01, "default": 1.0, "apply": "immediate"},
            {"key": "color_mode", "type": "select", "label": "Color Mode", "options": ["species", "velocity", "mono"], "default": "species", "apply": "immediate"},
            {"key": "pbc_tiling", "type": "toggle", "label": "3×3 PBC View", "default": False, "apply": "immediate"},
            {"key": "show_hud", "type": "toggle", "label": "Show HUD", "default": True, "apply": "immediate"},
        ],
    },
    {
        "key": "random",
        "label": "Random / Seed",
        "controls": [
            {"key": "seed", "type": "number", "label": "Seed", "min": 0, "max": 2147483647, "step": 1, "default": 0, "apply": "reset"},
        ],
    },
]

PRESETS = {
    "Default": {},
    "Dense": {"particles_per_species": 260, "interaction_radius": 0.09, "force_scale": 0.35},
    "Sparse": {"particles_per_species": 90, "interaction_radius": 0.15, "force_scale": 0.55},
    "Chaotic": {"dt": 0.03, "damping": 0.94, "force_scale": 0.9, "repel_radius": 0.014},
}


class ConfigUpdate(BaseModel):
    updates: dict[str, float | int | bool | str]


class PresetLoad(BaseModel):
    name: str


class PauseUpdate(BaseModel):
    paused: bool


class ParticleLifeSim:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.paused = False
        self.matrix_version = 0
        self.reset_state(random_matrix=True)

    def reset_state(self, random_matrix: bool = False) -> None:
        cfg = self.cfg
        self.count = cfg.species_count * cfg.particles_per_species
        self.positions = self.rng.random((self.count, 2), dtype=np.float32) * cfg.world_size
        self.velocities = np.zeros((self.count, 2), dtype=np.float32)
        self.species = np.repeat(np.arange(cfg.species_count), cfg.particles_per_species).astype(np.int32)
        self.rng.shuffle(self.species)
        if random_matrix or not hasattr(self, "matrix") or self.matrix.shape[0] != cfg.species_count:
            self.matrix = self.rng.uniform(-1.0, 1.0, (cfg.species_count, cfg.species_count)).astype(np.float32)
            np.fill_diagonal(self.matrix, self.rng.uniform(0.2, 1.0, cfg.species_count))

    def set_matrix(self, matrix: list[list[float]] | np.ndarray) -> None:
        self.matrix = _sanitize_matrix(matrix, self.cfg.species_count).copy()
        self.matrix_version += 1
        print(f"[sim] interaction matrix version -> {self.matrix_version}")

    def matrix_values(self) -> list[list[float]]:
        return self.matrix.astype(float).tolist()

    def step(self) -> None:
        cfg = self.cfg
        delta = self.positions[:, None, :] - self.positions[None, :, :]
        if cfg.boundary_mode == "wrap":
            delta -= np.round(delta / cfg.world_size) * cfg.world_size

        dist2 = np.sum(delta * delta, axis=2) + 1e-12
        np.fill_diagonal(dist2, np.inf)
        dist = np.sqrt(dist2)

        within = dist < cfg.interaction_radius
        influence = np.clip(1.0 - dist / cfg.interaction_radius, 0.0, 1.0)

        repel = np.clip(1.0 - dist / max(cfg.repel_radius, 1e-9), 0.0, 1.0)
        interaction = self.matrix[self.species[:, None], self.species[None, :]]
        strength = (interaction * influence - repel * 1.5) * within * cfg.force_scale

        direction = -delta / dist[:, :, None]
        force = np.sum(direction * strength[:, :, None], axis=1)
        self.velocities = self.velocities * cfg.damping + force * cfg.dt

        speed = np.linalg.norm(self.velocities, axis=1)
        over = speed > cfg.max_speed
        if np.any(over):
            self.velocities[over] *= (cfg.max_speed / speed[over])[:, None]

        self.positions = self.positions + self.velocities * cfg.dt
        if cfg.boundary_mode == "wrap":
            self.positions = np.mod(self.positions, cfg.world_size)
        else:
            for axis in (0, 1):
                low = self.positions[:, axis] < 0
                high = self.positions[:, axis] > cfg.world_size
                self.positions[low | high, axis] = np.clip(self.positions[low | high, axis], 0, cfg.world_size)
                self.velocities[low | high, axis] *= -1

    def snapshot(self) -> bytes:
        data = np.empty((self.count, 5), dtype=np.float32)
        data[:, :2] = self.positions / self.cfg.world_size
        data[:, 2] = self.species.astype(np.float32)
        data[:, 3:5] = self.velocities
        return data.tobytes()


def _defaults() -> dict:
    return asdict(SimConfig())


def _cast_control_value(control: dict, value):
    if control["type"] == "toggle":
        return bool(value)
    if control["type"] in {"range", "number"}:
        numeric = float(value)
        if not np.isfinite(numeric):
            numeric = float(control.get("default", 0))
        if isinstance(control.get("step"), int) or float(control.get("step", 0)).is_integer():
            return int(_clamp_numeric(control, int(numeric)))
        return float(_clamp_numeric(control, numeric))
    if control["type"] == "select":
        v = str(value)
        return v if v in control["options"] else control["default"]
    return value


def _build_control_index() -> dict[str, dict]:
    return {c["key"]: c for section in CONFIG_SECTIONS for c in section["controls"]}


def _clamp_numeric(control: dict, value: float | int):
    low = control.get("min")
    high = control.get("max")
    out = value
    if low is not None:
        out = max(low, out)
    if high is not None:
        out = min(high, out)
    return out


def _sanitize_matrix(matrix, species_count: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape != (species_count, species_count):
        raise ValueError("matrix shape mismatch")
    if not np.all(np.isfinite(arr)):
        raise ValueError("matrix has non-finite values")
    return np.clip(arr, -1.0, 1.0).astype(np.float32)


app = FastAPI(title="Particle Life")
app.mount("/static", StaticFiles(directory="src/particle_life/static"), name="static")
control_index = _build_control_index()
sim = ParticleLifeSim(SimConfig())


def _config_values() -> dict:
    values = asdict(sim.cfg)
    values["interaction_matrix"] = sim.matrix_values()
    return values


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("src/particle_life/static/index.html")


@app.get("/api/config")
async def get_config() -> dict:
    return {"sections": CONFIG_SECTIONS, "values": _config_values(), "presets": list(PRESETS.keys())}


@app.post("/api/config/update")
async def update_config(payload: ConfigUpdate) -> dict:
    values = asdict(sim.cfg)
    next_matrix: np.ndarray | None = None
    needs_reset = False

    for key, value in payload.updates.items():
        control = control_index.get(key)
        if control:
            values[key] = _cast_control_value(control, value)
            if control.get("apply") == "reset":
                needs_reset = True

    next_species_count = int(values["species_count"])
    if "interaction_matrix" in payload.updates:
        next_matrix = _sanitize_matrix(payload.updates["interaction_matrix"], next_species_count)

    sim.cfg = SimConfig(**values)
    if needs_reset:
        sim.rng = np.random.default_rng(sim.cfg.seed)
        sim.reset_state(random_matrix=False)
    if next_matrix is not None:
        sim.set_matrix(next_matrix)

    return {"values": _config_values(), "reset_applied": needs_reset}


@app.post("/api/config/reset")
async def reset_config() -> dict:
    sim.cfg = SimConfig(**_defaults())
    sim.rng = np.random.default_rng(sim.cfg.seed)
    sim.reset_state(random_matrix=True)
    return {"values": _config_values()}


@app.post("/api/config/randomize")
async def randomize_seed() -> dict:
    sim.cfg.seed = int(np.random.randint(0, 2**31 - 1))
    sim.rng = np.random.default_rng(sim.cfg.seed)
    sim.reset_state(random_matrix=True)
    return {"values": _config_values()}


@app.post("/api/config/preset")
async def load_preset(payload: PresetLoad) -> dict:
    if payload.name not in PRESETS:
        return {"values": _config_values(), "error": "unknown preset"}
    values = _defaults()
    values.update(PRESETS[payload.name])
    sim.cfg = SimConfig(**values)
    sim.rng = np.random.default_rng(sim.cfg.seed)
    sim.reset_state(random_matrix=True)
    return {"values": _config_values()}


@app.post("/api/sim/pause")
async def set_pause(payload: PauseUpdate) -> dict:
    sim.paused = bool(payload.paused)
    return {"paused": sim.paused}


@app.websocket("/ws")
async def stream_particles(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            if not sim.paused:
                for _ in range(sim.cfg.steps_per_frame):
                    sim.step()
            await websocket.send_bytes(sim.snapshot())
            await asyncio.sleep(1 / 60)
    except WebSocketDisconnect:
        return


def main() -> None:
    uvicorn.run("particle_life.realtime:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
