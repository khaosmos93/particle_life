from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from particle_life.simulation import ParticleLifeSim, SimConfig
from particle_life.simulation.config import sanitize_matrix, sanitize_particle_counts
from particle_life.storage import ReplayReader, list_runs

CONFIG_SECTIONS = [
    {
        "key": "simulation",
        "label": "Simulation",
        "controls": [
            {"key": "species_count", "type": "range", "label": "Species", "min": 2, "max": 12, "step": 1, "default": 6, "apply": "reset"},
            {"key": "particles_per_species", "type": "range", "label": "Particles / Species", "min": 20, "max": 2000, "step": 10, "default": 160, "apply": "reset"},
            {"key": "world_size", "type": "range", "label": "World Size", "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.0, "apply": "reset"},
            {"key": "interaction_radius", "type": "range", "label": "Interaction Radius", "min": 0.02, "max": 0.35, "step": 0.005, "default": 0.11, "apply": "immediate"},
            {"key": "repel_radius", "type": "range", "label": "Repel Radius", "min": 0.005, "max": 0.08, "step": 0.001, "default": 0.025, "apply": "immediate"},
            {"key": "force_scale", "type": "range", "label": "Force", "min": 0.05, "max": 1.2, "step": 0.01, "default": 0.42, "apply": "immediate"},
            {"key": "dt", "type": "range", "label": "dt", "min": 0.001, "max": 0.06, "step": 0.001, "default": 0.015, "apply": "immediate"},
            {"key": "damping", "type": "range", "label": "Damping", "min": 0.85, "max": 0.999, "step": 0.001, "default": 0.975, "apply": "immediate"},
            {"key": "max_speed", "type": "range", "label": "Max Speed", "min": 0.005, "max": 0.2, "step": 0.001, "default": 0.05, "apply": "immediate"},
            {"key": "noise_strength", "type": "range", "label": "Noise", "min": 0.0, "max": 1.0, "step": 0.001, "default": 0.0, "apply": "immediate"},
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
]
PRESETS = {
    "Default": {},
    "Dense": {"particles_per_species": 900, "interaction_radius": 0.09, "force_scale": 0.35},
    "Sparse": {"particles_per_species": 90, "interaction_radius": 0.15, "force_scale": 0.55},
    "Chaotic": {"dt": 0.03, "damping": 0.94, "force_scale": 0.9, "repel_radius": 0.014},
}

INITIAL_CONDITION_DIR = Path("data/initial_condition")
REPLAY_DIR = Path("data/replays")


class ConfigUpdate(BaseModel):
    updates: dict[str, Any]


class PresetLoad(BaseModel):
    name: str


class PauseUpdate(BaseModel):
    paused: bool



class InitialConditionSave(BaseModel):
    name: str
    input_json: dict[str, Any]


class InitialConditionLoad(BaseModel):
    name: str

app = FastAPI(title="Particle Life")
app.mount("/static", StaticFiles(directory="src/particle_life/static"), name="static")
sim = ParticleLifeSim(SimConfig())


def _build_control_index() -> dict[str, dict]:
    return {c["key"]: c for section in CONFIG_SECTIONS for c in section["controls"]}


control_index = _build_control_index()


def _defaults() -> dict:
    return asdict(SimConfig())


def _clamp_numeric(control: dict, value: float | int):
    low = control.get("min")
    high = control.get("max")
    out = value
    if low is not None:
        out = max(low, out)
    if high is not None:
        out = min(high, out)
    return out


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


def _config_values() -> dict:
    values = asdict(sim.cfg)
    values["interaction_matrix"] = sim.matrix_values()
    values["matrix_version"] = sim.matrix_version
    return values


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("src/particle_life/static/index.html")


@app.get("/editor")
async def editor() -> FileResponse:
    return FileResponse("src/particle_life/static/editor.html")


@app.get("/api/config")
async def get_config() -> dict:
    return {"sections": CONFIG_SECTIONS, "values": _config_values(), "presets": list(PRESETS.keys())}




def _validate_filename(name: str) -> str:
    cleaned = str(name).strip()
    if not cleaned:
        raise ValueError("name is required")
    if not cleaned.endswith(".json"):
        cleaned = f"{cleaned}.json"
    if "/" in cleaned or "\\" in cleaned or cleaned.startswith("."):
        raise ValueError("invalid file name")
    return cleaned


@app.get("/api/initial_condition/list")
async def list_initial_conditions() -> dict:
    INITIAL_CONDITION_DIR.mkdir(parents=True, exist_ok=True)
    return {"items": sorted([p.name for p in INITIAL_CONDITION_DIR.glob("*.json")])}


@app.post("/api/initial_condition/save")
async def save_initial_condition(payload: InitialConditionSave) -> dict:
    try:
        filename = _validate_filename(payload.name)
        INITIAL_CONDITION_DIR.mkdir(parents=True, exist_ok=True)
        (INITIAL_CONDITION_DIR / filename).write_text(json.dumps(payload.input_json, separators=(",", ":")), encoding="utf-8")
        return {"ok": True, "name": filename}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/api/initial_condition/load")
async def load_initial_condition(payload: InitialConditionLoad) -> dict:
    try:
        filename = _validate_filename(payload.name)
        path = INITIAL_CONDITION_DIR / filename
        if not path.exists():
            return {"values": _config_values(), "error": "preset not found"}
        input_json = json.loads(path.read_text(encoding="utf-8"))
        cfg_raw = input_json.get("config", {}) if isinstance(input_json, dict) else {}
        values = _defaults()
        for key, control in control_index.items():
            if key in cfg_raw:
                values[key] = _cast_control_value(control, cfg_raw[key])
        values["particle_counts"] = sanitize_particle_counts(
            cfg_raw.get("particle_counts", values.get("particle_counts", [])),
            int(values["species_count"]),
            int(values["particles_per_species"]),
        )
        sim.cfg = SimConfig(**values)
        sim.reset_state(random_matrix=False)
        if isinstance(input_json, dict) and "interaction_matrix" in input_json:
            sim.set_matrix(sanitize_matrix(input_json["interaction_matrix"], sim.cfg.species_count))
        return {"values": _config_values()}
    except (ValueError, json.JSONDecodeError) as exc:
        return {"values": _config_values(), "error": str(exc)}

@app.get("/api/replay/list")
async def replay_list() -> dict:
    return {"items": list_runs(REPLAY_DIR)}


@app.get("/api/replay/{name}/meta")
async def replay_meta(name: str) -> dict:
    try:
        return ReplayReader(REPLAY_DIR, name).meta
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/replay/{name}/frames")
async def replay_frames(name: str, start: int = Query(default=0, ge=0), count: int = Query(default=1, ge=1, le=600)) -> Response:
    try:
        reader = ReplayReader(REPLAY_DIR, name)
        payload = reader.read_frames(start, count)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    headers = {"x-frame-start": str(start), "x-frame-count": str(min(count, max(0, reader.frame_count - start))), "x-particle-count": str(reader.count)}
    return Response(content=payload, media_type="application/octet-stream", headers=headers)


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
    if "particle_counts" in payload.updates:
        values["particle_counts"] = sanitize_particle_counts(payload.updates["particle_counts"], next_species_count, int(values["particles_per_species"]))
        needs_reset = True
    if "interaction_matrix" in payload.updates:
        next_matrix = sanitize_matrix(payload.updates["interaction_matrix"], next_species_count)

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
                sim.step_many(sim.cfg.steps_per_frame)
            await websocket.send_bytes(sim.snapshot())
            await websocket.send_text(json.dumps({"type": "stats", "entropy": sim.information_entropy()}))
            await asyncio.sleep(1 / 60)
    except WebSocketDisconnect:
        return


def main() -> None:
    uvicorn.run("particle_life.realtime:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
