import asyncio
import argparse
import contextlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from particle_life.sim import SimulationConfig, init_particles, step

BASE_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = BASE_DIR / "web"


class RealtimeSimulation:
    def __init__(self) -> None:
        self.cfg = SimulationConfig(
            n_particles=400,
            state_dim=3,
            dt=0.01,
            steps=0,
            seed=0,
            box_size=1.0,
            out_path=None,
        )
        self.rng = np.random.default_rng(self.cfg.seed)
        self.particles = init_particles(self.cfg)
        self.running = True
        self.substeps = 10
        self.send_every = 1
        self.clients: set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def set_running(self, value: bool) -> None:
        async with self.lock:
            self.running = bool(value)

    async def reset(self) -> None:
        async with self.lock:
            self.rng = np.random.default_rng(self.cfg.seed)
            self.particles = init_particles(self.cfg)

    async def set_params(self, params: dict) -> None:
        async with self.lock:
            cfg_dict = asdict(self.cfg)
            for key, value in params.items():
                if key in cfg_dict and key not in {"steps", "chunk", "out_path"}:
                    cfg_dict[key] = value

            if "state_dim" in cfg_dict:
                cfg_dict["state_dim"] = max(3, int(cfg_dict["state_dim"]))

            self.cfg = SimulationConfig(**cfg_dict)
            self.rng = np.random.default_rng(self.cfg.seed)
            self.particles = init_particles(self.cfg)

    async def tick(self, frame_interval_idx: int) -> bytes | None:
        async with self.lock:
            if not self.running:
                return None

            dt_phys = self.cfg.dt / self.substeps
            cfg_dict = asdict(self.cfg)
            cfg_dict["dt"] = dt_phys
            step_cfg = SimulationConfig(**cfg_dict)

            for _ in range(self.substeps):
                step(self.particles, step_cfg, self.rng)

            if frame_interval_idx % self.send_every == 0:
                return pack_frame(self.particles, self.cfg.box_size)

            return None


sim = RealtimeSimulation()
app = FastAPI()
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


def pack_frame(particles, box_size):
    pos = np.stack([p.pos for p in particles])
    state = np.stack([p.state for p in particles])

    if pos.shape[1] == 2:
        pos = np.pad(pos, ((0, 0), (0, 1)))

    if state.shape[1] < 3:
        state = np.pad(state, ((0, 0), (0, 3 - state.shape[1])))

    posDim = pos.shape[1]
    stateDim = state.shape[1]
    N = pos.shape[0]

    header_i = np.array([posDim, stateDim, N], dtype=np.int32)
    header_f = np.array([box_size], dtype=np.float32)

    body = np.concatenate([pos, state], axis=1).astype(np.float32)

    return header_i.tobytes() + header_f.tobytes() + body.tobytes()


async def simulation_loop() -> None:
    frame_interval_idx = 0
    while True:
        frame = await sim.tick(frame_interval_idx)

        frame_interval_idx += 1

        if frame is not None and sim.clients:
            stale = []
            for ws in sim.clients:
                try:
                    await ws.send_bytes(frame)
                except Exception:
                    stale.append(ws)
            for ws in stale:
                sim.clients.discard(ws)

        await asyncio.sleep(1 / 60)


@app.on_event("startup")
async def on_startup() -> None:
    app.state.sim_task = asyncio.create_task(simulation_loop())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    task = getattr(app.state, "sim_task", None)
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    sim.clients.add(ws)
    try:
        while True:
            text = await ws.receive_text()
            msg = json.loads(text)
            cmd = msg.get("type")

            if cmd == "set_running":
                await sim.set_running(bool(msg.get("running", True)))
            elif cmd == "set_params":
                await sim.set_params(msg.get("params", {}))
            elif cmd == "reset":
                await sim.reset()
    except (WebSocketDisconnect, json.JSONDecodeError):
        pass
    finally:
        sim.clients.discard(ws)


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Run realtime Particle Life WebGL server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dt", type=float, default=sim.cfg.dt)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--send-every", type=int, default=1)
    args = parser.parse_args()

    sim.substeps = max(1, args.substeps)
    sim.send_every = max(1, args.send_every)

    cfg_dict = asdict(sim.cfg)
    cfg_dict["dt"] = args.dt
    sim.cfg = SimulationConfig(**cfg_dict)

    uvicorn.run("particle_life.realtime:app", host=args.host, port=args.port)
