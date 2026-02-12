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
            n_particles=100,  #400,
            state_dim=3,
            dt=1.0,
            steps=0,
            seed=0,
            box_size=1.0,
            out_path=None,
        )
        self.rng = np.random.default_rng(self.cfg.seed)
        self.realtime_speed = 0.12
        self.particles = self._init_realtime_particles()
        self.running = True
        self.substeps = 1
        self.send_every = 1
        self.point_size = 3.0
        self.physics_time = 0.0
        self.dt_phys = self.cfg.dt / self.substeps
        self.clients: set[WebSocket] = set()
        self.lock = asyncio.Lock()

    def _init_realtime_particles(self):
        particles = init_particles(self.cfg)
        for p in particles:
            p.vel = p.vel + self.realtime_speed * self.rng.normal(size=2)
        return particles

    def _recompute_dt_phys(self) -> None:
        self.dt_phys = self.cfg.dt / self.substeps

    def _sim_time_per_interval(self) -> float:
        return self.cfg.dt

    def _params_payload(self) -> dict:
        return {
            "type": "params",
            "dt": self.cfg.dt,
            "substeps": self.substeps,
            "send_every": self.send_every,
            "point_size": self.point_size,
            "physics_t": self.physics_time,
        }

    async def set_running(self, value: bool) -> None:
        async with self.lock:
            self.running = bool(value)

    async def reset(self) -> None:
        async with self.lock:
            self.rng = np.random.default_rng(self.cfg.seed)
            self.particles = self._init_realtime_particles()
            self.physics_time = 0.0

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
            self.particles = self._init_realtime_particles()
            self.physics_time = 0.0
            self._recompute_dt_phys()

    async def update_realtime_params(
        self,
        dt: float,
        substeps: int,
        send_every: int,
        point_size: float | None = None,
    ) -> dict:
        async with self.lock:
            if dt > 0:
                cfg_dict = asdict(self.cfg)
                cfg_dict["dt"] = float(dt)
                self.cfg = SimulationConfig(**cfg_dict)
            self.substeps = max(1, int(substeps))
            self.send_every = max(1, int(send_every))
            if point_size is not None and point_size > 0:
                self.point_size = float(point_size)
            self._recompute_dt_phys()
            return self._params_payload()

    async def get_params_payload(self) -> dict:
        async with self.lock:
            return self._params_payload()

    async def get_stats_payload(self) -> dict:
        async with self.lock:
            return {"type": "stats", "physics_t": self.physics_time}

    async def tick(self, frame_interval_idx: int) -> bytes | None:
        async with self.lock:
            if not self.running:
                return None

            cfg_dict = asdict(self.cfg)
            cfg_dict["dt"] = self.dt_phys
            step_cfg = SimulationConfig(**cfg_dict)

            for _ in range(self.substeps):
                step(self.particles, step_cfg, self.rng)
            self.physics_time += self._sim_time_per_interval()

            if frame_interval_idx % self.send_every == 0:
                return pack_frame(self.particles, self.cfg.box_size)

            return None


sim = RealtimeSimulation()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.sim_task = asyncio.create_task(simulation_loop())
    try:
        yield
    finally:
        task = getattr(app.state, "sim_task", None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


app = FastAPI(lifespan=lifespan)
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


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    sim.clients.add(ws)
    last_stats_sent = asyncio.get_running_loop().time()
    try:
        await ws.send_text(json.dumps(await sim.get_params_payload()))

        while True:
            now = asyncio.get_running_loop().time()
            if now - last_stats_sent >= 1.0:
                await ws.send_text(json.dumps(await sim.get_stats_payload()))
                last_stats_sent = now

            try:
                message = await asyncio.wait_for(ws.receive(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            if message.get("text") is not None:
                msg = json.loads(message["text"])
                cmd = msg.get("type")

                if cmd == "set_running":
                    await sim.set_running(bool(msg.get("running", True)))
                elif cmd == "set_params":
                    await sim.set_params(msg.get("params", {}))
                    await ws.send_text(json.dumps(await sim.get_params_payload()))
                elif cmd == "update_params":
                    dt = float(msg.get("dt", sim.cfg.dt))
                    substeps = int(msg.get("substeps", sim.substeps))
                    send_every = int(msg.get("send_every", sim.send_every))
                    point_size = msg.get("point_size")
                    if dt <= 0:
                        dt = sim.cfg.dt
                    if substeps < 1:
                        substeps = sim.substeps
                    if send_every < 1:
                        send_every = sim.send_every

                    params = await sim.update_realtime_params(
                        dt=dt,
                        substeps=substeps,
                        send_every=send_every,
                        point_size=None if point_size is None else float(point_size),
                    )
                    await ws.send_text(json.dumps(params))
                elif cmd == "reset":
                    await sim.reset()
                    await ws.send_text(json.dumps(await sim.get_params_payload()))
            elif message.get("bytes") is not None:
                continue
    except (WebSocketDisconnect, json.JSONDecodeError, TypeError, ValueError):
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
    sim._recompute_dt_phys()

    uvicorn.run("particle_life.realtime:app", host=args.host, port=args.port)
