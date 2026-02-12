import asyncio
import argparse
import contextlib
import json
import time
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
        self.send_every = 1
        self.target_fps = 60
        self.point_size = 3.0
        self.physics_time = 0.0
        self.speed = 1.0
        self.clients: set[WebSocket] = set()
        self.lock = asyncio.Lock()
        self.latest_bytes: bytes | None = None
        self.latest_physics_t = 0.0
        self.latest_frame_id = 0
        self.dirty = False

    def _init_realtime_particles(self):
        particles = init_particles(self.cfg)
        # for p in particles:
        #     p.vel = p.vel + self.realtime_speed * self.rng.normal(size=2)
        return particles

    def _params_payload(self) -> dict:
        return {
            "type": "params",
            "dt": self.cfg.dt,
            "speed": self.speed,
            "send_every": self.send_every,
            "target_fps": self.target_fps,
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
            self.latest_physics_t = 0.0
            self.latest_bytes = pack_frame(self.particles, self.cfg.box_size)
            self.latest_frame_id += 1
            self.dirty = True

    async def set_params(self, params: dict) -> None:
        async with self.lock:
            cfg_dict = asdict(self.cfg)
            for key, value in params.items():
                if key in cfg_dict and key not in {"steps", "chunk", "out_path"}:
                    cfg_dict[key] = value

            if "state_dim" in cfg_dict:
                cfg_dict["state_dim"] = max(3, int(cfg_dict["state_dim"]))

            self.cfg = SimulationConfig(**cfg_dict)
            self.target_fps = max(1, int(params.get("target_fps", self.target_fps)))
            self.rng = np.random.default_rng(self.cfg.seed)
            self.particles = self._init_realtime_particles()
            self.physics_time = 0.0
            self.latest_physics_t = 0.0
            self.latest_bytes = pack_frame(self.particles, self.cfg.box_size)
            self.latest_frame_id += 1
            self.dirty = True

    async def update_realtime_params(
        self,
        dt: float,
        speed: float,
        send_every: int,
        target_fps: int | None = None,
        point_size: float | None = None,
    ) -> dict:
        async with self.lock:
            if dt > 0:
                cfg_dict = asdict(self.cfg)
                cfg_dict["dt"] = float(dt)
                self.cfg = SimulationConfig(**cfg_dict)
            self.speed = min(10.0, max(0.1, float(speed)))
            self.send_every = max(1, int(send_every))
            if target_fps is not None:
                self.target_fps = max(1, int(target_fps))
            if point_size is not None and point_size > 0:
                self.point_size = float(point_size)
            return self._params_payload()

    async def get_params_payload(self) -> dict:
        async with self.lock:
            return self._params_payload()

    async def get_stats_payload(self) -> dict:
        async with self.lock:
            return {
                "type": "stats",
                "physics_t": self.latest_physics_t,
                "target_fps": self.target_fps,
            }

    async def physics_tick(self, frame_interval_idx: int) -> None:
        async with self.lock:
            if not self.running:
                return

            dt_eff = self.cfg.dt * self.speed
            cfg_dict = asdict(self.cfg)
            cfg_dict["dt"] = dt_eff
            step_cfg = SimulationConfig(**cfg_dict)

            step(self.particles, step_cfg, self.rng)
            self.physics_time += dt_eff
            self.latest_physics_t = self.physics_time

            if frame_interval_idx % self.send_every == 0:
                self.latest_bytes = pack_frame(self.particles, self.cfg.box_size)
                self.latest_frame_id += 1
                self.dirty = True

    async def sender_loop(self, ws: WebSocket) -> None:
        last_sent_frame_id = -1
        last_stats_sent = time.monotonic()
        next_deadline = time.monotonic()
        while True:
            async with self.lock:
                target_fps = max(1, int(self.target_fps))

            next_deadline += 1.0 / target_fps
            sleep_s = max(0.0, next_deadline - time.monotonic())
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

            frame = None
            async with self.lock:
                if self.dirty and self.latest_bytes is not None and self.latest_frame_id != last_sent_frame_id:
                    frame = self.latest_bytes
                    last_sent_frame_id = self.latest_frame_id

            if frame is not None:
                await ws.send_bytes(frame)

            now = time.monotonic()
            if now - last_stats_sent >= 1.0:
                await ws.send_text(json.dumps(await self.get_stats_payload()))
                last_stats_sent = now


sim = RealtimeSimulation()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.sim_task = asyncio.create_task(physics_loop())
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


async def physics_loop() -> None:
    frame_interval_idx = 0
    while True:
        await sim.physics_tick(frame_interval_idx)
        frame_interval_idx += 1
        await asyncio.sleep(0)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    sim.clients.add(ws)
    sender_task = asyncio.create_task(sim.sender_loop(ws))
    try:
        await ws.send_text(json.dumps(await sim.get_params_payload()))

        while True:
            message = await ws.receive()

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
                    speed = float(msg.get("speed", sim.speed))
                    send_every = int(msg.get("send_every", sim.send_every))
                    target_fps = int(msg.get("target_fps", sim.target_fps))
                    point_size = msg.get("point_size")
                    if dt <= 0:
                        dt = sim.cfg.dt
                    speed = min(10.0, max(0.1, speed))
                    if send_every < 1:
                        send_every = sim.send_every
                    if target_fps < 1:
                        target_fps = sim.target_fps

                    params = await sim.update_realtime_params(
                        dt=dt,
                        speed=speed,
                        send_every=send_every,
                        target_fps=target_fps,
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
        sender_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sender_task
        sim.clients.discard(ws)


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Run realtime Particle Life WebGL server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dt", type=float, default=sim.cfg.dt)
    parser.add_argument("--speed", type=float, default=sim.speed)
    parser.add_argument("--send-every", type=int, default=1)
    args = parser.parse_args()

    sim.speed = min(10.0, max(0.1, args.speed))
    sim.send_every = max(1, args.send_every)

    cfg_dict = asdict(sim.cfg)
    cfg_dict["dt"] = args.dt
    sim.cfg = SimulationConfig(**cfg_dict)

    uvicorn.run("particle_life.realtime:app", host=args.host, port=args.port)
