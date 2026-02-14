import argparse
import asyncio
import contextlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from particle_life.initializers import build_initial_state
from particle_life.particles import Interaction
from particle_life.sim import SimulationConfig

BASE_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = BASE_DIR / "web"



def pack_frame(particles, box_size):
    pos = np.stack([p.pos for p in particles])
    state = np.stack([p.state for p in particles])

    if pos.shape[1] == 2:
        pos = np.pad(pos, ((0, 0), (0, 1)))

    if state.shape[1] < 3:
        state = np.pad(state, ((0, 0), (0, 3 - state.shape[1])))

    pos_dim = pos.shape[1]
    state_dim = state.shape[1]
    n_particles = pos.shape[0]

    header_i = np.array([pos_dim, state_dim, n_particles], dtype=np.int32)
    header_f = np.array([box_size], dtype=np.float32)

    body = np.concatenate([pos, state], axis=1).astype(np.float32)

    return header_i.tobytes() + header_f.tobytes() + body.tobytes()


def pack_frame_arrays(pos: np.ndarray, state: np.ndarray, box_size: float) -> bytes:
    if pos.shape[1] == 2:
        pos = np.pad(pos, ((0, 0), (0, 1)))

    if state.shape[1] < 3:
        state = np.pad(state, ((0, 0), (0, 3 - state.shape[1])))

    pos_dim = pos.shape[1]
    state_dim = state.shape[1]
    n_particles = pos.shape[0]

    header_i = np.array([pos_dim, state_dim, n_particles], dtype=np.int32)
    header_f = np.array([box_size], dtype=np.float32)
    body = np.concatenate([pos, state], axis=1).astype(np.float32)
    return header_i.tobytes() + header_f.tobytes() + body.tobytes()


class RealtimeSimulation:
    def __init__(self) -> None:
        self.cfg = SimulationConfig(
            n_particles=100,
            state_dim=3,
            dt=0.01,
            steps=0,
            seed=0, #int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0]),
            box_size=1.0,
            out_path=None,
        )
        self.realtime_speed = 0.12
        self.running = True
        self.send_every = 1
        self.target_fps = 60
        self.point_size = 3.0
        self.physics_time = 0.0
        self.speed = 1.0
        self.color_scheme = "direct_clamp"
        self.clients: set[WebSocket] = set()
        self.lock = asyncio.Lock()
        self.latest_bytes: bytes | None = None
        self.latest_physics_t = 0.0
        self.latest_frame_id = 0
        self.dirty = False
        self.perf_acc = {
            "lock_wait_ms": 0.0,
            "lock_wait_n": 0,
            "pack_ms": 0.0,
            "pack_n": 0,
            "send_ms": 0.0,
            "send_n": 0,
        }

        self.rng = np.random.default_rng(self.cfg.seed)
        self.interaction = self._make_interaction(self.rng)
        self.particles = self._init_realtime_particles()
        self.latest_bytes = pack_frame(self.particles, self.cfg.box_size)

    @contextlib.asynccontextmanager
    async def locked(self):
        t0 = time.monotonic()
        await self.lock.acquire()
        self.perf_acc["lock_wait_ms"] += (time.monotonic() - t0) * 1000.0
        self.perf_acc["lock_wait_n"] += 1
        try:
            yield
        finally:
            self.lock.release()

    def _init_realtime_particles(self):
        _, particles, _ = build_initial_state(self.cfg, preset_id=None, seed=self.cfg.seed)
        return particles

    def _make_interaction(self, rng: np.random.Generator) -> Interaction:
        coupling_matrix = np.array([
            [0.0, 0.0, 1 / np.sqrt(2)],
            [0.0, 0.0, 0.0],
            [-np.sqrt(2), 0.0, 0.0],
        ], dtype=np.float64)
        return Interaction(
            box_size=self.cfg.box_size,
            wrap=self.cfg.wrap,
            r_min=self.cfg.r_min,
            r0=self.cfg.r_min,
            r_cut=self.cfg.r_cut,
            k_rep=1.0,
            k_mid=self.cfg.k_mid,
            gamma=self.cfg.gamma,
            sigma=self.cfg.sigma,
            rng=rng,
            coupling_matrix=coupling_matrix,
        )

    def _params_payload(self) -> dict:
        return {
            "type": "params",
            "dt": self.cfg.dt,
            "speed": self.speed,
            "send_every": self.send_every,
            "target_fps": self.target_fps,
            "point_size": self.point_size,
            "physics_t": self.physics_time,
            "seed": self.cfg.seed,
            "color_scheme": self.color_scheme,
        }

    def _reset_buffers(self) -> None:
        self.physics_time = 0.0
        self.latest_physics_t = 0.0
        self.latest_bytes = None
        self.dirty = False

    async def set_running(self, value: bool) -> None:
        async with self.locked():
            self.running = bool(value)

    async def reset(self) -> None:
        async with self.locked():
            cfg = self.cfg
        cfg_new, particles_new, meta = await asyncio.to_thread(
            build_initial_state, cfg, None, cfg.seed
        )
        async with self.locked():
            self.cfg = cfg_new
            self.rng = np.random.default_rng(self.cfg.seed)
            self.particles = particles_new
            self.speed = float(meta.get("speed", self.speed))
            self.interaction = self._make_interaction(self.rng)
            self._reset_buffers()

    async def _apply_initial_state(self, seed: int | None) -> dict:
        async with self.locked():
            cfg = self.cfg
        cfg_new, particles_new, meta = await asyncio.to_thread(build_initial_state, cfg, None, seed)
        async with self.locked():
            self.cfg = cfg_new
            self.particles = particles_new
            self.rng = np.random.default_rng(self.cfg.seed)
            self.speed = float(meta.get("speed", self.speed))
            self.interaction = self._make_interaction(self.rng)
            self._reset_buffers()
            return {
                "type": "seed",
                "seed": self.cfg.seed,
                "dt": self.cfg.dt,
                "speed": self.speed,
            }

    async def set_seed(self, seed: int | None) -> dict:
        return await self._apply_initial_state(seed)

    async def set_params(self, params: dict) -> None:
        async with self.locked():
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
            self.interaction = self._make_interaction(self.rng)
            self._reset_buffers()

    async def update_realtime_params(
        self,
        dt: float,
        speed: float,
        send_every: int,
        target_fps: int | None = None,
        point_size: float | None = None,
        color_scheme: str | None = None,
    ) -> dict:
        async with self.locked():
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
            if color_scheme:
                self.color_scheme = str(color_scheme)
            return self._params_payload()

    async def get_params_payload(self) -> dict:
        async with self.locked():
            return self._params_payload()

    async def get_stats_payload(self) -> dict:
        async with self.locked():
            lock_n = max(1, int(self.perf_acc["lock_wait_n"]))
            pack_n = max(1, int(self.perf_acc["pack_n"]))
            send_n = max(1, int(self.perf_acc["send_n"]))
            payload = {
                "type": "perf",
                "physics_t": self.latest_physics_t,
                "target_fps": self.target_fps,
                "particles": len(self.particles),
                "payload_bytes": 0 if self.latest_bytes is None else len(self.latest_bytes),
                "lock_wait_ms": self.perf_acc["lock_wait_ms"] / lock_n,
                "pack_ms": self.perf_acc["pack_ms"] / pack_n,
                "send_ms": self.perf_acc["send_ms"] / send_n,
            }
            self.perf_acc["lock_wait_ms"] = 0.0
            self.perf_acc["lock_wait_n"] = 0
            self.perf_acc["pack_ms"] = 0.0
            self.perf_acc["pack_n"] = 0
            self.perf_acc["send_ms"] = 0.0
            self.perf_acc["send_n"] = 0
            return {
                **payload,
            }

    def _compute_forces(self) -> np.ndarray:
        return self.interaction.compute_net_forces(self.particles)

    async def physics_tick(self, frame_interval_idx: int) -> None:
        need_pack = False
        pos_snapshot = None
        state_snapshot = None
        box_size = 0.0
        async with self.locked():
            if not self.running:
                return

            dt_eff = self.cfg.dt * self.speed
            forces = self._compute_forces()
            for i, p in enumerate(self.particles):
                p.vel = p.vel + (forces[i] / p.m) * dt_eff
                p.pos = p.pos + p.vel * dt_eff
                if self.cfg.wrap:
                    p.pos %= self.cfg.box_size

            self.physics_time += dt_eff
            self.latest_physics_t = self.physics_time

            if frame_interval_idx % self.send_every == 0:
                pos_snapshot = np.asarray([p.pos.copy() for p in self.particles], dtype=np.float64)
                state_snapshot = np.asarray([p.state.copy() for p in self.particles], dtype=np.float64)
                box_size = self.cfg.box_size
                need_pack = True

        if need_pack and pos_snapshot is not None and state_snapshot is not None:
            t0 = time.monotonic()
            frame_bytes = await asyncio.to_thread(pack_frame_arrays, pos_snapshot, state_snapshot, box_size)
            pack_ms = (time.monotonic() - t0) * 1000.0
            async with self.locked():
                self.latest_bytes = frame_bytes
                self.latest_frame_id += 1
                self.dirty = True
                self.perf_acc["pack_ms"] += pack_ms
                self.perf_acc["pack_n"] += 1

    async def sender_loop(self, ws: WebSocket) -> None:
        last_sent_frame_id = -1
        last_stats_sent = time.monotonic()
        next_deadline = time.monotonic()
        while True:
            async with self.locked():
                target_fps = max(1, int(self.target_fps))

            next_deadline += 1.0 / target_fps
            sleep_s = max(0.0, next_deadline - time.monotonic())
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

            frame = None
            async with self.locked():
                if self.dirty and self.latest_bytes is not None and self.latest_frame_id != last_sent_frame_id:
                    frame = self.latest_bytes
                    last_sent_frame_id = self.latest_frame_id

            if frame is not None:
                t0 = time.monotonic()
                await ws.send_bytes(frame)
                send_ms = (time.monotonic() - t0) * 1000.0
                async with self.locked():
                    self.perf_acc["send_ms"] += send_ms
                    self.perf_acc["send_n"] += 1

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


async def physics_loop() -> None:
    frame_interval_idx = 0
    while True:
        await sim.physics_tick(frame_interval_idx)
        frame_interval_idx += 1
        await asyncio.sleep(0)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    print("[ws] client connected")
    sim.clients.add(ws)
    tasks = [asyncio.create_task(sim.sender_loop(ws))]
    try:
        await ws.send_text(json.dumps(await sim.get_params_payload()))

        while True:
            try:
                message = await ws.receive()
            except WebSocketDisconnect:
                print("[ws] client disconnected")
                break
            if message.get("type") == "websocket.disconnect":
                print("[ws] client disconnected")
                break

            if message.get("text") is not None:
                msg = json.loads(message["text"])
                cmd = msg.get("type")

                if cmd == "set_running":
                    await sim.set_running(bool(msg.get("running", True)))
                elif cmd == "set_params":
                    await sim.set_params(msg.get("params", {}))
                    await ws.send_text(json.dumps(await sim.get_params_payload()))
                elif cmd == "set_seed":
                    seed_raw = msg.get("seed", None)
                    seed_ack = await sim.set_seed(None if seed_raw is None else int(seed_raw))
                    await ws.send_text(json.dumps(seed_ack))
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
                        color_scheme=msg.get("color_scheme"),
                    )
                    await ws.send_text(json.dumps(params))
                elif cmd == "reset":
                    await sim.reset()
                    await ws.send_text(json.dumps(await sim.get_params_payload()))
            elif message.get("bytes") is not None:
                continue
    except WebSocketDisconnect:
        print("[ws] client disconnected")
    except (json.JSONDecodeError, TypeError, ValueError, HTTPException):
        pass
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
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
