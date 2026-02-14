from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from particle_life.core import ParticleLifeSim, SimulationParams, pack_binary_frame

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"


class Hub:
    def __init__(self) -> None:
        self.params = SimulationParams()
        self.sim = ParticleLifeSim(self.params)
        self.running = True
        self.speed = 1.0
        self.send_every = 1
        self.target_fps = 60
        self.point_size = 15.0
        self.color_scheme = "direct_clamp"
        self.physics_t = 0.0
        self.frame_id = 0
        self.frame = pack_binary_frame(self.sim.pos, self.sim.state, self.params.box_size)
        self.lock = asyncio.Lock()

    def payload(self) -> dict:
        return {
            "type": "params",
            "dt": self.params.dt,
            "speed": self.speed,
            "send_every": self.send_every,
            "target_fps": self.target_fps,
            "point_size": self.point_size,
            "physics_t": self.physics_t,
            "seed": self.params.seed,
            "color_scheme": self.color_scheme,
        }

    async def update(self, msg: dict) -> dict:
        async with self.lock:
            if "dt" in msg and float(msg["dt"]) > 0:
                self.params.dt = float(msg["dt"])
            if "speed" in msg:
                self.speed = min(10.0, max(0.1, float(msg["speed"])))
            if "send_every" in msg:
                self.send_every = max(1, int(msg["send_every"]))
            if "target_fps" in msg:
                self.target_fps = max(1, int(msg["target_fps"]))
            if "point_size" in msg and float(msg["point_size"]) > 0:
                self.point_size = float(msg["point_size"])
            if msg.get("color_scheme"):
                self.color_scheme = str(msg["color_scheme"])
            return self.payload()


hub = Hub()
app = FastAPI()


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/static/app.js")
async def static_js() -> FileResponse:
    return FileResponse(WEB_DIR / "app.js")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    sender = asyncio.create_task(sender_loop(ws))
    try:
        await ws.send_text(json.dumps(hub.payload()))
        while True:
            msg = await ws.receive_json()
            t = msg.get("type")
            if t == "set_running":
                async with hub.lock:
                    hub.running = bool(msg.get("running", True))
            elif t == "reset":
                async with hub.lock:
                    hub.sim = ParticleLifeSim(hub.params)
                    hub.physics_t = 0.0
                    hub.frame = pack_binary_frame(hub.sim.pos, hub.sim.state, hub.params.box_size)
                    hub.frame_id += 1
                await ws.send_text(json.dumps(hub.payload()))
            elif t == "set_seed":
                seed = int(msg.get("seed", 0))
                async with hub.lock:
                    hub.sim.reseed(seed)
                    hub.physics_t = 0.0
                    hub.frame = pack_binary_frame(hub.sim.pos, hub.sim.state, hub.params.box_size)
                    hub.frame_id += 1
                await ws.send_text(json.dumps({"type": "seed", "seed": seed}))
                await ws.send_text(json.dumps(hub.payload()))
            elif t == "update_params":
                await ws.send_text(json.dumps(await hub.update(msg)))
    except (WebSocketDisconnect, RuntimeError, ValueError):
        pass
    finally:
        sender.cancel()


async def physics_loop() -> None:
    frame_tick = 0
    while True:
        async with hub.lock:
            if hub.running:
                dt_eff = hub.params.dt * hub.speed
                hub.sim.step(dt_eff)
                hub.physics_t += dt_eff
                if frame_tick % hub.send_every == 0:
                    hub.frame = pack_binary_frame(hub.sim.pos, hub.sim.state, hub.params.box_size)
                    hub.frame_id += 1
            target_fps = hub.target_fps
        frame_tick += 1
        await asyncio.sleep(1.0 / max(1, target_fps))


async def sender_loop(ws: WebSocket) -> None:
    last = -1
    while True:
        async with hub.lock:
            if hub.frame_id != last:
                frame = hub.frame
                last = hub.frame_id
                perf = {"type": "perf", "physics_t": hub.physics_t}
            else:
                frame = None
                perf = None
        if frame is not None:
            await ws.send_bytes(frame)
        if perf is not None:
            await ws.send_text(json.dumps(perf))
        await asyncio.sleep(1.0 / 60.0)


@app.on_event("startup")
async def startup() -> None:
    app.state.physics_task = asyncio.create_task(physics_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    app.state.physics_task.cancel()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("particle_life.realtime:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
