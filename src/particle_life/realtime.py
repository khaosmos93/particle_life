from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


@dataclass
class SimConfig:
    species_count: int = 5
    particles_per_species: int = 220
    world_size: float = 1.0
    interaction_radius: float = 0.11
    dt: float = 0.015
    damping: float = 0.975


class ParticleLifeSim:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.count = cfg.species_count * cfg.particles_per_species

        self.positions = np.random.rand(self.count, 2).astype(np.float32)
        self.velocities = np.zeros((self.count, 2), dtype=np.float32)
        self.species = np.repeat(np.arange(cfg.species_count), cfg.particles_per_species).astype(np.int32)
        np.random.shuffle(self.species)

        self.matrix = np.random.uniform(-1.0, 1.0, (cfg.species_count, cfg.species_count)).astype(np.float32)
        np.fill_diagonal(self.matrix, np.random.uniform(0.2, 1.0, cfg.species_count))

    def step(self) -> None:
        cfg = self.cfg

        delta = self.positions[:, None, :] - self.positions[None, :, :]
        delta -= np.round(delta / cfg.world_size) * cfg.world_size

        dist2 = np.sum(delta * delta, axis=2) + 1e-9
        within = dist2 < (cfg.interaction_radius * cfg.interaction_radius)
        np.fill_diagonal(within, False)

        dist = np.sqrt(dist2)
        influence = np.clip(1.0 - dist / cfg.interaction_radius, 0.0, 1.0)

        interaction = self.matrix[self.species[:, None], self.species[None, :]]
        strength = interaction * influence * within

        inv_dist = 1.0 / dist
        direction = -delta * inv_dist[:, :, None]
        force = np.sum(direction * strength[:, :, None], axis=1)

        self.velocities = self.velocities * cfg.damping + force * cfg.dt
        self.positions = (self.positions + self.velocities * cfg.dt) % cfg.world_size

    def snapshot(self) -> bytes:
        data = np.empty((self.count, 3), dtype=np.float32)
        data[:, :2] = self.positions
        data[:, 2] = self.species.astype(np.float32)
        return data.tobytes()


app = FastAPI(title="Particle Life")
app.mount("/static", StaticFiles(directory="src/particle_life/static"), name="static")
sim = ParticleLifeSim(SimConfig())


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("src/particle_life/static/index.html")


@app.websocket("/ws")
async def stream_particles(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            sim.step()
            await websocket.send_bytes(sim.snapshot())
            await asyncio.sleep(1 / 60)
    except WebSocketDisconnect:
        return


def main() -> None:
    uvicorn.run("particle_life.realtime:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
