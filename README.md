# Particle Life (FastAPI + WebGL)

Performance-focused Particle Life with a refactored CPU simulation core, realtime streaming, offline replay generation, and in-browser replay playback.

## Setup

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

## Run realtime app

```bash
PYTHONPATH=src python -m particle_life.realtime
```

Open `http://localhost:8000`.

## Architecture overview

- `particle_life/physics/`
  - `neighbors.py`: uniform-grid cell binning + compact pair-chunk construction (local interactions, no dense NxN matrix).
  - `kernels.py`: vectorized force accumulation over pair chunks with cutoff + PBC support.
  - `integrator.py`: SoA state integration (damping, noise, max-speed clamp, boundary handling).
- `particle_life/simulation/`
  - `config.py`: simulation config and sanitizers.
  - `engine.py`: orchestration loop and state container (`pos`, `vel`, `species`, matrix).
- `particle_life/realtime.py`
  - FastAPI app, realtime websocket streaming, config APIs, replay metadata/chunk APIs.
- `particle_life/storage.py`
  - Offline replay writer/reader (`meta.json` + contiguous `frames.f32`).
- `particle_life/offline.py`
  - CLI for high-throughput offline generation and 10k-scale profiling.

## Realtime + Web UI modes

The UI provides:
- **Live mode** (current websocket stream).
- **Replay mode**:
  - load recorded run
  - play / pause
  - scrub via timeline
  - chunked frame fetching from backend APIs
  - same WebGL renderer/frame decode path as live mode

## Offline run generation

Generate a replay with the same simulation algorithm as realtime:

```bash
PYTHONPATH=src python -m particle_life.offline generate \
  --name run_10k \
  --output data/replays \
  --frames 2400 \
  --species-count 6 \
  --particles-per-species 1667 \
  --steps-per-frame 1 \
  --seed 42
```

Output format per run directory:
- `meta.json` (config/seed/interaction matrix/particle count/frame count/storage layout)
- `frames.f32` (append-friendly contiguous float32 frame records)

## Replay playback

1. Generate one or more runs in `data/replays`.
2. Start realtime app.
3. In UI: switch to **Replay** mode.
4. Select run → **Load run**.
5. Use **Play/Pause** and timeline scrub.

## Profiling (N≈10,000)

```bash
PYTHONPATH=src python -m particle_life.offline profile --particles 10000 --steps 120 --warmup 30
```

This prints total runtime and average step milliseconds.

## Notes

- Core hot path avoids dense NxN memory and per-particle Python objects.
- Data layout is structure-of-arrays (`positions`, `velocities`, `species`).
- Realtime wire format remains float32 frame records compatible with the existing renderer semantics.
