# Particle Life (FastAPI + WebGL)

Minimal realtime particle-life framework with a Python backend simulation and WebGL frontend rendering.

## Setup

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

## Run

```bash
PYTHONPATH=src python -m particle_life.realtime
```

Then open: `http://localhost:8000`

## Manual verification (pause + live matrix updates)

1. Start the app and wait for particles to move.
2. Click **Pause**. Confirm particles stop moving and Physics FPS drops near zero while Graphics FPS keeps updating.
3. While paused, edit one or more interaction matrix cells (or use **Apply preset**).
4. Click **Resume**. Confirm motion continues from the paused state and reflects the new matrix behavior.
5. While running, edit matrix values again and confirm behavior changes without reset/reload.

Backend sanity signal: when matrix updates are applied, server logs print a single-line version bump (`[sim] interaction matrix version -> N`).

## Codespaces

- Start the dev container (post-create runs setup automatically).
- In **Ports**, ensure port `8000` is forwarded/public as needed.
- Open the forwarded URL in your browser.
