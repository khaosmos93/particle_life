# Particle Life

Minimal Particle Life simulation with:
- periodic boundary conditions using the minimum image convention,
- cell-list based neighbor candidate search,
- Parquet dataset trajectory output.

## Run

```bash
python -m particle_life.sim --out data/raw/run_ds --steps 200
```

Example sanity run:

```bash
python -m particle_life.sim --out data/raw/sample_ds --steps 5 --n 50 --dt 0.01 --seed 0 --box 1.0
```

Chunked output (50 steps per Parquet file):

```bash
python -m particle_life.sim --out data/raw/run_ds --steps 200 --chunk 50
```

## Realtime WebGL (Binary Streaming)

Run locally:

```bash
python -m particle_life.realtime
```

Open `http://localhost:8000`.

Binary frame protocol is unchanged:
- Header: `int32[posDim, stateDim, N]` + `float32[box_size]`
- Body: `float32` rows of `[pos..., state...]`

Control messages stay JSON text over WebSocket.

## Realtime controls

The Web UI supports:
- Dynamic `dt`, `speed`, `send_every`, and `point_size` updates.
- Seed control:
  - numeric seed input,
  - **Randomize Seed** button,
  - **Apply Seed** button (sends `{ "type": "set_seed", "seed": <int> }`).
- Color scheme selection (`direct_clamp`, `normalize`, `abs`, `softmax`, `hsv_like`) for state-to-RGB mapping on the frontend.

Initialization is random (seeded) only.

Color mapping is render-side only; backend continues to stream raw states in binary frames.
