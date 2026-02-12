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

## Output format (Parquet dataset)

Long-format table with one row per particle per step.

```python
import pandas as pd

df = pd.read_parquet("data/raw/run_ds")  # reads all chunk partitions
print(df.head())
print(df["pos"].iloc[0])

df.groupby("step").size()
```

Read only a single chunk partition:

```python
df = pd.read_parquet("data/raw/run_ds/chunk_id=0")
```

## Animation

Render a simple animation directly from a Parquet dataset directory (including Hive-style partitions such as `chunk_id=0/part.parquet`):

```bash
python -m particle_life.analysis --in data/raw/run_ds --out data/figures/run.gif --fps 30 --stride 2
```

Notes:
- GIF output is the easiest/most portable default.
- MP4 output is supported (`--out ...mp4`) but may require ffmpeg to be available.

## Realtime WebGL (Binary Streaming)

Run locally:

```bash
python -m particle_life.realtime
```

Open:

```text
http://localhost:8000
```

In GitHub Codespaces:
- Port 8000 auto-forwards.
- Use the forwarded browser link.
- WebSocket automatically uses `wss` when the page is loaded over HTTPS.

Notes:
- `--dt` is the base timestep for one frame interval.
- Physics uses substeps, so effective physics timestep is `dt / substeps`.
- Binary frames are sent every `--send-every` frame intervals (network decimation only).
- Realtime mode adds a small random initial velocity kick so motion is visible immediately after startup/reset.
- Simulation frames are streamed as binary `Float32` data over WebSocket.
- The frame protocol is dimension-agnostic for position and state vectors.
- Existing batch simulation workflow remains unchanged (`particle_life.sim`).

Examples:

```bash
python -m particle_life.realtime --dt 1.0 --substeps 20 --send-every 1
```

```bash
python -m particle_life.realtime --dt 1.0 --substeps 20 --send-every 3
```


## Realtime controls

- Sliders in the WebGL UI allow dynamic tuning of `dt`, `substeps`, and `send_every`.
- Changes apply immediately to the running simulation stream.
- No server restart is required.

## Environment setup (.venv)

This project standardizes on a project-local virtual environment at `.venv/`.

Use the exact setup sequence below (POSIX shell):

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U 'pip<25.3'
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
pip install -e .
```

### GitHub Codespaces

The devcontainer is configured for Python 3.12.1 and runs the same setup sequence automatically when the Codespace is created, including regenerating `requirements.txt` from `requirements.in` and installing editable project sources.

## Realtime stats and visuals

- The UI stats panel shows:
  - `FPS`: recent binary frame receive/render rate.
  - `realtime t`: wall-clock seconds since WebSocket open.
  - `physics t`: simulated time accumulated on the backend.
- `point_size` slider updates the existing WebGL `u_pointSize` uniform live and is synchronized with backend params acks.
