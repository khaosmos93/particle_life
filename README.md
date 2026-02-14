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


## Initial condition JSON schema (stable v1)

The app now supports a full simulation input JSON that deterministically defines a run.

Top-level fields:

- `schema_version`: must be `1`
- `config`: full simulation/render config (same keys as `/api/config` values)
- `num_types` (optional): alias for `config.species_count` for editor-friendly presets
- `interaction_matrix`: `species_count x species_count` values in `[-1, 1]`
- `particles`: array of `{ "position": [x, y], "velocity": [vx, vy], "type": int }`

Notes:
- `config.seed` is included so RNG-dependent behavior is reproducible.
- Particle positions are absolute world coordinates in `[0, world_size]`.
- On load, the file fully replaces current simulation config, matrix, and particle state.

Example:

```json
{"schema_version":1,"num_types":2,"config":{"species_count":2,"particles_per_species":10,"particle_counts":[2,1],"world_size":1.0,"interaction_radius":0.11,"repel_radius":0.025,"force_scale":0.42,"dt":0.015,"damping":0.975,"max_speed":0.05,"steps_per_frame":1,"boundary_mode":"wrap","point_size":3.0,"point_opacity":0.95,"background_alpha":1.0,"show_hud":true,"pbc_tiling":false,"color_mode":"species","type_colors":["#ff6f5f","#56c3ff"],"seed":0},"interaction_matrix":[[1,0.2],[-0.2,1]],"particles":[{"position":[0.2,0.3],"velocity":[0,0],"type":0},{"position":[0.8,0.6],"velocity":[0,0],"type":1}]}
```

## Initial condition editor

Open `http://localhost:8000/editor`.

- Set **Number of Types**, then pick a type, brush radius, and per-stroke density (defaults to `1`).
- Click-drag on canvas to paint particles for that type.
- Repeat with different types to create per-type spatial distributions.
- Click **Save JSON** to write a preset under `data/initial_condition/`.

## Loading presets in main UI

In `http://localhost:8000`:

- Use **Load input JSON** to load files from `data/initial_condition/`.
- This applies all config values, interaction matrix, and initial particles atomically.
- Use **Open editor** to create/edit new initial-condition files.
