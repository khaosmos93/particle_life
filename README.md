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

## Initial-condition presets

Preset files live in:

```text
data/initial_conditions/
```

The realtime server exposes:
- `GET /api/presets`: list available preset metadata.
- WebSocket `{ "type": "load_preset", "preset": "<file>.json" }`: load and reset to that preset.

### Preset JSON schema

```json
{
  "meta": {
    "name": "string",
    "description": "string"
  },
  "world": {
    "dim": 2,
    "box_size": 1.0
  },
  "model": {
    "state_dim": 3,
    "canonical_states": [
      [1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0]
    ],
    "interaction": {
      "r_repulse": 0.02,
      "r_cut": 0.25,
      "strength": 1.0,
      "noise": 0.0,
      "damping": 0.0
    },
    "coupling": {
      "fn": "two_state_asymmetric",
      "params": {
        "same": 1.0,
        "s0_to_s1": 1.0,
        "s1_to_s0": -1.0
      }
    }
  },
  "sim": {
    "dt": 0.01,
    "speed": 1.0,
    "seed": 0
  },
  "particles": [
    {
      "id": 0,
      "m": 1.0,
      "pos": [0.1, 0.2],
      "vel": [0.0, 0.0],
      "state": [1.0, 0.0, 0.0]
    }
  ]
}
```

Rules implemented by realtime backend:
- No A/B labels are used anywhere.
- Exactly two canonical state vectors are required in presets.
- Every `particles[i].state` must exactly match either `canonical_states[0]` or `canonical_states[1]`.
- Coupling is selected by function name via `model.coupling.fn`.

### Coupling function

`"fn": "two_state_asymmetric"` is the built-in coupling rule used by presets:
- same-state pairs use `params.same`
- canonical state index `0 -> 1` uses `params.s0_to_s1`
- canonical state index `1 -> 0` uses `params.s1_to_s0`

This produces the required asymmetric interaction behavior without labels.

## Realtime controls

The Web UI now supports:
- Dynamic `dt`, `speed`, `send_every`, and `point_size` updates.
- Seed control:
  - numeric seed input,
  - **Randomize Seed** button,
  - **Apply Seed** button (sends `{ "type": "set_seed", "seed": <int> }`).
- Preset selection:
  - preset dropdown populated from `/api/presets`,
  - **Load Preset** button.
- Color scheme selection (`direct_clamp`, `normalize`, `abs`, `softmax`, `hsv_like`) for state-to-RGB mapping on the frontend.

Color mapping is render-side only; backend continues to stream raw states in binary frames.

## Included structured presets

Ten structured presets are provided:
1. `two_clusters.json`
2. `checkerboard.json`
3. `concentric_rings.json`
4. `stripe_bands.json`
5. `spiral_arms.json`
6. `yin_yang.json`
7. `four_quadrants.json`
8. `line_vs_cloud.json`
9. `two_lanes.json`
10. `radial_burst.json`

All presets are 2D, use `box_size=1.0`, `state_dim=3`, and the two canonical state vectors.
