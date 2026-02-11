# Particle Life

Minimal Particle Life simulation with:
- periodic boundary conditions using the minimum image convention,
- cell-list based neighbor candidate search,
- JSONL trajectory output.

## Run

```bash
python -m particle_life.sim --out data/raw/run.jsonl --steps 200
```

Example sanity run:

```bash
python -m particle_life.sim --out data/raw/sample.jsonl --steps 5 --n 50 --dt 0.01 --seed 0 --box 1.0
```

## Output format (JSONL)

One JSON object per line:

```json
{
  "step": 0,
  "particles": [
    {"id": 0, "m": 1.0, "pos": [x, y], "vel": [vx, vy], "state": [s1, s2]}
  ]
}
```
