# Particle Life

Minimal Particle Life simulation with:
- periodic boundary conditions using the minimum image convention,
- cell-list based neighbor candidate search,
- CSV trajectory output.

## Run

```bash
python -m particle_life.sim --out data/raw/run.csv --steps 200
```

Example sanity run:

```bash
python -m particle_life.sim --out data/raw/sample.csv --steps 5 --n 50 --dt 0.01 --seed 0 --box 1.0
```

## Output format (CSV)

Long-format table with one row per particle per step.

```python
import pandas as pd

df = pd.read_csv("data/raw/run.csv")

df.groupby("step").size()
df.groupby("id")[["x", "y"]].plot()
```
