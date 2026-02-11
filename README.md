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
