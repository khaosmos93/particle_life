# Particle Life

Minimal Particle Life simulation with:
- periodic boundary conditions using the minimum image convention,
- cell-list based neighbor candidate search,
- chunked parquet dataset output.

## Run

```bash
python -m particle_life.sim --out data/raw/run_ds --steps 200
```

Chunked output example (50 steps per parquet file):

```bash
python -m particle_life.sim --out data/raw/run_ds --steps 200 --chunk 50
```

Example sanity run:

```bash
python -m particle_life.sim --out data/raw/sample_ds --steps 7 --n 5 --dt 0.01 --seed 0 --chunk 3
```

## Output format (Parquet dataset)

Long-format table with one row per particle per step, partitioned by `chunk_id`.

```python
import pandas as pd

df = pd.read_parquet("data/raw/run_ds")
print(df.head())
print(df["pos"].iloc[0])
```

Read only one chunk partition:

```python
df = pd.read_parquet("data/raw/run_ds/chunk_id=0")
```
