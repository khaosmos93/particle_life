"""Simple trajectory visualization for Particle Life outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import imageio.v2 as imageio
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError("analysis.py requires imageio. Install it to write GIF/MP4 output.") from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError("analysis.py requires matplotlib. Install it to render frames.") from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError("analysis.py requires numpy.") from exc

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "analysis.py requires pandas with parquet support (pyarrow engine assumed present)."
    ) from exc


REQUIRED_COLUMNS = {"step", "id", "pos"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a simple trajectory animation.")
    parser.add_argument("--in", dest="input_dir", required=True, help="Input parquet dataset directory")
    parser.add_argument("--out", dest="output_path", required=True, help="Output animation path (.gif or .mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--stride", type=int, default=1, help="Render every k-th step")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of frames")
    parser.add_argument("--size", type=int, default=800, help="Output image size (pixels, square)")
    parser.add_argument("--point_size", type=float, default=4.0, help="Scatter point size")
    parser.add_argument(
        "--box",
        type=float,
        default=None,
        help="Override axis box size. If omitted, infer from data and clamp near-1 values to exactly 1.0.",
    )
    return parser.parse_args()


def read_dataset(input_dir: str) -> pd.DataFrame:
    df = pd.read_parquet(input_dir)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        needed = ", ".join(sorted(missing))
        raise ValueError(f"Missing required column(s): {needed}")

    df = df.copy()
    df["x"] = df["pos"].apply(lambda v: float(v[0]))
    df["y"] = df["pos"].apply(lambda v: float(v[1]))
    return df


def steps_in_order(df: pd.DataFrame) -> list[int]:
    return [int(s) for s in sorted(df["step"].unique())]


def infer_box_size(df: pd.DataFrame, override: float | None) -> float:
    if override is not None:
        return float(override)

    max_pos = float(np.nanmax(df[["x", "y"]].to_numpy()))
    box = max(1.0, max_pos)
    if abs(box - 1.0) < 1e-6:
        return 1.0
    return box


def render_animation(
    df: pd.DataFrame,
    steps: list[int],
    output_path: str,
    fps: int,
    size: int,
    point_size: float,
    box: float,
) -> None:
    suffix = Path(output_path).suffix.lower()
    if suffix not in {".gif", ".mp4"}:
        raise ValueError("Output path must end with .gif or .mp4")

    fig_inches = size / 100.0
    fig, ax = plt.subplots(figsize=(fig_inches, fig_inches), dpi=100)
    ax.set_xlim(0.0, box)
    ax.set_ylim(0.0, box)
    ax.set_aspect("equal", adjustable="box")
    scatter = ax.scatter([], [], s=point_size)

    writer_kwargs = {"fps": fps}
    if suffix == ".gif":
        writer_kwargs["mode"] = "I"

    try:
        writer = imageio.get_writer(output_path, **writer_kwargs)
    except Exception as exc:
        if suffix == ".mp4":
            raise RuntimeError("MP4 requires ffmpeg; try output .gif instead.") from exc
        raise

    with writer:
        for step in steps:
            frame_df = df[df["step"] == step]
            x = frame_df["x"].to_numpy()
            y = frame_df["y"].to_numpy()
            scatter.set_offsets(np.column_stack([x, y]))
            ax.set_title(f"step={step}")

            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(img)

    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max_steps must be positive when provided")
    if args.size <= 0:
        raise ValueError("--size must be positive")
    if args.point_size <= 0:
        raise ValueError("--point_size must be positive")

    df = read_dataset(args.input_dir)
    steps = steps_in_order(df)
    steps = steps[:: args.stride]
    if args.max_steps is not None:
        steps = steps[: args.max_steps]
    if not steps:
        raise ValueError("No steps selected for rendering. Check --stride/--max_steps and dataset content.")

    box = infer_box_size(df, args.box)
    render_animation(
        df=df,
        steps=steps,
        output_path=args.output_path,
        fps=args.fps,
        size=args.size,
        point_size=args.point_size,
        box=box,
    )


if __name__ == "__main__":
    main()
