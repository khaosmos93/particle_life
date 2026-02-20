from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ReplayWriter:
    def __init__(self, root: Path, run_name: str, metadata: dict):
        self.dir = root / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "meta.json"
        self.frames_path = self.dir / "frames.f32"
        self.metadata = metadata
        self.frame_count = 0
        self._fp = self.frames_path.open("wb")

    def write_frame(self, frame: bytes) -> None:
        self._fp.write(frame)
        self.frame_count += 1

    def close(self) -> None:
        self._fp.flush()
        self._fp.close()
        meta = dict(self.metadata)
        meta["frame_count"] = self.frame_count
        meta["storage"] = {"dtype": "float32", "cols": 5, "layout": "interleaved", "file": "frames.f32"}
        self.meta_path.write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")


class ReplayReader:
    def __init__(self, root: Path, run_name: str):
        self.dir = root / run_name
        self.meta_path = self.dir / "meta.json"
        self.frames_path = self.dir / "frames.f32"
        if not self.meta_path.exists() or not self.frames_path.exists():
            raise FileNotFoundError("run not found")
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.count = int(self.meta["particle_count"])
        self.frame_count = int(self.meta["frame_count"])
        self.frame_bytes = self.count * 5 * np.dtype(np.float32).itemsize

    def read_frames(self, start: int, count: int) -> bytes:
        start = max(0, min(start, self.frame_count))
        count = max(0, min(count, self.frame_count - start))
        if count == 0:
            return b""
        with self.frames_path.open("rb") as fp:
            fp.seek(start * self.frame_bytes)
            return fp.read(count * self.frame_bytes)


def list_runs(root: Path) -> list[str]:
    if not root.exists():
        return []
    out = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "meta.json").exists() and (p / "frames.f32").exists():
            out.append(p.name)
    return out
