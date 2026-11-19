"""Small shared helpers: paths, hashing, atomic JSON, ordering, statistics."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "paper_v1.json"
SETS_DIR = ROOT / "data" / "paper_v1"
RESULTS_DIR = ROOT / "results" / "paper_v1"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    return json.loads(Path(path or CONFIG_PATH).read_text())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: Any) -> str:
    """Stable hash of any JSON value (same scheme as the course studies)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def order_key(record_id: str, salt: str = "paper_v1", seed: int = 566) -> str:
    return digest([seed, salt, record_id])


def hash_order(ids: Iterable[str], salt: str = "paper_v1", seed: int = 566) -> list[str]:
    return sorted(ids, key=lambda x: order_key(x, salt, seed))


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=_default))
    os.replace(tmp, path)


def _default(value: Any):
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=_default) + "\n")
    os.replace(tmp, path)


def git_commit(root: Path = ROOT) -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", "src", "configs", "scripts"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": sha, "dirty": dirty}


def environment() -> dict[str, Any]:
    """Versions and GPU recorded with every result."""
    import importlib.metadata as md
    import platform

    info: dict[str, Any] = {"python": platform.python_version(), "host": platform.node()}
    for name in ("torch", "transformers", "Pillow", "numpy", "paddleocr"):
        try:
            info[name] = md.version(name)
        except md.PackageNotFoundError:
            info[name] = None
    try:
        import torch

        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    except ImportError:
        info["gpu"] = None
    info["slurm_job"] = os.environ.get("SLURM_JOB_ID")
    return info


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval; (nan, nan) for n == 0."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
