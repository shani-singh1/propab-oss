"""Disk cache for expensive matbench dielectric frame derivations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SYMMETRY_CACHE = "matbench_dielectric_symmetry.json"


def symmetry_cache_path(data_dir: Path) -> Path:
    return data_dir / _SYMMETRY_CACHE


def load_symmetry_cache(data_dir: Path) -> dict[str, dict[str, Any]] | None:
    path = symmetry_cache_path(data_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    rows = payload.get("rows_by_index")
    return rows if isinstance(rows, dict) else None


def save_symmetry_cache(data_dir: Path, rows_by_index: dict[int, dict[str, Any]]) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = symmetry_cache_path(data_dir)
    path.write_text(
        json.dumps({"rows_by_index": {str(k): v for k, v in rows_by_index.items()}}, indent=2),
        encoding="utf-8",
    )
    return path
