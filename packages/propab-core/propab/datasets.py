"""Curated public dataset descriptors and fetch helpers (Phase 5 connectors, v1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CuratedDataset:
    dataset_id: str
    title: str
    description: str
    source: str
    format: str
    url: str | None


# Stable raw URLs suitable for CI and sandboxes (no auth).
CURATED: dict[str, CuratedDataset] = {
    "iris": CuratedDataset(
        dataset_id="iris",
        title="Iris (tabular)",
        description="Classic 3-class flower measurements (4 features).",
        source="seaborn-data mirror",
        format="csv",
        url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    ),
    "penguins": CuratedDataset(
        dataset_id="penguins",
        title="Palmer Penguins",
        description="Species classification with bill and body measurements.",
        source="seaborn-data mirror",
        format="csv",
        url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv",
    ),
    "synthetic_gaussian": CuratedDataset(
        dataset_id="synthetic_gaussian",
        title="Synthetic Gaussian clusters",
        description="Generated locally for offline tests; no network fetch.",
        source="propab.builtin",
        format="numpy",
        url=None,
    ),
}


def list_curated_dataset_ids() -> list[str]:
    return sorted(CURATED.keys())


def describe_dataset(dataset_id: str) -> CuratedDataset | None:
    return CURATED.get(dataset_id)


def synthetic_gaussian_rows(*, n_rows: int, seed: int = 0) -> tuple[list[str], list[dict[str, Any]]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    n = max(5, min(int(n_rows), 5000))
    x = rng.standard_normal((n, 3))
    labels = (x[:, 0] + x[:, 1] > 0).astype(int)
    cols = ["f0", "f1", "f2", "label"]
    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append(
            {
                "f0": float(x[i, 0]),
                "f1": float(x[i, 1]),
                "f2": float(x[i, 2]),
                "label": int(labels[i]),
            }
        )
    return cols, rows
