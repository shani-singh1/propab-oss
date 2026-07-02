"""Generic tabular data prep for sweep engine."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def prepare_sweep_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_column: str,
    family_column: str,
    exclude_columns: frozenset[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return X, y, families, and columns actually used (after NA drop)."""
    skip = exclude_columns or frozenset()
    cols = [c for c in feature_columns if c in df.columns and c not in skip]
    if not cols:
        raise ValueError("No usable feature columns in dataframe")

    sub = df[[target_column, family_column, *cols]].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=cols + [target_column, family_column])
    if len(sub) < 8:
        raise ValueError(f"Too few complete rows after NA drop: {len(sub)}")

    X = sub[cols].to_numpy(dtype=float)
    y = sub[target_column].to_numpy(dtype=float)
    families = sub[family_column].astype(str).to_numpy()
    return X, y, families, cols
