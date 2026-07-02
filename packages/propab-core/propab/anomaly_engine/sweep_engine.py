"""Sweep engine — feature-subset experiments without LLM or hypotheses."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from propab.anomaly_engine.data_loader import prepare_sweep_matrix
from propab.anomaly_engine.subset_generator import generate_feature_subsets, generate_grouped_subsets

ALLOWED_MODELS = ("LinearRegression", "Ridge", "RandomForestRegressor")


@dataclass
class SweepConfig:
    target_column: str
    family_column: str
    feature_columns: list[str]
    max_subset_size: int = 3
    model_names: tuple[str, ...] = ALLOWED_MODELS
    random_state: int = 42
    # When set: singles/pairs/triplets within each group only (not full cross-product).
    feature_groups: dict[str, list[str]] | None = None
    exclude_columns: frozenset[str] | None = None


@dataclass
class SweepResult:
    feature_subset: list[str]
    model_name: str
    within_family_r2: float
    leave_one_family_out_r2: float
    global_r2: float
    family_baseline_r2: float
    surprise_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SweepResult:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


def _make_model(name: str, *, random_state: int) -> Any:
    if name == "LinearRegression":
        return Pipeline([
            ("scale", StandardScaler()),
            ("reg", LinearRegression()),
        ])
    if name == "Ridge":
        return Pipeline([
            ("scale", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=random_state)),
        ])
    if name == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=50,
            max_depth=3,
            min_samples_leaf=3,
            n_jobs=1,
            random_state=random_state,
        )
    raise ValueError(f"Model not allowed: {name}")


def _clip_r2(value: float) -> float:
    return float(max(-1.0, min(1.0, value)))


def _family_baseline_r2(y: np.ndarray, families: np.ndarray) -> float:
    preds = np.zeros_like(y, dtype=float)
    for fam in np.unique(families):
        mask = families == fam
        preds[mask] = float(np.mean(y[mask]))
    return _clip_r2(float(r2_score(y, preds))) if len(y) > 1 else 0.0


def _within_family_r2(X: np.ndarray, y: np.ndarray, families: np.ndarray, model: Any) -> float:
    scores: list[float] = []
    for fam in np.unique(families):
        mask = families == fam
        if int(mask.sum()) < 4:
            continue
        m = clone(model)
        m.fit(X[mask], y[mask])
        pred = m.predict(X[mask])
        scores.append(_clip_r2(float(r2_score(y[mask], pred))))
    return float(np.mean(scores)) if scores else 0.0


def _leave_one_family_out_r2(X: np.ndarray, y: np.ndarray, families: np.ndarray, model: Any) -> float:
    scores: list[float] = []
    for fam in np.unique(families):
        test = families == fam
        train = ~test
        if int(test.sum()) < 2 or int(train.sum()) < 4:
            continue
        m = clone(model)
        m.fit(X[train], y[train])
        pred = m.predict(X[test])
        scores.append(_clip_r2(float(r2_score(y[test], pred))))
    return float(np.mean(scores)) if scores else 0.0


def _per_family_lofo_r2(
    X: np.ndarray, y: np.ndarray, families: np.ndarray, model: Any,
) -> dict[str, float]:
    """Per-family LOFO R² for cross-family anomaly detection (fixes.md P1)."""
    scores: dict[str, float] = {}
    for fam in np.unique(families):
        test = families == fam
        train = ~test
        if int(test.sum()) < 2 or int(train.sum()) < 4:
            continue
        m = clone(model)
        m.fit(X[train], y[train])
        pred = m.predict(X[test])
        scores[str(fam)] = _clip_r2(float(r2_score(y[test], pred)))
    return scores


def _global_r2(X: np.ndarray, y: np.ndarray, model: Any) -> float:
    m = clone(model)
    m.fit(X, y)
    pred = m.predict(X)
    return _clip_r2(float(r2_score(y, pred)))


def run_single_experiment(
    X: np.ndarray,
    y: np.ndarray,
    families: np.ndarray,
    feature_indices: list[int],
    feature_names: list[str],
    model_name: str,
    *,
    random_state: int = 42,
) -> SweepResult:
    Xi = X[:, feature_indices]
    model = _make_model(model_name, random_state=random_state)
    fam_base = _family_baseline_r2(y, families)
    lofo = _leave_one_family_out_r2(Xi, y, families, model)
    per_family = _per_family_lofo_r2(Xi, y, families, model)
    pf_vals = list(per_family.values())
    lofo_std = float(np.std(pf_vals)) if len(pf_vals) > 1 else 0.0
    return SweepResult(
        feature_subset=feature_names,
        model_name=model_name,
        within_family_r2=_within_family_r2(Xi, y, families, model),
        leave_one_family_out_r2=lofo,
        global_r2=_global_r2(Xi, y, model),
        family_baseline_r2=fam_base,
        surprise_score=lofo - fam_base,
        metadata={
            "n_samples": len(y),
            "n_families": len(np.unique(families)),
            "per_family_lofo": per_family,
            "lofo_family_std": round(lofo_std, 4),
        },
    )


def _subset_list_for_config(config: SweepConfig, available: set[str]) -> list[list[str]]:
    if config.feature_groups:
        return generate_grouped_subsets(
            config.feature_groups,
            max_subset_size=config.max_subset_size,
            available_columns=available,
        )
    cols = [c for c in config.feature_columns if c in available]
    return generate_feature_subsets(cols, max_subset_size=config.max_subset_size)


def run_sweep(
    df: pd.DataFrame,
    config: SweepConfig,
    *,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    families: np.ndarray | None = None,
    active_columns: list[str] | None = None,
) -> list[SweepResult]:
    """Run feature-subset × model experiments. No LLM, no hypotheses."""
    if X is None or y is None or families is None:
        X, y, families, active_columns = prepare_sweep_matrix(
            df,
            config.feature_columns,
            target_column=config.target_column,
            family_column=config.family_column,
            exclude_columns=config.exclude_columns,
        )
    cols = active_columns or list(config.feature_columns)
    name_to_idx = {c: i for i, c in enumerate(cols)}
    subset_list = _subset_list_for_config(config, set(cols))

    results: list[SweepResult] = []
    for subset in subset_list:
        indices = [name_to_idx[c] for c in subset if c in name_to_idx]
        if not indices:
            continue
        for model_name in config.model_names:
            results.append(
                run_single_experiment(
                    X, y, families, indices, subset, model_name,
                    random_state=config.random_state,
                )
            )
    return results


def sweep_results_to_dataframe(results: list[SweepResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = r.to_dict()
        row["feature_subset"] = "|".join(r.feature_subset)
        rows.append(row)
    return pd.DataFrame(rows)


def dataframe_to_sweep_results(df: pd.DataFrame) -> list[SweepResult]:
    out: list[SweepResult] = []
    for _, row in df.iterrows():
        subset_raw = row.get("feature_subset") or ""
        subset = (
            [s for s in subset_raw.split("|") if s]
            if isinstance(subset_raw, str)
            else list(subset_raw)
        )
        d = row.to_dict()
        d["feature_subset"] = subset
        out.append(SweepResult.from_dict(d))
    return out
