"""LOFO verification for enzyme kinetics across EC classes."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from propab.domain_modules.enzyme_kinetics.adapter import EnzymeExperimentSpec, EnzymeKineticsAdapter


def _lofo_r2(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    unique = sorted(set(groups.tolist()))
    if len(unique) < 3:
        return 0.0
    model = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    scores: list[float] = []
    for held in unique:
        tr, te = groups != held, groups == held
        if tr.sum() < 10 or te.sum() < 5:
            continue
        model.fit(X[tr], y[tr])
        scores.append(float(r2_score(y[te], model.predict(X[te]))))
    return float(np.mean(scores)) if scores else 0.0


def _label_shuffle_null(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, n_perm: int = 40) -> tuple[float, list[float], float]:
    """Permutation null that shuffles the TARGET (y) within each EC-class group.

    The observed statistic is the leave-one-family-out R² of X→y. The null must
    break that relationship, so we permute ``y`` *within each group* (preserving
    each family's marginal outcome distribution and the group partition) and
    recompute the LOFO R². The previous implementation shuffled ``groups`` — the
    split variable — which left X→y fully intact, so the null R² tracked the
    observed value and the test had no power (the genomics/enzyme "label shuffle
    shuffled the split, not the label" bug class).
    """
    observed = _lofo_r2(X, y, groups)
    nulls: list[float] = []
    rng = np.random.default_rng(42)
    unique = np.unique(groups)
    for _ in range(n_perm):
        yp = y.copy()
        for g in unique:
            mask = groups == g
            yp[mask] = rng.permutation(yp[mask])
        nulls.append(_lofo_r2(X, yp, groups))
    p = float(np.mean(np.asarray(nulls) >= observed)) if nulls else 1.0
    return observed, nulls, p


def run_enzyme_experiment(spec: EnzymeExperimentSpec) -> dict[str, Any]:
    df = EnzymeKineticsAdapter().load_frame()
    if spec.target_column not in df.columns:
        raise ValueError(f"Unknown target: {spec.target_column}")
    # Never let the target leak in as its own predictor (trivial R²=1.0). Defence
    # in depth against a spec that lists the target in feature_subset.
    feat_cols = [c for c in spec.feature_subset if c in df.columns and c != spec.target_column]
    if not feat_cols:
        raise ValueError(f"No usable enzyme features: {spec.feature_subset}")
    sub = df[[spec.group_column, spec.target_column, *feat_cols]].dropna()
    if len(sub) < 40:
        raise ValueError(f"Too few enzymes: {len(sub)}")
    groups = sub[spec.group_column].astype(str).to_numpy()
    X = sub[feat_cols].to_numpy(dtype=float)
    y = sub[spec.target_column].to_numpy(dtype=float)
    if float(np.nanvar(y)) < 1e-12:
        # Constant target → R² degenerates to 1.0 and permuting it changes nothing.
        # Report no signal so a degenerate metric can never masquerade as a result.
        return {
            "lofo_r2": 0.0,
            "label_shuffle_null_p95": 1.0,
            "label_shuffle_null_p": 1.0,
            "verification_method": "leave_family_out",
            "metric_name": "lofo_r2",
            "metric_value": 0.0,
            "n_samples": len(sub),
            "n_groups": len(set(groups.tolist())),
            "feature_subset": feat_cols,
            "verified_true_steps": 0,
            "degenerate_target": True,
        }
    lofo_r2, nulls, p = _label_shuffle_null(X, y, groups)
    p95 = float(np.percentile(nulls, 95)) if nulls else 1.0
    return {
        "lofo_r2": lofo_r2,
        "label_shuffle_null_p95": p95,
        "label_shuffle_null_p": p,
        "verification_method": "leave_family_out",
        "metric_name": "lofo_r2",
        "metric_value": lofo_r2,
        "n_samples": len(sub),
        "n_groups": len(set(groups.tolist())),
        "feature_subset": feat_cols,
        "verified_true_steps": 1 if lofo_r2 > 0 and p < 0.05 else 0,
    }


def classify_enzyme_verdict(hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
    lofo = float(result.get("lofo_r2") or 0.0)
    p = float(result.get("label_shuffle_null_p") or 1.0)
    if result.get("verified_true_steps", 0) >= 1 and lofo >= 0.12 and p < 0.05:
        return "confirmed", f"EC-class LOFO R²={lofo:.3f}, shuffle p={p:.3f}", 0.86
    if lofo < 0.0 or p > 0.5:
        return "refuted", f"no cross-EC signal (LOFO R²={lofo:.3f})", 0.80
    return "inconclusive", f"weak cross-EC evidence (LOFO R²={lofo:.3f})", 0.55
