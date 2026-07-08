"""Leave-one-protein-family-out stability verification with a within-family shuffle null.

Honesty model (Type B, same family as ``network_diffusion`` / genomics / enzyme):

1. **Holdout statistic** — leave-one-family-out R²: fit a ridge model on every
   protein family (fold) but one and score R² on the held-out family. Measures
   whether a sequence->stability rule *generalizes to an unseen fold family*.
2. **Label-shuffle permutation null** — the ``tm_celsius`` target is shuffled
   *within each family* (preserving the per-family Tm marginal, destroying the
   feature->stability pairing) and the held-out R² recomputed. Confirmed only when
   observed beats the null: ``p = float(np.mean(np.asarray(nulls) >= observed)) < 0.05``.

The finding is "this sequence-property->thermostability relationship holds on a
held-out protein family and beats its within-family shuffle null" — never a raw
predicted Tm.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from propab.domain_modules.proteomics.adapter import ProteomicsAdapter, ProteomicsExperimentSpec


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


def _within_group_shuffle_null(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, n_perm: int = 40
) -> tuple[float, list[float], float]:
    observed = _lofo_r2(X, y, groups)
    rng = np.random.default_rng(42)
    unique = sorted(set(groups.tolist()))
    nulls: list[float] = []
    for _ in range(n_perm):
        ys = y.copy()
        for g in unique:
            idx = np.where(groups == g)[0]
            ys[idx] = y[rng.permutation(idx)]
        nulls.append(_lofo_r2(X, ys, groups))
    p = float(np.mean(np.asarray(nulls) >= observed)) if nulls else 1.0
    return observed, nulls, p


def run_proteomics_experiment(spec: ProteomicsExperimentSpec) -> dict[str, Any]:
    df = ProteomicsAdapter().load_frame()
    feat_cols = [c for c in spec.feature_subset if c in df.columns]
    if not feat_cols:
        raise ValueError(f"No usable proteomics features: {spec.feature_subset}")
    if spec.target_column not in df.columns:
        raise ValueError(f"Unknown target: {spec.target_column}")
    sub = df[[spec.group_column, spec.target_column, *feat_cols]].dropna()
    if len(sub) < 40:
        raise ValueError(f"Too few proteins: {len(sub)}")
    groups = sub[spec.group_column].astype(str).to_numpy()
    X = sub[feat_cols].to_numpy(dtype=float)
    y = sub[spec.target_column].to_numpy(dtype=float)
    lofo_r2, nulls, p = _within_group_shuffle_null(X, y, groups)
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
        "verified_false_steps": 0 if (lofo_r2 > 0 and p < 0.05) else 1,
    }


def classify_proteomics_verdict(hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
    _ = hypothesis_text
    # Explicit None checks, NOT ``or`` defaults: a legitimate perm p of 0.0 (the
    # null never beat the observed R²) is falsy, and ``x or 1.0`` would silently
    # turn the strongest confirmation into a spurious refutation.
    r2_raw = result.get("lofo_r2")
    p_raw = result.get("label_shuffle_null_p")
    r2 = float(r2_raw) if r2_raw is not None else 0.0
    p = float(p_raw) if p_raw is not None else 1.0
    steps = int(result.get("verified_true_steps") or 0)
    if steps >= 1 and r2 >= 0.12 and p < 0.05:
        return "confirmed", f"leave-family-out R²={r2:.3f}, within-family shuffle p={p:.3f}", 0.87
    if r2 < 0.02 or p > 0.5:
        return "refuted", f"no cross-family stability signal (LOFO R²={r2:.3f}, p={p:.3f})", 0.81
    return "inconclusive", f"weak cross-family stability evidence (LOFO R²={r2:.3f})", 0.55
