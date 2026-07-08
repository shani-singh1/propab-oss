"""Cross-tissue leave-one-tissue-out verification for genomics hypotheses."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from propab.domain_modules.genomics.adapter import GenomicsAdapter, GenomicsExperimentSpec


def _leave_one_tissue_out_r2(
    X: np.ndarray,
    y: np.ndarray,
    tissues: np.ndarray,
) -> float:
    unique = sorted(set(tissues.tolist()))
    if len(unique) < 3:
        return 0.0
    scores: list[float] = []
    model = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    for held in unique:
        tr = tissues != held
        te = tissues == held
        if tr.sum() < 5 or te.sum() < 3:
            continue
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        scores.append(float(r2_score(y[te], pred)))
    return float(np.mean(scores)) if scores else 0.0


def _target_label_shuffle_null(
    X: np.ndarray,
    y: np.ndarray,
    tissues: np.ndarray,
    *,
    n_perm: int = 50,
) -> tuple[float, list[float]]:
    """Permutation null that shuffles the TARGET labels (``y``) across genes.

    The observed statistic is the leave-one-tissue-out R² of X→y. To ask whether
    that predictive relationship is real we must break it — permuting ``y``
    destroys the X→y correspondence while preserving the LOFO-by-tissue split
    structure. Shuffling ``tissues`` instead (the old behaviour) only re-randomised
    the train/test split and left X→y fully intact, so the null R² always matched
    the observed value and the test had no statistical power.
    """
    observed = _leave_one_tissue_out_r2(X, y, tissues)
    nulls: list[float] = []
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        y_shuffled = y.copy()
        rng.shuffle(y_shuffled)
        nulls.append(_leave_one_tissue_out_r2(X, y_shuffled, tissues))
    return observed, nulls


def run_genomics_experiment(spec: GenomicsExperimentSpec) -> dict[str, Any]:
    raw = GenomicsAdapter().load_frame()
    pivot = raw.pivot_table(index="gene_id", columns="tissue", values="expression", aggfunc="mean")
    dominant = pivot.idxmax(axis=1)
    gene_df = raw.groupby("gene_id", as_index=False).first()
    # Defence in depth against target leakage (see GenomicsExperimentSpec): the
    # target must never also be a predictor or LOFO R² is trivially 1.0.
    feat_cols = [c for c in spec.feature_subset if c in gene_df.columns and c != spec.target_column]
    if not feat_cols:
        raise ValueError(f"No usable genomics features: {spec.feature_subset}")
    if spec.target_column not in gene_df.columns:
        raise ValueError(f"Unknown target column: {spec.target_column}")
    gene_df = gene_df[["gene_id", spec.target_column, *feat_cols]].dropna()
    if len(gene_df) < 30:
        raise ValueError(f"Too few genes: {len(gene_df)}")
    tissues = gene_df["gene_id"].map(dominant).fillna("Unknown").astype(str).to_numpy()
    X = gene_df[feat_cols].to_numpy(dtype=float)
    y = gene_df[spec.target_column].to_numpy(dtype=float)
    if float(np.nanvar(y)) < 1e-12:
        # A (near-)constant target carries no learnable signal. sklearn's r2_score
        # degenerates to 1.0 when a constant is "predicted", and permuting a
        # constant y leaves it unchanged, so the shuffle null is also 1.0. Report
        # no signal outright so a broken/degenerate feature can never masquerade
        # as a perfect cross-tissue result.
        return {
            "lofo_r2": 0.0,
            "label_shuffle_null_p95": 1.0,
            "label_shuffle_null_p": 1.0,
            "verification_method": "leave_tissue_out",
            "n_genes": len(gene_df),
            "n_features": len(feat_cols),
            "feature_subset": feat_cols,
            "verified_true_steps": 0,
            "degenerate_target": True,
        }
    lofo_r2, nulls = _target_label_shuffle_null(X, y, tissues)
    label_shuffle_p95 = float(np.percentile(nulls, 95)) if nulls else 1.0
    label_shuffle_null_p = float(np.mean(np.asarray(nulls) >= lofo_r2)) if nulls else 1.0
    return {
        "lofo_r2": lofo_r2,
        "label_shuffle_null_p95": label_shuffle_p95,
        "label_shuffle_null_p": label_shuffle_null_p,
        "verification_method": "leave_tissue_out",
        "n_genes": len(gene_df),
        "n_features": len(feat_cols),
        "feature_subset": feat_cols,
        "verified_true_steps": 1 if lofo_r2 > 0 and label_shuffle_null_p < 0.05 else 0,
    }


def classify_genomics_verdict(hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
    lofo = float(result.get("lofo_r2") or 0.0)
    p = float(result.get("label_shuffle_null_p") or 1.0)
    steps = int(result.get("verified_true_steps") or 0)
    if steps >= 1 and lofo >= 0.15 and p < 0.05:
        return "confirmed", f"LOFO R²={lofo:.3f}, tissue-shuffle p={p:.3f}", 0.88
    if lofo < 0.05 or p > 0.5:
        return "refuted", f"no cross-tissue signal (LOFO R²={lofo:.3f}, p={p:.3f})", 0.82
    return "inconclusive", f"weak cross-tissue evidence (LOFO R²={lofo:.3f})", 0.55
