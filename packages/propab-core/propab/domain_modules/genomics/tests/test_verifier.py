"""Regression tests for the genomics verifier.

Guards the two latent bugs a live campaign exposed:
  1. target leakage — the target column was always also a predictor, so the
     leave-one-tissue-out R² came back trivially 1.0 for every hypothesis;
  2. a powerless null — the "label shuffle" permuted the tissue *split* variable
     instead of the target labels, so the null R² always matched the observed
     value (p ~= 1.0) and no real cross-tissue signal could ever be confirmed.
"""
from __future__ import annotations

import numpy as np
import pytest

from propab import config
from propab.domain_modules.genomics import verifier as V
from propab.domain_modules.genomics.adapter import KNOWN_FEATURES, GenomicsExperimentSpec


@pytest.fixture
def tmp_data(monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))


@pytest.mark.parametrize(
    "text",
    [
        "tissue specificity tau across housekeeping genes",
        "housekeeping constitutive cross-tissue expression",
        "cross-tissue expression conservation of a gene",
    ],
)
def test_target_never_leaks_into_features(text):
    spec = GenomicsExperimentSpec.from_hypothesis({"text": text})
    assert spec.target_column not in spec.feature_subset, (
        f"target {spec.target_column} leaked into features {spec.feature_subset}"
    )
    assert spec.feature_subset, "no predictors left after removing the target"
    assert all(f in KNOWN_FEATURES for f in spec.feature_subset)


def test_experiment_r2_not_trivially_one(tmp_data):
    # Before the leakage fix, every branch put the target in the feature set and
    # LOFO R^2 came back exactly 1.0 with a powerless (p=1.0) null.
    spec = GenomicsExperimentSpec.from_hypothesis({"text": "housekeeping genes cross-tissue"})
    res = V.run_genomics_experiment(spec)
    assert spec.target_column not in res["feature_subset"]
    assert res["lofo_r2"] < 0.999, f"degenerate leakage R^2={res['lofo_r2']}"
    # A degenerate metric of exactly 1.0 with a 1.0 null is the failure signature.
    assert not (res["lofo_r2"] >= 0.999 and res["label_shuffle_null_p"] >= 0.999)


def test_label_shuffle_null_permutes_target_not_split():
    # A genuine X->y signal must beat the shuffled-y null. The null must shuffle
    # y (destroying the relationship); the old tissue-split shuffle left X->y
    # intact so the null matched the observed value and had no power.
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 2))
    y = X[:, 0] * 1.5 + rng.normal(scale=0.1, size=n)  # strong real signal
    tissues = np.array([f"T{i % 5}" for i in range(n)])
    observed, nulls = V._target_label_shuffle_null(X, y, tissues, n_perm=30)
    p = float(np.mean(np.asarray(nulls) >= observed))
    assert observed > 0.5, observed
    assert p < 0.05, f"null has no power, p={p}"
    # Shuffling y destroys the signal, so null R^2 collapses toward ~0 — unlike
    # the old tissue-shuffle null, which stayed near the observed value.
    assert float(np.mean(nulls)) < 0.2, f"null R^2 not destroyed: {np.mean(nulls)}"


def test_tau_index_is_not_constant(tmp_data):
    # The Yanai tau must vary across genes. The old rank-sum formula collapsed it
    # to a constant 0.5 for every gene (a full ranking always sums to n(n+1)/2),
    # which made tau useless as a feature and degenerate (R^2=1.0) as a target.
    import numpy as np
    import pandas as pd

    from propab.domain_modules.genomics.adapter import compute_gene_features

    rng = np.random.default_rng(3)
    tissues = ["A", "B", "C", "D", "E"]
    rows = []
    for gi in range(50):
        # Half housekeeping (flat), half tissue-specific (one tissue dominates).
        peak = gi % len(tissues)
        for ti, t in enumerate(tissues):
            base = 5.0 if gi % 2 == 0 else (10.0 if ti == peak else 0.5)
            rows.append({"gene_id": f"G{gi}", "tissue": t, "expression": base + rng.normal(0, 0.05)})
    feats = compute_gene_features(pd.DataFrame(rows))
    tau = feats["tissue_specificity_tau"].to_numpy()
    assert float(np.var(tau)) > 1e-4, f"tau is constant: var={np.var(tau)}"
    assert len(np.unique(np.round(tau, 3))) > 1


def test_constant_target_is_refuted(tmp_data):
    # A degenerate (constant) target must report no signal, never a spurious 1.0.
    import numpy as np
    import pandas as pd

    from propab.domain_modules.genomics.adapter import GenomicsAdapter

    df = GenomicsAdapter().load_frame().copy()
    df["const_col"] = 0.5  # constant target
    # Build a spec pointing at the constant column and monkeypatch the frame.
    spec = GenomicsExperimentSpec(feature_subset=["expression_variance"], target_column="const_col")
    orig = GenomicsAdapter.load_frame
    try:
        GenomicsAdapter.load_frame = lambda self: df  # type: ignore[assignment]
        out = V.run_genomics_experiment(spec)
    finally:
        GenomicsAdapter.load_frame = orig  # type: ignore[assignment]
    assert out.get("degenerate_target") is True
    assert out["lofo_r2"] == 0.0 and out["verified_true_steps"] == 0


def test_pure_noise_target_is_not_confirmed():
    # No real X->y relationship: observed LOFO R^2 must not beat the null.
    rng = np.random.default_rng(1)
    n = 300
    X = rng.normal(size=(n, 2))
    y = rng.normal(size=n)  # independent of X
    tissues = np.array([f"T{i % 5}" for i in range(n)])
    observed, nulls = V._target_label_shuffle_null(X, y, tissues, n_perm=30)
    p = float(np.mean(np.asarray(nulls) >= observed))
    assert p > 0.05, f"noise wrongly looks significant, p={p}"
