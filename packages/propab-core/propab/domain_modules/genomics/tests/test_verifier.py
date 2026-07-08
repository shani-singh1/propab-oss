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
