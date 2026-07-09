"""Regression: statistical-domain label-shuffle nulls must permute the TARGET
(within group), not the split/group variable.

This is the genomics/enzyme bug class: shuffling the group/split leaves the X→y
relationship intact, so the null R² tracks the observed value and the test has no
power. A correct null permutes y (within group), which destroys X→y — so a planted
signal beats the null AND the null distribution collapses toward ~0.
"""
from __future__ import annotations

import numpy as np


def _planted(n=400, k=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = X[:, 0] * 1.5 + rng.normal(scale=0.1, size=n)  # strong real signal
    groups = np.array([f"g{i % k}" for i in range(n)])
    return X, y, groups


def _noise(n=400, k=4, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = rng.normal(size=n)  # independent of X
    groups = np.array([f"g{i % k}" for i in range(n)])
    return X, y, groups


def test_enzyme_null_permutes_target_and_has_power():
    from propab.domain_modules.enzyme_kinetics.verifier import _label_shuffle_null

    X, y, g = _planted()
    obs, nulls, p = _label_shuffle_null(X, y, g, n_perm=30)
    assert obs > 0.3, obs
    assert p < 0.05, p
    # The relationship was actually destroyed → null collapses (old group-shuffle
    # left it intact and null tracked obs).
    assert float(np.mean(nulls)) < 0.15, float(np.mean(nulls))


def test_enzyme_null_rejects_pure_noise():
    from propab.domain_modules.enzyme_kinetics.verifier import _label_shuffle_null

    X, y, g = _noise()
    _obs, _nulls, p = _label_shuffle_null(X, y, g, n_perm=30)
    assert p > 0.05, p


def test_mandrake_family_null_permutes_target_and_has_power():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from propab.domain_adapters.mandrake_adapter import _family_label_shuffle_null

    model = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    X, y, fam = _planted()
    obs, p_ge, nulls = _family_label_shuffle_null(X, y, fam, model, n_perm=30)
    assert obs > 0.3, obs
    assert p_ge < 0.05, p_ge
    assert float(np.mean(nulls)) < 0.2, float(np.mean(nulls))


def test_mandrake_family_null_rejects_pure_noise():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from propab.domain_adapters.mandrake_adapter import _family_label_shuffle_null

    model = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    X, y, fam = _noise()
    _obs, p_ge, _nulls = _family_label_shuffle_null(X, y, fam, model, n_perm=30)
    assert p_ge > 0.05, p_ge
