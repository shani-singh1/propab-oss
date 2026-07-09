"""S1 tool: label_shuffle_null is the CORRECT within-group target-shuffle LOFO null.

Honesty checks (fail-before/pass-after):
  * has POWER on a planted signal (observed beats the null; null_p < 0.05 and the
    null distribution collapses toward ~0 because X->y was actually destroyed);
  * REJECTS pure noise (null_p not significant);
  * is DEGENERATE-guarded (constant target -> no signal, never a spurious result).

These are exactly the properties a split-shuffle (wrong) null fails.
"""
from __future__ import annotations

import numpy as np

from propab.tools.statistics.label_shuffle_null import label_shuffle_null


def _planted(n: int = 400, k: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = X[:, 0] * 1.5 + rng.normal(scale=0.1, size=n)  # strong real signal
    groups = [f"g{i % k}" for i in range(n)]
    return X.tolist(), y.tolist(), groups


def _noise(n: int = 400, k: int = 4, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = rng.normal(size=n)  # independent of X
    groups = [f"g{i % k}" for i in range(n)]
    return X.tolist(), y.tolist(), groups


def test_null_has_power_on_planted_signal() -> None:
    X, y, g = _planted()
    r = label_shuffle_null(X=X, y=y, groups=g, n_perm=40)
    assert r.success, r.error
    o = r.output
    assert o["observed_lofo_r2"] > 0.3, o["observed_lofo_r2"]
    assert o["null_p"] < 0.05, o["null_p"]
    # Real relationship destroyed -> observed clears the 95th null percentile.
    assert o["observed_lofo_r2"] > o["null_p95"]
    assert o["significant"] is True
    assert o["degenerate_target"] is False


def test_null_rejects_pure_noise() -> None:
    X, y, g = _noise()
    r = label_shuffle_null(X=X, y=y, groups=g, n_perm=40)
    assert r.success, r.error
    assert r.output["null_p"] > 0.05, r.output["null_p"]
    assert r.output["significant"] is False


def test_null_degenerate_target_guard() -> None:
    X, _y, g = _planted()
    n = len(X)
    r = label_shuffle_null(X=X, y=[3.0] * n, groups=g, n_perm=40)
    assert r.success, r.error
    o = r.output
    assert o["degenerate_target"] is True
    assert o["observed_lofo_r2"] == 0.0
    assert o["null_p"] == 1.0
    assert o["significant"] is False


def test_null_requires_inputs() -> None:
    r = label_shuffle_null(X=None, y=None, groups=None)
    assert not r.success
    assert r.error.type == "validation_error"
