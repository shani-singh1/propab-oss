"""resolve_model must bridge the build -> train handle mismatch.

Agents pass either a base build id or a ``<id>:trained`` handle to downstream tools.
Both should resolve, and profiling/counting must never silently see 0 params.
"""
from __future__ import annotations

import propab.tools.model_registry as mr


def _reset() -> None:
    mr._STORE.clear()
    mr._redis_client = None


def test_resolve_base_from_trained_handle() -> None:
    _reset()
    mr._STORE["m1"] = {"kind": "mlp", "param_count": 1234, "dims": [16, 8, 2], "activation": "relu"}
    # Trained entry lacks param_count (older writers) — resolver backfills from base.
    mr._STORE["m1:trained"] = {"kind": "mlp_trained", "base": "m1", "dims": [16, 8, 2]}

    info = mr.resolve_model("m1:trained")
    assert info is not None
    assert info["param_count"] == 1234


def test_resolve_trained_from_base_handle() -> None:
    _reset()
    mr._STORE["m2:trained"] = {"kind": "mlp_trained", "base": "m2", "dims": [4, 4, 2], "param_count": 42}

    # Caller passes the base id but only the trained variant exists.
    info = mr.resolve_model("m2")
    assert info is not None
    assert info["param_count"] == 42


def test_resolve_backfills_param_count_from_dims() -> None:
    _reset()
    # dims [10, 5, 2] -> (10*5+5) + (5*2+2) = 55 + 12 = 67
    mr._STORE["m3:trained"] = {"kind": "mlp_trained", "base": "m3", "dims": [10, 5, 2]}

    info = mr.resolve_model("m3:trained")
    assert info is not None
    assert info["param_count"] == 67


def test_resolve_unknown_returns_none() -> None:
    _reset()
    assert mr.resolve_model("does-not-exist") is None
    assert mr.resolve_model("") is None


def test_param_count_from_dims_helper() -> None:
    assert mr._param_count_from_dims([16, 8, 2]) == (16 * 8 + 8) + (8 * 2 + 2)
    assert mr._param_count_from_dims([]) == 0
    assert mr._param_count_from_dims([5]) == 0
