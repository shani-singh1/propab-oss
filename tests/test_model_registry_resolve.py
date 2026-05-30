"""resolve_model must bridge the build -> train handle mismatch.

Agents pass either a base build id or a ``<id>:trained`` handle to downstream tools.
Both should resolve, and profiling/counting must never silently see 0 params.
"""
from __future__ import annotations

import propab.tools.model_registry as mr

# These are pure in-process unit tests; disable the optional Redis slow-path so a
# live Redis (e.g. a running dev stack) can't leak stale "latest model" keys in.
mr._get_redis = lambda: None  # type: ignore[assignment]


def _reset() -> None:
    mr._STORE.clear()
    mr._redis_client = None
    mr._LAST_BUILT = None
    mr._LAST_TRAINED = None


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


def test_placeholder_id_resolves_to_latest_built() -> None:
    _reset()
    # LLM copied the literal "x"/"dummy" example id; fall back to the real build.
    mr.put_model("real-1", {"kind": "mlp", "param_count": 100, "dims": [8, 4, 2]})
    for placeholder in ("x", "dummy", "model_id", "auto", "mlp"):
        info = mr.resolve_model(placeholder)
        assert info is not None, placeholder
        assert info["param_count"] == 100


def test_unknown_id_resolves_to_latest() -> None:
    _reset()
    mr.put_model("real-2", {"kind": "mlp", "param_count": 55, "dims": [5, 5, 2]})
    # A wholly unknown handle still resolves to the most recent model.
    info = mr.resolve_model("totally-made-up-id")
    assert info is not None
    assert info["param_count"] == 55


def test_eval_placeholder_prefers_trained() -> None:
    _reset()
    mr.put_model("b1", {"kind": "mlp", "param_count": 100, "dims": [8, 4, 2]})
    mr.put_model("b1:trained", {"kind": "mlp_trained", "base": "b1", "dims": [8, 4, 2],
                                "param_count": 100, "val_losses": [0.5, 0.4], "final_val_loss": 0.4})
    # 'x:trained' is the literal evaluate_model example — prefer the trained handle.
    info = mr.resolve_model("x:trained", prefer_trained=True)
    assert info is not None
    assert info.get("kind") == "mlp_trained"
    assert info.get("val_losses") == [0.5, 0.4]


def test_empty_registry_returns_none_for_placeholder() -> None:
    _reset()
    # No models built yet → placeholder cannot resolve to anything.
    assert mr.resolve_model("x") is None
    assert mr.resolve_model("auto", prefer_trained=True) is None


def test_latest_built_updates_with_recency() -> None:
    _reset()
    mr.put_model("first", {"kind": "mlp", "param_count": 10, "dims": [4, 2]})
    mr.put_model("second", {"kind": "mlp", "param_count": 20, "dims": [6, 2]})
    # Most recent build wins for a placeholder.
    info = mr.resolve_model("dummy")
    assert info is not None
    assert info["param_count"] == 20
