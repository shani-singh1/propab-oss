"""evaluate_model / inspect_gradients must measure REAL trained models — or fail closed.

Regression guard for issue TOOL1/TOOL2: evaluate_model used to fabricate ``eval_losses``
as ``stored_final_val + torch.randn(n) * 2%`` (cosmetic jitter of a single number) and,
in its fallback, evaluate a RANDOM-initialized net on RANDOM labels. Because those numbers
are computed in-sandbox (``stat_input_provenance == "computed"`` → trusted), the W1b verdict
gate does NOT catch them, so a DL hypothesis could "confirm" on manufactured variance fed to
statistical_significance.

The fix: train_model persists the trained ``state_dict`` + a held-out eval split; evaluate_model
runs real bootstrap evaluation passes on those weights (genuine across-pass variance) or returns
``success=False`` with NO ``eval_losses``. inspect_gradients loads the persisted weights or fails
closed rather than reporting gradients of a random net.
"""
from __future__ import annotations

import pytest

# Skip cleanly if torch isn't installed on this worker (the tools themselves also fail-soft).
torch = pytest.importorskip("torch")

import propab.tools.model_registry as mr
from propab.tools.deep_learning.build_mlp import build_mlp
from propab.tools.deep_learning.evaluate_model import evaluate_model
from propab.tools.deep_learning.inspect_gradients import inspect_gradients
from propab.tools.deep_learning.train_model import train_model

# Pure in-process tests: disable the optional Redis slow-path so a live dev Redis
# can't leak stale "latest model" keys or persisted state between tests.
mr._get_redis = lambda: None  # type: ignore[assignment]


def _reset() -> None:
    mr._STORE.clear()
    mr._redis_client = None
    mr._LAST_BUILT = None
    mr._LAST_TRAINED = None


def _train_small() -> tuple[str, dict]:
    """Build + train a tiny synthetic classifier; return (trained_id, train_output)."""
    br = build_mlp(input_dim=8, hidden_dims=[16], output_dim=3, activation="relu")
    assert br.success, br.error
    mid = br.output["model_id"]
    tr = train_model(model_id=mid, task="classification", dataset="synthetic", n_steps=60)
    assert tr.success, tr.error
    return tr.output["trained_model_id"], tr.output


def test_train_model_persists_weights_and_eval_data() -> None:
    _reset()
    tid, _ = _train_small()
    info = mr.get_model(tid)
    assert info is not None
    assert info.get("kind") == "mlp_trained"
    # Real trained weights persisted as JSON-serializable nested lists.
    sd = info.get("state_dict")
    assert sd and all(isinstance(k, str) for k in sd)
    # Held-out eval split persisted for honest re-evaluation.
    ed = info.get("eval_data")
    assert ed and ed.get("x") and ed.get("y") is not None
    assert ed.get("loss_kind") == "cross_entropy"


def test_evaluate_model_returns_real_bootstrap_variance() -> None:
    _reset()
    tid, tr_out = _train_small()

    er = evaluate_model(model_id=tid, task="classification")
    assert er.success, er.error
    el = er.output["eval_losses"]
    assert len(el) >= 2
    # There is genuine across-pass variance (not a single repeated value).
    assert len(set(el)) > 1
    assert float(torch.tensor(el).std(unbiased=False)) > 0.0

    # --- The load-bearing regression assertion --------------------------------
    # The REAL eval_losses are exactly reproducible by running the *persisted trained
    # net* on bootstrap resamples of the *persisted held-out data*. The fabricated
    # ``final_val + randn*2%`` construction cannot reproduce them.
    info = mr.get_model(tid)
    dims = info["dims"]
    import torch.nn as nn

    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    net.load_state_dict({k: torch.tensor(v) for k, v in info["state_dict"].items()})
    net.eval()
    x = torch.tensor(info["eval_data"]["x"], dtype=torch.float32)
    y = torch.tensor(info["eval_data"]["y"], dtype=torch.long)
    n_eval = x.shape[0]
    gen = torch.Generator().manual_seed(abs(hash(str(tid) + "eval")) & 0x7FFFFFFF)
    fn = nn.CrossEntropyLoss()
    repro: list[float] = []
    with torch.no_grad():
        for _ in range(len(el)):
            idx = torch.randint(0, n_eval, (min(64, n_eval),), generator=gen)
            repro.append(round(float(fn(net(x[idx]), y[idx])), 6))
    assert el == repro, "eval_losses must be the real bootstrap losses over held-out data"

    # And they must NOT be the fabricated stored_final_val ± 2% jitter construction.
    fv = float(tr_out["final_val_loss"])
    fab_gen = torch.Generator().manual_seed(abs(hash(str(tid) + "eval")) & 0x7FFFFFFF)
    noise = torch.randn(len(el), generator=fab_gen) * max(0.002, fv * 0.02)
    fabricated = [round(float(fv + noise[i]), 6) for i in range(len(el))]
    assert el != fabricated, "eval_losses must not match the old jitter-of-one-number fabrication"


def test_evaluate_model_fails_closed_without_weights() -> None:
    _reset()
    # A trained-kind entry that predates weight persistence: has a loss curve but
    # NO state_dict / eval_data. Must fail closed and emit no eval_losses.
    mr.put_model(
        "legacy:trained",
        {
            "kind": "mlp_trained",
            "base": "legacy",
            "dims": [8, 16, 3],
            "activation": "relu",
            "val_losses": [0.9, 0.8, 0.7],
            "final_val_loss": 0.7,
        },
    )
    er = evaluate_model(model_id="legacy:trained", task="classification")
    assert er.success is False
    assert er.error is not None
    assert er.error.type == "no_trained_weights"
    # Nothing for statistical_significance to consume.
    assert er.output is None or "eval_losses" not in (er.output or {})


def test_evaluate_model_fails_closed_without_eval_data() -> None:
    _reset()
    # Weights present but no held-out eval split → still fail closed (no fabrication).
    mr.put_model(
        "noeval:trained",
        {
            "kind": "mlp_trained",
            "base": "noeval",
            "dims": [4, 4, 2],
            "activation": "relu",
            "state_dict": {
                "0.weight": [[0.1] * 4] * 4,
                "0.bias": [0.0] * 4,
                "2.weight": [[0.1] * 4] * 2,
                "2.bias": [0.0] * 2,
            },
            "final_val_loss": 0.5,
        },
    )
    er = evaluate_model(model_id="noeval:trained", task="classification")
    assert er.success is False
    assert er.error is not None
    assert er.error.type == "no_eval_data"
    assert er.output is None or "eval_losses" not in (er.output or {})


def test_inspect_gradients_fails_closed_without_weights() -> None:
    _reset()
    # An untrained build (kind == "mlp") has no persisted weights → must fail closed
    # rather than reporting gradients of a random-initialized network.
    mr.put_model("build-only", {"kind": "mlp", "dims": [8, 16, 3], "activation": "relu", "param_count": 1})
    ig = inspect_gradients(model_id="build-only")
    assert ig.success is False
    assert ig.error is not None
    assert ig.error.type == "no_trained_weights"


def test_inspect_gradients_succeeds_on_trained_weights() -> None:
    _reset()
    tid, _ = _train_small()
    ig = inspect_gradients(model_id=tid)
    assert ig.success, ig.error
    assert ig.output["total_norm"] >= 0.0
    assert len(ig.output["per_layer"]) > 0


def test_fabricated_jitter_code_path_is_gone() -> None:
    """Regression: the ``stored_final_val + torch.randn * 2%`` fabrication must not exist."""
    import inspect

    import propab.tools.deep_learning.evaluate_model as em

    src = inspect.getsource(em)
    # The old fabrication multiplied randn by ``stored_final_val * 0.02`` and summed onto
    # a single stored number. None of those tell-tale fragments may remain.
    assert "stored_final_val" not in src
    assert "* 0.02" not in src
    assert "val_losses_from_training" not in src
    # The misleading "use with statistical_significance" promise must not sit on a
    # fabricated path; the only remaining eval path is the real bootstrap one.
    assert "bootstrap_over_heldout" in src
