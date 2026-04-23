from propab.tools.deep_learning.hyperparameter_sweep import hyperparameter_sweep


def test_hyperparameter_sweep_grid() -> None:
    r = hyperparameter_sweep(
        model_type="mlp",
        search_space={"learning_rate": [1e-3, 1e-2], "dropout": [0.0, 0.2]},
        search_type="grid",
        n_trials=99,
        metric="val_loss",
    )
    assert r.success
    out = r.output or {}
    assert len(out["trials"]) == 4
    assert out["best_score"] == min(t["score"] for t in out["trials"])
    assert "learning_rate" in out["best_config"]


def test_hyperparameter_sweep_random() -> None:
    r = hyperparameter_sweep(
        model_type="transformer",
        search_space={"lr": [1e-4, 1e-3]},
        search_type="random",
        n_trials=5,
    )
    assert r.success
    assert len((r.output or {})["trials"]) == 5
