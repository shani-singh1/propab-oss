from propab.tools.ml_research.plot_training_curves import plot_training_curves


def test_plot_training_curves_returns_series() -> None:
    r = plot_training_curves(
        [
            {"label": "a", "steps": [0, 1, 2], "values": [1.0, 0.5, 0.25]},
        ],
        title="T",
        smoothing=0.3,
    )
    assert r.success
    assert "series" in r.output["plot_data"]
