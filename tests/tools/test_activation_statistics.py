from propab.tools.deep_learning.activation_statistics import activation_statistics


def test_activation_statistics_runs() -> None:
    r = activation_statistics("test-model", n_batches=3)
    assert r.success
    assert len(r.output["by_layer"]) >= 1
    assert "dead_neuron_pct" in r.output
