from propab.tools.registry import ToolRegistry


def test_registry_discovers_core_tools() -> None:
    reg = ToolRegistry()
    names = {s["name"] for s in reg.get_all_specs()}
    assert "json_extract" in names
    assert "vector_dot" in names
    assert "category_counts" in names
    assert "numeric_summary" in names
    assert "statistical_significance" in names
    assert "run_experiment_grid" in names
    assert "compare_gradient_methods" in names
    assert "scaling_law_fit" in names
    assert "numerical_stability_test" in names
    assert "activation_statistics" in names
    assert "regularization_effect" in names
    assert "lr_range_test" in names
    assert "gradient_noise_scale" in names
    assert "reproduce_result" in names
    assert "plot_training_curves" in names
