from propab.tools.registry import (
    ToolEntry,
    ToolRegistry,
    audience_matches,
    normalize_audience,
)


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
    assert "compare_attention_variants" in names
    assert "hyperparameter_sweep" in names
    assert "compare_implementations" in names
    assert "convergence_analysis" in names
    assert "literature_baseline_compare" in names
    assert "load_curated_dataset" in names


# ---- Audience scoping (orchestrator / worker / both) ----------------------------

def _stub_entry(name: str, domain: str, audience: str = "both") -> ToolEntry:
    return ToolEntry(
        spec={"name": name, "domain": domain, "audience": audience},
        fn=lambda **_kw: None,
        domain=domain,
        audience=audience,
    )


def test_normalize_and_match_audience() -> None:
    # Absent / empty / unrecognised all degrade to "both" (never hidden).
    assert normalize_audience(None) == "both"
    assert normalize_audience("") == "both"
    assert normalize_audience("ORCHESTRATOR") == "orchestrator"
    assert normalize_audience("nonsense") == "both"
    # "both" is visible to any requested audience; a scoped tool only to its own.
    assert audience_matches("both", "orchestrator")
    assert audience_matches("both", "worker")
    assert audience_matches("orchestrator", "orchestrator")
    assert not audience_matches("worker", "orchestrator")
    assert not audience_matches("orchestrator", "worker")


def test_unscoped_tools_default_to_both_audience() -> None:
    reg = ToolRegistry()
    all_names = {s["name"] for s in reg.get_all_specs()}
    orch = {s["name"] for s in reg.get_for("orchestrator")}
    worker = {s["name"] for s in reg.get_for("worker")}
    # S1 added a few worker-scoped trusted primitives; everything else defaults to
    # "both". A worker sees {worker} + {both} == everything; the orchestrator sees
    # everything EXCEPT the worker-only tools.
    worker_only = {e.spec["name"] for e in reg._registry.values() if e.audience == "worker"}
    assert worker == all_names
    assert orch == all_names - worker_only
    # The S1 primitives are worker-scoped (workers call them; orchestrator does not).
    assert {"extremal_set_search", "certify_b3_record", "label_shuffle_null"} <= worker_only
    # The literature tool stays "both" (orchestrator literature stream is separate).
    assert "literature_baseline_compare" in orch
    assert "literature_baseline_compare" in worker


def test_get_for_filters_by_audience() -> None:
    reg = ToolRegistry()
    reg._registry["_t_orch"] = _stub_entry("_t_orch", "unit_test", "orchestrator")
    reg._registry["_t_worker"] = _stub_entry("_t_worker", "unit_test", "worker")
    reg._registry["_t_both"] = _stub_entry("_t_both", "unit_test", "both")

    orch = {s["name"] for s in reg.get_for("orchestrator")}
    assert "_t_orch" in orch and "_t_both" in orch
    assert "_t_worker" not in orch

    worker = {s["name"] for s in reg.get_for("worker")}
    assert "_t_worker" in worker and "_t_both" in worker
    assert "_t_orch" not in worker


def test_get_cluster_for_is_audience_filtered() -> None:
    reg = ToolRegistry()
    reg._registry["_c_orch"] = _stub_entry("_c_orch", "unit_test_cluster", "orchestrator")
    reg._registry["_c_worker"] = _stub_entry("_c_worker", "unit_test_cluster", "worker")
    reg._registry["_c_both"] = _stub_entry("_c_both", "unit_test_cluster", "both")

    # Unfiltered cluster returns everything in the domain.
    assert {s["name"] for s in reg.get_cluster("unit_test_cluster")} == {
        "_c_orch", "_c_worker", "_c_both",
    }
    # Audience-filtered cluster returns own-audience + both.
    assert {s["name"] for s in reg.get_cluster_for("unit_test_cluster", "orchestrator")} == {
        "_c_orch", "_c_both",
    }
    assert {s["name"] for s in reg.get_cluster_for("unit_test_cluster", "worker")} == {
        "_c_worker", "_c_both",
    }
