from pathlib import Path

import pytest

from propab.domain_adapters.mandrake_adapter import (
    MandrakeAdapter,
    MandrakeExperimentSpec,
    classify_mandrake_verdict,
    is_mandrake_campaign,
)
from propab.hypothesis_tree import HypothesisTree
from propab.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
MANDRAKE = ROOT / "mandrake-data"


@pytest.mark.skipif(not (MANDRAKE / "handcrafted_features.csv").is_file(), reason="no mandrake-data")
def test_lofo_experiment_runs():
    result = MandrakeAdapter(data_dir=MANDRAKE).run_experiment(
        MandrakeExperimentSpec(feature_subset=["t70_raw", "t75_raw"], methodology="LOFO")
    )
    assert result["n_samples"] >= 8
    assert result.get("permutation_p") is not None


@pytest.mark.skipif(not (MANDRAKE / "handcrafted_features.csv").is_file(), reason="no mandrake-data")
def test_mandrake_verification_tool():
    r = ToolRegistry().call("mandrake_verification", {"feature_subset": ["t55_raw", "t70_raw"]})
    assert r.success
    assert r.output.get("mean_r2") is not None


def test_tree_preserves_methodology():
    n = HypothesisTree().add_seeds([{"text": "x", "test_methodology": "plan", "feature_subset": ["t70_raw"]}])[0]
    assert n.test_methodology == "plan"
    assert n.feature_subset == ["t70_raw"]


def test_classify_null():
    result = {"mean_r2": -0.12, "family_baseline_r2": 0.29, "lofo_gap": 0.55, "permutation_p": 0.08}
    v, _, _ = classify_mandrake_verdict("Null: group identity fully explains", result)
    assert v == "confirmed"


def test_is_mandrake():
    assert is_mandrake_campaign(payload={"seed_source": "anomaly"})


def test_mandrake_evidence_parses_for_tree_diagnostics():
    from services.orchestrator.campaign_diagnostics import parse_evidence_obj
    from services.worker.sub_agent_loop import _build_mandrake_evidence
    from propab.research_quality import compute_evidence_hash, is_valid_evidence_for_hash

    output = {"mean_r2": 0.12, "family_baseline_r2": 0.29, "lofo_gap": 0.41, "permutation_p": 0.04}
    obj = _build_mandrake_evidence(
        output=output,
        verdict="confirmed",
        reason="LOFO=0.120 vs baseline 0.290",
        baseline={"value": 0.29},
    )
    evidence = f"evidence={__import__('json').dumps(obj)}; mandrake_verification"
    parsed = parse_evidence_obj(evidence)
    assert parsed.get("metric_value") == 0.12
    assert parsed.get("verified_true_steps") == 1
    assert is_valid_evidence_for_hash(parsed)
    assert compute_evidence_hash(parsed) is not None
