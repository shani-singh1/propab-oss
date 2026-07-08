"""Root-cause-A fix (campaign 1ae74abd post-mortem): a computational/verification
domain must NOT be scored against a trained ML baseline.

Before the fix, launching math_combinatorics with the default breakthrough metric
`val_accuracy` made the orchestrator classify it as an ML campaign, measure a
meaningless MLP baseline (0.875), and leave best_metric=0.0 for the whole run — no
Sidon/cap-set construction could ever register a record. The domain now declares an
`objective_spec()` with `is_ml=False`, and both campaign creation and
`_is_ml_campaign()` honor it.
"""
from __future__ import annotations

from propab.campaign import BreakthroughCriteria, ResearchCampaign
from propab.domain_modules.registry import get_domain_plugin, resolve_domain_plugin


def test_base_default_objective_spec_is_none():
    # The BASE default is None (a domain with no numeric extremal objective falls
    # back to request criteria). Asserted against a bare subclass rather than a real
    # domain, since most verification domains now override it (math with is_ml=False;
    # biology + graph_invariants with is_ml=False — see their dedicated tests).
    from propab.domain_modules.base import DomainPlugin

    class _Bare(DomainPlugin):
        domain_id = "bare_objective_test"

        def available_features(self) -> list[str]:
            return []

    assert _Bare().objective_spec() is None


def test_math_objective_spec_is_non_ml_deterministic():
    plugin = get_domain_plugin("math_combinatorics")
    assert plugin is not None
    obj = plugin.objective_spec()
    assert obj is not None
    assert obj["is_ml"] is False
    # metric label must not carry an ML token or core mis-classifies the run as ML
    assert "accuracy" not in obj["metric_name"].lower()
    assert "loss" not in obj["metric_name"].lower()
    assert obj["direction"] == "higher_is_better"


def test_math_question_routes_to_math_plugin():
    plugin = resolve_domain_plugin(
        question="Maximize Sidon set density under greedy search "
        "[domain_profile:math_combinatorics]"
    )
    assert plugin is not None and plugin.domain_id == "math_combinatorics"


def test_is_ml_campaign_false_for_math_even_with_val_accuracy_default():
    from services.orchestrator.campaign_loop import _is_ml_campaign

    campaign = ResearchCampaign(
        id="test-math",
        question="Maximize Sidon set density [domain_profile:math_combinatorics]",
        breakthrough_criteria=BreakthroughCriteria(metric_name="val_accuracy"),
        compute_budget_seconds=60,
    )
    # The explicit domain objective (is_ml=False) overrides the "accuracy" token.
    assert _is_ml_campaign(campaign) is False


def test_is_ml_campaign_true_for_genuine_ml_question():
    from services.orchestrator.campaign_loop import _is_ml_campaign

    campaign = ResearchCampaign(
        id="test-ml",
        question="Find the optimal MLP architecture for MNIST classification",
        breakthrough_criteria=BreakthroughCriteria(metric_name="val_accuracy"),
        compute_budget_seconds=60,
    )
    assert _is_ml_campaign(campaign) is True


# --- coding_theory (deterministic best-known-beating domain) ----------------

def test_coding_theory_objective_spec_is_non_ml_deterministic():
    plugin = get_domain_plugin("coding_theory")
    assert plugin is not None
    obj = plugin.objective_spec()
    assert obj is not None
    assert obj["is_ml"] is False
    # metric label must match what coding_theory/verifier.py emits and carry no ML token
    assert obj["metric_name"] == "code_minimum_distance"
    assert "accuracy" not in obj["metric_name"].lower()
    assert "loss" not in obj["metric_name"].lower()
    assert obj["direction"] == "higher_is_better"


def test_coding_theory_question_routes_to_coding_theory_plugin():
    plugin = resolve_domain_plugin(
        question="Construct a binary linear code beating the best-known minimum "
        "distance [domain_profile:coding_theory]"
    )
    assert plugin is not None and plugin.domain_id == "coding_theory"


def test_is_ml_campaign_false_for_coding_theory_even_with_val_accuracy_default():
    from services.orchestrator.campaign_loop import _is_ml_campaign

    campaign = ResearchCampaign(
        id="test-coding",
        question="Construct a binary linear code with improved minimum distance "
        "[domain_profile:coding_theory]",
        breakthrough_criteria=BreakthroughCriteria(metric_name="val_accuracy"),
        compute_budget_seconds=60,
    )
    # The explicit domain objective (is_ml=False) overrides the "accuracy" token.
    assert _is_ml_campaign(campaign) is False


# --- materials (leave-one-crystal-system-out holdout domain) ----------------

def test_materials_objective_spec_is_non_ml_holdout():
    plugin = get_domain_plugin("materials")
    assert plugin is not None
    obj = plugin.objective_spec()
    assert obj is not None
    assert obj["is_ml"] is False
    # metric label must match materials_adapter.py's emitted key (a holdout R², not accuracy)
    assert obj["metric_name"] == "lofo_r2"
    assert "accuracy" not in obj["metric_name"].lower()
    assert "loss" not in obj["metric_name"].lower()
    assert obj["direction"] == "higher_is_better"


def test_materials_question_routes_to_materials_plugin():
    plugin = resolve_domain_plugin(
        question="Does the descriptor-dielectric relationship survive "
        "leave-one-crystal-system-out holdout? [domain_profile:materials]"
    )
    assert plugin is not None and plugin.domain_id == "materials"


def test_is_ml_campaign_false_for_materials_even_with_val_accuracy_default():
    from services.orchestrator.campaign_loop import _is_ml_campaign

    campaign = ResearchCampaign(
        id="test-materials",
        question="Predict matbench dielectric constant across crystal systems "
        "[domain_profile:materials]",
        breakthrough_criteria=BreakthroughCriteria(metric_name="val_accuracy"),
        compute_budget_seconds=60,
    )
    # The explicit domain objective (is_ml=False) overrides the "accuracy" token.
    assert _is_ml_campaign(campaign) is False
