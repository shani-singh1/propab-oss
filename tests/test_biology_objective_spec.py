"""Biology domains must not be mis-scored as ML training campaigns.

Companion to test_domain_objective_spec.py (which covers math_combinatorics).
The biology-family domains — genomics (leave-one-tissue-out R²), enzyme_kinetics
(leave-one-EC-class-out R²), network_diffusion (held-out cross-family diffusion
correlation) and mandrake (leave-one-family-out R²) — are statistical / simulation
holdout domains. None of them trains an MLP, so each declares an ``objective_spec``
with ``is_ml=False`` and a non-ML metric label. This guarantees ``_is_ml_campaign``
short-circuits to False and core never measures a meaningless trained baseline for
a biology campaign (the 1ae74abd mis-scoring, generalized to biology).
"""
from __future__ import annotations

import pytest

from propab.campaign import BreakthroughCriteria, ResearchCampaign
from propab.domain_modules.registry import get_domain_plugin, resolve_domain_plugin

# Mirror of the orchestrator's ML-metric token guard: an objective_spec metric
# label containing any of these would make core classify the run as ML.
_ML_METRIC_TOKENS = (
    "accuracy", "loss", "error", "f1", "auc", "perplexity", "flops",
    "bleu", "mse", "rmse", "recall", "precision",
)

_BIOLOGY_DOMAINS = ("genomics", "enzyme_kinetics", "network_diffusion", "mandrake")

# Representative questions that route to each biology domain, paired with the
# default ML-shaped breakthrough metric a campaign would otherwise carry.
_BIOLOGY_CAMPAIGNS = {
    "genomics": "Does a cross-tissue gene expression pattern survive GTEx leave-tissue-out holdout? [domain_profile:genomics]",
    "enzyme_kinetics": "Does a kcat/Km signal survive leave-one-EC-class-out holdout across BRENDA enzyme families? [domain_profile:enzyme_kinetics]",
    "network_diffusion": "Does degree heterogeneity predict SIR outbreak size on a held-out network topology family?",
    "mandrake": "Does RT activity share one reverse transcriptase biophysical mechanism across evolutionary family groups, surviving leave-one-family-out holdout?",
}


@pytest.mark.parametrize("domain_id", _BIOLOGY_DOMAINS)
def test_biology_objective_spec_is_non_ml(domain_id: str):
    plugin = get_domain_plugin(domain_id)
    assert plugin is not None
    obj = plugin.objective_spec()
    # Biology domains have a genuine numeric holdout objective; each must declare it.
    assert obj is not None, f"{domain_id} must declare a non-ML objective_spec"
    assert obj["is_ml"] is False
    assert obj["direction"] in ("higher_is_better", "lower_is_better")
    # measured (statistical holdout), never a best-known extremal table.
    assert obj.get("baseline_kind") == "measured"
    metric = str(obj["metric_name"]).lower()
    assert metric  # non-empty label
    for tok in _ML_METRIC_TOKENS:
        assert tok not in metric, f"{domain_id} metric_name {metric!r} carries ML token {tok!r}"


@pytest.mark.parametrize("domain_id,question", list(_BIOLOGY_CAMPAIGNS.items()))
def test_biology_campaign_is_not_classified_ml(domain_id: str, question: str):
    from services.orchestrator.campaign_loop import _is_ml_campaign

    # The question routes to the intended biology domain...
    routed = resolve_domain_plugin(question=question)
    assert routed is not None and routed.domain_id == domain_id, (
        f"expected {domain_id}, routed to {getattr(routed, 'domain_id', None)}"
    )

    # ...and even with the default val_accuracy breakthrough metric, the domain's
    # is_ml=False objective_spec short-circuits _is_ml_campaign to False.
    campaign = ResearchCampaign(
        id=f"test-{domain_id}",
        question=question,
        breakthrough_criteria=BreakthroughCriteria(metric_name="val_accuracy"),
        compute_budget_seconds=60,
    )
    assert _is_ml_campaign(campaign) is False
