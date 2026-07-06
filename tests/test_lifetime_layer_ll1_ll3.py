"""Regression tests for lifetime-learning layer fixes LL1 and LL3.

LL1: ``theme_success_rates`` must fold in ``FailureRecord``s so a mostly-failing
theme scores well below 1.0 (previously it was structurally always 1.0, which
made the policy penalty/saturation branch dead code).

LL3: a single drifted/renamed field on a persisted record must NOT wipe the
whole lifetime store — ``from_dict`` field-filters, and ``load`` fails closed
(loud log + raise) rather than returning an empty graph a caller would then
``save`` over the good data.
"""
from __future__ import annotations

import json
import logging

import pytest

from propab import config
from propab.knowledge_graph import (
    Claim,
    FailureRecord,
    KnowledgeGraph,
    knowledge_store_path,
    new_id,
)
from propab.meta_science import CampaignObservation, MetaScienceLedger
from propab.policy_mutation import mutate_policy_params


# ── LL1 ───────────────────────────────────────────────────────────────────────


def _confirmed(theme: str, campaign_id: str = "c") -> Claim:
    return Claim(
        id=new_id("claim"), text="t", verdict="confirmed", theme=theme,
        confidence=0.8, campaign_id=campaign_id,
    )


def _failure(theme: str, campaign_id: str = "c", sig: str | None = None) -> FailureRecord:
    return FailureRecord(
        id=new_id("fail"), text=new_id("txt"), reason="refuted",
        failure_signature=sig, theme=theme, verdict="refuted",
        campaign_id=campaign_id,
    )


def test_theme_success_rate_counts_failures_not_always_one():
    """1 confirmed + 3 failures in theme X -> 0.25 (was structurally 1.0)."""
    g = KnowledgeGraph()
    g.add_claim(_confirmed("X"))
    for _ in range(3):
        g.add_failure(_failure("X"))
    rates = g.theme_success_rates()
    assert rates["X"] == 0.25


def test_theme_success_rate_all_confirmed_is_one():
    g = KnowledgeGraph()
    for _ in range(4):
        g.add_claim(_confirmed("Y"))
    rates = g.theme_success_rates()
    assert rates["Y"] == 1.0


def test_theme_counts_confirmed_and_failed():
    g = KnowledgeGraph()
    g.add_claim(_confirmed("X"))
    for _ in range(3):
        g.add_failure(_failure("X"))
    counts = g.theme_counts()
    assert counts["X"] == {"confirmed": 1, "failed": 3}


def test_bucket_filter_applies_to_confirmed_and_failed():
    g = KnowledgeGraph()
    # in-bucket: 1 confirmed + 3 failed -> 0.25
    g.add_claim(_confirmed("X", campaign_id="in"))
    for _ in range(3):
        g.add_failure(_failure("X", campaign_id="in"))
    # out-of-bucket noise that must be excluded from both numerator/denominator
    g.add_claim(_confirmed("X", campaign_id="out"))
    for _ in range(9):
        g.add_failure(_failure("X", campaign_id="out"))
    rates = g.theme_success_rates(campaign_ids={"in"})
    assert rates["X"] == 0.25


def test_policy_mutation_penalizes_low_rate_theme():
    """The previously-dead rate<0.15 penalty/saturation branch now fires."""
    g = KnowledgeGraph()
    # theme "bad": 1 confirmed + 9 failures -> rate 0.1 (<0.15), n=10 (>=5)
    g.add_claim(_confirmed("bad", campaign_id="run"))
    for _ in range(9):
        g.add_failure(_failure("bad", campaign_id="run"))
    # theme "good": all confirmed -> rate 1.0, gets a boost
    for _ in range(3):
        g.add_claim(_confirmed("good", campaign_id="run"))

    meta = MetaScienceLedger()
    meta.record(CampaignObservation(
        campaign_id="run", question="q", tested=13, confirmed=4, refuted=9,
        inconclusive=0, closure_ratio=0.3, theme_entropy=0.5,
        general_theme_fraction=0.0, compute_seconds=10800, policy_generation=1,
        knowledge_claims=4, knowledge_failures=9,
        budget_bucket="3h", domain_bucket="graphs",
    ))

    params = mutate_policy_params(
        g, meta, budget_bucket="3h", domain_bucket="graphs", parent=None,
    )
    assert "bad" in params["penalties"], params
    assert params["penalties"]["bad"] == 0.25
    assert "bad" in params["saturated_themes"]
    assert "good" in params["boosts"]
    # a healthy theme must never be penalized
    assert "good" not in params["penalties"]


# ── LL3 ───────────────────────────────────────────────────────────────────────


def test_from_dict_drops_unknown_field_and_keeps_others():
    """A drifted extra key on one claim must not raise; other records survive."""
    good_claim = _confirmed("X")
    drifted = _confirmed("Y").to_dict()
    drifted["obsolete_field"] = "drifted-value"  # renamed/removed in a later schema
    data = {
        "version": 1,
        "claims": {"drifted": drifted, "good": good_claim.to_dict()},
        "failures": {"f1": _failure("Z").to_dict()},
    }
    g = KnowledgeGraph.from_dict(data)  # must NOT raise
    assert "good" in g.claims
    assert "drifted" in g.claims
    assert g.claims["drifted"].theme == "Y"       # known fields preserved
    assert not hasattr(g.claims["drifted"], "obsolete_field")  # extra dropped
    assert "f1" in g.failures                      # other record types survive


def test_from_dict_drift_survives_all_record_types():
    data = {
        "version": 1,
        "claims": {"c": {**_confirmed("X").to_dict(), "junk": 1}},
        "mechanisms": {"m": {"id": "m", "claim_id": "c", "cause": "a",
                             "effect": "b", "conditions": "x", "junk": 1}},
        "failures": {"f": {**_failure("X").to_dict(), "junk": 1}},
        "theories": {"t": {"id": "t", "name": "n", "assumptions": [],
                           "mechanism_summary": "s", "predictions": [],
                           "failure_regions": [], "junk": 1}},
        "questions": {"q": {"id": "q", "text": "t", "junk": 1}},
    }
    g = KnowledgeGraph.from_dict(data)
    assert g.claims and g.mechanisms and g.failures and g.theories and g.questions


def test_load_drifted_file_does_not_wipe_and_loads_fields(monkeypatch, tmp_path):
    """A file with a drifted extra key loads cleanly (no wipe, fields kept)."""
    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))
    g = KnowledgeGraph()
    g.add_claim(_confirmed("X"))
    g.add_failure(_failure("X"))
    g.save()

    p = knowledge_store_path()
    raw = json.loads(p.read_text(encoding="utf-8"))
    # simulate schema drift: an extra key appears on the persisted claim
    only_claim = next(iter(raw["claims"]))
    raw["claims"][only_claim]["future_only_field"] = 123
    p.write_text(json.dumps(raw), encoding="utf-8")

    loaded = KnowledgeGraph.load()
    assert len(loaded.claims) == 1   # not wiped
    assert len(loaded.failures) == 1
    assert next(iter(loaded.claims.values())).theme == "X"


def test_load_corrupt_file_raises_and_does_not_return_empty(
    monkeypatch, tmp_path, caplog,
):
    """Genuine corruption of an existing store fails closed (raise + log),
    so a caller cannot save an empty graph over good data."""
    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))
    p = knowledge_store_path()
    p.write_text("{ this is not valid json", encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(json.JSONDecodeError):
            KnowledgeGraph.load()
    assert any("Refusing to load" in r.message for r in caplog.records)
    # the corrupt file is left intact for recovery (not overwritten)
    assert p.read_text(encoding="utf-8") == "{ this is not valid json"


def test_load_missing_file_returns_empty_first_run(monkeypatch, tmp_path):
    """No store yet -> empty graph is safe (nothing on disk to clobber)."""
    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))
    g = KnowledgeGraph.load()
    assert g.claims == {} and g.failures == {}


def test_ingest_would_not_clobber_on_corrupt_store(monkeypatch, tmp_path):
    """End-to-end LL3 invariant: a corrupt store aborts the load-then-save
    cycle in ingest_campaign instead of overwriting it with an empty graph."""
    from propab.campaign import ResearchCampaign
    from services.orchestrator.lifetime_knowledge import ingest_campaign

    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))
    p = knowledge_store_path()
    p.write_text("{ corrupt", encoding="utf-8")

    campaign = ResearchCampaign(
        id="00000000-0000-0000-0000-0000000000ab",
        question="Which network metrics predict contagion speed on graphs?",
        compute_budget_seconds=10800,
    )
    with pytest.raises(json.JSONDecodeError):
        ingest_campaign(campaign)
    # save() never ran against an empty graph -> corrupt file untouched
    assert p.read_text(encoding="utf-8") == "{ corrupt"
