"""Regression tests for two further latent bugs in the campaign core.

Both are the "val_accuracy class": silent, semantically-corrupting, invisible to the
existing suite. Each fails against the pre-fix code and passes after the fix.

1. HypothesisTree.from_dict rehydrates ``_used_evidence_hashes`` but NOT
   ``_used_confirmed_claim_keys``. After a checkpoint resume the confirmed-claim dedup
   gate (``register_confirmed_claim``, used by ``_apply_result_diagnostics``) therefore
   no-ops, so a re-confirmed DUPLICATE claim is accepted as a fresh confirmed finding —
   silently inflating the confirmed count and the replication proxy.

2. ``measure_baseline`` fell back to reading ``val_accuracy`` from the worker result
   whenever the DECLARED metric was absent — even for a non-accuracy metric like
   ``val_loss``. That substitutes an accuracy value (~0.9) as the loss baseline, so a
   ``lower_is_better`` campaign then treats any real (small) loss as a massive win and
   hands out false breakthroughs. This violates the O1 domain-general extraction
   contract (never substitute a differently-named metric).
"""
from __future__ import annotations

import asyncio

from propab.campaign import BreakthroughCriteria, ResearchCampaign
from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.research_quality import compute_claim_dedup_key


# ── Bug 1: confirmed-claim dedup set must survive a tree roundtrip ────────────

def test_confirmed_claim_dedup_survives_tree_roundtrip() -> None:
    tree = HypothesisTree()
    node = HypothesisNode(
        id="n1",
        text="Greedy Sidon ratio below 0.90 at n=1000",
        parent_id=None,
        depth=0,
        verdict="confirmed",
    )
    tree.nodes["n1"] = node
    tree.confirmed.append("n1")
    key = compute_claim_dedup_key(node.text)
    # Live registration (as _apply_result_diagnostics would do on first confirm).
    assert tree.register_confirmed_claim(key) is True

    # Checkpoint -> resume.
    reloaded = HypothesisTree.from_dict(tree.to_dict())

    # The SAME claim must still be recognized as a duplicate after reload.
    # Before the fix, the dedup set was empty on reload → this returned True,
    # letting a re-confirmed duplicate inflate the confirmed count.
    assert reloaded.register_confirmed_claim(key) is False
    # A genuinely different claim is still admitted.
    other = compute_claim_dedup_key("Greedy Sidon ratio below 0.80 at n=2000")
    assert reloaded.register_confirmed_claim(other) is True


# ── Bug 2: a non-accuracy baseline must not borrow val_accuracy ──────────────

class _FakeDB:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, *a, **k):
        return None

    async def commit(self):
        return None


def _fake_session_factory():
    return _FakeDB()


class _FakeEmitter:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def emit(self, **kwargs) -> None:
        self.events.append(kwargs)


class _FakeAccuracyOnlyResult:
    """Worker produced ONLY an accuracy metric (no val_loss)."""

    def get(self, timeout=None):  # noqa: ARG002
        return {"val_accuracy": 0.9}


class _FakeAccuracyOnlyTask:
    def delay(self, payload):  # noqa: ARG002
        return _FakeAccuracyOnlyResult()


def test_loss_baseline_does_not_borrow_val_accuracy(monkeypatch) -> None:
    from services.orchestrator import campaign_loop as cl
    import propab.domain_modules.registry as reg

    # Force the plugin-less path so the keyword heuristic classifies this as ML
    # ("loss" is an ML metric token) — isolates the metric-substitution bug.
    monkeypatch.setattr(reg, "resolve_domain_plugin", lambda **_kw: None)
    monkeypatch.setattr(cl.settings, "campaign_baseline_mode", "fast_tool", raising=False)
    monkeypatch.setattr(cl, "run_sub_agent_task", _FakeAccuracyOnlyTask(), raising=True)

    campaign = ResearchCampaign(
        id="00000000-0000-0000-0000-0000000000c6",
        question="Minimize validation loss training an MLP on synthetic data",
        breakthrough_criteria=BreakthroughCriteria(
            metric_name="val_loss", direction="lower_is_better"
        ),
        compute_budget_seconds=60,
    )
    assert cl._is_ml_campaign(campaign) is True

    measured = asyncio.run(
        cl.measure_baseline(campaign, object(), _FakeEmitter(), _fake_session_factory)
    )
    # Before the fix: 0.9 (the accuracy value borrowed as the loss baseline).
    # After the fix: None (fail closed — the declared val_loss was never produced).
    assert measured is None


def test_accuracy_baseline_fallback_still_resolves(monkeypatch) -> None:
    """The accuracy-family path is unaffected: a val_accuracy campaign still reads it."""
    from services.orchestrator import campaign_loop as cl
    import propab.domain_modules.registry as reg

    monkeypatch.setattr(reg, "resolve_domain_plugin", lambda **_kw: None)
    monkeypatch.setattr(cl.settings, "campaign_baseline_mode", "fast_tool", raising=False)
    monkeypatch.setattr(cl, "run_sub_agent_task", _FakeAccuracyOnlyTask(), raising=True)

    campaign = ResearchCampaign(
        id="00000000-0000-0000-0000-0000000000c7",
        question="Find the optimal MLP architecture for MNIST classification",
        breakthrough_criteria=BreakthroughCriteria(metric_name="val_accuracy"),
        compute_budget_seconds=60,
    )
    assert cl._is_ml_campaign(campaign) is True
    measured = asyncio.run(
        cl.measure_baseline(campaign, object(), _FakeEmitter(), _fake_session_factory)
    )
    assert measured == 0.9
