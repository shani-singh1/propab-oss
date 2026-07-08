"""C1 — one verdict authority + one confidence function.

Two regressions, both of the silent "val_accuracy class":

TASK 1 (split-brain): the worker writes its OWN verdict to the DB ``hypotheses``
row, then the orchestrator's ``_apply_result_diagnostics`` can DOWNGRADE the
effective verdict on the in-memory tree (control_calibration / metric_missing /
duplicate_evidence). Pre-C1 that downgrade was applied to the TREE ONLY and never
written back, so the DB verdict (which the paper + API read) could permanently say
"confirmed" while the tree said "inconclusive". After C1 the async caller mirrors
the corrected verdict/confidence to the DB, so DB == tree effective verdict.

TASK 2 (one confidence function): the worker's ``_compute_confidence`` and the core
pipeline's confidence used to be two separate implementations. They are now a single
canonical function (``propab.verdict_pipeline.compute_confidence``); the worker is a
thin adapter. These tests pin that both call sites return identical results.
"""
from __future__ import annotations

import asyncio
import json

from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.research_quality import compute_claim_dedup_key
from propab.verdict_pipeline import compute_confidence
from services.orchestrator.campaign_loop import (
    DiagnosticsOutcome,
    _apply_result_diagnostics,
    _persist_effective_verdict,
)
from services.worker.sub_agent_loop import _compute_confidence


# ── DB test double: capture every UPDATE hypotheses write ────────────────────

class _CapturingDB:
    def __init__(self, sink: list[dict]) -> None:
        self._sink = sink

    async def __aenter__(self) -> "_CapturingDB":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def execute(self, query: object, params: dict | None = None) -> None:
        self._sink.append({"query": str(query), "params": dict(params or {})})

    async def commit(self) -> None:
        return None


def _capturing_factory() -> tuple[object, list[dict]]:
    sink: list[dict] = []

    def factory() -> _CapturingDB:
        return _CapturingDB(sink)

    return factory, sink


def _make_node(tree: HypothesisTree, node_id: str, text: str) -> HypothesisNode:
    node = HypothesisNode(id=node_id, text=text, parent_id=None, depth=0)
    tree.nodes[node_id] = node
    tree.frontier.append(node_id)
    return node


def _valid_confirm_evidence() -> str:
    return json.dumps(
        {
            "verified_true_steps": 1,
            "verified_false_steps": 0,
            "n_metric_steps": 3,
            "metric_value": 0.94,
            "baseline_value": 0.90,
            "p_value": 0.001,
            "effect_size": 0.8,
            "verdict_reason": "significance replicated",
        }
    )


def _diag_then_persist(
    tree: HypothesisTree,
    node_id: str,
    worker_verdict: str,
    worker_confidence: float,
    evidence: str,
) -> tuple[DiagnosticsOutcome, bool, list[dict]]:
    """Mirror the live call site: update_node -> diagnostics -> persist."""
    tree.update_node(node_id, worker_verdict, worker_confidence, evidence)
    diag = _apply_result_diagnostics(tree, node_id, worker_verdict, worker_confidence, evidence)
    factory, sink = _capturing_factory()
    wrote = asyncio.run(
        _persist_effective_verdict(
            factory,
            f"hyp-{node_id}",
            worker_verdict=worker_verdict,
            worker_confidence=worker_confidence,
            diagnostics=diag,
        )
    )
    return diag, wrote, sink


# ── TASK 1: DB verdict must mirror the tree's effective verdict ──────────────

def test_control_downgrade_persists_to_db() -> None:
    """A control node the worker CONFIRMED is downgraded to inconclusive on the tree
    (control_calibration). The correction must be written back to the DB."""
    tree = HypothesisTree()
    _make_node(tree, "c1", "Null hypothesis: no effect beyond random noise")

    diag, wrote, sink = _diag_then_persist(tree, "c1", "confirmed", 0.9, _valid_confirm_evidence())

    # Tree is the source of truth: effective verdict was downgraded.
    assert tree.nodes["c1"].verdict == "inconclusive"
    assert diag.verdict == "inconclusive"
    assert diag.inconclusive_reason == "control_calibration"
    # Pre-C1 bug: no write happened and the DB kept the worker's "confirmed".
    assert wrote is True
    assert len(sink) == 1
    persisted = sink[0]["params"]
    assert persisted["verdict"] == "inconclusive"
    # The load-bearing invariant: DB verdict == tree effective verdict.
    assert persisted["verdict"] == tree.nodes["c1"].verdict
    assert "hypotheses" in sink[0]["query"].lower()


def test_metric_missing_downgrade_persists_to_db() -> None:
    """A discovery confirm with no metric-bearing evidence (ev_hash is None) is
    downgraded to inconclusive/metric_missing; the DB must mirror it."""
    tree = HypothesisTree()
    _make_node(tree, "d1", "Deeper networks improve validation accuracy on MNIST")

    # Empty evidence -> not valid for hashing -> ev_hash None -> metric_missing.
    diag, wrote, sink = _diag_then_persist(tree, "d1", "confirmed", 0.8, "")

    assert tree.nodes["d1"].verdict == "inconclusive"
    assert diag.verdict == "inconclusive"
    assert diag.inconclusive_reason == "metric_missing"
    assert wrote is True
    assert sink[0]["params"]["verdict"] == "inconclusive"
    assert sink[0]["params"]["verdict"] == tree.nodes["d1"].verdict
    assert "d1" not in tree.confirmed


def test_duplicate_evidence_downgrade_persists_verdict_and_confidence() -> None:
    """A duplicate discovery confirm is downgraded to inconclusive AND has its
    confidence capped at 0.4. Both corrections must reach the DB."""
    tree = HypothesisTree()
    text = "Attention dropout of 0.1 improves generalization on the transfer set"
    _make_node(tree, "d2", text)
    # Pre-register the claim so this node's own registration reads as a duplicate.
    assert tree.register_confirmed_claim(compute_claim_dedup_key(text)) is True

    diag, wrote, sink = _diag_then_persist(tree, "d2", "confirmed", 0.9, _valid_confirm_evidence())

    assert tree.nodes["d2"].verdict == "inconclusive"
    assert diag.verdict == "inconclusive"
    assert diag.inconclusive_reason == "duplicate_evidence"
    # Confidence capped at 0.4 by the downgrade; effective confidence tracks the node.
    assert tree.nodes["d2"].confidence == 0.4
    assert diag.confidence == 0.4
    assert wrote is True
    persisted = sink[0]["params"]
    assert persisted["verdict"] == "inconclusive"
    assert persisted["verdict"] == tree.nodes["d2"].verdict
    assert persisted["confidence"] == 0.4
    assert persisted["confidence"] == tree.nodes["d2"].confidence


def test_genuine_confirm_does_not_rewrite_db() -> None:
    """No downgrade -> effective verdict == worker verdict -> NO extra DB write
    (behaviour-preserving: the worker's row is already correct)."""
    tree = HypothesisTree()
    _make_node(tree, "d3", "Wider hidden layers raise validation accuracy on MNIST")

    diag, wrote, sink = _diag_then_persist(tree, "d3", "confirmed", 0.9, _valid_confirm_evidence())

    assert tree.nodes["d3"].verdict == "confirmed"
    assert diag.verdict == "confirmed"
    assert wrote is False
    assert sink == []


def test_persist_helper_no_write_when_identical() -> None:
    """Unit: _persist_effective_verdict is a no-op when nothing diverged."""
    factory, sink = _capturing_factory()
    diag = DiagnosticsOutcome(verdict="confirmed", confidence=0.9)
    wrote = asyncio.run(
        _persist_effective_verdict(
            factory, "hyp-x", worker_verdict="confirmed", worker_confidence=0.9, diagnostics=diag,
        )
    )
    assert wrote is False
    assert sink == []


def test_persist_helper_skips_empty_hypothesis_id() -> None:
    """A missing hypothesis_id must not attempt a write (guards a malformed result)."""
    factory, sink = _capturing_factory()
    diag = DiagnosticsOutcome(verdict="inconclusive", confidence=0.4)
    wrote = asyncio.run(
        _persist_effective_verdict(
            factory, "", worker_verdict="confirmed", worker_confidence=0.9, diagnostics=diag,
        )
    )
    assert wrote is False
    assert sink == []


# ── TASK 2: the single confidence function is identical at both call sites ───

def _confidence_cases() -> list[tuple[dict, str]]:
    return [
        ({"metric_value": 0.9, "baseline_value": 0.8, "p_value": 0.01,
          "effect_size": 0.5, "n_metric_steps": 3, "relevance_score": 0.4}, "confirmed"),
        ({"metric_value": 0.9, "baseline_value": None, "p_value": 0.2,
          "effect_size": 0.1, "n_metric_steps": 1, "relevance_score": 0.1}, "inconclusive"),
        ({"metric_value": None, "baseline_value": None, "p_value": None,
          "effect_size": None, "n_metric_steps": 0, "relevance_score": 0.0}, "inconclusive"),
        ({"verified_true_steps": 1, "metric_value": 0.9, "baseline_value": 0.8,
          "p_value": 0.2, "effect_size": 0.1, "n_metric_steps": 1, "relevance_score": 0.0}, "confirmed"),
        ({"verified_false_steps": 2, "metric_value": None, "n_metric_steps": 0,
          "relevance_score": 0.0}, "refuted"),
    ]


def _full_evidence(partial: dict) -> dict:
    """Fill the HypothesisEvidence keys the worker's TypedDict access path expects."""
    base = {
        "metric_value": None, "baseline_value": None, "delta": None, "delta_pct": None,
        "p_value": None, "effect_size": None, "confidence_interval": None,
        "n_tool_steps": 0, "n_metric_steps": 0, "relevance_score": 0.0,
        "verdict_reason": "", "verified_true_steps": 0, "verified_false_steps": 0,
    }
    base.update(partial)
    return base


def test_worker_and_core_confidence_are_identical() -> None:
    """The worker adapter and the canonical core function must return the SAME
    number for the same evidence + verdict — they are now one implementation."""
    for partial, verdict in _confidence_cases():
        ev = _full_evidence(partial)
        worker_val = _compute_confidence(ev, verdict)
        core_val = compute_confidence(dict(ev), verdict)
        assert worker_val == core_val, (partial, verdict, worker_val, core_val)


def test_confidence_respects_verified_steps_preamble() -> None:
    """Reconciled behaviour: a verified proof step + a matching verdict short-circuits
    to 0.95 (the core rule) through the worker adapter too."""
    ev = _full_evidence({"verified_true_steps": 1, "metric_value": 0.9, "baseline_value": 0.8})
    assert _compute_confidence(ev, "confirmed") == 0.95
    assert compute_confidence(dict(ev), "confirmed") == 0.95
    # Same evidence but a non-matching verdict falls through to the additive score.
    assert _compute_confidence(ev, "inconclusive") == compute_confidence(dict(ev), "inconclusive")
    assert _compute_confidence(ev, "inconclusive") < 0.95


def test_confidence_additive_score_matches_known_value() -> None:
    """Pin the additive-score path: metric(0.2)+baseline(0.2)+p<0.05(0.25)+
    |es|>0.2(0.15)+n>=3(0.10)+relevance>0.30(0.10) = 1.0 capped to 0.95."""
    ev = _full_evidence({
        "metric_value": 0.9, "baseline_value": 0.8, "p_value": 0.01,
        "effect_size": 0.5, "n_metric_steps": 3, "relevance_score": 0.4,
    })
    assert _compute_confidence(ev, "inconclusive") == 0.95
    assert compute_confidence(dict(ev), "inconclusive") == 0.95
