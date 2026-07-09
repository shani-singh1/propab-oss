"""C3a — the orchestrator is made VISIBLE via first-class reasoning events.

This is a purely-additive observability layer: the orchestrator emits events that
DESCRIBE decisions it already makes (per-result verdicts, hypothesis generation,
baseline, literature, finalize). Nothing here changes any verdict, expansion, or
control-flow decision.

These tests pin two things:
  1. The new ``EventType`` members exist with the enum's ``category.subcategory``
     string convention.
  2. The pure payload builders produce the expected structured, plain-language
     shapes for representative inputs, and the async emit helpers (a) emit the
     right event types/payloads through a recording emitter and (b) are fully
     failure-isolated — a raising emitter must NEVER propagate out of the loop.
"""
from __future__ import annotations

import asyncio

import pytest

from propab.hypothesis_tree import HypothesisNode
from propab.types import EventType
from services.orchestrator.campaign_loop import (
    DiagnosticsOutcome,
    _decision_event_payload,
    _emit_orchestrator_decision,
    _emit_orchestrator_generation,
    _emit_orchestrator_literature,
    _emit_orchestrator_reasoning,
    _hypothesis_node_kind,
    _hypothesis_written_payload,
    _null_p_from_evidence,
    _reasoning_event_payload,
    _safe_emit,
    _short_text,
    _verdict_action_and_why,
)


# ── test doubles ──────────────────────────────────────────────────────────────

class _RecordingEmitter:
    """Captures every emit(**kwargs) call."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    async def emit(self, **kwargs: object) -> None:
        self.events.append(dict(kwargs))
        return None

    def of_type(self, event_type: EventType) -> list[dict]:
        return [e for e in self.events if e.get("event_type") == event_type]


class _RaisingEmitter:
    """Fails on every emit — used to prove failure-isolation."""

    def __init__(self) -> None:
        self.calls = 0

    async def emit(self, **kwargs: object) -> None:
        self.calls += 1
        raise RuntimeError("emit boom")


CID = "11111111-2222-3333-4444-555555555555"


def _node(node_id: str, text: str, *, parent_id: str | None = None,
          expansion_type: str | None = None, generation: int = 0,
          inconclusive_reason: str | None = None) -> HypothesisNode:
    return HypothesisNode(
        id=node_id, text=text, parent_id=parent_id,
        depth=0 if parent_id is None else 1,
        expansion_type=expansion_type, generation=generation,
        inconclusive_reason=inconclusive_reason,
    )


# ── 1. EventType members exist with the right convention ──────────────────────

def test_orchestrator_event_types_exist() -> None:
    assert EventType.ORCHESTRATOR_DECISION.value == "orchestrator.decision"
    assert EventType.ORCHESTRATOR_REASONING.value == "orchestrator.reasoning"
    assert EventType.ORCHESTRATOR_HYPOTHESIS_WRITTEN.value == "orchestrator.hypothesis_written"
    assert EventType.ORCHESTRATOR_LITERATURE.value == "orchestrator.literature"


def test_orchestrator_event_values_follow_dotted_convention() -> None:
    for et in (
        EventType.ORCHESTRATOR_DECISION,
        EventType.ORCHESTRATOR_REASONING,
        EventType.ORCHESTRATOR_HYPOTHESIS_WRITTEN,
        EventType.ORCHESTRATOR_LITERATURE,
    ):
        assert et.value.startswith("orchestrator.")
        assert "." in et.value and et.value == et.value.lower()


# ── 2. small pure helpers ─────────────────────────────────────────────────────

def test_short_text_collapses_and_truncates() -> None:
    assert _short_text("  a\n  b   c ") == "a b c"
    long = "word " * 100
    out = _short_text(long, limit=20)
    assert len(out) <= 20
    assert out.endswith("…")
    assert _short_text(None) == ""


@pytest.mark.parametrize("evidence, expected", [
    ({"p_value": 0.01}, 0.01),
    ({"permutation_p": 0.2}, 0.2),
    ({"label_shuffle_permutation_p": 0.03}, 0.03),
    ({"label_shuffle_null_p": 0.7}, 0.7),
    ({"p_value": 0.01, "permutation_p": 0.9}, 0.01),  # priority order
    ({"no_p": 1}, None),
    ({"p_value": True}, None),  # bool is not a p-value
    ("not-a-dict", None),
])
def test_null_p_from_evidence(evidence, expected) -> None:
    assert _null_p_from_evidence(evidence) == expected


@pytest.mark.parametrize("node, expected_kind", [
    (_node("s", "seed", parent_id=None), "seed"),
    (_node("c", "child", parent_id="p", expansion_type="boundary"), "child"),
    (_node("c2", "child", parent_id="p", expansion_type="mechanistic"), "child"),
    (_node("l", "lateral", parent_id="p", expansion_type="alternative"), "lateral"),
    (_node("l2", "lateral", parent_id="p", expansion_type="generalization"), "lateral"),
    (_node("c3", "child no type", parent_id="p"), "child"),
])
def test_hypothesis_node_kind(node, expected_kind) -> None:
    assert _hypothesis_node_kind(node) == expected_kind


# ── verdict → plain language ─────────────────────────────────────────────────

def test_verdict_action_and_why_confirmed() -> None:
    action, why = _verdict_action_and_why("confirmed", "confirmed", None)
    assert action == "confirmed"
    assert "honesty gate" in why
    assert "downgraded" not in why  # no downgrade when worker agreed


def test_verdict_action_and_why_refuted() -> None:
    action, why = _verdict_action_and_why("refuted", "refuted", None)
    assert action == "refuted"
    assert "null" in why.lower()


def test_verdict_action_and_why_inconclusive_maps_reason() -> None:
    action, why = _verdict_action_and_why("inconclusive", "confirmed", "duplicate_evidence")
    assert action == "marked inconclusive"
    assert "duplicate evidence" in why
    # a worker→effective disagreement is surfaced as a downgrade note
    assert "downgraded from the worker's 'confirmed'" in why


def test_verdict_action_and_why_unknown_reason_falls_back() -> None:
    _, why = _verdict_action_and_why("inconclusive", "inconclusive", "some_new_reason")
    assert "insufficient evidence" in why


# ── 3. decision payload shape ─────────────────────────────────────────────────

def test_decision_event_payload_confirmed_shape() -> None:
    node = _node("n1", "Expression variance predicts cross-tissue specificity", parent_id="root")
    payload = _decision_event_payload(
        node=node,
        worker_verdict="confirmed",
        verdict="confirmed",
        effective_verdict="confirmed",
        inconclusive_reason=None,
        evidence_obj={"metric_value": 0.31, "p_value": 0.01},
        metric_name="lofo_r2",
        metric_value=0.31,
    )
    assert set(payload) == {
        "node_id", "hypothesis_text", "verdict", "effective_verdict",
        "worker_verdict", "downgraded", "action", "why",
        "metric_name", "metric_value", "null_p", "inconclusive_reason",
    }
    assert payload["node_id"] == "n1"
    assert payload["verdict"] == "confirmed"
    assert payload["effective_verdict"] == "confirmed"
    assert payload["action"] == "confirmed"
    assert payload["downgraded"] is False
    assert payload["metric_name"] == "lofo_r2"
    assert payload["metric_value"] == 0.31
    assert payload["null_p"] == 0.01
    assert payload["inconclusive_reason"] is None
    assert payload["hypothesis_text"]  # non-empty short text


def test_decision_event_payload_downgrade_from_confirmed() -> None:
    node = _node("n2", "dup claim", parent_id="root", inconclusive_reason="duplicate_evidence")
    payload = _decision_event_payload(
        node=node,
        worker_verdict="confirmed",
        verdict="confirmed",
        effective_verdict="inconclusive",
        inconclusive_reason="duplicate_evidence",
        evidence_obj={"metric_value": 0.31},
        metric_name="lofo_r2",
    )
    assert payload["downgraded"] is True
    assert payload["worker_verdict"] == "confirmed"
    assert payload["effective_verdict"] == "inconclusive"
    assert payload["action"] == "marked inconclusive"
    assert "duplicate evidence" in payload["why"]
    assert payload["inconclusive_reason"] == "duplicate_evidence"
    # metric_value falls back to evidence when not passed explicitly
    assert payload["metric_value"] == 0.31


def test_decision_event_payload_ignores_bool_metric() -> None:
    node = _node("n3", "x")
    payload = _decision_event_payload(
        node=node, worker_verdict="refuted", verdict="refuted",
        effective_verdict="refuted", inconclusive_reason=None,
        evidence_obj={"metric_value": True}, metric_name="acc",
    )
    assert payload["metric_value"] is None


# ── hypothesis-written + reasoning payload shapes ─────────────────────────────

def test_hypothesis_written_payload_shape() -> None:
    node = _node("h1", "A larger B_3 Sidon set exists at n=20", parent_id="p0",
                 expansion_type="alternative", generation=2)
    payload = _hypothesis_written_payload(node)
    assert payload == {
        "node_id": "h1",
        "parent_id": "p0",
        "text": "A larger B_3 Sidon set exists at n=20",
        "kind": "lateral",
        "expansion_type": "alternative",
        "generation": 2,
    }


def test_hypothesis_written_payload_kind_override() -> None:
    node = _node("h2", "seed text", parent_id=None)
    payload = _hypothesis_written_payload(node, kind="seed")
    assert payload["kind"] == "seed"
    assert payload["parent_id"] is None


def test_reasoning_event_payload_drops_none_and_truncates() -> None:
    payload = _reasoning_event_payload(
        decision="synthesize follow-ups",
        detail="  frontier  low ",
        count=3,
        generation=1,
        source=None,  # dropped
    )
    assert payload["decision"] == "synthesize follow-ups"
    assert payload["detail"] == "frontier low"
    assert payload["count"] == 3
    assert payload["generation"] == 1
    assert "source" not in payload


# ── 4. async emit helpers: recording + failure isolation ──────────────────────

def test_emit_decision_records_event() -> None:
    em = _RecordingEmitter()
    node = _node("n1", "hyp text", parent_id="root")
    diag = DiagnosticsOutcome(verdict="inconclusive", confidence=0.4,
                              inconclusive_reason="duplicate_evidence")
    asyncio.run(_emit_orchestrator_decision(
        em, campaign_id=CID, node=node, worker_verdict="confirmed",
        verdict="confirmed", diagnostics=diag,
        evidence_obj={"metric_value": 0.3, "p_value": 0.02},
        metric_name="lofo_r2", metric_value=0.3,
    ))
    evs = em.of_type(EventType.ORCHESTRATOR_DECISION)
    assert len(evs) == 1
    ev = evs[0]
    assert ev["session_id"] == CID
    assert ev["step"] == "orchestrator.decision"
    assert ev["hypothesis_id"]  # node row id attached
    p = ev["payload"]
    assert p["effective_verdict"] == "inconclusive"
    assert p["downgraded"] is True
    assert p["null_p"] == 0.02


def test_emit_generation_emits_reasoning_plus_one_per_node() -> None:
    em = _RecordingEmitter()
    nodes = [
        _node("a", "child a", parent_id="p", expansion_type="boundary"),
        _node("b", "lateral b", parent_id="p", expansion_type="alternative"),
    ]
    asyncio.run(_emit_orchestrator_generation(
        em, campaign_id=CID, generation=2, decision="synthesize follow-ups",
        detail="queue low", nodes=nodes, source="synthesis",
    ))
    reasoning = em.of_type(EventType.ORCHESTRATOR_REASONING)
    written = em.of_type(EventType.ORCHESTRATOR_HYPOTHESIS_WRITTEN)
    assert len(reasoning) == 1
    assert reasoning[0]["payload"]["count"] == 2
    assert reasoning[0]["payload"]["source"] == "synthesis"
    assert reasoning[0]["payload"]["generation"] == 2
    assert len(written) == 2
    assert {w["payload"]["node_id"] for w in written} == {"a", "b"}
    assert {w["payload"]["kind"] for w in written} == {"child", "lateral"}


def test_emit_generation_with_no_nodes_still_emits_reasoning() -> None:
    em = _RecordingEmitter()
    asyncio.run(_emit_orchestrator_generation(
        em, campaign_id=CID, generation=0, decision="seed hypotheses",
        detail="frontier empty", nodes=[],
    ))
    assert len(em.of_type(EventType.ORCHESTRATOR_REASONING)) == 1
    assert len(em.of_type(EventType.ORCHESTRATOR_HYPOTHESIS_WRITTEN)) == 0


def test_emit_literature_summarizes_prior() -> None:
    em = _RecordingEmitter()
    prior = {
        "established_facts": ["f1", "f2"],
        "open_gaps": ["g1"],
        "contested_claims": [],
        "key_papers": ["p1", "p2", "p3"],
        "evidence_status": "READY",
    }
    asyncio.run(_emit_orchestrator_literature(em, campaign_id=CID, prior_dict=prior))
    evs = em.of_type(EventType.ORCHESTRATOR_LITERATURE)
    assert len(evs) == 1
    p = evs[0]["payload"]
    assert p["established_facts"] == 2
    assert p["open_gaps"] == 1
    assert p["key_papers"] == 3
    assert p["evidence_status"] == "READY"
    assert "2 established fact" in p["detail"]


def test_emit_reasoning_records_event() -> None:
    em = _RecordingEmitter()
    asyncio.run(_emit_orchestrator_reasoning(
        em, campaign_id=CID, decision="baseline measured",
        detail="measured baseline acc = 0.9", metric_name="acc",
    ))
    evs = em.of_type(EventType.ORCHESTRATOR_REASONING)
    assert len(evs) == 1
    assert evs[0]["payload"]["metric_name"] == "acc"


# ── failure isolation: a raising emitter must NEVER propagate ─────────────────

def test_safe_emit_swallows_errors() -> None:
    em = _RaisingEmitter()
    asyncio.run(_safe_emit(em, session_id=CID, event_type=EventType.ORCHESTRATOR_DECISION,
                           step="x", payload={}))
    assert em.calls == 1  # it tried, and the error was swallowed


def test_all_emit_helpers_are_failure_isolated() -> None:
    em = _RaisingEmitter()
    node = _node("n1", "t", parent_id="root")
    diag = DiagnosticsOutcome(verdict="confirmed", confidence=0.9)

    # None of these may raise, even though the emitter always throws.
    asyncio.run(_emit_orchestrator_decision(
        em, campaign_id=CID, node=node, worker_verdict="confirmed", verdict="confirmed",
        diagnostics=diag, evidence_obj={}, metric_name="acc"))
    asyncio.run(_emit_orchestrator_generation(
        em, campaign_id=CID, generation=0, decision="seed hypotheses",
        detail="x", nodes=[node]))
    asyncio.run(_emit_orchestrator_reasoning(
        em, campaign_id=CID, decision="finalize campaign", detail="done"))
    asyncio.run(_emit_orchestrator_literature(
        em, campaign_id=CID, prior_dict={"established_facts": ["f"]}))

    assert em.calls >= 4  # every helper attempted at least one emit


def test_emit_generation_survives_a_bad_node() -> None:
    """A single malformed node must not abort the batch or raise."""
    em = _RecordingEmitter()

    class _BadNode:
        @property
        def id(self):  # noqa: D401 - raises on access
            raise ValueError("boom")

    good = _node("ok", "good node", parent_id="p")
    asyncio.run(_emit_orchestrator_generation(
        em, campaign_id=CID, generation=1, decision="synthesize follow-ups",
        detail="mixed", nodes=[_BadNode(), good],
    ))
    # reasoning still emitted; the one good node still produced its event.
    assert len(em.of_type(EventType.ORCHESTRATOR_REASONING)) == 1
    written = em.of_type(EventType.ORCHESTRATOR_HYPOTHESIS_WRITTEN)
    assert [w["payload"]["node_id"] for w in written] == ["ok"]
