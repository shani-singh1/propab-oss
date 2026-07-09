"""C3b — the orchestrator's LLM REASONING step decides STRATEGY (what to test next).

The reasoning step chooses one strategic action for an ALREADY-judged node —
``deepen`` / ``retune`` / ``spawn_related`` / ``drop`` — with full tree context. It
NEVER decides the honesty verdict (that stays deterministic, C2) and it is inert
unless ``orchestrator_reasoning_enabled`` is True.

These tests pin:
  1. The config flag exists and defaults to False.
  2. Flag OFF preserves current behavior: the mechanical Tier-2 synthesis path is
     reached only when the flag is off; the flag ON short-circuits it (so reasoning
     drives expansion instead), proving the two never double-expand.
  3. ``tree_context_for_reasoning`` builds the right context (pure).
  4. ``parse_reasoning_decision`` handles good + malformed LLM output (pure).
  5. Each action maps to the right tree mutation via ``apply_reasoning_decision``
     (pure), driven by scripted decisions.
  6. ``orchestrator_reason_next`` + ``_run_orchestrator_reasoning_step`` with a FAKE
     LLM returning scripted decisions: right events, right mutations, failure-isolated.
  7. The ``attempts`` field round-trips through serialization.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from propab.config import Settings, settings
from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.types import EventType
from services.orchestrator.orchestrator_reasoning import (
    ACTION_DEEPEN,
    ACTION_DROP,
    ACTION_RETUNE,
    ACTION_SPAWN_RELATED,
    ReasoningDecision,
    apply_reasoning_decision,
    build_reasoning_prompt,
    orchestrator_reason_next,
    parse_reasoning_decision,
    tree_context_for_reasoning,
)


# ── test doubles ──────────────────────────────────────────────────────────────

class _RecordingEmitter:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def emit(self, **kwargs: object) -> None:
        self.events.append(dict(kwargs))

    def of_type(self, event_type: EventType) -> list[dict]:
        return [e for e in self.events if e.get("event_type") == event_type]


class _RaisingEmitter:
    def __init__(self) -> None:
        self.calls = 0

    async def emit(self, **kwargs: object) -> None:
        self.calls += 1
        raise RuntimeError("emit boom")


class _ScriptedLLM:
    """Returns a fixed raw string for every call; records the prompts it saw."""

    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.prompts: list[str] = []

    async def call(self, *, prompt: str, purpose: str, session_id: str,
                   hypothesis_id: str | None = None) -> str:
        self.prompts.append(prompt)
        return self.raw


class _BoomLLM:
    async def call(self, **kwargs: object) -> str:
        raise RuntimeError("llm boom")


CID = "11111111-2222-3333-4444-555555555555"


def _tree_with_parent_child() -> tuple[HypothesisTree, HypothesisNode, HypothesisNode]:
    """Parent (confirmed) → target node (inconclusive), plus one sibling."""
    tree = HypothesisTree()
    parent = HypothesisNode(id="p", text="Variance predicts specificity", parent_id=None,
                            depth=0, verdict="confirmed", confidence=0.9,
                            primary_theme="variance")
    node = HypothesisNode(id="n", text="Effect holds in tissue A", parent_id="p",
                          depth=1, verdict="inconclusive", confidence=0.4,
                          evidence_summary="metric=0.1; null_p=0.4")
    sibling = HypothesisNode(id="s", text="Effect holds in tissue B", parent_id="p",
                             depth=1, verdict="refuted", confidence=0.2)
    tree.nodes = {"p": parent, "n": node, "s": sibling}
    parent.children = ["n", "s"]
    tree.confirmed = ["p"]
    tree.frontier = []
    return tree, parent, node


def _campaign_with_tree(tree: HypothesisTree) -> SimpleNamespace:
    return SimpleNamespace(id=CID, question="Does expression variance predict tissue specificity?",
                           hypothesis_tree=tree)


# ── 1. config flag ────────────────────────────────────────────────────────────

def test_reasoning_flag_defaults_false() -> None:
    assert Settings().orchestrator_reasoning_enabled is False
    # and the live process settings default off too (opt-in only)
    assert getattr(settings, "orchestrator_reasoning_enabled") is False


# ── 2. flag OFF preserves the mechanical synthesis path ───────────────────────

def _run_synthesis_probe(monkeypatch, *, reasoning_enabled: bool) -> int:
    """Call _maybe_run_campaign_synthesis and count how many times the mechanical
    should_trigger_synthesis check is reached. 0 ⇒ the reasoning guard short-circuited."""
    from services.orchestrator import campaign_loop as cl

    calls = {"n": 0}

    def _spy_should_trigger(*args, **kwargs):
        calls["n"] += 1
        return False  # never actually run the real synthesis pass

    monkeypatch.setattr(cl, "should_trigger_synthesis", _spy_should_trigger)
    monkeypatch.setattr(cl.settings, "orchestrator_reasoning_enabled", reasoning_enabled)
    monkeypatch.setattr(cl.settings, "campaign_synthesis_enabled", True)

    tree = HypothesisTree()
    belief = SimpleNamespace(branch_exhausted=False, results_since_last_synthesis=10)
    campaign = SimpleNamespace(id=CID, question="q", hypothesis_tree=tree, belief_state=belief)
    em = _RecordingEmitter()

    asyncio.run(cl._maybe_run_campaign_synthesis(
        campaign=campaign, llm=None, emitter=em, generation=0,
        max_concurrent=2, inflight_ids=set(), prior_snippets=None,
    ))
    return calls["n"]


def test_flag_off_reaches_mechanical_synthesis(monkeypatch) -> None:
    # Flag OFF (default): the mechanical trigger check IS reached → unchanged behavior.
    assert _run_synthesis_probe(monkeypatch, reasoning_enabled=False) == 1


def test_flag_on_short_circuits_mechanical_synthesis(monkeypatch) -> None:
    # Flag ON: the reasoning guard returns before the mechanical trigger check →
    # reasoning drives expansion instead of synthesis (no double-expansion).
    assert _run_synthesis_probe(monkeypatch, reasoning_enabled=True) == 0


# ── 3. tree_context_for_reasoning (pure) ──────────────────────────────────────

def test_tree_context_shape_and_content() -> None:
    tree, parent, node = _tree_with_parent_child()
    node.attempts = [{"round": 1, "changes": "more samples", "rationale": "underpowered"}]
    ctx = tree_context_for_reasoning(tree, node, question="Q?")

    assert ctx["question"] == "Q?"
    this = ctx["this_node"]
    assert this["node_id"] == "n"
    assert this["verdict"] == "inconclusive"
    assert this["attempts"] == 1
    assert "metric" in this["evidence_summary"]

    assert ctx["parent"]["node_id"] == "p"
    assert ctx["parent"]["verdict"] == "confirmed"

    sib_ids = {s["node_id"] for s in ctx["siblings"]}
    assert sib_ids == {"s"}  # excludes the node itself

    assert ctx["confirmed"]["count"] == 1
    assert any("Variance predicts specificity" in c for c in ctx["confirmed"]["claims"])
    assert "variance" in ctx["whats_working"]


def test_tree_context_for_seed_has_no_parent() -> None:
    tree = HypothesisTree()
    seed = HypothesisNode(id="root", text="seed", parent_id=None, depth=0, verdict="refuted")
    tree.nodes = {"root": seed}
    ctx = tree_context_for_reasoning(tree, seed, question="Q")
    assert ctx["parent"] is None
    assert ctx["siblings"] == []
    assert ctx["confirmed"]["count"] == 0
    assert "no confirmed findings" in ctx["whats_working"]


def test_build_reasoning_prompt_is_pure_and_mentions_final_verdict() -> None:
    tree, parent, node = _tree_with_parent_child()
    ctx = tree_context_for_reasoning(tree, node, question="Q?")
    prompt = build_reasoning_prompt(ctx, verdict="inconclusive", confidence=0.4, max_retune_rounds=3)
    assert "FINAL" in prompt  # verdict framed as final / not re-judged
    assert "inconclusive" in prompt
    assert "deepen" in prompt and "retune" in prompt and "spawn_related" in prompt and "drop" in prompt
    assert "Effect holds in tissue A" in prompt  # this node's text
    assert "Variance predicts specificity" in prompt  # parent text


# ── 4. parse_reasoning_decision (pure, defensive) ─────────────────────────────

def test_parse_deepen_with_child() -> None:
    d = parse_reasoning_decision('{"action": "deepen", "rationale": "narrow it", "child_hypothesis_text": "in cortex only"}')
    assert d.action == ACTION_DEEPEN
    assert d.parse_error is False
    assert d.child_hypothesis_text == "in cortex only"
    assert d.rationale == "narrow it"


def test_parse_retune_with_changes() -> None:
    d = parse_reasoning_decision('{"action": "retune", "retune_changes": "more replications"}')
    assert d.action == ACTION_RETUNE
    assert d.retune_changes == "more replications"


def test_parse_action_aliases() -> None:
    assert parse_reasoning_decision('{"action": "lateral"}').action == ACTION_SPAWN_RELATED
    assert parse_reasoning_decision('{"action": "abandon"}').action == ACTION_DROP
    assert parse_reasoning_decision('{"action": "narrow"}').action == ACTION_DEEPEN
    assert parse_reasoning_decision('{"action": "retry"}').action == ACTION_RETUNE


def test_parse_fenced_json_block() -> None:
    raw = "```json\n{\"action\": \"drop\", \"rationale\": \"dead end\"}\n```"
    d = parse_reasoning_decision(raw)
    assert d.action == ACTION_DROP
    assert d.parse_error is False


def test_parse_json_embedded_in_prose() -> None:
    raw = 'Here is my decision: {"action": "deepen", "child_hypothesis_text": "x"} — done.'
    d = parse_reasoning_decision(raw)
    assert d.action == ACTION_DEEPEN
    assert d.child_hypothesis_text == "x"


def test_parse_malformed_is_safe_drop() -> None:
    d = parse_reasoning_decision("not json at all")
    assert d.action == ACTION_DROP
    assert d.parse_error is True


def test_parse_unknown_action_is_drop_but_not_parse_error() -> None:
    d = parse_reasoning_decision('{"action": "teleport"}')
    assert d.action == ACTION_DROP
    assert d.parse_error is False  # the model spoke; it just said something off-vocab


# ── 5. apply_reasoning_decision → tree mutations (scripted) ────────────────────

def test_apply_deepen_creates_narrower_child() -> None:
    tree, parent, node = _tree_with_parent_child()
    dec = ReasoningDecision(action=ACTION_DEEPEN, child_hypothesis_text="in cortex only")
    out = apply_reasoning_decision(tree, node, dec, generation=2, max_retune_rounds=3, max_depth=8)
    assert out.action == ACTION_DEEPEN
    assert len(out.new_nodes) == 1
    child = out.new_nodes[0]
    assert child.parent_id == "n"
    assert child.depth == node.depth + 1
    assert child.expansion_type == "boundary"
    assert child.verdict == "pending"
    assert child.id in tree.frontier
    assert child.id in tree.nodes["n"].children


def test_apply_deepen_at_max_depth_degrades_to_drop() -> None:
    tree, parent, node = _tree_with_parent_child()
    dec = ReasoningDecision(action=ACTION_DEEPEN, child_hypothesis_text="deeper")
    out = apply_reasoning_decision(tree, node, dec, generation=0, max_retune_rounds=3, max_depth=1)
    assert out.action == ACTION_DROP
    assert out.dropped is True
    assert out.note == "max_depth_reached"
    assert node.id in tree.exhausted
    assert out.new_nodes == []


def test_apply_deepen_without_text_is_noop() -> None:
    tree, parent, node = _tree_with_parent_child()
    before = dict(tree.nodes)
    dec = ReasoningDecision(action=ACTION_DEEPEN, child_hypothesis_text=None)
    out = apply_reasoning_decision(tree, node, dec, generation=0, max_retune_rounds=3)
    assert out.action == "noop"
    assert tree.nodes.keys() == before.keys()  # nothing added
    assert node.id not in tree.exhausted  # not closed either


def test_apply_spawn_related_makes_a_sibling() -> None:
    tree, parent, node = _tree_with_parent_child()
    dec = ReasoningDecision(action=ACTION_SPAWN_RELATED, child_hypothesis_text="a related idea")
    out = apply_reasoning_decision(tree, node, dec, generation=1, max_retune_rounds=3, max_depth=8)
    assert out.action == ACTION_SPAWN_RELATED
    child = out.new_nodes[0]
    # a lateral attaches to the node's PARENT → same depth, sibling of node
    assert child.parent_id == "p"
    assert child.depth == node.depth
    assert child.expansion_type == "alternative"
    assert child.id in tree.nodes["p"].children


def test_apply_spawn_related_on_seed_attaches_to_seed() -> None:
    tree = HypothesisTree()
    seed = HypothesisNode(id="root", text="seed", parent_id=None, depth=0, verdict="refuted")
    tree.nodes = {"root": seed}
    dec = ReasoningDecision(action=ACTION_SPAWN_RELATED, child_hypothesis_text="lateral idea")
    out = apply_reasoning_decision(tree, seed, dec, generation=0, max_retune_rounds=3, max_depth=8)
    child = out.new_nodes[0]
    assert child.parent_id == "root"
    assert child.depth == 1


def test_apply_retune_records_attempt_and_requeues() -> None:
    tree, parent, node = _tree_with_parent_child()
    assert node.attempts == []
    dec = ReasoningDecision(action=ACTION_RETUNE, retune_changes="more samples", rationale="underpowered")
    out = apply_reasoning_decision(tree, node, dec, generation=0, max_retune_rounds=3)
    assert out.action == ACTION_RETUNE
    assert out.retuned is True
    assert len(node.attempts) == 1
    assert node.attempts[0]["round"] == 1
    assert node.attempts[0]["changes"] == "more samples"
    assert node.verdict == "pending"          # re-run: back on the frontier
    assert node.id in tree.frontier
    assert node.expansion_type == "retest"
    assert "retune 1" in (node.test_methodology or "")  # change carried into methodology


def test_apply_retune_past_budget_degrades_to_drop() -> None:
    tree, parent, node = _tree_with_parent_child()
    node.attempts = [{"round": 1}, {"round": 2}, {"round": 3}]  # budget already used
    dec = ReasoningDecision(action=ACTION_RETUNE, retune_changes="again")
    out = apply_reasoning_decision(tree, node, dec, generation=0, max_retune_rounds=3)
    assert out.action == ACTION_DROP
    assert out.dropped is True
    assert out.note == "retune_budget_exhausted"
    assert len(node.attempts) == 3  # no new attempt recorded
    assert node.id in tree.exhausted


def test_apply_retune_reconfirm_path_unconfirms_until_rerun() -> None:
    # A retune of a confirmed node resets it to pending and drops it from confirmed
    # (the deterministic verdict will be recomputed on the re-run).
    tree, parent, node = _tree_with_parent_child()
    node.verdict = "confirmed"
    tree.confirmed = ["p", "n"]
    dec = ReasoningDecision(action=ACTION_RETUNE, retune_changes="tighter control")
    apply_reasoning_decision(tree, node, dec, generation=0, max_retune_rounds=3)
    assert node.verdict == "pending"
    assert "n" not in tree.confirmed


def test_apply_drop_exhausts_node() -> None:
    tree, parent, node = _tree_with_parent_child()
    tree.frontier = ["n"]
    dec = ReasoningDecision(action=ACTION_DROP, rationale="dead end")
    out = apply_reasoning_decision(tree, node, dec, generation=0, max_retune_rounds=3)
    assert out.action == ACTION_DROP
    assert out.dropped is True
    assert node.id in tree.exhausted
    assert node.id not in tree.frontier


# ── 6. orchestrator_reason_next + wiring with a FAKE LLM ───────────────────────

def test_reason_next_returns_scripted_decision() -> None:
    tree, parent, node = _tree_with_parent_child()
    campaign = _campaign_with_tree(tree)
    llm = _ScriptedLLM('{"action": "deepen", "child_hypothesis_text": "narrower"}')
    dec = asyncio.run(orchestrator_reason_next(
        campaign=campaign, node=node, evidence={"metric_value": 0.1}, verdict="inconclusive",
        confidence=0.4, llm=llm, max_retune_rounds=3,
    ))
    assert dec.action == ACTION_DEEPEN
    assert dec.child_hypothesis_text == "narrower"
    assert llm.prompts and "FINAL" in llm.prompts[0]  # prompt carried the final-verdict framing


def test_reason_next_is_failure_isolated() -> None:
    tree, parent, node = _tree_with_parent_child()
    campaign = _campaign_with_tree(tree)
    dec = asyncio.run(orchestrator_reason_next(
        campaign=campaign, node=node, evidence=None, verdict="inconclusive",
        confidence=0.4, llm=_BoomLLM(), max_retune_rounds=3,
    ))
    assert dec.action == ACTION_DROP
    assert dec.parse_error is True  # a failed call is a safe no-strategy fallback


def test_run_reasoning_step_deepen_emits_events_and_mutates_tree() -> None:
    from services.orchestrator import campaign_loop as cl

    tree, parent, node = _tree_with_parent_child()
    campaign = _campaign_with_tree(tree)
    em = _RecordingEmitter()
    llm = _ScriptedLLM('{"action": "deepen", "rationale": "narrow it", "child_hypothesis_text": "in cortex only"}')

    out = asyncio.run(cl._run_orchestrator_reasoning_step(
        emitter=em, campaign=campaign, node=node, evidence_obj={"metric_value": 0.1},
        verdict="inconclusive", confidence=0.4, llm=llm, generation=2, hypothesis_id="hid",
    ))
    assert out.action == ACTION_DEEPEN
    # tree got a new child
    assert any(n.parent_id == "n" for n in tree.nodes.values())
    # events: one reasoning + one hypothesis-written
    reasoning = em.of_type(EventType.ORCHESTRATOR_REASONING)
    written = em.of_type(EventType.ORCHESTRATOR_HYPOTHESIS_WRITTEN)
    assert len(reasoning) == 1
    assert reasoning[0]["payload"]["decision"] == ACTION_DEEPEN
    assert "narrow it" in reasoning[0]["payload"]["detail"]
    assert len(written) == 1


def test_run_reasoning_step_parse_error_leaves_tree_untouched() -> None:
    from services.orchestrator import campaign_loop as cl

    tree, parent, node = _tree_with_parent_child()
    n_before = len(tree.nodes)
    campaign = _campaign_with_tree(tree)
    em = _RecordingEmitter()
    llm = _ScriptedLLM("garbage not json")

    out = asyncio.run(cl._run_orchestrator_reasoning_step(
        emitter=em, campaign=campaign, node=node, evidence_obj=None,
        verdict="inconclusive", confidence=0.4, llm=llm, generation=0,
    ))
    assert out is None                       # skipped
    assert len(tree.nodes) == n_before       # no mutation
    assert node.id not in tree.exhausted     # not closed
    # narrated as "reasoning unavailable"
    reasoning = em.of_type(EventType.ORCHESTRATOR_REASONING)
    assert len(reasoning) == 1
    assert reasoning[0]["payload"].get("parse_error") is True


def test_run_reasoning_step_is_failure_isolated_against_raising_emitter() -> None:
    from services.orchestrator import campaign_loop as cl

    tree, parent, node = _tree_with_parent_child()
    campaign = _campaign_with_tree(tree)
    llm = _ScriptedLLM('{"action": "drop", "rationale": "dead end"}')
    # A raising emitter must not propagate out of the reasoning step.
    asyncio.run(cl._run_orchestrator_reasoning_step(
        emitter=_RaisingEmitter(), campaign=campaign, node=node, evidence_obj={},
        verdict="refuted", confidence=0.1, llm=llm, generation=0,
    ))
    # the drop still took effect on the tree despite the emit failing
    assert node.id in tree.exhausted


# ── 7. attempts serialization round-trip ──────────────────────────────────────

def test_attempts_round_trip_through_serialization() -> None:
    node = HypothesisNode(id="n", text="t", parent_id=None, depth=0)
    node.attempts = [{"round": 1, "changes": "x", "rationale": "y"}]
    restored = HypothesisNode.from_dict(node.to_dict())
    assert restored.attempts == [{"round": 1, "changes": "x", "rationale": "y"}]


def test_old_checkpoint_without_attempts_defaults_empty() -> None:
    # A pre-C3b serialized node has no "attempts" key; from_dict must default it.
    data = {"id": "n", "text": "t", "parent_id": None, "depth": 0}
    restored = HypothesisNode.from_dict(data)
    assert restored.attempts == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
