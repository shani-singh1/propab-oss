"""Tests for the hypothesis-trajectory telemetry substrate.

Covers:
  1. ``build_trajectories`` derives well-formed records from a synthetic campaign
     tree + event stream (verdict, cost, parent lineage, failure_reason, branch).
  2. Records round-trip through ``save_trajectories`` / ``load_trajectories``
     against an in-memory fake session that emulates the Postgres upsert
     semantics (no real DB needed — keeps the suite green everywhere).
"""
from __future__ import annotations

import asyncio

from propab.campaign import ResearchCampaign
from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.telemetry import HypothesisTrajectory, _db_row_id, build_trajectories

CID = "11111111-2222-3333-4444-555555555555"


def _ev(event_type: str, node_id: str | None, payload: dict) -> dict:
    """An emitter-shape event; hypothesis_id is the DB row id (uuid5 of node)."""
    return {
        "event_type": event_type,
        "step": "x",
        "hypothesis_id": _db_row_id(CID, node_id) if node_id else None,
        "payload": payload,
    }


def _build_synthetic() -> tuple[ResearchCampaign, list[dict]]:
    tree = HypothesisTree()

    a = HypothesisNode(id="A", text="root confirmed claim", parent_id=None, depth=0,
                       verdict="confirmed", confidence=0.9, generation=0,
                       verification_method="combinatorial_computation", children=["B"])
    b = HypothesisNode(id="B", text="refuted child", parent_id="A", depth=1,
                       verdict="refuted", confidence=0.6, generation=1,
                       expansion_type="boundary", failure_signature="below_baseline")
    c = HypothesisNode(id="C", text="inconclusive root", parent_id=None, depth=0,
                       verdict="inconclusive", confidence=0.3, generation=0,
                       inconclusive_reason="timeout")
    d = HypothesisNode(id="D", text="pending root", parent_id=None, depth=0,
                       verdict="pending", confidence=0.0, generation=0)
    for n in (a, b, c, d):
        tree.nodes[n.id] = n
    tree.confirmed = ["A"]

    campaign = ResearchCampaign(
        id=CID,
        question="find records [domain_profile:math_combinatorics]",
        hypothesis_tree=tree,
    )

    events = [
        _ev("llm.response", "A", {"tokens_in": 100, "tokens_out": 40, "duration_ms": 1200, "round": 0}),
        _ev("llm.response", "A", {"tokens_in": 50, "tokens_out": 20, "duration_ms": 800}),
        _ev("tool.called", "A", {"tool": "combinatorial_verification", "domain": "math_combinatorics"}),
        _ev("code.submitted", "A", {}),
        _ev("finding.certified", "A", {"certified": True, "round": 0,
                                       "vs_best_known": "exceeds_best_known",
                                       "domain": "math_combinatorics"}),
        _ev("agent.completed", "A", {"verdict": "confirmed", "round": 0, "domain": "math_combinatorics"}),
        _ev("tool.called", "B", {"tool": "materials_verification"}),
        _ev("agent.completed", "B", {"verdict": "refuted", "round": 1}),
        _ev("code.timeout", "C", {}),
        _ev("agent.completed", "C", {"verdict": "inconclusive", "round": 2}),
    ]
    return campaign, events


def _by_id(trajs: list[HypothesisTrajectory]) -> dict[str, HypothesisTrajectory]:
    return {t.hypothesis_id: t for t in trajs}


def test_build_trajectories_derives_expected_fields() -> None:
    campaign, events = _build_synthetic()
    trajs = build_trajectories(campaign, events)
    assert len(trajs) == 4
    by = _by_id(trajs)

    # ── Confirmed root A: identity, cost, novelty, lineage ──────────────────
    a = by["A"]
    assert a.campaign_id == CID
    assert a.parent_id is None
    assert a.generation == 0
    assert a.round == 0
    assert a.domain == "math_combinatorics"
    assert a.verdict == "confirmed"
    assert a.confidence == 0.9
    assert a.discovery_worthy is True
    assert a.was_novel is True
    assert a.literature_predicted is False  # exceeds best known → not predicted
    assert a.llm_calls == 2
    assert a.tool_calls == 1
    assert a.code_runs == 1
    assert a.tokens_in == 150
    assert a.tokens_out == 60
    assert a.duration_sec == 2.0
    assert a.experiment_informative is True
    assert a.failure_reason is None
    assert a.verifier_that_exposed_failure is None  # confirmed → no failure
    assert a.branch_outcome == "confirmed"

    # ── Refuted child B: parent lineage + failure diagnosis ─────────────────
    b = by["B"]
    assert b.parent_id == "A"
    assert b.generation == 1
    assert b.expansion_type == "boundary"
    assert b.verdict == "refuted"
    assert b.failure_reason == "below_baseline"
    assert b.verifier_that_exposed_failure == "materials_verification"
    assert b.tool_calls == 1
    assert b.was_novel is None
    assert b.branch_outcome == "dead_end"

    # ── Inconclusive C: failure_reason + code-sandbox verifier ──────────────
    c = by["C"]
    assert c.verdict == "inconclusive"
    assert c.failure_reason == "timeout"
    assert c.verifier_that_exposed_failure == "code_sandbox"
    assert c.experiment_informative is False
    assert c.code_runs is None  # code.timeout is not a code.submitted run
    assert c.branch_outcome == "dead_end"

    # ── Pending D: honest Nones, never fabricated ───────────────────────────
    d = by["D"]
    assert d.verdict == "pending"
    assert d.confidence is None
    assert d.llm_calls is None
    assert d.duration_sec is None
    assert d.experiment_informative is None
    assert d.branch_outcome is None


def test_build_trajectories_reads_db_row_shape() -> None:
    """Events in the raw DB shape (payload_json string) also attribute correctly."""
    campaign, _ = _build_synthetic()
    db_events = [
        {
            "event_type": "llm.response",
            "hypothesis_id": _db_row_id(CID, "A"),
            "payload_json": '{"tokens_in": 7, "tokens_out": 3, "duration_ms": 500}',
        }
    ]
    trajs = build_trajectories(campaign, db_events)
    a = _by_id(trajs)["A"]
    assert a.llm_calls == 1
    assert a.tokens_in == 7
    assert a.tokens_out == 3
    assert a.duration_sec == 0.5


# ── Round-trip through save/load with an in-memory Postgres-upsert emulator ───

class _FakeResult:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def mappings(self) -> "_FakeResult":
        return self

    def all(self) -> list[dict]:
        return self._rows


class _FakeSession:
    """Emulates just enough of an async SQLAlchemy session to round-trip the
    upsert keyed by (campaign_id, hypothesis_id)."""

    def __init__(self, store: dict) -> None:
        self._store = store

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def execute(self, sql, params=None):
        sql_text = str(sql)
        if "INSERT INTO hypothesis_trajectories" in sql_text:
            key = (params["campaign_id"], params["hypothesis_id"])
            self._store[key] = dict(params)  # ON CONFLICT upsert == overwrite
            return _FakeResult([])
        if "SELECT * FROM hypothesis_trajectories" in sql_text:
            rows = list(self._store.values())
            if params and params.get("cid") is not None:
                rows = [r for r in rows if r["campaign_id"] == params["cid"]]
            return _FakeResult(rows)
        return _FakeResult([])

    async def commit(self) -> None:
        return None


def _fake_factory(store: dict):
    def factory():
        return _FakeSession(store)
    return factory


def test_trajectories_round_trip_through_save_load() -> None:
    from propab.telemetry_db import load_trajectories, save_trajectories

    campaign, events = _build_synthetic()
    trajs = build_trajectories(campaign, events)
    store: dict = {}
    factory = _fake_factory(store)

    async def run():
        n = await save_trajectories(trajs, factory)
        assert n == len(trajs)
        # Idempotent: saving again upserts the same rows, not duplicates.
        await save_trajectories(trajs, factory)
        assert len(store) == len(trajs)
        loaded_all = await load_trajectories(factory)
        loaded_one = await load_trajectories(factory, campaign_id=CID)
        return loaded_all, loaded_one

    loaded_all, loaded_one = asyncio.run(run())
    assert len(loaded_all) == len(trajs)
    assert len(loaded_one) == len(trajs)

    original = {t.hypothesis_id: t.to_dict() for t in trajs}
    restored = {t.hypothesis_id: t.to_dict() for t in loaded_one}
    assert original == restored
