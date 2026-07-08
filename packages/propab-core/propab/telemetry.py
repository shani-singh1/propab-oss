"""
Research-trajectory telemetry — the per-hypothesis "dataset nobody else has".

For EVERY hypothesis in a campaign we persist one structured
:class:`HypothesisTrajectory` record. Accumulated across thousands of future
campaigns on a FIXED engine, the corpus can answer meta-questions like "which
generators produce confirmed hypotheses?", "which failures repeat?", and "where
is compute wasted?".

This module is **pure instrumentation**: :func:`build_trajectories` DERIVES the
records from a campaign's hypothesis tree + the campaign's event stream. It never
mutates the campaign and never changes verdict / breakthrough / honesty logic.
Fields that cannot be honestly derived are left ``None`` — never fabricated.

Event-stream contract
---------------------
``events`` is an iterable of event dicts. Each event is read defensively so both
the DB row shape (``payload_json`` string) and the emitter shape (``payload``
dict, ``event_type`` may be an :class:`~propab.types.EventType`) work:

- ``event_type``   → str (``"llm.response"`` …)
- ``payload``      → dict (or JSON string under ``payload_json``)
- ``hypothesis_id``→ the DB hypotheses-row id the worker/orchestrator stamped on
  the event (a uuid5 of ``campaign_id:node_id``); may instead live inside the
  payload for some orchestrator events.

The worker/orchestrator reference hypotheses by their DB row id
(``uuid5(_NODE_ID_NAMESPACE, "{campaign_id}:{node_id}")``), not by the tree's own
node id. We therefore index every tree node under BOTH ids so events attribute
correctly regardless of which id a given event carries.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable
from uuid import UUID, uuid5

from propab.campaign import ResearchCampaign
from propab.hypothesis_tree import HypothesisNode, HypothesisTree

# Must match services.orchestrator.campaign_loop._NODE_ID_NAMESPACE so the DB
# hypotheses-row id can be reconstructed from a tree node id (events reference the
# row id). Replicated here (not imported) to keep this module pure core with no
# dependency on the orchestrator service package.
_NODE_ID_NAMESPACE = UUID("c3e7a1f0-4b2d-4e8a-95cf-0d1e3f5a7c9b")

_DOMAIN_TAG_RE = re.compile(r"\[domain_profile:([a-z0-9_]+)\]", re.I)


def _db_row_id(campaign_id: str, node_id: str) -> str:
    return str(uuid5(_NODE_ID_NAMESPACE, f"{campaign_id}:{node_id}"))


# ── The record ───────────────────────────────────────────────────────────────

@dataclass
class HypothesisTrajectory:
    """One research-trajectory record per hypothesis (the telemetry primitive)."""

    # identity
    campaign_id: str
    hypothesis_id: str
    parent_id: str | None = None
    generation: int | None = None       # campaign generation that spawned the node
    round: int | None = None            # authoritative round from the event stream
    domain: str | None = None

    # outcome
    verdict: str | None = None          # confirmed | refuted | inconclusive | failed | pending
    confidence: float | None = None
    discovery_worthy: bool | None = None

    # novelty
    was_novel: bool | None = None            # novel vs. prior confirmations
    literature_predicted: bool | None = None  # did the prior/literature anticipate it

    # cost
    llm_calls: int | None = None
    tool_calls: int | None = None
    code_runs: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    duration_sec: float | None = None

    # process
    generator_strategy: str | None = None
    reasoning_strategy: str | None = None
    expansion_type: str | None = None

    # diagnosis
    experiment_informative: bool | None = None
    failure_reason: str | None = None
    verifier_that_exposed_failure: str | None = None

    # lineage
    branch_outcome: str | None = None   # confirmed | dead_end | None (still open)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HypothesisTrajectory":
        fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in fields})


# ── Event-reading helpers (defensive) ────────────────────────────────────────

def _ev_type(ev: dict[str, Any]) -> str:
    et = ev.get("event_type")
    if et is None:
        return ""
    val = getattr(et, "value", et)  # accept EventType or str
    return str(val)


def _ev_payload(ev: dict[str, Any]) -> dict[str, Any]:
    p = ev.get("payload")
    if isinstance(p, dict):
        return p
    raw = ev.get("payload_json")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _ev_hypothesis_id(ev: dict[str, Any], payload: dict[str, Any]) -> str | None:
    hid = ev.get("hypothesis_id")
    if hid:
        return str(hid)
    hid = payload.get("hypothesis_id")
    return str(hid) if hid else None


def _as_int(v: Any) -> int | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    return None


# ── Derivations ──────────────────────────────────────────────────────────────

def _index_events_by_node(
    campaign_id: str,
    tree: HypothesisTree,
    events: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group events by tree-node id.

    Each node is registered under its tree id AND its derived DB row id so events
    attribute correctly whichever id they carry.
    """
    key_to_node: dict[str, str] = {}
    for nid in tree.nodes:
        key_to_node[str(nid)] = str(nid)
        key_to_node[_db_row_id(campaign_id, str(nid))] = str(nid)

    grouped: dict[str, list[dict[str, Any]]] = {nid: [] for nid in tree.nodes}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        payload = _ev_payload(ev)
        hid = _ev_hypothesis_id(ev, payload)
        if hid is None:
            continue
        node_id = key_to_node.get(hid)
        if node_id is not None:
            # Stash the normalized payload so downstream derivations don't re-parse.
            grouped[node_id].append({"_type": _ev_type(ev), "_payload": payload})
    return grouped


def _derive_cost(evs: list[dict[str, Any]]) -> dict[str, Any]:
    llm_calls = tool_calls = code_runs = 0
    tok_in_sum = tok_out_sum = 0
    saw_tok_in = saw_tok_out = False
    dur_ms_sum = 0.0
    saw_dur = False
    for e in evs:
        et = e["_type"]
        p = e["_payload"]
        if et == "llm.response":
            llm_calls += 1
            ti = _as_int(p.get("tokens_in"))
            if ti is not None:
                tok_in_sum += ti
                saw_tok_in = True
            to = _as_int(p.get("tokens_out"))
            if to is not None:
                tok_out_sum += to
                saw_tok_out = True
        elif et == "tool.called":
            tool_calls += 1
        elif et == "code.submitted":
            code_runs += 1
        dur = p.get("duration_ms")
        if isinstance(dur, (int, float)) and not isinstance(dur, bool):
            dur_ms_sum += float(dur)
            saw_dur = True
    return {
        "llm_calls": llm_calls or None,
        "tool_calls": tool_calls or None,
        "code_runs": code_runs or None,
        "tokens_in": tok_in_sum if saw_tok_in else None,
        "tokens_out": tok_out_sum if saw_tok_out else None,
        "duration_sec": round(dur_ms_sum / 1000.0, 3) if saw_dur else None,
    }


def _first_payload_value(
    evs: list[dict[str, Any]], key: str, *, types: set[str] | None = None
) -> Any:
    for e in evs:
        if types is not None and e["_type"] not in types:
            continue
        val = e["_payload"].get(key)
        if val is not None and val != "":
            return val
    return None


def _derive_round(evs: list[dict[str, Any]]) -> int | None:
    for e in evs:
        r = _as_int(e["_payload"].get("round"))
        if r is not None:
            return r
    return None


def _derive_domain(node: HypothesisNode, evs: list[dict[str, Any]], question: str) -> str | None:
    dom = _first_payload_value(evs, "domain")
    if dom:
        return str(dom)
    m = _DOMAIN_TAG_RE.search(question or "")
    return m.group(1).lower() if m else None


def _derive_discovery_worthy(node: HypothesisNode, evs: list[dict[str, Any]]) -> bool | None:
    certs = [e for e in evs if e["_type"] == "finding.certified"]
    for e in certs:
        if e["_payload"].get("certified") is True:
            return True
    if isinstance(node.finding, dict) and "discovery_worthy" in node.finding:
        return bool(node.finding.get("discovery_worthy"))
    if certs:  # certification ran but nothing certified
        return False
    return None


def _derive_was_novel(node: HypothesisNode, verdict: str) -> bool | None:
    reason = (node.inconclusive_reason or "").lower()
    if "duplicate" in reason:
        return False
    if verdict == "confirmed":
        # A confirmed discovery node passed the confirmed-claim dedup gate.
        return True
    return None


def _derive_literature_predicted(evs: list[dict[str, Any]]) -> bool | None:
    vs = _first_payload_value(evs, "vs_best_known", types={"finding.certified"})
    if not vs:
        return None
    s = str(vs).lower()
    if "match" in s:
        return True
    if "exceed" in s or "beat" in s:
        return False
    return None


def _derive_experiment_informative(node: HypothesisNode, verdict: str) -> bool | None:
    if verdict in ("confirmed", "refuted"):
        return True
    if node.children:
        # Even an inconclusive result that spawned children changed the tree.
        return True
    if verdict == "inconclusive":
        return False
    return None


def _derive_failure_reason(node: HypothesisNode, evs: list[dict[str, Any]], verdict: str) -> str | None:
    if node.inconclusive_reason:
        return node.inconclusive_reason
    if verdict == "refuted":
        return node.failure_signature or "refuted"
    if verdict in ("inconclusive", "failed"):
        types = {e["_type"] for e in evs}
        if "code.timeout" in types:
            return "timeout"
        if "agent.failed" in types:
            reason = _first_payload_value(evs, "reason", types={"agent.failed"})
            return str(reason) if reason else "agent_failed"
        if node.failure_signature:
            return node.failure_signature
    return None


def _derive_verifier(node: HypothesisNode, evs: list[dict[str, Any]], verdict: str) -> str | None:
    if verdict in ("confirmed", "pending"):
        return None
    if node.verification_method:
        return str(node.verification_method)
    tools = [
        e["_payload"].get("tool")
        for e in evs
        if e["_type"] == "tool.called" and e["_payload"].get("tool")
    ]
    if tools:
        return str(tools[-1])
    if any(e["_type"].startswith("code.") for e in evs):
        return "code_sandbox"
    return None


def _derive_verdict(node: HypothesisNode, evs: list[dict[str, Any]]) -> str:
    verdict = node.verdict or "pending"
    if verdict == "pending" and any(e["_type"] == "agent.failed" for e in evs):
        return "failed"
    return verdict


def _build_branch_index(tree: HypothesisTree) -> tuple[dict[str, set[str]], set[str]]:
    """Return (subtree-descendants-incl-self per node, confirmed node ids)."""
    confirmed = {nid for nid, n in tree.nodes.items() if (n.verdict or "") == "confirmed"}
    subtree: dict[str, set[str]] = {}

    def descendants(root: str, seen: set[str]) -> set[str]:
        if root in subtree:
            return subtree[root]
        if root in seen:  # cycle guard
            return {root}
        seen = seen | {root}
        acc = {root}
        node = tree.nodes.get(root)
        if node is not None:
            for cid in node.children:
                if cid in tree.nodes and cid != root:
                    acc |= descendants(cid, seen)
        subtree[root] = acc
        return acc

    for nid in tree.nodes:
        descendants(nid, set())
    return subtree, confirmed


def _derive_branch_outcome(
    node_id: str,
    tree: HypothesisTree,
    subtree: dict[str, set[str]],
    confirmed: set[str],
) -> str | None:
    reach = subtree.get(node_id, {node_id})
    if reach & confirmed:
        return "confirmed"
    # Fully evaluated with no confirmation anywhere in the branch → dead end.
    all_evaluated = all(
        (tree.nodes[nid].verdict or "pending") != "pending"
        for nid in reach
        if nid in tree.nodes
    )
    return "dead_end" if all_evaluated else None


def build_trajectories(
    campaign: ResearchCampaign,
    events: Iterable[dict[str, Any]],
) -> list[HypothesisTrajectory]:
    """Derive one :class:`HypothesisTrajectory` per hypothesis-tree node.

    Pure and defensive: never mutates ``campaign``; any field that cannot be
    honestly derived from the tree or the event stream is left ``None``.
    """
    tree = campaign.hypothesis_tree
    events_list = list(events or [])
    by_node = _index_events_by_node(campaign.id, tree, events_list)
    subtree, confirmed = _build_branch_index(tree)

    out: list[HypothesisTrajectory] = []
    for node_id, node in tree.nodes.items():
        evs = by_node.get(node_id, [])
        verdict = _derive_verdict(node, evs)
        cost = _derive_cost(evs)
        conf = float(node.confidence) if node.confidence else None
        traj = HypothesisTrajectory(
            campaign_id=str(campaign.id),
            hypothesis_id=str(node_id),
            parent_id=str(node.parent_id) if node.parent_id else None,
            generation=int(node.generation) if node.generation is not None else None,
            round=_derive_round(evs),
            domain=_derive_domain(node, evs, campaign.question),
            verdict=verdict,
            confidence=conf,
            discovery_worthy=_derive_discovery_worthy(node, evs),
            was_novel=_derive_was_novel(node, verdict),
            literature_predicted=_derive_literature_predicted(evs),
            llm_calls=cost["llm_calls"],
            tool_calls=cost["tool_calls"],
            code_runs=cost["code_runs"],
            tokens_in=cost["tokens_in"],
            tokens_out=cost["tokens_out"],
            duration_sec=cost["duration_sec"],
            generator_strategy=(
                _first_payload_value(evs, "generator_strategy")
                or _first_payload_value(evs, "strategy")
            ),
            reasoning_strategy=_first_payload_value(evs, "reasoning_strategy"),
            expansion_type=node.expansion_type,
            experiment_informative=_derive_experiment_informative(node, verdict),
            failure_reason=_derive_failure_reason(node, evs, verdict),
            verifier_that_exposed_failure=_derive_verifier(node, evs, verdict),
            branch_outcome=_derive_branch_outcome(node_id, tree, subtree, confirmed),
        )
        out.append(traj)
    return out
