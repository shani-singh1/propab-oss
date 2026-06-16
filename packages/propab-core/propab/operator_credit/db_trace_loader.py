"""DB-backed operator trace extraction — HypothesisTree → NodeOperatorTrace (fixes.md #1)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.operator_credit.operator_trace import NodeOperatorTrace, OperatorTraceLedger
from propab.operator_credit.trace_extractor import extract_trace_for_node


@dataclass
class ToolCallRecord:
    tool_name: str
    success: bool
    duration_ms: int
    hypothesis_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "hypothesis_id": self.hypothesis_id,
        }


@dataclass
class CampaignDBBundle:
    """Full campaign bundle for accurate operator traces."""

    campaign_id: str
    question: str = ""
    tree: HypothesisTree | None = None
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    hypothesis_db_ids: dict[str, str] = field(default_factory=dict)
    compute_seconds: int = 0
    baseline_campaign_id: str | None = None
    policy_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "question": self.question,
            "has_tree": self.tree is not None,
            "n_snapshots": len(self.snapshots),
            "n_tool_calls": len(self.tool_calls),
            "compute_seconds": self.compute_seconds,
            "baseline_campaign_id": self.baseline_campaign_id,
            "policy_id": self.policy_id,
        }


def _tool_calls_for_node(
    node_id: str,
    tool_calls: list[ToolCallRecord],
    hypothesis_db_ids: dict[str, str],
) -> list[dict[str, Any]]:
    db_id = hypothesis_db_ids.get(node_id)
    if not db_id:
        return []
    return [
        tc.to_dict()
        for tc in tool_calls
        if tc.hypothesis_id == db_id
    ]


def enrich_trace_from_db(
    trace: NodeOperatorTrace,
    *,
    node: HypothesisNode,
    tree: HypothesisTree,
    tool_calls: list[ToolCallRecord],
    hypothesis_db_ids: dict[str, str],
) -> NodeOperatorTrace:
    micro = _tool_calls_for_node(node.id, tool_calls, hypothesis_db_ids)
    duration = sum(int(t.get("duration_ms") or 0) for t in micro)
    trace.parent_node_id = node.parent_id
    trace.child_node_ids = list(node.children)
    trace.expansion_type = node.expansion_type
    trace.claim_type = node.claim_type
    trace.primary_theme = node.primary_theme
    trace.depth = node.depth
    trace.tool_calls = micro
    trace.duration_ms = duration
    trace.lineage_length = node.lineage_length
    if duration > 0:
        trace.cost = round(trace.cost + duration / 1000.0, 2)
    return trace


def extract_traces_from_db_bundle(bundle: CampaignDBBundle) -> OperatorTraceLedger:
    """Extract accurate traces from DB-backed campaign bundle."""
    if not bundle.tree:
        from propab.operator_credit.trace_extractor import extract_traces_from_snapshots

        return extract_traces_from_snapshots(
            campaign_id=bundle.campaign_id,
            snapshots=bundle.snapshots,
        )

    ledger = OperatorTraceLedger()
    tested = [
        n for n in bundle.tree.nodes.values()
        if n.verdict in ("confirmed", "refuted", "inconclusive")
    ]
    tested.sort(key=lambda n: (n.generation, n.depth))
    for i, node in enumerate(tested):
        trace = extract_trace_for_node(
            campaign_id=bundle.campaign_id,
            node=node,
            tree=bundle.tree,
            order=i,
        )
        enrich_trace_from_db(
            trace,
            node=node,
            tree=bundle.tree,
            tool_calls=bundle.tool_calls,
            hypothesis_db_ids=bundle.hypothesis_db_ids,
        )
        trace.source = "db" if bundle.tool_calls else "tree"
        ledger.add(trace)
    return ledger


async def load_campaign_db_bundle(campaign_id: str) -> CampaignDBBundle:
    """Load tree, snapshots, tool calls, and metadata from Postgres."""
    from sqlalchemy import text

    from propab.config import settings
    from propab.db import create_engine, create_session_factory
    from propab.policy_store import PolicyStore

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    bundle = CampaignDBBundle(campaign_id=campaign_id)

    async with session_factory() as db:
        camp = (
            await db.execute(
                text(
                    """
                    SELECT question, hypothesis_tree_json, compute_seconds_used
                    FROM research_campaigns
                    WHERE id = CAST(:id AS uuid)
                    """
                ),
                {"id": campaign_id},
            )
        ).fetchone()
        if camp:
            bundle.question = str(camp[0] or "")
            if camp[1]:
                raw = camp[1] if isinstance(camp[1], dict) else json.loads(camp[1])
                bundle.tree = HypothesisTree.from_dict(raw)
            bundle.compute_seconds = int(camp[2] or 0)

        snap_rows = (
            await db.execute(
                text(
                    """
                    SELECT payload_json
                    FROM events
                    WHERE session_id = CAST(:id AS uuid)
                      AND step = 'campaign.frontier_snapshot'
                    ORDER BY created_at ASC
                    """
                ),
                {"id": campaign_id},
            )
        ).fetchall()
        for row in snap_rows:
            payload = row[0]
            if isinstance(payload, str):
                payload = json.loads(payload)
            if isinstance(payload, dict):
                bundle.snapshots.append(payload)

        hyp_rows = (
            await db.execute(
                text(
                    """
                    SELECT id, text, status
                    FROM hypotheses
                    WHERE session_id = CAST(:id AS uuid)
                    """
                ),
                {"id": campaign_id},
            )
        ).fetchall()
        text_to_db: dict[str, str] = {}
        for hid, htext, _status in hyp_rows:
            text_to_db[str(htext or "")[:80]] = str(hid)

        if bundle.tree:
            for nid, node in bundle.tree.nodes.items():
                key = node.text[:80]
                if key in text_to_db:
                    bundle.hypothesis_db_ids[nid] = text_to_db[key]

        tc_rows = (
            await db.execute(
                text(
                    """
                    SELECT tc.tool_name, tc.success, tc.duration_ms, tc.hypothesis_id
                    FROM tool_calls tc
                    JOIN hypotheses h ON h.id = tc.hypothesis_id
                    WHERE h.session_id = CAST(:id AS uuid)
                    ORDER BY tc.duration_ms DESC
                    """
                ),
                {"id": campaign_id},
            )
        ).fetchall()
        for tool_name, success, duration_ms, hyp_id in tc_rows:
            bundle.tool_calls.append(ToolCallRecord(
                tool_name=str(tool_name or "unknown"),
                success=bool(success),
                duration_ms=int(duration_ms or 0),
                hypothesis_id=str(hyp_id),
            ))

    await engine.dispose()

    store = PolicyStore.load()
    binding = store.active_bindings.get(campaign_id)
    if binding:
        bundle.policy_id = binding.policy_id
        bundle.baseline_campaign_id = binding.baseline_campaign_id

    return bundle


async def load_bundles_from_db(campaign_ids: list[str]) -> list[CampaignDBBundle]:
    """Load multiple campaign bundles from Postgres."""
    return [await load_campaign_db_bundle(cid) for cid in campaign_ids]


def campaign_ids_from_trajectory(path: Path | str) -> list[str]:
    """Extract campaign IDs from entropy trajectory JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids: list[str] = []
    for camp in data.get("campaigns") or []:
        cid = camp.get("campaign_id")
        if cid:
            ids.append(str(cid))
    return ids


def load_bundles_from_trajectory_file(
    path: str,
    *,
    trees: dict[str, HypothesisTree | dict[str, Any]] | None = None,
) -> list[CampaignDBBundle]:
    """Offline bundle loader when DB is unavailable."""
    from pathlib import Path

    from propab.layer05.replay_loader import load_snapshots_from_json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    by_snaps = load_snapshots_from_json(path)
    trees = trees or {}
    bundles: list[CampaignDBBundle] = []
    for camp in data.get("campaigns") or []:
        cid = camp.get("campaign_id")
        if not cid:
            continue
        tree_raw = trees.get(cid)
        tree = None
        if tree_raw is not None:
            tree = tree_raw if isinstance(tree_raw, HypothesisTree) else HypothesisTree.from_dict(tree_raw)
        bundles.append(CampaignDBBundle(
            campaign_id=cid,
            tree=tree,
            snapshots=by_snaps.get(cid, camp.get("trajectory") or []),
            baseline_campaign_id=camp.get("baseline_campaign_id"),
        ))
    global_baseline = data.get("baseline_campaign_id")
    if global_baseline:
        for b in bundles:
            if not b.baseline_campaign_id:
                b.baseline_campaign_id = global_baseline
    return bundles
