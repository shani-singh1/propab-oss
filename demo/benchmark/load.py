"""Load campaign data from Postgres for demo benchmark."""
from __future__ import annotations

import json
from typing import Any

from demo.benchmark.metric import CampaignMetrics, metrics_from_db_row
from demo.benchmark.report import DemoFinding


async def load_campaign_metrics(campaign_id: str) -> CampaignMetrics | None:
    from sqlalchemy import text

    from propab.config import settings
    from propab.db import create_engine, create_session_factory

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    async with session_factory() as db:
        row = (
            await db.execute(
                text(
                    """
                    SELECT
                        rc.id::text AS campaign_id,
                        rc.question,
                        rc.status,
                        rc.baseline_metric,
                        rc.best_metric,
                        rc.improvement_pct,
                        rc.total_confirmed,
                        rc.total_hypotheses,
                        rc.compute_seconds_used,
                        rc.hypothesis_tree_json,
                        (SELECT COUNT(*) FROM events e
                         WHERE e.session_id = rc.id AND e.event_type = 'paper.ready') AS paper_count,
                        (SELECT MAX((e.payload_json->>'closure_ratio')::float)
                         FROM events e
                         WHERE e.session_id = rc.id
                           AND e.step = 'campaign.frontier_snapshot'
                           AND e.payload_json->>'closure_ratio' IS NOT NULL) AS max_closure_ratio,
                        (SELECT payload_json->>'pdf_url' FROM events e
                         WHERE e.session_id = rc.id AND e.event_type = 'paper.ready'
                         ORDER BY e.created_at DESC LIMIT 1) AS paper_url
                    FROM research_campaigns rc
                    WHERE rc.id = CAST(:id AS uuid)
                    """
                ),
                {"id": campaign_id},
            )
        ).mappings().first()
    await engine.dispose()
    if not row:
        return None

    tree = row.get("hypothesis_tree_json")
    if isinstance(tree, str):
        tree = json.loads(tree)
    n_nodes, n_frontier, max_depth = _tree_stats(tree if isinstance(tree, dict) else None)

    d = dict(row)
    d["n_tree_nodes"] = n_nodes
    d["n_frontier"] = n_frontier
    d["max_depth"] = max_depth
    return metrics_from_db_row(d)


async def load_tree_summary_and_findings(
    campaign_id: str,
) -> tuple[dict[str, Any], list[DemoFinding], str | None]:
    from sqlalchemy import text

    from propab.config import settings
    from propab.db import create_engine, create_session_factory

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    tree_summary: dict[str, Any] = {}
    findings: list[DemoFinding] = []
    paper_url: str | None = None

    async with session_factory() as db:
        row = (
            await db.execute(
                text(
                    """
                    SELECT hypothesis_tree_json,
                           (SELECT payload_json->>'pdf_url' FROM events e
                            WHERE e.session_id = rc.id AND e.event_type = 'paper.ready'
                            ORDER BY e.created_at DESC LIMIT 1) AS paper_url
                    FROM research_campaigns rc
                    WHERE rc.id = CAST(:id AS uuid)
                    """
                ),
                {"id": campaign_id},
            )
        ).mappings().first()
    await engine.dispose()

    if not row:
        return tree_summary, findings, paper_url

    paper_url = row.get("paper_url")
    raw = row.get("hypothesis_tree_json")
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        return tree_summary, findings, paper_url

    nodes = raw.get("nodes") or {}
    if isinstance(nodes, dict):
        node_list = list(nodes.values())
    else:
        node_list = nodes

    n_nodes, n_frontier, max_depth = _tree_stats(raw)
    confirmed = [n for n in node_list if isinstance(n, dict) and n.get("verdict") == "confirmed"]
    confirmed.sort(key=lambda n: float(n.get("confidence") or 0), reverse=True)

    for n in confirmed[:8]:
        findings.append(DemoFinding(
            node_id=str(n.get("id") or ""),
            text=str(n.get("text") or "")[:300],
            verdict="confirmed",
            theme=str(n.get("primary_theme") or "general"),
            confidence=float(n.get("confidence") or 0),
        ))

    tree_summary = {
        "total_nodes": n_nodes,
        "max_depth": max_depth,
        "frontier_size": n_frontier,
        "confirmed_count": len(confirmed),
    }
    return tree_summary, findings, paper_url


def _tree_stats(tree: dict[str, Any] | None) -> tuple[int, int, int]:
    if not tree:
        return 0, 0, 0
    nodes = tree.get("nodes") or {}
    if isinstance(nodes, dict):
        node_list = list(nodes.values())
    else:
        node_list = list(nodes)
    frontier = tree.get("frontier") or []
    depths = [int(n.get("depth") or 0) for n in node_list if isinstance(n, dict)]
    return len(node_list), len(frontier), max(depths) if depths else 0
