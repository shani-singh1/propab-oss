#!/usr/bin/env python3
"""
P5/P6 — Collect main demo run artifacts after campaign completion.

Collects: paper, tree, traces, lineage, metrics
Outputs: artifacts/demo/main/ demo report + assessment bundle
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "orchestrator"))

from demo.benchmark.domain import DEMO_DOMAIN
from demo.benchmark.gold import archive_count, gold_campaign_ids, load_baseline_metrics
from demo.benchmark.load import load_campaign_metrics, load_tree_summary_and_findings
from demo.benchmark.report import build_campaign_asset, build_demo_report, write_report
from demo.benchmark.verifier import verify_main


def _campaign_id_from_state(state_path: Path) -> str:
    data = json.loads(state_path.read_text(encoding="utf-8"))
    if data.get("campaign_id"):
        return str(data["campaign_id"])
    campaigns = data.get("campaigns") or []
    if not campaigns:
        raise ValueError(f"No campaign_id in {state_path}")
    return str(campaigns[0]["campaign_id"])


async def _load_raw_tree(campaign_id: str) -> dict[str, Any] | None:
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
                    SELECT hypothesis_tree_json
                    FROM research_campaigns
                    WHERE id = CAST(:id AS uuid)
                    """
                ),
                {"id": campaign_id},
            )
        ).fetchone()
    await engine.dispose()
    if not row or not row[0]:
        return None
    raw = row[0]
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw) if isinstance(raw, dict) else None


async def _event_summary(campaign_id: str) -> dict[str, int]:
    from sqlalchemy import text

    from propab.config import settings
    from propab.db import create_engine, create_session_factory

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    async with session_factory() as db:
        rows = (
            await db.execute(
                text(
                    """
                    SELECT event_type, COUNT(*)::int
                    FROM events
                    WHERE session_id = CAST(:id AS uuid)
                    GROUP BY event_type
                    ORDER BY COUNT(*) DESC
                    """
                ),
                {"id": campaign_id},
            )
        ).fetchall()
    await engine.dispose()
    return {str(r[0]): int(r[1]) for r in rows}


def _lineage_paths(tree: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = tree.get("nodes") or {}
    if isinstance(nodes, dict):
        by_id = {str(k): v for k, v in nodes.items() if isinstance(v, dict)}
    else:
        by_id = {str(n.get("id")): n for n in nodes if isinstance(n, dict) and n.get("id")}

    def path_to_root(node_id: str) -> list[str]:
        chain: list[str] = []
        cur: str | None = node_id
        seen: set[str] = set()
        while cur and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            node = by_id.get(cur)
            if not node:
                break
            cur = node.get("parent_id")
        chain.reverse()
        return chain

    confirmed = [
        n for n in by_id.values()
        if n.get("verdict") == "confirmed"
    ]
    confirmed.sort(key=lambda n: float(n.get("confidence") or 0), reverse=True)

    paths: list[dict[str, Any]] = []
    for n in confirmed[:12]:
        nid = str(n.get("id") or "")
        chain_ids = path_to_root(nid)
        paths.append({
            "node_id": nid,
            "confidence": n.get("confidence"),
            "primary_theme": n.get("primary_theme"),
            "text": str(n.get("text") or "")[:200],
            "lineage_length": len(chain_ids) - 1,
            "path_node_ids": chain_ids,
            "path": [
                {
                    "id": cid,
                    "verdict": by_id.get(cid, {}).get("verdict"),
                    "depth": by_id.get(cid, {}).get("depth"),
                    "text": str(by_id.get(cid, {}).get("text") or "")[:120],
                }
                for cid in chain_ids
            ],
        })
    return paths


async def _harvest_traces(campaign_id: str) -> dict[str, Any]:
    from propab.operator_credit.db_trace_loader import (
        extract_traces_from_db_bundle,
        load_campaign_db_bundle,
    )

    bundle = await load_campaign_db_bundle(campaign_id)
    ledger = extract_traces_from_db_bundle(bundle)
    traces = [t.to_dict() for t in ledger.traces]
    return {
        "campaign_id": campaign_id,
        "bundle": bundle.to_dict(),
        "n_traces": len(traces),
        "traces": traces,
    }


def _main_assessment(
    *,
    campaign_id: str,
    metrics: dict[str, Any],
    verification: dict[str, Any],
    event_summary: dict[str, int],
    report_paths: dict[str, str],
) -> dict[str, Any]:
    passed = verification.get("passed", False)
    status = metrics.get("status", "")
    confirmed = int(metrics.get("total_confirmed") or 0)
    imp = metrics.get("improvement_pct")

    if passed and status == "breakthrough" and confirmed >= 2:
        verdict = "breakthrough_demo"
        assessment = (
            f"Main demo succeeded: breakthrough with {confirmed} confirmed hypotheses, "
            f"{imp}% improvement over baseline, paper generated. "
            "P5 artifacts collected; P6 demo report ready."
        )
    elif passed:
        verdict = "main_complete"
        assessment = (
            "Main demo completed and passed verification checks. "
            "Review findings and demo report for presentation."
        )
    else:
        verdict = "verification_failed"
        assessment = (
            "Main run finished but failed one or more verification checks: "
            f"{verification.get('failures')}. Review artifacts before demo."
        )

    return {
        "main_campaign_id": campaign_id,
        "domain": DEMO_DOMAIN.domain_id,
        "verdict": verdict,
        "status": status,
        "compute_seconds_used": metrics.get("compute_seconds_used"),
        "total_hypotheses": metrics.get("total_hypotheses"),
        "total_confirmed": confirmed,
        "best_metric": metrics.get("best_metric"),
        "improvement_pct": imp,
        "tree_nodes": metrics.get("n_tree_nodes"),
        "has_paper": metrics.get("has_paper"),
        "verification_passed": passed,
        "verification_mode": "main_strict",
        "verification": verification,
        "event_summary": event_summary,
        "assessment": assessment,
        "report_paths": report_paths,
    }


async def _build(
    campaign_id: str,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = await load_baseline_metrics()
    db_metrics = await load_campaign_metrics(campaign_id)
    if not db_metrics:
        raise RuntimeError(f"Campaign not found in Postgres: {campaign_id}")

    tree_summary, findings, paper_url = await load_tree_summary_and_findings(campaign_id)
    verification = verify_main(db_metrics)
    asset = build_campaign_asset(
        db_metrics,
        verification,
        baseline,
        tree_summary=tree_summary,
        top_findings=findings,
        paper_url=paper_url,
    )
    report = build_demo_report(
        [asset],
        gold_corpus_size=len(gold_campaign_ids()),
        archive_size=archive_count(),
    )
    report_paths = write_report(report, out_dir)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(db_metrics.to_dict(), indent=2),
        encoding="utf-8",
    )

    raw_tree = await _load_raw_tree(campaign_id)
    tree_path = out_dir / "hypothesis_tree.json"
    if raw_tree:
        tree_path.write_text(json.dumps(raw_tree, indent=2), encoding="utf-8")

    lineage = _lineage_paths(raw_tree) if raw_tree else []
    lineage_path = out_dir / "lineage.json"
    lineage_path.write_text(
        json.dumps({
            "campaign_id": campaign_id,
            "confirmed_lineages": lineage,
            "tree_summary": tree_summary,
        }, indent=2),
        encoding="utf-8",
    )

    trace_bundle = await _harvest_traces(campaign_id)
    traces_path = out_dir / "traces.json"
    traces_path.write_text(json.dumps(trace_bundle, indent=2), encoding="utf-8")

    event_summary = await _event_summary(campaign_id)
    assessment = _main_assessment(
        campaign_id=campaign_id,
        metrics=db_metrics.to_dict(),
        verification=verification.to_dict(),
        event_summary=event_summary,
        report_paths={k: str(v) for k, v in report_paths.items()},
    )
    assessment_path = out_dir / "main_assessment.json"
    assessment_path.write_text(json.dumps(assessment, indent=2), encoding="utf-8")

    return {
        "campaign_id": campaign_id,
        "verification_passed": verification.passed,
        "verdict": assessment["verdict"],
        "outputs": {
            "demo_report_md": str(report_paths["markdown"]),
            "demo_report_json": str(report_paths["json"]),
            "metrics": str(metrics_path),
            "hypothesis_tree": str(tree_path) if raw_tree else None,
            "lineage": str(lineage_path),
            "traces": str(traces_path),
            "main_assessment": str(assessment_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build P5/P6 assets for completed main demo run")
    parser.add_argument(
        "--state-file",
        default=str(ROOT / "artifacts" / "demo" / "main_latest.json"),
        help="State file from run_demo_main.py",
    )
    parser.add_argument("--campaign-id", help="Override campaign ID")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "artifacts" / "demo" / "main"),
    )
    args = parser.parse_args()

    if args.campaign_id:
        campaign_id = args.campaign_id
    else:
        state = Path(args.state_file)
        if not state.is_file():
            print(f"State file not found: {state}", file=sys.stderr)
            return 1
        campaign_id = _campaign_id_from_state(state)

    result = asyncio.run(_build(campaign_id, Path(args.out_dir)))
    print(json.dumps(result, indent=2))
    return 0 if result["verification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
