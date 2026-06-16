#!/usr/bin/env python3
"""
P6 — Build demo assets from gold corpus campaigns (no new runs required).

Shows: Question → Tree → Verification → Finding → Improvement over baseline
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "orchestrator"))

from demo.benchmark.gold import archive_count, gold_campaign_ids, load_baseline_metrics
from demo.benchmark.load import load_campaign_metrics, load_tree_summary_and_findings
from demo.benchmark.report import build_campaign_asset, build_demo_report, write_report
from demo.benchmark.verifier import verify_main


async def _build(out_dir: Path) -> dict:
    baseline = await load_baseline_metrics()
    ids = gold_campaign_ids()
    assets = []

    for cid in ids:
        metrics = await load_campaign_metrics(cid)
        if not metrics:
            continue
        tree_summary, findings, paper_url = await load_tree_summary_and_findings(cid)
        verification = verify_main(metrics)
        assets.append(build_campaign_asset(
            metrics,
            verification,
            baseline,
            tree_summary=tree_summary,
            top_findings=findings,
            paper_url=paper_url,
        ))

    report = build_demo_report(
        assets,
        gold_corpus_size=len(ids),
        archive_size=archive_count(),
    )
    paths = write_report(report, out_dir)
    return {
        "gold_corpus_size": len(ids),
        "n_assets": len(assets),
        "best_campaign_id": report.best_campaign_id,
        "outputs": {k: str(v) for k, v in paths.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build demo assets from gold corpus")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "artifacts" / "demo"),
    )
    args = parser.parse_args()

    result = asyncio.run(_build(Path(args.out_dir)))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
