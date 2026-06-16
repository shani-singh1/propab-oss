"""P3/P6 — Demo benchmark and asset reports."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from demo.benchmark.domain import DEMO_DOMAIN
from demo.benchmark.metric import CampaignMetrics, compare_to_baseline
from demo.benchmark.verifier import VerificationResult


@dataclass
class DemoFinding:
    node_id: str
    text: str
    verdict: str
    theme: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DemoCampaignAsset:
    campaign_id: str
    question: str
    metrics: dict[str, Any]
    verification: dict[str, Any]
    baseline_comparison: dict[str, Any]
    tree_summary: dict[str, Any]
    top_findings: list[dict[str, Any]] = field(default_factory=list)
    paper_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DemoReport:
    domain: dict[str, Any]
    gold_corpus_size: int
    archive_size: int
    campaigns: list[DemoCampaignAsset] = field(default_factory=list)
    best_campaign_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "gold_corpus_size": self.gold_corpus_size,
            "archive_size": self.archive_size,
            "best_campaign_id": self.best_campaign_id,
            "campaigns": [c.to_dict() for c in self.campaigns],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Propab Demo — Graph Contagion Benchmark",
            "",
            f"**Domain:** {self.domain.get('domain_id')}",
            "",
            f"**Question:** {self.domain.get('question')}",
            "",
            f"Gold corpus: {self.gold_corpus_size} campaigns | Archive: {self.archive_size} campaigns",
            "",
        ]
        if self.best_campaign_id:
            lines.append(f"**Best run:** `{self.best_campaign_id}`")
            lines.append("")

        for asset in self.campaigns:
            lines.extend(_asset_markdown(asset))
            lines.append("---")
            lines.append("")

        lines.extend([
            "## What Propab Discovered",
            "",
            "The demo shows autonomous research on **network contagion**:",
            "which structural properties (degree, spectral gap, clustering) ",
            "most strongly determine spreading speed under competing diffusion models.",
            "",
        ])
        return "\n".join(lines)


def _asset_markdown(asset: DemoCampaignAsset) -> list[str]:
    m = asset.metrics
    bc = asset.baseline_comparison
    lines = [
        f"## Campaign `{asset.campaign_id[:8]}…`",
        "",
        "### Question → Tree → Verification → Finding",
        "",
        f"- **Confirmed:** {m.get('total_confirmed')} / {m.get('total_hypotheses')} hypotheses",
        f"- **Best metric ({DEMO_DOMAIN.primary_metric}):** {m.get('best_metric')}",
        f"- **Improvement over baseline:** {bc.get('improvement_pct')}% (Δ={bc.get('delta')})",
        f"- **Tree:** {asset.tree_summary.get('total_nodes')} nodes, "
        f"depth {asset.tree_summary.get('max_depth')}",
        f"- **Paper:** {'yes' if asset.paper_url else 'no'}",
        "",
    ]
    if asset.top_findings:
        lines.append("### Top findings")
        for f in asset.top_findings[:5]:
            lines.append(f"- **{f.get('verdict')}** ({f.get('theme')}): {f.get('text', '')[:120]}")
        lines.append("")
    return lines


def build_demo_report(
    assets: list[DemoCampaignAsset],
    *,
    gold_corpus_size: int,
    archive_size: int,
) -> DemoReport:
    best = None
    best_score = -1.0
    for asset in assets:
        imp = asset.baseline_comparison.get("improvement_pct")
        conf = asset.metrics.get("total_confirmed") or 0
        score = (float(imp) if imp is not None else 0) + conf * 0.01
        if score > best_score:
            best_score = score
            best = asset.campaign_id

    return DemoReport(
        domain=DEMO_DOMAIN.to_dict(),
        gold_corpus_size=gold_corpus_size,
        archive_size=archive_size,
        campaigns=assets,
        best_campaign_id=best,
    )


def build_campaign_asset(
    metrics: CampaignMetrics,
    verification: VerificationResult,
    baseline: dict[str, Any],
    *,
    tree_summary: dict[str, Any] | None = None,
    top_findings: list[DemoFinding] | None = None,
    paper_url: str | None = None,
) -> DemoCampaignAsset:
    return DemoCampaignAsset(
        campaign_id=metrics.campaign_id,
        question=metrics.question or DEMO_DOMAIN.question,
        metrics=metrics.to_dict(),
        verification=verification.to_dict(),
        baseline_comparison=compare_to_baseline(metrics, baseline),
        tree_summary=tree_summary or {},
        top_findings=[f.to_dict() for f in (top_findings or [])],
        paper_url=paper_url,
    )


def write_report(report: DemoReport, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "demo_report.json"
    md_path = out_dir / "demo_report.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
