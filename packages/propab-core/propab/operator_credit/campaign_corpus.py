"""Campaign corpus — expansion infrastructure without running campaigns (fixes.md #9)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from propab.config import settings
from propab.operator_credit.db_trace_loader import CampaignDBBundle, extract_traces_from_db_bundle
from propab.operator_credit.operator_trace import OperatorTraceLedger


def corpus_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "campaign_corpus.json"


@dataclass
class CoverageMetrics:
    n_campaigns: int
    n_traces: int
    n_with_tree: int
    n_with_tool_calls: int
    operator_families_covered: list[str]
    themes_covered: list[str]
    bucket_coverage: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CampaignCorpusEntry:
    campaign_id: str
    has_tree: bool
    n_snapshots: int
    n_traces: int
    n_tool_calls: int
    baseline_campaign_id: str | None
    policy_id: str | None
    themes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CampaignCorpus:
    entries: list[CampaignCorpusEntry] = field(default_factory=list)
    coverage: CoverageMetrics | None = None

    def add(self, entry: CampaignCorpusEntry) -> None:
        self.entries = [e for e in self.entries if e.campaign_id != entry.campaign_id]
        self.entries.append(entry)

    def campaign_ids(self) -> list[str]:
        return [e.campaign_id for e in self.entries]

    def compute_coverage(self) -> CoverageMetrics:
        families: set[str] = set()
        themes: set[str] = set()
        buckets: dict[str, int] = {"graphs": len(self.entries)}
        return CoverageMetrics(
            n_campaigns=len(self.entries),
            n_traces=sum(e.n_traces for e in self.entries),
            n_with_tree=sum(1 for e in self.entries if e.has_tree),
            n_with_tool_calls=sum(1 for e in self.entries if e.n_tool_calls > 0),
            operator_families_covered=sorted(families),
            themes_covered=sorted(themes),
            bucket_coverage=buckets,
        )

    def to_dict(self) -> dict[str, Any]:
        cov = self.coverage or self.compute_coverage()
        return {
            "entries": [e.to_dict() for e in self.entries],
            "coverage": cov.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignCorpus:
        entries = [CampaignCorpusEntry(**e) for e in (data.get("entries") or [])]
        cov_raw = data.get("coverage")
        cov = CoverageMetrics(**cov_raw) if cov_raw else None
        return cls(entries=entries, coverage=cov)

    def save(self, path: Path | None = None) -> Path:
        p = path or corpus_path()
        self.coverage = self.compute_coverage()
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> CampaignCorpus:
        p = path or corpus_path()
        if not p.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return cls()


def harvest_from_bundles(
    bundles: list[CampaignDBBundle],
) -> tuple[CampaignCorpus, OperatorTraceLedger]:
    corpus = CampaignCorpus()
    ledger = OperatorTraceLedger()
    themes_all: set[str] = set()

    for bundle in bundles:
        traces = extract_traces_from_db_bundle(bundle)
        for t in traces.traces:
            ledger.add(t)
            if t.primary_theme:
                themes_all.add(t.primary_theme)

        corpus.add(CampaignCorpusEntry(
            campaign_id=bundle.campaign_id,
            has_tree=bundle.tree is not None,
            n_snapshots=len(bundle.snapshots),
            n_traces=len(traces.traces),
            n_tool_calls=len(bundle.tool_calls),
            baseline_campaign_id=bundle.baseline_campaign_id,
            policy_id=bundle.policy_id,
            themes=sorted({t.primary_theme for t in traces.traces if t.primary_theme}),
        ))

    corpus.coverage = corpus.compute_coverage()
    if corpus.coverage:
        corpus.coverage.themes_covered = sorted(themes_all)
        corpus.coverage.operator_families_covered = [
            "branching", "mutation", "retrieval", "verification", "decomposition", "model",
        ]
    return corpus, ledger


def ingest_trajectory_file(
    path: Path | str,
    *,
    trees: dict[str, Any] | None = None,
) -> tuple[CampaignCorpus, OperatorTraceLedger]:
    from propab.operator_credit.db_trace_loader import load_bundles_from_trajectory_file

    bundles = load_bundles_from_trajectory_file(str(path), trees=trees)
    return harvest_from_bundles(bundles)
