"""Campaign era partitioning — separate history from current learning (fixes.md P0–P6)."""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import IntEnum
from pathlib import Path
from typing import Any

from propab.config import settings
from propab.operator_credit.difference_rewards import DifferenceRewardLedger
from propab.operator_credit.operator_statistics import OperatorStatistics, OperatorStatCell
from propab.operator_credit.operator_trace import OperatorTraceLedger


class EraId(IntEnum):
    ERA_0 = 0  # pre-literature fixes, contaminated priors
    ERA_1 = 1  # literature repaired, claim typing, frontier fixes
    ERA_2 = 2  # replay + simulator
    ERA_3 = 3  # operator credit
    ERA_4 = 4  # DB-backed traces (current architecture)


ERA_LABELS: dict[int, str] = {
    0: "pre_fix_contaminated",
    1: "literature_claim_typing",
    2: "replay_simulator",
    3: "operator_credit",
    4: "db_backed_traces",
}

ERA_DESCRIPTIONS: dict[int, str] = {
    0: "Pre-literature fixes; contaminated priors; ML drift; fake timeout bug",
    1: "Literature repaired; claim typing; frontier fixes",
    2: "Replay + simulator calibration",
    3: "Operator credit assignment",
    4: "DB-backed traces; current architecture",
}

# Recent eras dominate; very old eras decay (P2).
ERA_BASE_TRUST: dict[int, float] = {
    EraId.ERA_0: 0.05,
    EraId.ERA_1: 0.15,
    EraId.ERA_2: 0.40,
    EraId.ERA_3: 0.75,
    EraId.ERA_4: 1.00,
}

CURRENT_SIMULATOR_VERSION = "sim_v2"
CURRENT_SEARCH_STATE_VERSION = "SearchStateV3"
CURRENT_OPERATOR_CREDIT_GENERATION = 1
CURRENT_TRACE_SOURCE = "db"


def era_store_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "campaign_eras.json"


def gold_corpus_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "gold_corpus.json"


def current_git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip()[:12] or None
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return None


@dataclass
class CampaignEra:
    """Architecture era definition (P0)."""

    era_id: int
    label: str
    description: str
    git_commit: str | None = None
    architecture_features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CampaignEraMetadata:
    """Per-campaign era metadata (P1)."""

    campaign_id: str
    era_id: int
    started_at: str | None = None
    git_commit: str | None = None
    simulator_version: str = "unknown"
    search_state_version: str = "unknown"
    trace_source: str = "unknown"
    policy_generation: int = 0
    policy_id: str | None = None
    operator_credit_generation: int = 0
    budget_bucket: str | None = None
    domain_bucket: str | None = None
    total_confirmed: int = 0
    total_hypotheses: int = 0
    has_paper: bool = False
    max_closure_ratio: float = 0.0
    n_tool_calls: int = 0
    n_snapshots: int = 0
    trust_weight: float = 0.0
    quality_score: float = 0.0
    in_gold_corpus: bool = False
    in_archive: bool = False
    architecture_features: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        return None


def classify_era(meta: CampaignEraMetadata) -> int:
    """
    Heuristic era assignment from observable campaign metadata.
    Uses date breakpoints + architecture feature flags.
    """
    started = _parse_date(meta.started_at)
    feats = meta.architecture_features

    if started and started < date(2026, 5, 10):
        return EraId.ERA_0
    if started and started < date(2026, 5, 29):
        return EraId.ERA_1
    if started and started < date(2026, 6, 6):
        return EraId.ERA_2

    # June 6+ — current architecture generations
    if (
        feats.get("db_traces")
        or (meta.n_tool_calls >= 50 and meta.n_snapshots >= 10 and meta.policy_id)
    ):
        return EraId.ERA_4
    if meta.policy_id or feats.get("operator_credit"):
        return EraId.ERA_3
    return EraId.ERA_2


def compute_quality_score(meta: CampaignEraMetadata) -> float:
    """Quality heuristic for gold corpus selection (P5)."""
    hyp = max(1, meta.total_hypotheses)
    closure = meta.max_closure_ratio
    confirmed_rate = meta.total_confirmed / hyp
    paper_bonus = 2.0 if meta.has_paper else 0.0
    tool_bonus = min(1.0, meta.n_tool_calls / 300.0)
    snap_bonus = min(0.5, meta.n_snapshots / 40.0)
    return round(
        confirmed_rate * 3.0 + closure * 2.0 + paper_bonus + tool_bonus + snap_bonus,
        4,
    )


def trust_weight(meta: CampaignEraMetadata, *, reference: date | None = None) -> float:
    """
    Trust weight for a campaign (P2).
    Recent eras dominate; quality modulates within era.
    """
    base = ERA_BASE_TRUST.get(meta.era_id, 0.1)
    quality_factor = min(1.0, 0.5 + meta.quality_score / 5.0)

    started = _parse_date(meta.started_at)
    age_decay = 1.0
    if started and reference:
        age_days = max(0, (reference - started).days)
        age_decay = math.exp(-age_days / 120.0)

    return round(base * quality_factor * age_decay, 4)


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def build_campaign_metadata(raw: dict[str, Any]) -> CampaignEraMetadata:
    """Build metadata record from DB row dict."""
    n_tools = int(raw.get("n_tool_calls") or 0)
    n_snaps = int(raw.get("n_snapshots") or 0)
    policy_id = raw.get("policy_id")
    eval_mode = raw.get("entropy_eval_mode")

    architecture_features = {
        "literature_repaired": bool(raw.get("total_confirmed", 0) > 0 or n_tools > 20),
        "claim_typing": n_tools > 50,
        "simulator_eval": eval_mode == "dynamics" or bool(policy_id),
        "operator_credit": bool(policy_id) and n_snaps >= 10,
        "db_traces": n_tools >= 50 and n_snaps >= 10,
    }

    meta = CampaignEraMetadata(
        campaign_id=str(raw["campaign_id"]),
        era_id=0,
        started_at=str(raw.get("started_at") or "") or None,
        git_commit=raw.get("git_commit"),
        simulator_version=raw.get("simulator_version") or CURRENT_SIMULATOR_VERSION,
        search_state_version=(
            CURRENT_SEARCH_STATE_VERSION if n_snaps >= 10 else "legacy"
        ),
        trace_source="db" if n_tools >= 50 else ("tree" if n_tools > 0 else "snapshot"),
        policy_generation=_parse_int(raw.get("policy_generation")),
        policy_id=policy_id,
        operator_credit_generation=(
            CURRENT_OPERATOR_CREDIT_GENERATION if architecture_features["operator_credit"] else 0
        ),
        budget_bucket=raw.get("budget_bucket"),
        domain_bucket=raw.get("domain_bucket"),
        total_confirmed=int(raw.get("total_confirmed") or 0),
        total_hypotheses=int(raw.get("total_hypotheses") or 0),
        has_paper=bool(int(raw.get("paper_count") or 0) > 0),
        max_closure_ratio=float(raw.get("max_closure_ratio") or 0),
        n_tool_calls=n_tools,
        n_snapshots=n_snaps,
        architecture_features=architecture_features,
    )
    meta.era_id = classify_era(meta)
    meta.quality_score = compute_quality_score(meta)
    meta.trust_weight = trust_weight(meta)
    return meta


@dataclass
class EraLocalStatistics:
    """P(success|operator,state) computed inside each era (P3)."""

    era_id: int
    cells: dict[str, OperatorStatCell] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "era_id": self.era_id,
            "label": ERA_LABELS.get(self.era_id, "unknown"),
            "n_cells": len(self.cells),
            "cells": {k: v.to_dict() for k, v in self.cells.items()},
        }


@dataclass
class CrossEraComparison:
    """Operator ranking stability across eras (P4)."""

    family: str
    state_bucket: str
    rankings_by_era: dict[int, list[str]] = field(default_factory=dict)
    stable_operators: list[str] = field(default_factory=list)
    unstable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GoldCorpus:
    """Latest-architecture demo/training campaigns only (P5)."""

    campaign_ids: list[str] = field(default_factory=list)
    min_era: int = EraId.ERA_3
    selection_criteria: dict[str, Any] = field(default_factory=dict)
    entries: list[CampaignEraMetadata] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_ids": self.campaign_ids,
            "min_era": self.min_era,
            "selection_criteria": self.selection_criteria,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldCorpus:
        entries = [CampaignEraMetadata(**e) for e in (data.get("entries") or [])]
        return cls(
            campaign_ids=list(data.get("campaign_ids") or []),
            min_era=int(data.get("min_era") or EraId.ERA_3),
            selection_criteria=dict(data.get("selection_criteria") or {}),
            entries=entries,
        )

    def save(self, path: Path | None = None) -> Path:
        p = path or gold_corpus_path()
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> GoldCorpus:
        p = path or gold_corpus_path()
        if not p.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return cls()


@dataclass
class ExperienceArchive:
    """Historical campaigns — experience archive, not policy training (P6)."""

    campaign_ids: list[str] = field(default_factory=list)
    era_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CampaignEraPartition:
    """Full era partition for all campaigns."""

    git_commit: str | None = None
    eras: list[CampaignEra] = field(default_factory=list)
    campaigns: list[CampaignEraMetadata] = field(default_factory=list)
    era_local_stats: list[EraLocalStatistics] = field(default_factory=list)
    cross_era_comparisons: list[CrossEraComparison] = field(default_factory=list)
    gold_corpus: GoldCorpus = field(default_factory=GoldCorpus)
    archive: ExperienceArchive = field(default_factory=ExperienceArchive)

    def by_era(self, era_id: int) -> list[CampaignEraMetadata]:
        return [c for c in self.campaigns if c.era_id == era_id]

    def trusted_campaign_ids(self, *, min_trust: float = 0.3) -> list[str]:
        return [c.campaign_id for c in self.campaigns if c.trust_weight >= min_trust]

    def to_dict(self) -> dict[str, Any]:
        return {
            "git_commit": self.git_commit,
            "eras": [e.to_dict() for e in self.eras],
            "campaigns": [c.to_dict() for c in self.campaigns],
            "era_local_stats": [s.to_dict() for s in self.era_local_stats],
            "cross_era_comparisons": [c.to_dict() for c in self.cross_era_comparisons],
            "gold_corpus": self.gold_corpus.to_dict(),
            "archive": self.archive.to_dict(),
            "summary": self.summary(),
        }

    def summary(self) -> dict[str, Any]:
        by_era: dict[str, int] = {}
        for c in self.campaigns:
            label = ERA_LABELS.get(c.era_id, str(c.era_id))
            by_era[label] = by_era.get(label, 0) + 1
        return {
            "n_campaigns": len(self.campaigns),
            "by_era": by_era,
            "n_gold": len(self.gold_corpus.campaign_ids),
            "n_archive": len(self.archive.campaign_ids),
            "mean_trust_by_era": {
                ERA_LABELS.get(eid, str(eid)): round(
                    sum(c.trust_weight for c in self.campaigns if c.era_id == eid)
                    / max(1, len(self.by_era(eid))),
                    4,
                )
                for eid in sorted({c.era_id for c in self.campaigns})
            },
        }

    def save(self, path: Path | None = None) -> Path:
        p = path or era_store_path()
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> CampaignEraPartition:
        p = path or era_store_path()
        if not p.is_file():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls()
        campaigns = [CampaignEraMetadata(**c) for c in (data.get("campaigns") or [])]
        return cls(
            git_commit=data.get("git_commit"),
            eras=[CampaignEra(**e) for e in (data.get("eras") or [])],
            campaigns=campaigns,
            era_local_stats=[
                EraLocalStatistics(
                    era_id=s["era_id"],
                    cells={
                        k: OperatorStatCell(**v)
                        for k, v in (s.get("cells") or {}).items()
                    },
                )
                for s in (data.get("era_local_stats") or [])
            ],
            cross_era_comparisons=[
                CrossEraComparison(**c) for c in (data.get("cross_era_comparisons") or [])
            ],
            gold_corpus=GoldCorpus.from_dict(data.get("gold_corpus") or {}),
            archive=ExperienceArchive(**(data.get("archive") or {})),
        )


def build_era_definitions() -> list[CampaignEra]:
    commit = current_git_commit()
    return [
        CampaignEra(
            era_id=eid,
            label=ERA_LABELS[eid],
            description=ERA_DESCRIPTIONS[eid],
            git_commit=commit if eid >= EraId.ERA_3 else None,
            architecture_features={
                "simulator_version": CURRENT_SIMULATOR_VERSION if eid >= EraId.ERA_2 else None,
                "search_state_version": CURRENT_SEARCH_STATE_VERSION if eid >= EraId.ERA_3 else None,
                "trace_source": CURRENT_TRACE_SOURCE if eid >= EraId.ERA_4 else None,
                "operator_credit_generation": CURRENT_OPERATOR_CREDIT_GENERATION if eid >= EraId.ERA_3 else 0,
            },
        )
        for eid in EraId
    ]


def compute_era_local_statistics(
    *,
    partition: CampaignEraPartition,
    traces: OperatorTraceLedger,
    credits: DifferenceRewardLedger | None = None,
) -> list[EraLocalStatistics]:
    """Compute operator statistics separately per era (P3)."""
    campaign_era = {c.campaign_id: c.era_id for c in partition.campaigns}
    campaign_trust = {c.campaign_id: c.trust_weight for c in partition.campaigns}
    by_era: dict[int, OperatorStatistics] = {}

    for era_id in {c.era_id for c in partition.campaigns}:
        by_era[era_id] = OperatorStatistics()

    era_traces = OperatorTraceLedger()
    for trace in traces.traces:
        era_id = campaign_era.get(trace.campaign_id)
        if era_id is None:
            continue
        weight = campaign_trust.get(trace.campaign_id, 0.0)
        if weight < 0.05:
            continue
        era_traces.add(trace)

    for era_id, stats in by_era.items():
        era_campaign_ids = {c.campaign_id for c in partition.campaigns if c.era_id == era_id}
        era_trace_ledger = OperatorTraceLedger()
        for t in era_traces.traces:
            if t.campaign_id in era_campaign_ids:
                era_trace_ledger.add(t)
        era_credits = None
        if credits:
            from propab.operator_credit.difference_rewards import DifferenceRewardLedger as DRL

            era_credits = DRL()
            for cr in credits.credits:
                if cr.campaign_id in era_campaign_ids:
                    era_credits.add(cr)
        stats.update_from_traces(era_trace_ledger, era_credits)

    return [
        EraLocalStatistics(era_id=eid, cells=dict(stats.cells))
        for eid, stats in sorted(by_era.items())
    ]


def compute_cross_era_comparisons(
    era_stats: list[EraLocalStatistics],
    *,
    top_k: int = 3,
) -> list[CrossEraComparison]:
    """Measure operator ranking stability across eras (P4)."""
    family_buckets: dict[tuple[str, str], dict[int, list[tuple[str, float]]]] = {}

    for era in era_stats:
        for cell in era.cells.values():
            key = (cell.family, cell.state_bucket)
            score = cell.mean_contribution + cell.p_success - cell.p_timeout
            family_buckets.setdefault(key, {}).setdefault(era.era_id, []).append(
                (cell.operator, score)
            )

    comparisons: list[CrossEraComparison] = []
    for (family, bucket), era_rankings in family_buckets.items():
        rankings_by_era: dict[int, list[str]] = {}
        for era_id, ops in era_rankings.items():
            ops.sort(key=lambda x: x[1], reverse=True)
            rankings_by_era[era_id] = [op for op, _ in ops[:top_k]]

        all_tops: list[str] = []
        for ranks in rankings_by_era.values():
            all_tops.extend(ranks)
        stable = [op for op in set(all_tops) if all_tops.count(op) >= 2]

        comparisons.append(CrossEraComparison(
            family=family,
            state_bucket=bucket,
            rankings_by_era=rankings_by_era,
            stable_operators=sorted(stable),
            unstable=len(set(tuple(r) for r in rankings_by_era.values())) > 2,
        ))
    return comparisons


def select_gold_corpus(
    campaigns: list[CampaignEraMetadata],
    *,
    min_era: int = EraId.ERA_3,
    max_size: int = 10,
    min_quality: float = 1.0,
) -> GoldCorpus:
    """
    Select demo/training gold set from latest architecture (P5).
    Trust 5 latest-architecture campaigns over 50 obsolete ones.
    """
    candidates = [
        c for c in campaigns
        if c.era_id >= min_era and c.quality_score >= min_quality
    ]
    candidates.sort(
        key=lambda c: (c.era_id, c.quality_score, c.trust_weight, c.total_confirmed),
        reverse=True,
    )
    selected = candidates[:max_size]

    for c in campaigns:
        c.in_gold_corpus = c.campaign_id in {s.campaign_id for s in selected}
        c.in_archive = not c.in_gold_corpus

    return GoldCorpus(
        campaign_ids=[c.campaign_id for c in selected],
        min_era=min_era,
        selection_criteria={
            "min_era": min_era,
            "max_size": max_size,
            "min_quality": min_quality,
        },
        entries=selected,
    )


def build_experience_archive(campaigns: list[CampaignEraMetadata]) -> ExperienceArchive:
    """All non-gold campaigns become historical archive (P6)."""
    archive_ids = [c.campaign_id for c in campaigns if c.in_archive]
    era_counts: dict[str, int] = {}
    for c in campaigns:
        if c.in_archive:
            label = ERA_LABELS.get(c.era_id, str(c.era_id))
            era_counts[label] = era_counts.get(label, 0) + 1
    return ExperienceArchive(campaign_ids=archive_ids, era_counts=era_counts)


def build_weighted_statistics(
    era_stats: list[EraLocalStatistics],
    partition: CampaignEraPartition,
) -> OperatorStatistics:
    """Merge era-local stats with trust weights for gold-corpus priors."""
    merged = OperatorStatistics()
    trust_by_era = {
        eid: ERA_BASE_TRUST.get(eid, 0.1)
        for eid in {c.era_id for c in partition.campaigns}
    }

    for era in era_stats:
        weight = trust_by_era.get(era.era_id, 0.1)
        for key, cell in era.cells.items():
            existing = merged.cells.get(key)
            if existing:
                n_total = existing.n + cell.n
                if n_total == 0:
                    continue
                merged.cells[key] = OperatorStatCell(
                    family=cell.family,
                    operator=cell.operator,
                    state_bucket=cell.state_bucket,
                    n=n_total,
                    p_success=round(
                        (existing.p_success * existing.n + cell.p_success * cell.n * weight) / n_total,
                        4,
                    ),
                    p_refute=round(
                        (existing.p_refute * existing.n + cell.p_refute * cell.n * weight) / n_total,
                        4,
                    ),
                    p_timeout=round(
                        (existing.p_timeout * existing.n + cell.p_timeout * cell.n * weight) / n_total,
                        4,
                    ),
                    mean_contribution=round(
                        (existing.mean_contribution * existing.n + cell.mean_contribution * cell.n * weight)
                        / n_total,
                        4,
                    ),
                )
            else:
                merged.cells[key] = OperatorStatCell(
                    family=cell.family,
                    operator=cell.operator,
                    state_bucket=cell.state_bucket,
                    n=cell.n,
                    p_success=round(cell.p_success * weight, 4),
                    p_refute=round(cell.p_refute * weight, 4),
                    p_timeout=round(cell.p_timeout * weight, 4),
                    mean_contribution=round(cell.mean_contribution * weight, 4),
                )
    return merged


def build_gold_statistics(
    traces: OperatorTraceLedger,
    credits: DifferenceRewardLedger | None,
    gold: GoldCorpus,
) -> OperatorStatistics:
    """Statistics computed only from gold corpus campaigns."""
    gold_ids = set(gold.campaign_ids)
    gold_traces = OperatorTraceLedger()
    for t in traces.traces:
        if t.campaign_id in gold_ids:
            gold_traces.add(t)
    gold_credits = None
    if credits:
        from propab.operator_credit.difference_rewards import DifferenceRewardLedger as DRL

        gold_credits = DRL()
        for cr in credits.credits:
            if cr.campaign_id in gold_ids:
                gold_credits.add(cr)
    stats = OperatorStatistics()
    stats.update_from_traces(gold_traces, gold_credits)
    return stats
