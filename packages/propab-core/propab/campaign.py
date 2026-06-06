from __future__ import annotations

"""Campaign model — the persistent, long-running research primitive.

A campaign wraps the session loop with:
- A HypothesisTree that grows as findings accumulate
- A BreakthroughCriteria that defines when to stop with success
- A measured baseline (never assumed)
- Checkpoint/resume via the research_campaigns DB table
"""

import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from propab.hypothesis_tree import HypothesisTree
from propab.metric_normalize import normalize_accuracy_metric

# Default plausibility ceiling for accuracy-style metrics. A single-experiment result at or
# above this (after normalization) is usually instrumentation / unit confusion rather than a
# real win, so accuracy criteria adopt it by default. It is an explicit, overridable field on
# BreakthroughCriteria (set plausibility_max=None to disable, or a higher value for easy tasks) —
# NOT a hardwired assumption in the comparison logic, so the engine stays domain-agnostic.
ACC_METRIC_PLAUSIBILITY_MAX = 0.975


def _wall_seconds_since_started(started_at: str) -> float | None:
    """Wall-clock seconds since campaign ``started_at`` (for API snapshots and budget)."""
    if not (started_at or "").strip():
        return None
    try:
        s = started_at.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        t0 = datetime.fromisoformat(s)
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=UTC)
        return max(0.0, (datetime.now(UTC) - t0).total_seconds())
    except Exception:
        return None


# ── Breakthrough criteria ────────────────────────────────────────────────────

@dataclass
class BreakthroughCriteria:
    """
    Defines what counts as a breakthrough for this campaign.
    The baseline_value is measured by the system at campaign start.
    """

    metric_name: str                   # "val_accuracy" | "loss" | "flops" | custom
    baseline_value: float = 0.0        # filled in after baseline measurement
    improvement_threshold: float = 0.05  # 0.05 = 5% required
    direction: str = "higher_is_better"  # "higher_is_better" | "lower_is_better"
    min_confidence: float = 0.85       # significance confidence required
    min_replications: int = 3          # confirmed siblings before declaring
    # Optional declared ceiling: a higher_is_better result at/above this is rejected as
    # implausible (instrumentation guard). None disables the guard. Domain-agnostic.
    plausibility_max: float | None = None

    def _is_implausible(self, metric_val: float) -> bool:
        return (
            self.direction == "higher_is_better"
            and self.plausibility_max is not None
            and metric_val >= self.plausibility_max
        )

    def is_breakthrough(self, finding: dict[str, Any]) -> bool:
        """Return True only when ALL criteria are met."""
        if finding.get("confidence", 0.0) < self.min_confidence:
            return False
        if finding.get("replication_count", 1) < self.min_replications:
            return False
        metric_val = finding.get("metric_value") or finding.get(self.metric_name)
        if metric_val is None:
            return False
        try:
            metric_val = float(metric_val)
        except (TypeError, ValueError):
            return False
        metric_val = normalize_accuracy_metric(self.metric_name, metric_val)
        if metric_val is None:
            return False
        if self._is_implausible(metric_val):
            return False
        base = self.baseline_value
        if abs(base) < 1e-12:
            return False
        denom = max(abs(base), 1e-9)
        improvement = (metric_val - base) / denom
        if self.direction == "higher_is_better":
            return improvement >= self.improvement_threshold
        else:
            return improvement <= -self.improvement_threshold

    def improvement_pct(self, metric_value: float) -> float:
        """Return signed fractional improvement over baseline (positive = better for higher_is_better)."""
        mv = normalize_accuracy_metric(self.metric_name, float(metric_value))
        if mv is None:
            return 0.0
        base = self.baseline_value
        if abs(base) < 1e-12:
            return 0.0
        denom = max(abs(base), 1e-9)
        raw = (mv - base) / denom
        return raw if self.direction == "higher_is_better" else -raw

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "baseline_value": self.baseline_value,
            "improvement_threshold": self.improvement_threshold,
            "direction": self.direction,
            "min_confidence": self.min_confidence,
            "min_replications": self.min_replications,
            "plausibility_max": self.plausibility_max,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BreakthroughCriteria":
        return cls(
            metric_name=data["metric_name"],
            baseline_value=data.get("baseline_value", 0.0),
            improvement_threshold=data.get("improvement_threshold", 0.05),
            direction=data.get("direction", "higher_is_better"),
            min_confidence=data.get("min_confidence", 0.85),
            min_replications=data.get("min_replications", 3),
            plausibility_max=data.get("plausibility_max"),
        )

    @classmethod
    def default_accuracy(cls) -> "BreakthroughCriteria":
        return cls(
            metric_name="val_accuracy",
            direction="higher_is_better",
            improvement_threshold=0.05,
            plausibility_max=ACC_METRIC_PLAUSIBILITY_MAX,
        )

    @classmethod
    def default_loss(cls) -> "BreakthroughCriteria":
        return cls(metric_name="val_loss", direction="lower_is_better", improvement_threshold=0.05)


# ── Campaign status constants ────────────────────────────────────────────────

STATUS_ACTIVE = "active"
STATUS_PAUSED = "paused"
STATUS_BREAKTHROUGH = "breakthrough"
STATUS_BUDGET_EXHAUSTED = "budget_exhausted"


# ── ResearchCampaign ─────────────────────────────────────────────────────────

@dataclass
class ResearchCampaign:
    """
    A long-running research campaign.

    Unlike a session (ephemeral, minutes to hours), a campaign:
    - Persists indefinitely via DB checkpoints
    - Grows its hypothesis space via HypothesisTree
    - Measures its own baseline
    - Declares breakthrough only when BreakthroughCriteria are met
    """

    id: str
    question: str
    status: str = STATUS_ACTIVE

    # Tree & criteria
    hypothesis_tree: HypothesisTree = field(default_factory=HypothesisTree)
    breakthrough_criteria: BreakthroughCriteria = field(default_factory=BreakthroughCriteria.default_accuracy)

    # Metric tracking
    baseline_metric: float = 0.0
    best_metric: float = 0.0
    improvement_pct: float = 0.0
    best_finding: dict[str, Any] | None = None

    # Counts
    total_hypotheses: int = 0
    total_confirmed: int = 0

    # Budget (wall-clock seconds)
    compute_budget_seconds: int = 14400      # 4 hours default
    compute_seconds_used: int = 0

    # Lifetime policy: "accepted" (default) or "candidate" (calibration evaluation)
    policy_mode: str = "accepted"

    # Timestamps
    started_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    last_checkpoint: str = ""
    checkpoint_every: int = 300              # checkpoint every 5 minutes

    # Internal monotonic start for elapsed tracking
    _start_mono: float = field(default_factory=time.monotonic, repr=False)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid4())

    # ── Stopping ─────────────────────────────────────────────────────────────

    def should_stop(self) -> bool:
        if self.status in (STATUS_BREAKTHROUGH, STATUS_BUDGET_EXHAUSTED, STATUS_PAUSED):
            return True
        elapsed = time.monotonic() - self._start_mono
        if elapsed >= self.compute_budget_seconds:
            return True
        return False

    def elapsed_seconds(self) -> float:
        # Prefer wall clock from persisted started_at so GET /campaigns and db_save
        # stay correct while a long Celery batch is in flight (per-request objects
        # otherwise report only stale compute_seconds_used from monotonic math).
        wall = _wall_seconds_since_started(self.started_at)
        if wall is not None:
            return wall
        return time.monotonic() - self._start_mono

    def remaining_seconds(self) -> float:
        return max(0.0, self.compute_budget_seconds - self.elapsed_seconds())

    # ── Metric update ────────────────────────────────────────────────────────

    def update_best_metric(self, finding: dict[str, Any]) -> bool:
        """
        Update best_metric / improvement_pct if this finding is better than current best.
        Returns True if this is a new best.
        """
        crit = self.breakthrough_criteria
        metric_val = finding.get("metric_value") or finding.get(crit.metric_name)
        if metric_val is None:
            return False
        try:
            metric_val = float(metric_val)
        except (TypeError, ValueError):
            return False
        metric_val = normalize_accuracy_metric(crit.metric_name, metric_val)
        if metric_val is None:
            return False

        if crit._is_implausible(metric_val):
            return False

        is_better = (
            (crit.direction == "higher_is_better" and metric_val > self.best_metric)
            or (crit.direction == "lower_is_better" and (self.best_metric == 0.0 or metric_val < self.best_metric))
        )
        if is_better:
            self.best_metric = metric_val
            self.best_finding = finding
            self.improvement_pct = crit.improvement_pct(metric_val)
            return True
        return False

    def recount_from_tree(self) -> None:
        """Recompute outcome counts from distinct tree nodes.

        Counters must reflect *distinct evaluated hypotheses* — not the number of
        worker results — otherwise re-dispatched nodes double-count and the campaign
        summary diverges from the hypothesis tree and the DB-derived paper counts.
        Tree nodes are upserted one row per node in the ``hypotheses`` table, so this
        keeps ``summary()`` aligned with ``compile_session_findings``.
        """
        nodes = self.hypothesis_tree.nodes
        evaluated = [
            n for n in nodes.values()
            if n.verdict in ("confirmed", "refuted", "inconclusive")
        ]
        self.total_hypotheses = len(evaluated)
        from propab.research_quality import NODE_ROLE_CONTROL

        self.total_confirmed = sum(
            1
            for n in evaluated
            if n.verdict == "confirmed" and getattr(n, "node_role", NODE_ROLE_CONTROL) != NODE_ROLE_CONTROL
        )

    def count_replications(self, hypothesis_id: str) -> int:
        """
        Count confirmed siblings (same parent) as a replication proxy.
        Used to check min_replications before declaring breakthrough.
        """
        node = self.hypothesis_tree.nodes.get(hypothesis_id)
        if node is None or node.parent_id is None:
            return 1
        parent = self.hypothesis_tree.nodes.get(node.parent_id)
        if parent is None:
            return 1
        confirmed_siblings = [
            c for c in parent.children
            if self.hypothesis_tree.nodes.get(c) is not None
            and self.hypothesis_tree.nodes[c].verdict == "confirmed"
        ]
        return max(1, len(confirmed_siblings))

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "status": self.status,
            "breakthrough_criteria": self.breakthrough_criteria.to_dict(),
            "hypothesis_tree": self.hypothesis_tree.to_dict(),
            "baseline_metric": self.baseline_metric,
            "best_metric": self.best_metric,
            "improvement_pct": self.improvement_pct,
            "best_finding": self.best_finding,
            "total_hypotheses": self.total_hypotheses,
            "total_confirmed": self.total_confirmed,
            "compute_budget_seconds": self.compute_budget_seconds,
            "compute_seconds_used": self.compute_seconds_used,
            "started_at": self.started_at,
            "last_checkpoint": self.last_checkpoint,
            "checkpoint_every": self.checkpoint_every,
            "policy_mode": self.policy_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchCampaign":
        c = cls(
            id=data["id"],
            question=data["question"],
            status=data.get("status", STATUS_ACTIVE),
            hypothesis_tree=HypothesisTree.from_dict(data.get("hypothesis_tree") or {}),
            breakthrough_criteria=BreakthroughCriteria.from_dict(
                data.get("breakthrough_criteria") or {"metric_name": "val_accuracy"}
            ),
            baseline_metric=data.get("baseline_metric", 0.0),
            best_metric=data.get("best_metric", 0.0),
            improvement_pct=data.get("improvement_pct", 0.0),
            best_finding=data.get("best_finding"),
            total_hypotheses=data.get("total_hypotheses", 0),
            total_confirmed=data.get("total_confirmed", 0),
            compute_budget_seconds=data.get("compute_budget_seconds", 14400),
            compute_seconds_used=data.get("compute_seconds_used", 0),
            started_at=data.get("started_at", datetime.now(tz=UTC).isoformat()),
            last_checkpoint=data.get("last_checkpoint", ""),
            checkpoint_every=data.get("checkpoint_every", 300),
            policy_mode=data.get("policy_mode", "accepted"),
        )
        # Preserve elapsed wall-clock budget accounting when loading from DB.
        used = float(data.get("compute_seconds_used", 0) or 0)
        c._start_mono = time.monotonic() - max(0.0, used)
        c._sync_baseline_fields()
        return c

    def _sync_baseline_fields(self) -> None:
        """Keep DB column breakthrough_criteria.baseline_value and baseline_metric aligned."""
        bm = float(self.baseline_metric)
        bv = float(self.breakthrough_criteria.baseline_value)
        if bv > 1e-12 and abs(bm) < 1e-12:
            self.baseline_metric = bv
        elif bm > 1e-12 and abs(bv) < 1e-12:
            self.breakthrough_criteria.baseline_value = bm

    def _resolved_baseline_metric(self) -> float:
        self._sync_baseline_fields()
        bm, bv = float(self.baseline_metric), float(self.breakthrough_criteria.baseline_value)
        if abs(bm) > 1e-12:
            return bm
        return bv

    def summary(self) -> dict[str, Any]:
        self._sync_baseline_fields()
        base_ok = (
            abs(self.baseline_metric) > 1e-12 or abs(self.breakthrough_criteria.baseline_value) > 1e-12
        )
        imp_pct = None
        if base_ok:
            frac = (
                self.improvement_pct
                if math.isfinite(self.improvement_pct)
                else 0.0
            )
            imp_pct = round(frac * 100, 2)
        return {
            "id": self.id,
            "question": self.question[:120],
            "status": self.status,
            "total_hypotheses": self.total_hypotheses,
            "total_confirmed": self.total_confirmed,
            "baseline_metric": self._resolved_baseline_metric(),
            "best_metric": self.best_metric,
            "improvement_pct": imp_pct,
            "elapsed_sec": round(self.elapsed_seconds(), 1),
            "remaining_sec": round(self.remaining_seconds(), 1),
            "tree": self.hypothesis_tree.summary(),
            "breakthrough_threshold_pct": round(
                self.breakthrough_criteria.improvement_threshold * 100, 1
            ),
        }
