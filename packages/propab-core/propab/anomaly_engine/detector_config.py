"""Configuration for anomaly detection (domain-agnostic)."""
from __future__ import annotations

from dataclasses import dataclass, field

ANOMALY_BUCKETS = (
    "survivor",
    "collapse",
    "cross_family",
    "threshold",
    "symmetry_break",
    "outlier",
    "neighbor_disagreement",
)


@dataclass
class FalsifiableExpectation:
    """
    Optional domain rule: subsets matching feature_substrings should show
    large within-vs-LOFO degradation (prediction_failure under group removal).
    """

    feature_substrings: list[str]
    min_lofo_gap: float = 0.2
    note: str = ""


def default_bucket_slots(top_k: int) -> dict[str, int]:
    """Per-bucket quotas (fixes.md P0/P1) — primary buckets first, no global backfill."""
    top_k = max(1, top_k)
    if top_k <= 8:
        return {
            "survivor": max(1, top_k // 4),
            "collapse": max(1, top_k // 4),
            "cross_family": max(1, top_k // 4),
            "threshold": top_k - 3 * max(1, top_k // 4),
            "symmetry_break": 0,
            "outlier": 0,
            "neighbor_disagreement": 0,
        }
    primary = ("survivor", "collapse", "cross_family", "threshold")
    per_primary = max(1, top_k // 4)
    slots = {b: per_primary for b in primary}
    remainder = top_k - sum(slots.values())
    secondary = ("symmetry_break", "outlier", "neighbor_disagreement")
    for b in secondary:
        share = max(0, remainder // len(secondary))
        if share > 0:
            slots[b] = share
            remainder -= share
    # Assign any leftover to neighbor_disagreement (P4)
    if remainder > 0:
        slots["neighbor_disagreement"] = slots.get("neighbor_disagreement", 0) + remainder
    return slots


@dataclass
class DetectorConfig:
    top_k: int = 20
    min_lofo: float = -1.0
    min_surprise: float = -1.0  # allow negative surprise when LOFO is best available
    collapse_gap: float = 0.35
    lofo_survival: float = -0.25  # relative LOFO floor for "survivor" bucket
    min_within_for_collapse: float = 0.25
    cross_family_lofo_std: float = 0.18  # min per-family LOFO std for cross_family bucket
    survivor_slots: int | None = None  # legacy; overridden by bucket_slots when set
    collapse_slots: int | None = None
    bucket_slots: dict[str, int] | None = None
    feature_groups: dict[str, list[str]] | None = None
    max_group_fraction: float = 0.4  # fixes.md P2 — no group >40% of selected anomalies
    expectations: list[FalsifiableExpectation] = field(default_factory=list)

    def resolved_bucket_slots(self) -> dict[str, int]:
        if self.bucket_slots:
            positive = {k: max(0, v) for k, v in self.bucket_slots.items() if v > 0}
            total = sum(positive.values())
            if total == 0:
                return default_bucket_slots(self.top_k)
            if total <= self.top_k:
                return positive
            # Scale down proportionally while keeping at least 1 per bucket
            scaled: dict[str, int] = {}
            keys = list(positive.keys())
            remaining = self.top_k
            for i, k in enumerate(keys):
                if i == len(keys) - 1:
                    scaled[k] = max(0, remaining)
                else:
                    share = max(1, int(round(self.top_k * positive[k] / total)))
                    share = min(share, remaining - (len(keys) - i - 1))
                    scaled[k] = share
                    remaining -= share
            return {k: v for k, v in scaled.items() if v > 0}
        if self.survivor_slots is not None or self.collapse_slots is not None:
            surv = self.survivor_slots if self.survivor_slots is not None else max(1, self.top_k // 3)
            coll = self.collapse_slots if self.collapse_slots is not None else max(1, self.top_k // 3)
            slots = default_bucket_slots(self.top_k)
            slots["survivor"] = surv
            slots["collapse"] = coll
            used = surv + coll
            for b in ("cross_family", "threshold", "symmetry_break", "outlier", "neighbor_disagreement"):
                if b in slots and used + slots[b] <= self.top_k:
                    used += slots[b]
                elif b in slots:
                    slots[b] = max(0, self.top_k - used)
                    break
            return {k: v for k, v in slots.items() if v > 0}
        return default_bucket_slots(self.top_k)

    def group_cap(self) -> int:
        """Max anomalies from a single feature group (fixes.md P2)."""
        return max(1, int(self.top_k * self.max_group_fraction))

    def survivor_budget(self) -> int:
        return self.resolved_bucket_slots().get("survivor", max(1, self.top_k // 3))

    def collapse_budget(self) -> int:
        return self.resolved_bucket_slots().get("collapse", max(1, self.top_k // 3))
