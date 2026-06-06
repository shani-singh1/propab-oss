"""
Phase E — search policy learned from campaign history.

Campaign N+1 uses updated theme weights, blocked failure patterns, and saturation penalties.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from propab.config import settings
from propab.knowledge_graph import KnowledgeGraph


def policy_store_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "search_policy.json"


@dataclass
class SearchPolicy:
    """First-class policy object updated after each campaign."""

    version: int = 1
    generation: int = 0
    theme_boost: dict[str, float] = field(default_factory=dict)
    theme_penalty: dict[str, float] = field(default_factory=dict)
    blocked_failure_signatures: list[str] = field(default_factory=list)
    saturated_themes: list[str] = field(default_factory=list)
    prefer_replication_t2_plus: bool = True
    closure_target: float = 0.35
    notes: list[str] = field(default_factory=list)

    def theme_weight(self, theme: str) -> float:
        base = 1.0
        base += self.theme_boost.get(theme, 0.0)
        base -= self.theme_penalty.get(theme, 0.0)
        if theme in self.saturated_themes:
            base -= 0.15
        return max(0.1, min(2.0, base))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchPolicy:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path | None = None) -> Path:
        p = path or policy_store_path()
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> SearchPolicy:
        p = path or policy_store_path()
        if not p.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return cls()


def update_policy_from_graph(
    policy: SearchPolicy,
    graph: KnowledgeGraph,
    *,
    campaign_metrics: dict[str, Any] | None = None,
) -> SearchPolicy:
    """Derive policy deltas from accumulated knowledge + latest campaign metrics."""
    policy.generation += 1
    rates = graph.theme_success_rates()

    policy.theme_boost.clear()
    policy.theme_penalty.clear()
    for theme, rate in rates.items():
        if rate >= 0.4:
            policy.theme_boost[theme] = round(min(0.35, rate * 0.5), 3)
        elif rate < 0.15 and sum(1 for c in graph.claims.values() if c.theme == theme) >= 5:
            policy.theme_penalty[theme] = 0.25
            if theme not in policy.saturated_themes:
                policy.saturated_themes.append(theme)

    sig_counts: dict[str, int] = {}
    for f in graph.failures.values():
        if f.failure_signature:
            sig_counts[f.failure_signature] = sig_counts.get(f.failure_signature, 0) + 1
    policy.blocked_failure_signatures = [
        s for s, n in sig_counts.items() if n >= 3
    ][:12]

    if campaign_metrics:
        cr = float(campaign_metrics.get("closure_ratio") or 0)
        policy.notes.append(
            f"gen={policy.generation} closure={cr:.3f} campaigns={len(graph.campaign_ids)}"
        )
        if cr < policy.closure_target:
            policy.prefer_replication_t2_plus = True
        policy.notes = policy.notes[-20:]

    return policy
