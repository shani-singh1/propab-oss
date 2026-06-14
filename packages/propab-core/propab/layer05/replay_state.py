"""Replay and simulation state objects (fixes.md Component 1 & 2)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ReplayState:
    tested_count: int
    frontier_nodes: list[str]
    theme_histogram: dict[str, int]
    entropy: float
    pending_nodes: int
    closure_ratio: float
    mechanisms: list[str] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    frontier_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> ReplayState:
        hist = dict(snap.get("theme_histogram") or {})
        return cls(
            tested_count=int(snap.get("tested") or snap.get("executed") or 0),
            frontier_nodes=[],
            theme_histogram=hist,
            entropy=float(snap.get("theme_entropy") or 0),
            pending_nodes=int(snap.get("pending") or 0),
            closure_ratio=float(snap.get("closure_ratio") or 0),
            frontier_size=int(snap.get("frontier_size") or 0),
        )


@dataclass
class SearchState:
    frontier: list[str]
    entropy: float
    closure_ratio: float
    pending_nodes: int
    theme_histogram: dict[str, int]
    tested_count: int = 0
    mechanisms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_replay(cls, state: ReplayState) -> SearchState:
        return cls(
            frontier=list(state.frontier_nodes),
            entropy=state.entropy,
            closure_ratio=state.closure_ratio,
            pending_nodes=state.pending_nodes,
            theme_histogram=dict(state.theme_histogram),
            tested_count=state.tested_count,
            mechanisms=list(state.mechanisms),
        )

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> SearchState:
        return cls.from_replay(ReplayState.from_snapshot(snap))
