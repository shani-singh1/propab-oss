"""StateEmbeddingIndex — nearest-neighbor trajectory priors (fixes.md P4)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from propab.config import settings
from propab.layer05.replay_state import SearchState
from propab.layer05.simulator_hyperparams import SimulatorHyperparams
from propab.layer05.state_vector_v2 import (
    FeatureStats,
    build_state_vector,
    feature_distance,
    normalize_features,
    state_features,
)
from propab.layer05.trajectories import CampaignTrajectories


def state_index_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "state_embedding_index.json"


@dataclass
class StateIndexEntry:
    campaign_id: str
    features: list[float]
    features_v2: list[float] = field(default_factory=list)
    initial_state: dict[str, Any] = field(default_factory=dict)
    observed_trajectories: dict[str, Any] = field(default_factory=dict)
    snapshots: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StateEmbeddingIndex:
    entries: list[StateIndexEntry] = field(default_factory=list)
    max_tested: float = 250.0

    def add_campaign(
        self,
        *,
        campaign_id: str,
        snapshots: list[dict[str, Any]],
    ) -> None:
        if not snapshots:
            return
        state = SearchState.from_snapshot(snapshots[0])
        feats = state_features(state, max_tested=self.max_tested)
        feats_v2 = build_state_vector(
            state, version="v2", snapshots=snapshots, max_tested=self.max_tested,
        )
        traj = CampaignTrajectories.from_snapshots(snapshots).to_dict()
        self.entries = [e for e in self.entries if e.campaign_id != campaign_id]
        self.entries.append(StateIndexEntry(
            campaign_id=campaign_id,
            features=feats,
            features_v2=feats_v2,
            initial_state=state.to_dict(),
            observed_trajectories=traj,
            snapshots=snapshots,
        ))

    def build_from_snapshots_map(
        self,
        campaigns: dict[str, list[dict[str, Any]]],
    ) -> StateEmbeddingIndex:
        for cid, snaps in campaigns.items():
            self.add_campaign(campaign_id=cid, snapshots=snaps)
        tested = [float(s.get("tested") or 0) for snaps in campaigns.values() for s in snaps]
        if tested:
            self.max_tested = max(tested) or 250.0
        return self

    def _entry_features(
        self,
        entry: StateIndexEntry,
        *,
        feature_version: str,
    ) -> list[float]:
        if feature_version == "v2":
            if entry.features_v2:
                return entry.features_v2
            if entry.snapshots:
                state = SearchState.from_snapshot(entry.snapshots[0])
                return build_state_vector(
                    state,
                    version="v2",
                    snapshots=entry.snapshots,
                    max_tested=self.max_tested,
                )
        return entry.features

    def feature_stats(
        self,
        *,
        feature_version: str = "v2",
        normalization: str = "minmax",
    ) -> FeatureStats | None:
        if normalization != "minmax":
            return None
        rows = [
            self._entry_features(e, feature_version=feature_version)
            for e in self.entries
        ]
        return FeatureStats.from_matrix(rows) if rows else None

    def nearest(
        self,
        state: SearchState | dict[str, Any],
        *,
        k: int = 3,
        hyperparams: SimulatorHyperparams | None = None,
        query_snapshots: list[dict[str, Any]] | None = None,
    ) -> list[tuple[StateIndexEntry, float]]:
        hp = hyperparams or SimulatorHyperparams()
        query = build_state_vector(
            state,
            version=hp.feature_version,
            snapshots=query_snapshots,
            max_tested=self.max_tested,
        )
        stats = self.feature_stats(
            feature_version=hp.feature_version,
            normalization=hp.normalization,
        )
        query = normalize_features(query, hp.normalization, stats)
        scored: list[tuple[StateIndexEntry, float]] = []
        for entry in self.entries:
            feats = self._entry_features(entry, feature_version=hp.feature_version)
            feats = normalize_features(feats, hp.normalization, stats)
            scored.append((
                entry,
                feature_distance(query, feats, metric=hp.distance_metric),
            ))
        scored.sort(key=lambda x: x[1])
        return scored[:k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tested": self.max_tested,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateEmbeddingIndex:
        entries = [StateIndexEntry(**e) for e in (data.get("entries") or [])]
        return cls(entries=entries, max_tested=float(data.get("max_tested") or 250))

    def save(self, path: Path | None = None) -> Path:
        p = path or state_index_path()
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> StateEmbeddingIndex:
        p = path or state_index_path()
        if not p.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return cls()
