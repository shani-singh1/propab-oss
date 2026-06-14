"""DirectionErrorDataset — one row per direction error (fixes.md P0)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from propab.config import settings
from propab.layer05.neighbor_attribution import (
    NeighborAttribution,
    attribution_summary,
    build_neighbor_attribution,
)
from propab.layer05.replay_state import SearchState
from propab.layer05.state_embedding_index import StateEmbeddingIndex
from propab.layer05.state_vector_v2 import build_state_vector
from propab.layer05.trajectory_stages import stage_at_index


def direction_dataset_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "direction_error_dataset.json"


@dataclass
class DirectionErrorRow:
    campaign_id: str
    snapshot_index: int
    snapshot_id: int
    predicted_entropy: float
    observed_entropy: float
    sim_delta: float
    obs_delta: float
    residual: float
    state_vector: list[float]
    neighbors: list[dict[str, Any]]
    neighbor_summary: dict[str, Any]
    theme_histogram: dict[str, int]
    frontier_size: int
    stage: str
    simulator_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DirectionErrorDataset:
    rows: list[DirectionErrorRow] = field(default_factory=list)

    def add(self, row: DirectionErrorRow) -> None:
        self.rows.append(row)

    def by_campaign(self, campaign_id: str) -> list[DirectionErrorRow]:
        return [r for r in self.rows if r.campaign_id == campaign_id]

    def by_stage(self, stage: str) -> list[DirectionErrorRow]:
        return [r for r in self.rows if r.stage == stage]

    def to_dict(self) -> dict[str, Any]:
        return {"rows": [r.to_dict() for r in self.rows], "count": len(self.rows)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DirectionErrorDataset:
        return cls(rows=[DirectionErrorRow(**r) for r in (data.get("rows") or [])])

    def save(self, path: Path | None = None) -> Path:
        p = path or direction_dataset_path()
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> DirectionErrorDataset:
        p = path or direction_dataset_path()
        if not p.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return cls()

    @classmethod
    def build_from_simulation(
        cls,
        *,
        campaign_id: str,
        simulator_version: str,
        simulated_entropy_points: list[dict[str, Any]],
        observed_snapshots: list[dict[str, Any]],
        index: StateEmbeddingIndex,
        hyperparams=None,
    ) -> DirectionErrorDataset:
        from propab.layer05.simulator_hyperparams import SimulatorHyperparams

        hp = hyperparams or SimulatorHyperparams()
        ds = cls()
        obs_h = [float(s.get("theme_entropy") or 0) for s in observed_snapshots]
        sim_h = [float(p.get("theme_entropy") or 0) for p in simulated_entropy_points]
        n = min(len(obs_h), len(sim_h))
        steps = max(1, n - 1)
        state = SearchState.from_snapshot(observed_snapshots[0] if observed_snapshots else {})
        neighbors = index.nearest(
            state, k=hp.k_neighbors, hyperparams=hp, query_snapshots=observed_snapshots[:3],
        )
        attributions = build_neighbor_attribution(neighbors)
        attr_dicts = [a.to_dict() for a in attributions]
        n_summary = attribution_summary(attributions)
        state_vec = build_state_vector(
            state, version=hp.feature_version, snapshots=observed_snapshots[:3],
            max_tested=index.max_tested,
        )

        for i in range(1, n):
            ds_sim = sim_h[i] - sim_h[i - 1]
            ds_obs = obs_h[i] - obs_h[i - 1]
            if ds_sim * ds_obs >= 0 or (ds_sim == 0 and ds_obs == 0):
                continue
            snap = observed_snapshots[i] if i < len(observed_snapshots) else observed_snapshots[-1]
            ds.add(DirectionErrorRow(
                campaign_id=campaign_id,
                snapshot_index=i,
                snapshot_id=int(snap.get("tested") or i),
                predicted_entropy=round(sim_h[i], 4),
                observed_entropy=round(obs_h[i], 4),
                sim_delta=round(ds_sim, 4),
                obs_delta=round(ds_obs, 4),
                residual=round(sim_h[i] - obs_h[i], 4),
                state_vector=state_vec,
                neighbors=attr_dicts,
                neighbor_summary=n_summary,
                theme_histogram=dict(snap.get("theme_histogram") or state.theme_histogram),
                frontier_size=int(snap.get("frontier_size") or 0),
                stage=stage_at_index(i, steps),
                simulator_version=simulator_version,
            ))
        return ds
