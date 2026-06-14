"""Dispatch simulate_search by registry version (sim_v1 / sim_v2)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from propab.layer05.ensemble_simulator import SIM_V3, simulate_search_ensemble
from propab.layer05.stage_simulators import SIM_V4, simulate_search_stage_aware
from propab.layer05.hybrid_simulator import SIM_V2, simulate_search_hybrid
from propab.layer05.replay_state import SearchState
from propab.layer05.search_simulator import SimulationResult, simulate_search
from propab.layer05.simulator_hyperparams import SimulatorHyperparams
from propab.layer05.simulator_registry import SIM_V1, SimulatorRegistry
from propab.layer05.state_embedding_index import StateEmbeddingIndex
from propab.policy_record import PolicyRecord
from propab.search_policy import SearchPolicy


def resolve_simulator_version(
    *,
    version: str | None = None,
    registry: SimulatorRegistry | None = None,
) -> str:
    reg = registry or SimulatorRegistry.load()
    return version or reg.active_version or SIM_V1


def load_or_build_index(
    *,
    snapshots_by_campaign: dict[str, list[dict[str, Any]]] | None = None,
    trajectory_path: Path | str | None = None,
    exclude_campaign_id: str | None = None,
) -> StateEmbeddingIndex:
    index = StateEmbeddingIndex.load()
    if index.entries:
        if exclude_campaign_id:
            index = StateEmbeddingIndex(
                entries=[e for e in index.entries if e.campaign_id != exclude_campaign_id],
                max_tested=index.max_tested,
            )
        return index

    if snapshots_by_campaign:
        index = StateEmbeddingIndex().build_from_snapshots_map(snapshots_by_campaign)
    elif trajectory_path and Path(trajectory_path).is_file():
        from propab.layer05.replay_loader import load_snapshots_from_json

        index = StateEmbeddingIndex().build_from_snapshots_map(
            load_snapshots_from_json(trajectory_path),
        )
    if exclude_campaign_id:
        index = StateEmbeddingIndex(
            entries=[e for e in index.entries if e.campaign_id != exclude_campaign_id],
            max_tested=index.max_tested,
        )
    return index


def simulate_for_version(
    *,
    version: str,
    state: SearchState,
    policy: PolicyRecord | SearchPolicy,
    steps: int,
    index: StateEmbeddingIndex | None = None,
    snapshots_by_campaign: dict[str, list[dict[str, Any]]] | None = None,
    trajectory_path: Path | str | None = None,
    exclude_campaign_id: str | None = None,
    hyperparams: SimulatorHyperparams | None = None,
) -> SimulationResult:
    hp = hyperparams or _load_saved_hyperparams(version)
    if version in (SIM_V2, SIM_V3, SIM_V4):
        idx = index or load_or_build_index(
            snapshots_by_campaign=snapshots_by_campaign,
            trajectory_path=trajectory_path,
            exclude_campaign_id=exclude_campaign_id,
        )
        if version == SIM_V3:
            return simulate_search_ensemble(
                state=state,
                policy=policy,
                index=idx,
                steps=steps,
                hyperparams=hp,
            )
        if version == SIM_V4:
            return simulate_search_stage_aware(
                state=state,
                policy=policy,
                index=idx,
                steps=steps,
                hyperparams=hp,
            )
        return simulate_search_hybrid(
            state=state,
            policy=policy,
            index=idx,
            steps=steps,
            hyperparams=hp,
        )
    return simulate_search(state=state, policy=policy, steps=steps)


def _load_saved_hyperparams(version: str) -> SimulatorHyperparams | None:
    from propab.layer05.simulator_calibration_v3 import hyperparams_path

    p = hyperparams_path()
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if version in data:
            return SimulatorHyperparams.from_dict(data[version])
    except (json.JSONDecodeError, OSError, TypeError):
        return None
    return None
