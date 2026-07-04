"""Synthetic SNAP-style graph invariant dataset."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from propab.config import settings

GRAPH_FAMILIES = ("erdos_renyi", "barabasi_albert", "watts_strogatz", "grid_lattice")
GRAPHS_PER_FAMILY = 40
N_NODES = 200
RANDOM_SEED = 42

KNOWN_INVARIANTS: tuple[str, ...] = (
    "spectral_gap",
    "algebraic_connectivity",
    "clustering_coefficient",
    "diameter",
    "avg_degree",
    "modularity",
)


def cache_dir() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "graph_invariants"
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_path() -> Path:
    return cache_dir() / "snap_subset_v1.csv"


def _erdos_renyi(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    adj = (rng.random((n, n)) < p).astype(float)
    adj = np.triu(adj, 1)
    return adj + adj.T


def _barabasi_albert(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    adj = np.zeros((n, n))
    for i in range(1, n):
        deg = adj.sum(axis=0) + 1e-9
        probs = deg[:i] / deg[:i].sum()
        targets = rng.choice(i, size=min(m, i), replace=False, p=probs)
        for j in targets:
            adj[i, j] = adj[j, i] = 1.0
    return adj


def _watts_strogatz(n: int, k: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(1, k // 2 + 1):
            a, b = i, (i + j) % n
            adj[a, b] = adj[b, a] = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] and rng.random() < beta:
                adj[i, j] = adj[j, i] = 0.0
                t = rng.integers(0, n)
                adj[i, t] = adj[t, i] = 1.0
    return adj


def _grid_lattice(n: int) -> np.ndarray:
    side = int(np.sqrt(n))
    side = max(side, 2)
    adj = np.zeros((n, n))
    for i in range(n):
        r, c = divmod(i, side)
        for dr, dc in ((0, 1), (1, 0)):
            nr, nc = r + dr, c + dc
            if nr < side and nc < side:
                j = nr * side + nc
                if j < n:
                    adj[i, j] = adj[j, i] = 1.0
    return adj


def _bfs_farthest(adj: np.ndarray, start: int) -> tuple[int, int]:
    n = adj.shape[0]
    dist = np.full(n, -1, dtype=int)
    dist[start] = 0
    frontier = [start]
    farthest, farthest_dist = start, 0
    while frontier:
        nxt: list[int] = []
        for u in frontier:
            for v in np.where(adj[u] > 0)[0]:
                if dist[v] >= 0:
                    continue
                dist[v] = dist[u] + 1
                if dist[v] > farthest_dist:
                    farthest, farthest_dist = int(v), int(dist[v])
                nxt.append(int(v))
        frontier = nxt
    return farthest, farthest_dist


def _approx_diameter(adj: np.ndarray) -> float:
    if adj.shape[0] <= 1:
        return 0.0
    mid, _ = _bfs_farthest(adj, 0)
    _, diameter = _bfs_farthest(adj, mid)
    return float(diameter)


def _graph_metrics(adj: np.ndarray) -> dict[str, float]:
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    avg_deg = float(deg.mean())
    lap = np.diag(deg) - adj
    eigvals = np.linalg.eigvalsh(lap)
    spectral_gap = float(eigvals[1]) if len(eigvals) > 1 else 0.0
    tris = np.trace(adj @ adj @ adj) / 6.0
    possible = n * (n - 1) * (n - 2) / 6.0 if n > 2 else 1.0
    clustering = float(tris / possible) if possible else 0.0
    diameter = _approx_diameter(adj)
    modularity = float(0.25 * clustering + 0.1 * (avg_deg / max(n, 1)))
    return {
        "spectral_gap": spectral_gap,
        "algebraic_connectivity": spectral_gap,
        "clustering_coefficient": clustering,
        "diameter": float(diameter),
        "avg_degree": avg_deg,
        "modularity": modularity,
    }


def _synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    gid = 0
    for family in GRAPH_FAMILIES:
        for _ in range(GRAPHS_PER_FAMILY):
            if family == "erdos_renyi":
                adj = _erdos_renyi(N_NODES, 0.03, rng)
            elif family == "barabasi_albert":
                adj = _barabasi_albert(N_NODES, 3, rng)
            elif family == "watts_strogatz":
                adj = _watts_strogatz(N_NODES, 6, 0.1, rng)
            else:
                adj = _grid_lattice(N_NODES)
            metrics = _graph_metrics(adj)
            rows.append({"graph_id": f"G{gid:04d}", "network_family": family, **metrics})
            gid += 1
    return pd.DataFrame(rows)


@dataclass
class GraphInvariantSpec:
    source_invariant: str
    target_invariant: str
    claim_type: str = "correlation_positive"
    held_out_family: str | None = None

    @classmethod
    def from_hypothesis(cls, hypothesis: dict[str, Any]) -> GraphInvariantSpec:
        text = str(hypothesis.get("text") or hypothesis.get("statement") or "").lower()
        src, tgt = "spectral_gap", "clustering_coefficient"
        if "diameter" in text:
            tgt = "diameter"
        if "modularity" in text:
            src, tgt = "modularity", "clustering_coefficient"
        if "algebraic" in text:
            src = "algebraic_connectivity"
        claim = "correlation_positive"
        if "negative" in text or "inverse" in text or "decreases" in text:
            claim = "correlation_negative"
        if "monotonic" in text or "all families" in text:
            claim = "holds_all_families"
        held = None
        for fam in GRAPH_FAMILIES:
            if fam.replace("_", " ") in text or fam.replace("_", "-") in text:
                held = fam
                break
        return cls(source_invariant=src, target_invariant=tgt, claim_type=claim, held_out_family=held)


class GraphInvariantsAdapter:
    def ensure_cache(self) -> Path:
        path = cache_path()
        if path.is_file():
            return path
        df = _synthetic_frame()
        df.to_csv(path, index=False)
        cache_dir().joinpath("snap_subset_v1.meta.json").write_text(
            json.dumps({"graphs": len(df), "families": list(GRAPH_FAMILIES), "synthetic": True}, indent=2),
            encoding="utf-8",
        )
        return path

    def load_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.ensure_cache())
