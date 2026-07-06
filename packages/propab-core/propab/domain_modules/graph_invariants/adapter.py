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


def _fiedler_vector(lap: np.ndarray) -> np.ndarray:
    """Second-smallest eigenvector of the Laplacian (the Fiedler vector)."""
    _, eigvecs = np.linalg.eigh(lap)
    if eigvecs.shape[1] < 2:
        return np.zeros(lap.shape[0])
    return eigvecs[:, 1]


def _newman_modularity(adj: np.ndarray, communities: np.ndarray) -> float:
    """
    Real Newman modularity Q for a partition of nodes into communities.

    Q = (1 / 2m) * sum_ij [ A_ij - k_i k_j / 2m ] * delta(c_i, c_j)

    DOM2b honesty: this is a genuine structural quantity computed from the
    adjacency matrix and a community partition — NOT a closed-form function of
    the clustering coefficient. It measures how much more densely nodes connect
    within their assigned community than expected under a degree-preserving null,
    so ``modularity`` and ``clustering_coefficient`` are independent invariants
    (a "modularity↔clustering" finding is no longer a tautology).
    """
    deg = adj.sum(axis=1)
    two_m = float(deg.sum())
    if two_m <= 0:
        return 0.0
    same = (communities[:, None] == communities[None, :]).astype(float)
    expected = np.outer(deg, deg) / two_m
    q = float(((adj - expected) * same).sum() / two_m)
    return q


def _graph_metrics(adj: np.ndarray) -> dict[str, float]:
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    avg_deg = float(deg.mean())
    lap = np.diag(deg) - adj
    lap_eigvals = np.linalg.eigvalsh(lap)
    # Algebraic connectivity = Fiedler value = second-smallest Laplacian eigenvalue.
    algebraic_connectivity = float(lap_eigvals[1]) if len(lap_eigvals) > 1 else 0.0
    # Spectral gap: a DISTINCT quantity — the gap between the two largest
    # adjacency-matrix eigenvalues (λ1 - λ2). Previously ``spectral_gap`` was set
    # equal to ``algebraic_connectivity`` (both = lap_eigvals[1]), making them an
    # exact duplicate pair; this decouples them (DOM2b: no exposed-invariant pair
    # is a deterministic function of another).
    adj_eigvals = np.linalg.eigvalsh(adj)
    spectral_gap = (
        float(adj_eigvals[-1] - adj_eigvals[-2]) if len(adj_eigvals) > 1 else 0.0
    )
    tris = np.trace(adj @ adj @ adj) / 6.0
    possible = n * (n - 1) * (n - 2) / 6.0 if n > 2 else 1.0
    clustering = float(tris / possible) if possible else 0.0
    diameter = _approx_diameter(adj)
    # Real modularity of a Fiedler (spectral) bipartition — a structural community
    # measure, not 0.25*clustering + 0.1*(avg_deg/n). See _newman_modularity.
    communities = (_fiedler_vector(lap) >= 0).astype(int)
    modularity = _newman_modularity(adj, communities)
    return {
        "spectral_gap": spectral_gap,
        "algebraic_connectivity": algebraic_connectivity,
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


class GraphInvariantNotIdentified(ValueError):
    """
    Raised by :meth:`GraphInvariantSpec.from_hypothesis` when no graph invariant
    can be confidently identified in the hypothesis text.

    DOM4 honesty: the previous ``from_hypothesis`` ALWAYS fell back to
    ``spectral_gap → clustering_coefficient`` for any text lacking specific
    keywords, so an off-topic / misrouted hypothesis was silently verified
    against a fixed default pair and could produce a "confirmed" verdict
    decoupled from the actual claim. Refusing (raising) here lets the caller map
    the result to ``inconclusive`` instead of fabricating a check.
    """


# Phrases that confidently name an exposed invariant. Ordered so more-specific
# phrases match before bare ones (e.g. "clustering coefficient" before "cluster").
_INVARIANT_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("modularity", ("modularity",)),
    ("algebraic_connectivity", ("algebraic connectivity", "algebraic", "fiedler")),
    ("spectral_gap", ("spectral gap", "spectral")),
    ("clustering_coefficient", ("clustering coefficient", "clustering", "transitivity")),
    ("diameter", ("diameter",)),
    ("avg_degree", ("average degree", "avg degree", "average-degree", "degree distribution")),
)


@dataclass
class GraphInvariantSpec:
    source_invariant: str
    target_invariant: str
    claim_type: str = "correlation_positive"
    held_out_family: str | None = None

    @classmethod
    def from_hypothesis(cls, hypothesis: dict[str, Any]) -> GraphInvariantSpec:
        """
        Resolve source/target invariants from hypothesis text.

        DOM4: identify invariants only from explicit markers. If NO invariant is
        named at all, raise :class:`GraphInvariantNotIdentified` rather than
        silently defaulting to ``spectral_gap → clustering`` — a text mentioning
        no graph invariant is off-topic / misrouted and must not be verified
        against a fabricated default pair. When exactly ONE invariant is named
        (a threshold / band / cross-family self-check), it is paired with a
        distinct default correlate so the check is a real cross-invariant
        comparison rather than a trivial self-correlation.
        """
        text = str(hypothesis.get("text") or hypothesis.get("statement") or "").lower()

        # Collect invariants in the order they appear in the text so the leading
        # invariant becomes the source and the next distinct one the target.
        found: list[tuple[int, str]] = []
        seen: set[str] = set()
        for invariant, phrases in _INVARIANT_MARKERS:
            pos = min(
                (text.find(p) for p in phrases if p in text),
                default=-1,
            )
            if pos >= 0 and invariant not in seen:
                found.append((pos, invariant))
                seen.add(invariant)
        found.sort(key=lambda t: t[0])

        if not found:
            raise GraphInvariantNotIdentified(
                "could not confidently identify any graph invariant in hypothesis "
                "text; refusing default spectral_gap->clustering_coefficient"
            )

        if len(found) >= 2:
            src, tgt = found[0][1], found[1][1]
        else:
            # Exactly one invariant named: pair with a distinct default correlate
            # (never itself, which would be a trivial r=1 self-correlation).
            only = found[0][1]
            default_other = "clustering_coefficient" if only != "clustering_coefficient" else "spectral_gap"
            src, tgt = only, default_other

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
