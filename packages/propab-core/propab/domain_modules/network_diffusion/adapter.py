"""
Real-network loader and subgraph sampler for network-diffusion verification.

Provenance — REAL graphs, no synthetic topology (``uses_synthetic_data`` is
False for this domain):

* ``collaboration`` — ``ca-GrQc`` : collaboration network of the arXiv General
  Relativity category (Leskovec, Kleinberg & Faloutsos, 2007). SNAP dataset
  ``ca-GrQc``. Sparse, highly clustered, assortative.
* ``email`` — ``email-Eu-core`` : e-mail network of a large European research
  institution (Leskovec & Krevl, SNAP; Yin, Benson, Leskovec & Gleich, 2017).
  Denser, lower clustering, different degree-mixing than the collaboration net.

Both are shipped under ``data/v1_candidates/`` as gzipped SNAP edge lists. The
loader reads them directly; it never fabricates edges. To obtain many graph
*instances* of a given real topology family we sample **induced subgraphs**
(BFS-ball around random seed nodes) — every node and every edge in a sample is
a real node/edge of the source network, so the topology of each instance is
genuinely empirical. The source network is the *topology family* used for the
cross-topology-family holdout.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from propab.config import settings

# --- Real SNAP datasets shipped on disk ------------------------------------
# family_id -> (filename under data/v1_candidates, is_tab_separated, has_comments)
REAL_NETWORKS: dict[str, str] = {
    "collaboration": "ca-GrQc.txt.gz",
    "email": "email-Eu-core.txt.gz",
}

# Structural (topology) features computed per subgraph instance. These are the
# hypothesis-testable structural quantities; the diffusion outcome is simulated
# separately (see simulator.py).
STRUCTURAL_FEATURES: tuple[str, ...] = (
    "degree_gini",          # degree-inequality (heterogeneity)
    "degree_cv",            # coefficient of variation of degree
    "k2_over_k1",           # <k^2>/<k> — governs the epidemic threshold
    "max_degree_ratio",     # hub dominance: k_max / <k>
    "mean_degree",          # density proxy
    "clustering",           # average local clustering coefficient
    "assortativity",        # degree assortativity coefficient
)


def data_root() -> Path:
    """Directory holding the SNAP edge lists (``data/v1_candidates``)."""
    return Path(settings.propab_data_dir).resolve() / "v1_candidates"


@dataclass
class Subgraph:
    """A real induced subgraph instance: adjacency (as neighbor lists) + provenance."""

    family: str
    nodes: list[int]
    adj: dict[int, list[int]]
    source_network: str

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return sum(len(v) for v in self.adj.values()) // 2

    def degrees(self) -> np.ndarray:
        return np.array([len(self.adj[n]) for n in self.nodes], dtype=float)


def _read_edge_list(path: Path) -> dict[int, set[int]]:
    """Read a gzipped SNAP edge list into an undirected adjacency dict."""
    adj: dict[int, set[int]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace("\t", " ").split()
            if len(parts) < 2:
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if u == v:
                continue
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
    return adj


def _largest_component(adj: dict[int, set[int]]) -> dict[int, set[int]]:
    """Return the giant connected component (BFS)."""
    if not adj:
        return {}
    seen: set[int] = set()
    best: set[int] = set()
    for start in adj:
        if start in seen:
            continue
        comp: set[int] = set()
        frontier = [start]
        comp.add(start)
        while frontier:
            nxt: list[int] = []
            for u in frontier:
                for w in adj[u]:
                    if w not in comp:
                        comp.add(w)
                        nxt.append(w)
            frontier = nxt
        seen |= comp
        if len(comp) > len(best):
            best = comp
    return {u: (adj[u] & best) for u in best}


@lru_cache(maxsize=8)
def load_network(family: str) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """
    Load one real SNAP network's giant component, cached.

    Returned as hashable tuples so ``lru_cache`` can key on ``family`` alone.
    Use :func:`network_adjacency` for a usable dict.
    """
    if family not in REAL_NETWORKS:
        raise ValueError(f"Unknown network family: {family!r}")
    path = data_root() / REAL_NETWORKS[family]
    if not path.is_file():
        raise FileNotFoundError(f"SNAP edge list not found: {path}")
    adj = _largest_component(_read_edge_list(path))
    if not adj:
        raise ValueError(f"Empty graph for {family!r} at {path}")
    nodes = tuple(sorted(adj))
    adj_t = tuple(tuple(sorted(adj[n])) for n in nodes)
    return nodes, adj_t


def network_adjacency(family: str) -> dict[int, list[int]]:
    nodes, adj_t = load_network(family)
    return {n: list(neigh) for n, neigh in zip(nodes, adj_t)}


def sample_subgraphs(
    family: str,
    *,
    n_samples: int,
    target_size: int,
    rng: np.random.Generator,
    min_nodes: int = 25,
    min_edges: int = 30,
) -> list[Subgraph]:
    """
    Sample ``n_samples`` real induced subgraphs from a family via BFS-ball
    expansion around random seed nodes. Every returned node/edge is real.
    """
    adj = network_adjacency(family)
    nodes = list(adj)
    out: list[Subgraph] = []
    attempts = 0
    max_attempts = n_samples * 6
    while len(out) < n_samples and attempts < max_attempts:
        attempts += 1
        seed = nodes[int(rng.integers(len(nodes)))]
        visited = {seed}
        frontier = [seed]
        while frontier and len(visited) < target_size:
            rng.shuffle(frontier)
            nxt: list[int] = []
            done = False
            for u in frontier:
                for w in adj[u]:
                    if w not in visited:
                        visited.add(w)
                        nxt.append(w)
                        if len(visited) >= target_size:
                            done = True
                            break
                if done:
                    break
            frontier = nxt
        sub_adj = {n: [w for w in adj[n] if w in visited] for n in visited}
        n_edges = sum(len(v) for v in sub_adj.values()) // 2
        if len(visited) >= min_nodes and n_edges >= min_edges:
            out.append(
                Subgraph(
                    family=family,
                    nodes=sorted(visited),
                    adj=sub_adj,
                    source_network=REAL_NETWORKS[family],
                )
            )
    return out


def structural_features(sub: Subgraph) -> dict[str, float]:
    """Compute the structural (topology) features of a real subgraph instance."""
    deg = sub.degrees()
    k1 = float(deg.mean()) if deg.size else 0.0
    k2 = float((deg ** 2).mean()) if deg.size else 0.0
    s = np.sort(deg)
    n = len(s)
    tot = s.sum()
    gini = float((2.0 * np.sum(np.arange(1, n + 1) * s) / (n * tot)) - (n + 1) / n) if tot > 0 else 0.0
    return {
        "degree_gini": gini,
        "degree_cv": float(deg.std() / k1) if k1 > 0 else 0.0,
        "k2_over_k1": float(k2 / k1) if k1 > 0 else 0.0,
        "max_degree_ratio": float(deg.max() / k1) if k1 > 0 else 0.0,
        "mean_degree": k1,
        "clustering": _avg_clustering(sub.adj),
        "assortativity": _degree_assortativity(sub.adj),
    }


def _avg_clustering(adj: dict[int, list[int]]) -> float:
    total = 0.0
    count = 0
    neigh_sets = {n: set(v) for n, v in adj.items()}
    for n, neigh in adj.items():
        k = len(neigh)
        if k < 2:
            continue
        links = 0
        nl = list(neigh)
        for i in range(len(nl)):
            ni = neigh_sets[nl[i]]
            for j in range(i + 1, len(nl)):
                if nl[j] in ni:
                    links += 1
        total += 2.0 * links / (k * (k - 1))
        count += 1
    return float(total / count) if count else 0.0


def _degree_assortativity(adj: dict[int, list[int]]) -> float:
    """Pearson degree-degree correlation over edges (Newman assortativity)."""
    deg = {n: len(v) for n, v in adj.items()}
    xs: list[float] = []
    ys: list[float] = []
    for u, neigh in adj.items():
        for v in neigh:
            if u < v:
                xs.append(deg[u])
                ys.append(deg[v])
    if len(xs) < 3:
        return 0.0
    x = np.array(xs + ys, dtype=float)  # symmetrize
    y = np.array(ys + xs, dtype=float)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


@dataclass
class DiffusionSpec:
    """What structural feature vs. diffusion outcome to test, and the holdout family."""

    structural_feature: str = "k2_over_k1"
    outcome: str = "final_size"          # "final_size" | "outbreak_prob"
    simulator: str = "sir"               # "sir" | "cascade"
    claim_sign: str = "positive"         # "positive" | "negative"
    held_out_family: str | None = None
    families: tuple[str, ...] = field(default_factory=lambda: tuple(REAL_NETWORKS))

    @classmethod
    def from_hypothesis(cls, hypothesis: dict[str, Any]) -> DiffusionSpec:
        text = str(hypothesis.get("text") or hypothesis.get("statement") or "").lower()
        # Default to <k^2>/<k>, the moment that governs the epidemic threshold
        # (Pastor-Satorras & Vespignani 2001) and the canonical measure of degree
        # heterogeneity for contagion.
        feature = "k2_over_k1"
        if "gini" in text or "inequality" in text:
            feature = "degree_gini"
        elif "clustering" in text or "transitivity" in text:
            feature = "clustering"
        elif "assortativ" in text or "degree correlation" in text:
            feature = "assortativity"
        elif "hub" in text or "max degree" in text:
            feature = "max_degree_ratio"
        elif "coefficient of variation" in text or "degree variance" in text:
            feature = "degree_cv"

        outcome = "final_size"
        if "outbreak probability" in text or "outbreak prob" in text or "epidemic threshold" in text or "takeoff" in text:
            outcome = "outbreak_prob"

        simulator = "sir"
        if "cascade" in text or "independent cascade" in text or "adoption" in text or "influence" in text:
            simulator = "cascade"

        claim_sign = "positive"
        if "negative" in text or "inverse" in text or "decreas" in text or "suppress" in text or "lower" in text:
            claim_sign = "negative"

        held = None
        for fam in REAL_NETWORKS:
            if fam in text or REAL_NETWORKS[fam].split(".")[0].lower() in text:
                held = fam
                break

        return cls(
            structural_feature=feature,
            outcome=outcome,
            simulator=simulator,
            claim_sign=claim_sign,
            held_out_family=held,
        )
