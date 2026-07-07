"""Real SNAP network invariant dataset.

This module builds a cross-network invariant frame from **REAL** public networks
(Stanford SNAP edge lists shipped under ``data/v1_candidates/``), NOT from
seed-generated textbook graph families. Each row is an invariant fingerprint of a
genuine connected induced subgraph sampled (randomised snowball / BFS) from a real
network, so a cross-family invariant correlation tests whether the relationship
holds across *real* network types — a genuinely open question — rather than
re-deriving a Newman-2003 textbook fact from ER/BA/WS/lattice toys.

Provenance (see ``data/graph_invariants/PROVENANCE.md``):

* ``collaboration`` — ``ca-GrQc.txt.gz``: arXiv General Relativity (GR-QC)
  co-authorship network. Leskovec, Kleinberg, Faloutsos (2007); SNAP dataset
  ``ca-GrQc`` (https://snap.stanford.edu/data/ca-GrQc.html).
* ``communication`` — ``email-Eu-core.txt.gz``: e-mail network of a large European
  research institution. Leskovec, Kleinberg, Faloutsos (2007); Yin, Benson,
  Leskovec, Gleich (2017); SNAP dataset ``email-Eu-core``
  (https://snap.stanford.edu/data/email-Eu-core.html).

Both files were fetched from SNAP and cached on disk; no network access is needed
at run time. The families are structurally distinct real networks (a sparse,
high-clustering collaboration graph vs. a dense communication graph), so the
cross-family LOFO in :mod:`.verifier` is a real out-of-distribution test.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from propab.config import settings

# --- Real network sources (SNAP edge lists cached on disk) -------------------
# family_id -> (edge-list filename, human description, SNAP url for citation)
REAL_NETWORKS: tuple[tuple[str, str, str, str], ...] = (
    (
        "collaboration",
        "ca-GrQc.txt.gz",
        "arXiv GR-QC co-authorship (SNAP ca-GrQc)",
        "https://snap.stanford.edu/data/ca-GrQc.html",
    ),
    (
        "communication",
        "email-Eu-core.txt.gz",
        "EU research-institution e-mail (SNAP email-Eu-core)",
        "https://snap.stanford.edu/data/email-Eu-core.html",
    ),
)
GRAPH_FAMILIES = tuple(fam for fam, _, _, _ in REAL_NETWORKS)

# Sampling parameters. Each family contributes ``SUBGRAPHS_PER_FAMILY`` connected
# induced subgraphs of ``SUBGRAPH_NODES`` nodes, snowball-sampled from the largest
# connected component of the real network. This yields a distribution of invariant
# fingerprints per real family without ever fabricating topology — every subgraph
# is a genuine piece of the real network.
SUBGRAPHS_PER_FAMILY = 30
SUBGRAPH_NODES = 100
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


def _source_dir() -> Path:
    """Directory holding the cached real SNAP edge-list files.

    Resolved relative to the repo's ``data/v1_candidates`` so the real networks are
    found regardless of ``propab_data_dir`` (which points at a scratch cache dir in
    tests). ``adapter.py`` lives at
    ``packages/propab-core/propab/domain_modules/graph_invariants/adapter.py``; the
    repo root is five parents up.
    """
    repo_root = Path(__file__).resolve().parents[5]
    return repo_root / "data" / "v1_candidates"


class RealNetworkUnavailable(FileNotFoundError):
    """Raised when a cached real SNAP edge list cannot be found on disk.

    We fail closed (raise) rather than silently falling back to synthetic graphs —
    a "real" cross-family finding must never be computed from fabricated topology.
    """


def _load_real_network(filename: str) -> nx.Graph:
    """Load a real SNAP edge list (gzipped, whitespace-separated, ``#`` comments)."""
    path = _source_dir() / filename
    if not path.is_file():
        raise RealNetworkUnavailable(
            f"real SNAP network {filename!r} not found at {path}; refusing to "
            "fabricate synthetic topology in its place"
        )
    graph = nx.Graph()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            if a != b:  # drop self-loops
                graph.add_edge(int(a), int(b))
    return graph


def _largest_component(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph
    lcc = max(nx.connected_components(graph), key=len)
    return graph.subgraph(lcc).copy()


def _sample_connected_subgraph(
    graph: nx.Graph, size: int, rng: np.random.Generator, nodes: list[int]
) -> nx.Graph:
    """Randomised snowball (BFS) sample of a connected induced subgraph.

    Starts from a random seed node and expands over shuffled neighbours until
    ``size`` nodes are collected, then returns the induced subgraph — which is
    connected by construction. This is a standard real-network subsampling scheme;
    the result is a genuine piece of the real network, not a generated graph.
    """
    if not nodes:
        return graph.subgraph([]).copy()
    start = int(nodes[int(rng.integers(len(nodes)))])
    order: list[int] = [start]
    seen: set[int] = {start}
    i = 0
    while i < len(order) and len(order) < size:
        u = order[i]
        i += 1
        neighbours = list(graph.neighbors(u))
        rng.shuffle(neighbours)
        for v in neighbours:
            if v not in seen:
                seen.add(v)
                order.append(v)
                if len(order) >= size:
                    break
    return graph.subgraph(seen).copy()


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


def _graph_metrics(subgraph: nx.Graph) -> dict[str, float]:
    """Compute the six exposed invariants for a real (connected) subgraph.

    Uses the SAME definitions the verifier expects — real Newman modularity,
    transitivity (global clustering), algebraic connectivity (Fiedler value),
    adjacency spectral gap (λ1 − λ2), diameter, and average degree — computed
    directly on the real subgraph. DOM2b: the six invariants remain independent
    (none is a closed-form function of another).
    """
    graph = nx.convert_node_labels_to_integers(subgraph)
    n = graph.number_of_nodes()
    adj = nx.to_numpy_array(graph)
    deg = adj.sum(axis=1)
    avg_deg = float(deg.mean()) if n else 0.0

    lap = np.diag(deg) - adj
    lap_eigvals = np.linalg.eigvalsh(lap)
    # Algebraic connectivity = Fiedler value = second-smallest Laplacian eigenvalue.
    algebraic_connectivity = float(lap_eigvals[1]) if len(lap_eigvals) > 1 else 0.0
    # Spectral gap: a DISTINCT quantity — the gap between the two largest
    # adjacency-matrix eigenvalues (λ1 - λ2), decoupled from algebraic connectivity.
    adj_eigvals = np.linalg.eigvalsh(adj)
    spectral_gap = (
        float(adj_eigvals[-1] - adj_eigvals[-2]) if len(adj_eigvals) > 1 else 0.0
    )
    # Global clustering coefficient (transitivity) — real triangle density.
    clustering = float(nx.transitivity(graph)) if n > 2 else 0.0
    # Diameter of the (connected) subgraph.
    try:
        diameter = float(nx.diameter(graph)) if n > 1 else 0.0
    except (nx.NetworkXError, nx.NetworkXException):
        diameter = 0.0
    # Real modularity of a Fiedler (spectral) bipartition — a structural community
    # measure, independent of clustering. See _newman_modularity.
    communities = (_fiedler_vector(lap) >= 0).astype(int)
    modularity = _newman_modularity(adj, communities)
    return {
        "spectral_gap": spectral_gap,
        "algebraic_connectivity": algebraic_connectivity,
        "clustering_coefficient": clustering,
        "diameter": diameter,
        "avg_degree": avg_deg,
        "modularity": modularity,
    }


def _real_frame() -> pd.DataFrame:
    """Build the invariant frame from REAL SNAP networks (no synthetic topology)."""
    rows: list[dict[str, Any]] = []
    gid = 0
    for family, filename, _desc, _url in REAL_NETWORKS:
        graph = _largest_component(_load_real_network(filename))
        nodes = list(graph.nodes())
        # Per-family deterministic seed derived from the fixed RANDOM_SEED and the
        # family index (so the frame is reproducible without cross-family aliasing).
        fam_seed = RANDOM_SEED * 1000 + GRAPH_FAMILIES.index(family)
        rng = np.random.default_rng(fam_seed)
        size = min(SUBGRAPH_NODES, len(nodes))
        for _ in range(SUBGRAPHS_PER_FAMILY):
            subgraph = _sample_connected_subgraph(graph, size, rng, nodes)
            metrics = _graph_metrics(subgraph)
            rows.append(
                {"graph_id": f"G{gid:04d}", "network_family": family, **metrics}
            )
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
            if fam.replace("_", " ") in text or fam.replace("_", "-") in text or fam in text:
                held = fam
                break
        return cls(source_invariant=src, target_invariant=tgt, claim_type=claim, held_out_family=held)


class GraphInvariantsAdapter:
    def ensure_cache(self) -> Path:
        path = cache_path()
        if path.is_file():
            return path
        df = _real_frame()
        df.to_csv(path, index=False)
        cache_dir().joinpath("snap_subset_v1.meta.json").write_text(
            json.dumps(
                {
                    "graphs": len(df),
                    "families": list(GRAPH_FAMILIES),
                    "synthetic": False,
                    "data_provenance": "real",
                    "sources": [
                        {"family": fam, "file": fn, "description": desc, "url": url}
                        for fam, fn, desc, url in REAL_NETWORKS
                    ],
                    "sampling": {
                        "method": "randomised snowball (BFS) connected induced subgraph",
                        "subgraphs_per_family": SUBGRAPHS_PER_FAMILY,
                        "subgraph_nodes": SUBGRAPH_NODES,
                        "random_seed": RANDOM_SEED,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def load_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.ensure_cache())
