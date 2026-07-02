"""Reproducible multi-topology graph contagion benchmark for permutation audits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class GraphContagionRow:
    topology_family: str
    n_nodes: int
    avg_degree: float
    clustering: float
    modularity: float
    max_k_shell: float
    k_core_density: float
    assortativity: float
    spectral_gap: float
    bridge_density: float
    mixing_mu: float
    outbreak_fraction: float
    saturation_time: float

    def feature_dict(self) -> dict[str, float]:
        return {
            "avg_degree": self.avg_degree,
            "clustering": self.clustering,
            "modularity": self.modularity,
            "max_k_shell": self.max_k_shell,
            "k_core_density": self.k_core_density,
            "assortativity": self.assortativity,
            "spectral_gap": self.spectral_gap,
            "bridge_density": self.bridge_density,
            "mixing_mu": self.mixing_mu,
        }


def _clip_r2(r2: float) -> float:
    return float(max(min(r2, 1.0), -1.0))


def _k_core_density(G: nx.Graph) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    core = nx.core_number(G)
    if not core:
        return 0.0
    max_core = max(core.values())
    if max_core <= 0:
        return 0.0
    inner = sum(1 for v in core.values() if v >= max_core)
    return inner / G.number_of_nodes()


def _max_k_shell(G: nx.Graph) -> float:
    try:
        core = nx.core_number(G)
        return float(max(core.values()) if core else 0.0)
    except nx.NetworkXError:
        return 0.0


def _modularity(G: nx.Graph) -> float:
    try:
        comms = nx.community.greedy_modularity_communities(G, seed=42)
        return float(nx.community.modularity(G, comms))
    except Exception:
        return 0.0


def _spectral_gap(G: nx.Graph) -> float:
    if G.number_of_nodes() < 3:
        return 0.0
    try:
        A = nx.to_numpy_array(G, dtype=float)
        eig = np.linalg.eigvalsh(A)
        eig = np.sort(eig)[::-1]
        if len(eig) < 2:
            return 0.0
        return float(abs(eig[0] - eig[1]))
    except np.linalg.LinAlgError:
        return 0.0


def _bridge_density(G: nx.Graph) -> float:
    if G.number_of_edges() == 0:
        return 0.0
    try:
        bridges = set(nx.bridges(G))
        return len(bridges) / G.number_of_edges()
    except nx.NetworkXError:
        return 0.0


def _sis_outbreak_fraction(G: nx.Graph, *, beta: float, mu: float, trials: int, rng: np.random.Generator) -> tuple[float, float]:
    n = G.number_of_nodes()
    if n == 0:
        return 0.0, 0.0
    nodes = list(G.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    fracs: list[float] = []
    sat_times: list[float] = []
    max_steps = max(50, 3 * n)
    for _ in range(trials):
        state = np.zeros(n, dtype=bool)
        state[rng.integers(0, n)] = True
        peak = state.sum()
        t_peak = 0
        for t in range(1, max_steps + 1):
            new = state.copy()
            for i, node in enumerate(nodes):
                if state[i]:
                    if rng.random() < mu:
                        new[i] = False
                else:
                    for nb in G.neighbors(node):
                        j = idx[nb]
                        if state[j] and rng.random() < beta:
                            new[i] = True
                            break
            state = new
            if state.sum() > peak:
                peak = int(state.sum())
                t_peak = t
            if not state.any():
                break
        fracs.append(float(state.sum()) / n)
        sat_times.append(float(t_peak))
    return float(np.mean(fracs)), float(np.mean(sat_times))


def _make_graph(family: str, n: int, rng: np.random.Generator) -> nx.Graph:
    if family == "ER":
        p = float(rng.uniform(0.01, 0.08))
        G = nx.gnp_random_graph(n, p, seed=int(rng.integers(0, 2**31)))
    elif family == "BA":
        m = int(rng.integers(2, min(6, n // 2)))
        G = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(0, 2**31)))
    elif family == "WS":
        k = int(rng.integers(4, min(12, n // 2)))
        p = float(rng.uniform(0.01, 0.3))
        G = nx.watts_strogatz_graph(n, k, p, seed=int(rng.integers(0, 2**31)))
    elif family == "SBM":
        sizes = [n // 4, n // 4, n // 2]
        sizes[-1] += n - sum(sizes)
        p_in = float(rng.uniform(0.05, 0.2))
        p_out = float(rng.uniform(0.001, 0.02))
        probs = [
            [p_in, p_out, p_out],
            [p_out, p_in, p_out],
            [p_out, p_out, p_in],
        ]
        G = nx.stochastic_block_model(
            sizes, probs, seed=int(rng.integers(0, 2**31)), directed=False
        )
        G = nx.Graph(G)
    else:
        raise ValueError(f"unknown family {family!r}")
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_edges() == 0:
        G = nx.gnp_random_graph(n, 0.05, seed=int(rng.integers(0, 2**31)))
    return G


def generate_benchmark(
    *,
    per_family: int = 40,
    n_nodes: int = 300,
    sis_trials: int = 3,
    seed: int = 42,
) -> list[GraphContagionRow]:
    families = ("ER", "BA", "WS", "SBM")
    rng = np.random.default_rng(seed)
    rows: list[GraphContagionRow] = []
    for family in families:
        for _ in range(per_family):
            G = _make_graph(family, n_nodes, rng)
            beta = float(rng.uniform(0.15, 0.35))
            mu = float(rng.uniform(0.05, 0.2))
            outbreak, sat = _sis_outbreak_fraction(G, beta=beta, mu=mu, trials=sis_trials, rng=rng)
            rows.append(
                GraphContagionRow(
                    topology_family=family,
                    n_nodes=G.number_of_nodes(),
                    avg_degree=float(np.mean([d for _, d in G.degree()]) if G.number_of_nodes() else 0.0),
                    clustering=float(nx.average_clustering(G)),
                    modularity=_modularity(G),
                    max_k_shell=_max_k_shell(G),
                    k_core_density=_k_core_density(G),
                    assortativity=float(nx.degree_assortativity_coefficient(G))
                    if G.number_of_edges()
                    else 0.0,
                    spectral_gap=_spectral_gap(G),
                    bridge_density=_bridge_density(G),
                    mixing_mu=mu,
                    outbreak_fraction=outbreak,
                    saturation_time=sat,
                )
            )
    return rows


def benchmark_arrays(
    rows: list[GraphContagionRow],
    *,
    feature_cols: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([[r.feature_dict()[c] for c in feature_cols] for r in rows], dtype=float)
    if target == "outbreak_fraction":
        y = np.array([r.outbreak_fraction for r in rows], dtype=float)
    elif target == "saturation_time":
        y = np.array([r.saturation_time for r in rows], dtype=float)
    else:
        y = np.array([getattr(r, target) for r in rows], dtype=float)
    families = np.array([r.topology_family for r in rows])
    return X, y, families


def infer_finding_proxy(text: str, theme: str) -> dict[str, Any]:
    """Map a confirmed finding to benchmark features + target."""
    low = (text or "").lower()
    features: list[str] = []
    target = "outbreak_fraction"

    if any(k in low for k in ("k-shell", "k shell", "kshell")):
        features.append("max_k_shell")
    if any(k in low for k in ("k-core", "k core", "core density", "core population")):
        features.append("k_core_density")
    if "degree" in low or theme in ("degree_structure",):
        features.append("avg_degree")
    if "clustering" in low or theme == "clustering":
        features.append("clustering")
    if "modular" in low or "modularity" in low or "q >" in low:
        features.append("modularity")
    if "assortativity" in low or theme == "assortativity":
        features.append("assortativity")
    if "spectral" in low or theme == "spectral":
        features.append("spectral_gap")
    if any(k in low for k in ("mixing", " mu", "multy", "kendall")):
        features.append("mixing_mu")
    if "bridge" in low:
        features.append("bridge_density")
    if any(k in low for k in ("saturation", "speed", "time-to-peak", "t_peak")):
        target = "saturation_time"

    if not features:
        theme_defaults = {
            "diffusion_dynamics": ["max_k_shell", "modularity"],
            "degree_structure": ["k_core_density", "avg_degree"],
            "clustering": ["clustering", "mixing_mu"],
            "assortativity": ["assortativity", "avg_degree"],
            "centrality": ["max_k_shell", "avg_degree"],
            "spectral": ["spectral_gap", "clustering"],
            "general": ["max_k_shell", "mixing_mu"],
        }
        features = theme_defaults.get(theme, ["max_k_shell", "clustering", "spectral_gap"])

    # dedupe preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for f in features:
        if f not in seen:
            seen.add(f)
            deduped.append(f)

    compare_pair: tuple[str, str] | None = None
    if "max_k_shell" in deduped and "avg_degree" in deduped:
        if any(k in low for k in ("compared to", "than average degree", "than degree", "higher correlation")):
            compare_pair = ("max_k_shell", "avg_degree")

    return {
        "feature_cols": deduped,
        "target": target,
        "compare_pair": compare_pair,
    }
