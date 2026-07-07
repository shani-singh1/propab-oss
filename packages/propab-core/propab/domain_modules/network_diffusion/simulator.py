"""
Diffusion simulators run on REAL graphs.

Two independent stochastic dynamics — used both as the primary outcome
generator and as an *alternate simulator* robustness check (a confirmed finding
must not be an artifact of one particular dynamics):

* ``sir`` — discrete-time SIR (susceptible / infected / recovered). Each infected
  node infects each susceptible neighbour with probability ``beta`` per step and
  recovers with probability ``gamma``. Runs to extinction.
* ``cascade`` — independent-cascade (single-activation adoption). Each newly
  active node gets one chance to activate each inactive neighbour with
  probability ``p``. Runs until no new activations.

Neither uses a mean-field / closed-form approximation: they are Monte-Carlo
simulations over the actual adjacency of the real subgraph. The outcome
(``final_size`` = fraction ever infected/active; ``outbreak_prob`` = fraction of
seeds producing a macroscopic outbreak) is measured, not derived.
"""
from __future__ import annotations

import numpy as np

from propab.domain_modules.network_diffusion.adapter import Subgraph

# A run counts as a macroscopic "outbreak" if it reaches this fraction of nodes.
OUTBREAK_FRACTION = 0.10


def _sir_once(adj: dict[int, list[int]], nodes: list[int], seed: int,
              beta: float, gamma: float, rng: np.random.Generator) -> float:
    state = dict.fromkeys(nodes, 0)  # 0=S, 1=I, 2=R
    state[seed] = 1
    infected = 1
    n = len(nodes)
    while infected > 0:
        newly: list[int] = []
        for u in nodes:
            if state[u] != 1:
                continue
            for v in adj[u]:
                if state[v] == 0 and rng.random() < beta:
                    newly.append(v)
            if rng.random() < gamma:
                state[u] = 2
        for v in newly:
            if state[v] == 0:
                state[v] = 1
        infected = sum(1 for x in nodes if state[x] == 1)
    recovered = sum(1 for x in nodes if state[x] == 2)
    return recovered / n


def _cascade_once(adj: dict[int, list[int]], nodes: list[int], seed: int,
                  p: float, rng: np.random.Generator) -> float:
    active = {seed}
    newly = [seed]
    n = len(nodes)
    while newly:
        nxt: list[int] = []
        for u in newly:
            for v in adj[u]:
                if v not in active and rng.random() < p:
                    active.add(v)
                    nxt.append(v)
        newly = nxt
    return len(active) / n


def simulate(sub: Subgraph, *, simulator: str, outcome: str,
             beta: float, gamma: float, n_runs: int,
             rng: np.random.Generator) -> float:
    """
    Run ``n_runs`` Monte-Carlo diffusions from random seeds and aggregate.

    ``outcome='final_size'`` -> mean final infected/active fraction.
    ``outcome='outbreak_prob'`` -> fraction of runs reaching OUTBREAK_FRACTION.
    """
    nodes = sub.nodes
    adj = sub.adj
    n = len(nodes)
    if n == 0:
        return 0.0
    sizes: list[float] = []
    for _ in range(n_runs):
        seed = nodes[int(rng.integers(n))]
        if simulator == "cascade":
            frac = _cascade_once(adj, nodes, seed, beta, rng)
        else:
            frac = _sir_once(adj, nodes, seed, beta, gamma, rng)
        sizes.append(frac)
    arr = np.asarray(sizes, dtype=float)
    if outcome == "outbreak_prob":
        return float(np.mean(arr >= OUTBREAK_FRACTION))
    return float(arr.mean())
