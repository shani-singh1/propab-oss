"""evolve/targets/conjectures.py — the conjecture library for Target B (counterexample hunting).

Every entry is a *published* conjecture of the shape ``lhs(G) <= rhs(G)`` over a finite graph,
normalised so that

    margin(G) = lhs(G) - rhs(G)          margin > 0  <=>  G REFUTES the conjecture

That single number is the search objective. A positive margin is not "evidence"; it is a finite,
fully-checkable counterexample that settles the question. This is the whole reason Target B is the
asymmetric bet: the verifier is exact, and the object is a small graph.

------------------------------------------------------------------------------------------------
HONESTY RULES (this project has twice reported a "win" that dissolved — read before adding an entry)
------------------------------------------------------------------------------------------------
1. Every entry carries a `source` (a real citation) and a `status` that was CHECKED against the
   literature on `status_checked` — never recalled from a model's memory.
2. Only `Status.OPEN` conjectures are `live`. `GraphConjectureProblem` REFUSES to hunt a non-live
   conjecture unless the caller explicitly passes `allow_non_live=True`. A "refutation" of
   something already refuted — or since proven — is worthless and embarrassing.
3. Statuses go stale. `brouwer_laplacian` was listed as OPEN in the 2023 Liu-Ning survey and was
   PROVEN in 2026. It is kept here, marked PROVEN, precisely as the standing reminder to re-check.
4. Every conjecture below was validated against the exhaustive atlas of all 994 connected graphs on
   <= 7 vertices: each OPEN conjecture's margin is <= 0 on every one of them (as it must be — all
   four are verified in the literature for n <= 10). If an evaluator here fires on a small graph,
   the TRANSCRIPTION is wrong, not the conjecture. `tests/evolve/targets/` re-runs that check.

------------------------------------------------------------------------------------------------
WHY EVERY CONJECTURE HERE REQUIRES A **CONNECTED** GRAPH
------------------------------------------------------------------------------------------------
Not cosmetic — it closes a degeneracy that would manufacture a fake refutation:

  Bollobas-Nikiforov is usually stated "for G not a complete graph". Taken literally, K_n plus one
  isolated vertex is "not a complete graph" — yet lambda_1, lambda_2, m and omega are all unchanged
  by isolated vertices, so it inherits K_n's margin of +1. A naive implementation therefore
  "refutes" BN in one line, with a graph no referee would accept. Likewise EFGW
  (min{s+,s-} >= n-1) is broken trivially by any disconnected graph (two disjoint edges give
  s+ = 2 < 3 = n-1) — which is exactly why its authors state it for connected graphs.

So: connected, no isolated vertices, and complete graphs excluded where the source excludes them.
A counterexample found inside this class is unambiguously a real counterexample.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import networkx as nx
import numpy as np

# A refutation must clear this margin. Measured floating-point noise on the *tight* families (every
# tree is an exact equality case for EFGW; complete multipartite graphs are exact equality cases for
# BN) peaks at ~2e-12 for n <= 40, while the known counterexample to the positive control has margin
# ~8e-2. So 1e-6 is a >= 6-order-of-magnitude cushion in both directions: no floating-point artefact
# can clear it, and no real counterexample comes anywhere near it.
MARGIN_EPS = 1e-6

# Exact clique number is the only super-polynomial invariant in this library. Branch-and-bound
# max-clique (nx.max_weight_clique) is fast on the graphs a search actually produces, but adversarial
# inputs can blow up: a Moon-Moser graph (complete multipartite, parts of size 3) has 3^(n/3) maximal
# cliques. Measured on this machine, enumerating them (nx.find_cliques) takes 191 SECONDS at n=45,
# whereas branch-and-bound takes 168ms. Hence `Inv.omega` uses branch-and-bound, and the conjectures
# that need omega cap n at MAX_N_CLIQUE so the verifier stays in milliseconds on a worst-case mutant.
MAX_N_CLIQUE = 34
MAX_N_DEFAULT = 40


class Status(str, Enum):
    """Checked status of the conjecture in the literature."""

    OPEN = "open"                # neither proven nor refuted -> a live hunting target
    REFUTED = "refuted"          # a counterexample is published -> usable only as a positive control
    PROVEN = "proven"            # it is a theorem -> usable only as a negative control
    UNVERIFIED = "unverified"    # statement/status could not be confirmed -> NEVER live


@dataclass(frozen=True)
class GraphClass:
    """The class C in 'for all graphs G in C, f(G) <= g(G)'."""

    min_n: int = 3
    max_n: int = MAX_N_DEFAULT
    connected: bool = True
    exclude_complete: bool = False
    min_edges: int = 1

    def violation(self, g: nx.Graph) -> str | None:
        """None => g IS in the class. Otherwise, a human-readable reason it is not."""
        n = g.number_of_nodes()
        m = g.number_of_edges()
        if n < self.min_n:
            return f"n={n} < min_n={self.min_n}"
        if n > self.max_n:
            return f"n={n} > max_n={self.max_n}"
        if m < self.min_edges:
            return f"m={m} < min_edges={self.min_edges}"
        if self.connected and not nx.is_connected(g):
            return "graph is not connected"
        if self.exclude_complete and m == n * (n - 1) // 2:
            return "graph is complete (excluded by the conjecture's own statement)"
        return None

    def describe(self) -> str:
        bits = [f"{self.min_n} <= n <= {self.max_n} vertices"]
        if self.connected:
            bits.append("connected")
        if self.exclude_complete:
            bits.append("NOT the complete graph K_n")
        bits.append("simple, undirected, unweighted (no self-loops, no multi-edges)")
        return "; ".join(bits)


class Inv:
    """Graph invariants, computed lazily and memoised per graph. All exact. All polynomial except
    `omega` (exact branch-and-bound max-clique; capped via MAX_N_CLIQUE at the class level)."""

    def __init__(self, g: nx.Graph) -> None:
        self.g = g
        self.n = g.number_of_nodes()
        self.m = g.number_of_edges()
        self._eigs: np.ndarray | None = None
        self._leigs: np.ndarray | None = None
        self._omega: int | None = None
        self._matching: int | None = None

    @property
    def eigs(self) -> np.ndarray:
        """Adjacency eigenvalues, DESCENDING (lambda_1 >= ... >= lambda_n)."""
        if self._eigs is None:
            a = nx.to_numpy_array(self.g, dtype=float)
            self._eigs = np.sort(np.linalg.eigvalsh(a))[::-1]
        return self._eigs

    @property
    def laplacian_eigs(self) -> np.ndarray:
        """Laplacian eigenvalues, DESCENDING (mu_1 >= ... >= mu_n = 0)."""
        if self._leigs is None:
            lap = np.asarray(nx.laplacian_matrix(self.g).todense(), dtype=float)
            self._leigs = np.sort(np.linalg.eigvalsh(lap))[::-1]
        return self._leigs

    @property
    def lam1(self) -> float:
        return float(self.eigs[0])

    @property
    def lam2(self) -> float:
        return float(self.eigs[1]) if self.n > 1 else 0.0

    @property
    def s_plus(self) -> float:
        """s^+(G): sum of squares of the POSITIVE adjacency eigenvalues (positive square energy)."""
        e = self.eigs
        return float(np.sum(e[e > 0.0] ** 2))

    @property
    def s_minus(self) -> float:
        """s^-(G): sum of squares of the NEGATIVE adjacency eigenvalues (negative square energy)."""
        e = self.eigs
        return float(np.sum(e[e < 0.0] ** 2))

    @property
    def n_plus(self) -> int:
        """n^+(G): the number of positive adjacency eigenvalues. Uses a tolerance: eigenvalues of a
        0/1 matrix that are mathematically 0 come back as ~1e-16, and counting those as positive
        would silently inflate `ell` in the Elphick-Linz-Wocjan sum."""
        return int(np.sum(self.eigs > 1e-9))

    @property
    def omega(self) -> int:
        """Clique number (exact)."""
        if self._omega is None:
            if self.m == 0:
                self._omega = 1 if self.n else 0
            elif self.m == self.n * (self.n - 1) // 2:
                # Fast path: K_n. Branch-and-bound is at its SLOWEST on complete graphs (156ms at
                # n=40 on this machine), and it is the one case we can answer in O(1).
                self._omega = self.n
            else:
                _, w = nx.max_weight_clique(self.g, weight=None)
                self._omega = int(w)
        return self._omega

    @property
    def matching(self) -> int:
        """Matching number mu(G) — exact.

        Bipartite graphs take the Hopcroft-Karp path, which is exact and roughly an order of
        magnitude faster than general Blossom in networkx. This is not a micro-optimisation: the
        hunting ground for the matching-number conjecture is TREES (which are bipartite), so the
        inner loop of the search lands here on nearly every evaluation, and the entire approach rests
        on the verifier staying cheap. Non-bipartite graphs fall back to Blossom.
        """
        if self._matching is None:
            if self.m and nx.is_connected(self.g) and nx.is_bipartite(self.g):
                # hopcroft_karp_matching returns a dict containing BOTH directions of each edge.
                self._matching = len(nx.bipartite.hopcroft_karp_matching(self.g)) // 2
            else:
                self._matching = len(nx.max_weight_matching(self.g, maxcardinality=True))
        return self._matching


# Each `sides` returns a dict that MUST contain "lhs" and "rhs"; every other key is recorded in the
# witness as a supporting invariant. margin = lhs - rhs, so > 0 ALWAYS means "conjecture violated",
# whichever way the source happened to write the inequality.
SidesFn = Callable[[Inv], dict[str, float]]


def _sides_bollobas_nikiforov(iv: Inv) -> dict[str, float]:
    return {
        "lhs": iv.lam1**2 + iv.lam2**2,
        "rhs": 2.0 * iv.m * (1.0 - 1.0 / iv.omega),
        "lambda_1": iv.lam1,
        "lambda_2": iv.lam2,
        "omega": float(iv.omega),
        "m": float(iv.m),
    }


def _sides_elphick_linz_wocjan(iv: Inv) -> dict[str, float]:
    # ell = min(n^+, omega). The n^+ cap is NOT decoration. Summing lambda_i^2 over the omega largest
    # eigenvalues *including negative ones* is "refuted" instantly by K_n (the omega-1 eigenvalues
    # equal to -1 pile onto the left side), which is plainly not what the authors mean. The Liu-Ning
    # survey states it with ell = min{n^+(G), omega(G)}; that form is exactly tight on K_n, and is
    # what is implemented here. Getting this wrong is precisely how a fake "win" gets manufactured.
    ell = min(iv.n_plus, iv.omega)
    return {
        "lhs": float(np.sum(iv.eigs[:ell] ** 2)) if ell > 0 else 0.0,
        "rhs": 2.0 * iv.m * (1.0 - 1.0 / iv.omega),
        "ell": float(ell),
        "n_plus": float(iv.n_plus),
        "omega": float(iv.omega),
        "m": float(iv.m),
    }


def _sides_elphick_wocjan(iv: Inv) -> dict[str, float]:
    # sqrt(s^+) <= n(1 - 1/omega); equivalently omega >= n / (n - sqrt(s^+)), the conjectured
    # strengthening of Turan's theorem (replace the average degree d by sqrt(s^+)).
    return {
        "lhs": math.sqrt(iv.s_plus),
        "rhs": iv.n * (1.0 - 1.0 / iv.omega),
        "s_plus": iv.s_plus,
        "omega": float(iv.omega),
        "n": float(iv.n),
    }


def _sides_efgw(iv: Inv) -> dict[str, float]:
    # min{s^+, s^-} >= n - 1, rewritten as (n-1) - min{s^+, s^-} <= 0.
    return {
        "lhs": float(iv.n - 1),
        "rhs": min(iv.s_plus, iv.s_minus),
        "s_plus": iv.s_plus,
        "s_minus": iv.s_minus,
        "n": float(iv.n),
    }


def _sides_aouchiche_hansen_lam1_matching(iv: Inv) -> dict[str, float]:
    # lambda_1 + mu >= sqrt(n-1) + 1, rewritten as (sqrt(n-1) + 1) - (lambda_1 + mu) <= 0.
    return {
        "lhs": math.sqrt(iv.n - 1) + 1.0,
        "rhs": iv.lam1 + iv.matching,
        "lambda_1": iv.lam1,
        "matching_number": float(iv.matching),
        "n": float(iv.n),
    }


def _sides_hong(iv: Inv) -> dict[str, float]:
    # Hong's THEOREM: lambda_1 <= sqrt(2m - n + 1) whenever the minimum degree is >= 1.
    return {
        "lhs": iv.lam1,
        "rhs": math.sqrt(max(2.0 * iv.m - iv.n + 1.0, 0.0)),
        "lambda_1": iv.lam1,
        "m": float(iv.m),
        "n": float(iv.n),
    }


def _sides_brouwer(iv: Inv) -> dict[str, float]:
    # Brouwer: for EVERY k, sum_{i<=k} mu_i(L) <= m + C(k+1, 2). Reduced to a single number by taking
    # the worst k — any k with a positive margin would refute it.
    mu = iv.laplacian_eigs
    best_k, best_margin, running = 1, -math.inf, 0.0
    for k in range(1, iv.n + 1):
        running += float(mu[k - 1])
        margin = running - (iv.m + k * (k + 1) / 2.0)
        if margin > best_margin:
            best_margin, best_k = margin, k
    return {
        "lhs": float(np.sum(mu[:best_k])),
        "rhs": iv.m + best_k * (best_k + 1) / 2.0,
        "worst_k": float(best_k),
        "m": float(iv.m),
    }


@dataclass(frozen=True)
class Conjecture:
    """One named, published conjecture, normalised to `lhs <= rhs`."""

    key: str
    title: str
    statement: str          # the exact statement, as published
    source: str             # a real citation — never a model's memory
    status: Status
    status_checked: str     # ISO date on which `status` was checked against the literature
    status_note: str        # what is proven so far / who refuted it / where the status came from
    graph_class: GraphClass
    sides: SidesFn
    verified_up_to_n: int = 0   # exhaustively checked in the literature for all n <= this
    hunting_notes: str = ""     # the proven special cases => where a counterexample CANNOT be

    @property
    def live(self) -> bool:
        """Only an OPEN conjecture is a legitimate hunting target."""
        return self.status is Status.OPEN

    def margin(self, g: nx.Graph) -> float:
        """> 0  <=>  g refutes the conjecture."""
        s = self.sides(Inv(g))
        return float(s["lhs"] - s["rhs"])

    def evaluate(self, g: nx.Graph) -> dict[str, float]:
        """Both sides, the margin, and the supporting invariants (the witness payload)."""
        s = self.sides(Inv(g))
        s["margin"] = float(s["lhs"] - s["rhs"])
        return s


# ------------------------------------------------------------------------------------------------
# THE LIBRARY
#
# Statuses below were checked against the literature on 2026-07-14 (see each `status_note`). The four
# OPEN entries all live in the "sums of squares of eigenvalues vs the clique number" family; all four
# are catalogued as open in the Liu-Ning survey of unsolved problems in spectral graph theory
# (arXiv:2305.10290), and all four were re-confirmed open against 2024-2026 follow-up papers.
# ------------------------------------------------------------------------------------------------

_CONJECTURES: tuple[Conjecture, ...] = (
    # ---------------------------------------------------------------- OPEN (the live targets) ----
    Conjecture(
        key="bollobas_nikiforov",
        title="Bollobas-Nikiforov (2007)",
        statement=(
            "For every connected graph G with m edges and clique number omega(G), other than the "
            "complete graph:  lambda_1(G)^2 + lambda_2(G)^2 <= 2m (1 - 1/omega(G)),  where "
            "lambda_1 >= lambda_2 are the two largest adjacency eigenvalues."
        ),
        source=(
            "B. Bollobas, V. Nikiforov, 'Cliques and the spectral radius', J. Combin. Theory Ser. B "
            "97 (2007) 859-865. Catalogued as open in: Liu & Ning, 'Unsolved Problems in Spectral "
            "Graph Theory', arXiv:2305.10290, Conjecture 3."
        ),
        status=Status.OPEN,
        status_checked="2026-07-14",
        status_note=(
            "OPEN in full generality. Still being attacked special-case-by-special-case as of 2026: "
            "arXiv:2407.19341 (graphs with few triangles), arXiv:2501.07137 (asymptotically almost "
            "surely), arXiv:2603.26379 (complete multipartite and dense K_4-free graphs). "
            "arXiv:2411.08184 (Nov 2024) can only prove a weakened constant C ~ 1.4231 in place of "
            "1. No counterexample is known."
        ),
        graph_class=GraphClass(min_n=3, max_n=MAX_N_CLIQUE, connected=True, exclude_complete=True),
        sides=_sides_bollobas_nikiforov,
        verified_up_to_n=10,
        hunting_notes=(
            "PROVEN (so a counterexample CANNOT live there): triangle-free graphs (omega = 2), "
            "regular graphs, weakly perfect graphs (omega = chromatic number), complete multipartite "
            "graphs, dense K_4-free graphs, and graphs with O(m^{1.5-eps}) triangles (hence planar "
            "and book-free graphs). => Hunt IRREGULAR, TRIANGLE-RICH, NON-PLANAR graphs whose clique "
            "number is strictly BELOW their chromatic number. Equality holds for complete "
            "multipartite graphs, so perturbing those is the natural starting move."
        ),
    ),
    Conjecture(
        key="elphick_linz_wocjan",
        title="Elphick-Linz-Wocjan (2021) — strengthened Bollobas-Nikiforov",
        statement=(
            "For every connected graph G with m edges and clique number omega(G):  "
            "sum_{i=1}^{ell} lambda_i(G)^2 <= 2m (1 - 1/omega(G)),  where ell = min{n^+(G), "
            "omega(G)} and n^+(G) is the number of positive adjacency eigenvalues."
        ),
        source=(
            "C. Elphick, W. Linz, P. Wocjan, 'Two conjectured strengthenings of Turan's theorem', "
            "arXiv:2101.05229 (Conjecture 2). Catalogued as open in: Liu & Ning, arXiv:2305.10290, "
            "Conjecture 4."
        ),
        status=Status.OPEN,
        status_checked="2026-07-14",
        status_note=(
            "OPEN. Verified for weakly perfect graphs, Kneser graphs and Johnson graphs (Liu & Ning "
            "survey, Conjecture 4). No counterexample is known."
        ),
        graph_class=GraphClass(min_n=3, max_n=MAX_N_CLIQUE, connected=True),
        sides=_sides_elphick_linz_wocjan,
        verified_up_to_n=10,
        hunting_notes=(
            "STRICTLY STRONGER than Bollobas-Nikiforov: whenever omega >= 2 and n^+ >= 2 we have "
            "ell >= 2, so the left side is at least lambda_1^2 + lambda_2^2 while the right side is "
            "IDENTICAL to BN's. Hence every BN counterexample is an ELW counterexample, and ELW has "
            "a strictly larger violation space — it is the cheaper of the two to break, and the "
            "best value-for-compute target in this library. BN's pruning applies (weakly perfect "
            "graphs are proven), plus Kneser and Johnson graphs. => Hunt graphs with MANY positive "
            "eigenvalues and a LARGE clique number (both raise ell, hence the left side) but FEW "
            "edges (which lowers the right side)."
        ),
    ),
    Conjecture(
        key="elphick_wocjan",
        title="Elphick-Wocjan (2018) — Turan strengthening via positive square energy",
        statement=(
            "For every connected graph G on n vertices with clique number omega(G):  "
            "sqrt(s^+(G)) <= n (1 - 1/omega(G)),  where s^+(G) is the sum of the squares of the "
            "positive adjacency eigenvalues. Equivalently:  omega(G) >= n / (n - sqrt(s^+(G)))."
        ),
        source=(
            "C. Elphick, P. Wocjan, 'Conjectured lower bound for the clique number of a graph', "
            "arXiv:1804.03752; restated as Conjecture 1 of C. Elphick, W. Linz, P. Wocjan, 'Two "
            "conjectured strengthenings of Turan's theorem', arXiv:2101.05229. Catalogued as open "
            "in: Liu & Ning, arXiv:2305.10290, Conjecture 2."
        ),
        status=Status.OPEN,
        status_checked="2026-07-14",
        status_note=(
            "OPEN. Proven for triangle-free graphs and for almost all graphs; verified exhaustively "
            "for all graphs on <= 10 vertices. Software searches for a counterexample have so far "
            "found none."
        ),
        graph_class=GraphClass(min_n=3, max_n=MAX_N_CLIQUE, connected=True),
        sides=_sides_elphick_wocjan,
        verified_up_to_n=10,
        hunting_notes=(
            "A Turan-type strengthening: it says sqrt(s^+) may replace the average degree d in "
            "Turan's bound omega >= n/(n-d). PROVEN for triangle-free graphs and for almost all "
            "graphs => the counterexample, if it exists, is a STRUCTURED (non-random), "
            "TRIANGLE-BEARING graph with a small clique number but unusually large positive square "
            "energy. Equality at K_n."
        ),
    ),
    Conjecture(
        key="efgw_min_square_energy",
        title="Elphick-Farber-Goldberg-Wocjan (2016) — min square energy",
        statement=(
            "For every connected graph G on n vertices:  min{s^+(G), s^-(G)} >= n - 1,  where "
            "s^+(G) and s^-(G) are the sums of the squares of the positive and of the negative "
            "adjacency eigenvalues respectively."
        ),
        source=(
            "C. Elphick, M. Farber, F. Goldberg, P. Wocjan, 'Conjectured bounds for the sum of "
            "squares of positive eigenvalues of a graph', Discrete Math. 339 (2016) 2215-2223 "
            "(arXiv:1409.2079). Catalogued as open in: Liu & Ning, arXiv:2305.10290, Conjecture 1."
        ),
        status=Status.OPEN,
        status_checked="2026-07-14",
        status_note=(
            "OPEN. Proven for bipartite, regular and complete k-partite graphs, and for almost all "
            "graphs; verified for all connected graphs on <= 10 vertices. The best known general "
            "lower bound is min{s^+,s^-} >= n - gamma(G), where gamma is the domination number "
            "(arXiv:2409.15504). Work continues on special cases into 2026 (arXiv:2605.24668, "
            "unicyclic graphs)."
        ),
        graph_class=GraphClass(min_n=3, max_n=MAX_N_DEFAULT, connected=True),
        sides=_sides_efgw,
        verified_up_to_n=10,
        hunting_notes=(
            "CHEAPEST verifier in the library — adjacency eigenvalues only, no clique number — so it "
            "runs in well under a millisecond and tolerates n up to 40. It is TIGHT (margin exactly "
            "0) on EVERY tree: a bipartite graph has s^+ = s^- = m, and a tree has m = n-1, so the "
            "entire tree family sits exactly on the boundary. The hunt is therefore a perturbation "
            "problem — nudge a tree off the boundary in the right direction. PROVEN for bipartite "
            "and for regular graphs, and min{s^+,s^-} >= n - gamma(G) forces any counterexample to "
            "have DOMINATION NUMBER >= 2. => Hunt sparse, NON-bipartite (odd girth), irregular, "
            "near-tree graphs (unicyclic / bicyclic) on n >= 11 vertices with no dominating vertex."
        ),
    ),
    # ---------------------------------------------------- REFUTED (the positive control) ---------
    Conjecture(
        key="aouchiche_hansen_lam1_matching",
        title="Aouchiche-Hansen / AutoGraphiX — spectral radius + matching number [REFUTED]",
        statement=(
            "For every connected graph G on n >= 3 vertices:  lambda_1(G) + mu(G) >= sqrt(n-1) + 1, "
            "where lambda_1 is the adjacency spectral radius and mu is the matching number."
        ),
        source=(
            "M. Aouchiche, P. Hansen, 'A survey of automated conjectures in spectral graph theory', "
            "Linear Algebra Appl. 432 (2010) 2293-2322 (an AutoGraphiX conjecture). REFUTED by "
            "A. Z. Wagner, 'Constructions in combinatorics via neural networks', arXiv:2104.14516 "
            "(2021), with a counterexample on 19 vertices; independently re-refuted by the AMCS "
            "search of arXiv:2306.07956 and the Monte-Carlo search of arXiv:2207.03343."
        ),
        status=Status.REFUTED,
        status_checked="2026-07-14",
        status_note=(
            "POSITIVE CONTROL — never report a 'discovery' here; the counterexample is published. "
            "This is the target that proves the ENGINE works. Wagner's RL found a 19-vertex "
            "counterexample with no LLM at all, so if our search cannot rediscover one, the engine "
            "is broken — not the conjecture. See `known_counterexample()` in graph_conj.py."
        ),
        graph_class=GraphClass(min_n=3, max_n=MAX_N_DEFAULT, connected=True),
        sides=_sides_aouchiche_hansen_lam1_matching,
        verified_up_to_n=0,
        hunting_notes=(
            "TIGHT (margin exactly 0) at the star K_{1,n-1}, where lambda_1 = sqrt(n-1) and mu = 1. "
            "So the conjecture asserts the star is optimal, and the hunt is for a graph that buys a "
            "reduction of more than 1 in lambda_1 for the 1 extra unit of matching number it must "
            "pay. Wagner's counterexample (n = 19) is the smallest known. But the WHOLE FAMILY of "
            "balanced double stars S(a,a) -- two adjacent centres with a leaves each, n = 2a+2 -- "
            "refutes it for every a >= 13, with a closed form: mu = 2 and lambda_1 = "
            "(1 + sqrt(1+4a))/2, so margin = sqrt(2a+1) + 1 - (1+sqrt(1+4a))/2 - 2, which is exactly "
            "0 at a = 12 (n = 26, lambda_1 = 4 exactly) and strictly positive for a >= 13. Verified "
            "numerically against the closed form to 1e-9 in the test suite. NOTE the search-relevant "
            "moral: at FIXED n = 19 the star is a deceptive local optimum and greedy object-space "
            "search stalls on it, but sweeping n makes the counterexample trivially reachable."
        ),
    ),
    # ------------------------------------------ PROVEN (negative controls; NOT hunting targets) --
    Conjecture(
        key="hong_spectral_radius",
        title="Hong's bound (1988) [THEOREM — negative control]",
        statement=(
            "For every graph G with n vertices, m edges and minimum degree >= 1:  "
            "lambda_1(G) <= sqrt(2m - n + 1)."
        ),
        source="Y. Hong, 'Bounds of eigenvalues of graphs', Discrete Math. 61 (1988) 45-49.",
        status=Status.PROVEN,
        status_checked="2026-07-14",
        status_note=(
            "NEGATIVE CONTROL. This is a THEOREM, so the margin must be <= 0 on every graph, always. "
            "If the search ever reports a positive margin here, the bug is in our evaluator or our "
            "candidate decoding — not in mathematics. Cheap, elementary, and (unlike Brouwer below) "
            "in no danger of being overturned."
        ),
        graph_class=GraphClass(min_n=3, max_n=MAX_N_DEFAULT, connected=True),
        sides=_sides_hong,
        verified_up_to_n=0,
        hunting_notes="A theorem. Nothing to hunt — used only to detect a broken verifier.",
    ),
    Conjecture(
        key="brouwer_laplacian",
        title="Brouwer's Laplacian conjecture (2008) [PROVEN IN 2026 — do not hunt]",
        statement=(
            "For every graph G with m edges and every 1 <= k <= n:  "
            "sum_{i=1}^{k} mu_i(G) <= m + C(k+1, 2),  where mu_1 >= ... >= mu_n are the Laplacian "
            "eigenvalues. (Implemented as the worst k — i.e. the largest margin over all k.)"
        ),
        source=(
            "A. E. Brouwer, W. H. Haemers, 'Spectra of Graphs', Springer (2012), Conjecture 3.7.1. "
            "PROVED by Kothari and Tudose, arXiv:2606.12197 (2026); their proof is used as an "
            "established result by arXiv:2607.08452 (July 2026), which opens 'Brouwer conjectured "
            "that ... which was recently confirmed by Kothari and Tudose'."
        ),
        status=Status.PROVEN,
        status_checked="2026-07-14",
        status_note=(
            "*** THE CAUTIONARY ENTRY. *** Brouwer's conjecture was listed as OPEN by the 2023 "
            "Liu-Ning survey (Conjecture 16) and was still being treated as open in January 2026 by "
            "Lew ('An approximate version of Brouwer's Laplacian conjecture', arXiv:2601.17575). It "
            "was PROVEN in mid-2026. It was the FIRST target picked for this library, and shipping "
            "it as 'open' would have burned real compute hunting a counterexample that provably "
            "does not exist — and, worse, invited a 'win' that dissolved on contact with a referee. "
            "It is retained, marked PROVEN and non-live, as the standing reminder that a status "
            "taken from a survey is a snapshot, not a fact: RE-CHECK before hunting, and re-check "
            "again before claiming."
        ),
        graph_class=GraphClass(min_n=3, max_n=MAX_N_DEFAULT, connected=True),
        sides=_sides_brouwer,
        verified_up_to_n=10,
        hunting_notes="Now a theorem. Not a hunting target.",
    ),
)

REGISTRY: dict[str, Conjecture] = {c.key: c for c in _CONJECTURES}


def get(key: str) -> Conjecture:
    """Look up a conjecture by key. Raises KeyError listing the valid keys on a typo."""
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown conjecture {key!r}; known: {sorted(REGISTRY)}") from None


def live_conjectures() -> list[Conjecture]:
    """The OPEN conjectures — the only legitimate hunting targets."""
    return [c for c in _CONJECTURES if c.live]


def controls() -> dict[str, list[Conjecture]]:
    """The non-live entries by role: `positive` (a counterexample is known — the search MUST be able
    to find one) and `negative` (a theorem — the search must NEVER find one)."""
    return {
        "positive": [c for c in _CONJECTURES if c.status is Status.REFUTED],
        "negative": [c for c in _CONJECTURES if c.status is Status.PROVEN],
    }


def summary() -> list[dict[str, Any]]:
    """One row per conjecture — for the ledger or a status report."""
    return [
        {
            "key": c.key,
            "title": c.title,
            "status": c.status.value,
            "status_checked": c.status_checked,
            "live": c.live,
            "verified_up_to_n": c.verified_up_to_n,
            "source": c.source,
        }
        for c in _CONJECTURES
    ]
