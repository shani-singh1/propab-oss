"""evolve/targets/graph_conj.py — Target B: counterexample hunting on open graph conjectures.

The asymmetric bet. For a conjecture "for all graphs G in C, f(G) <= g(G)":

  * the verifier is the best one that exists anywhere — you just EVALUATE both sides on the
    candidate graph. No model, no proxy, no statistics, no p-value. Milliseconds.
  * a counterexample is a FINITE OBJECT that FULLY SETTLES the question. Not evidence for it — it
    settles it. A referee re-checks it in seconds from the edge list alone.
  * the frontier labs' pipelines are optimised to PROVE, which needs an argument and therefore needs
    frontier IQ. REFUTING needs an object, which needs search. That space is relatively empty.

The precedent: Adam Zsolt Wagner, "Constructions in combinatorics via neural networks"
(arXiv:2104.14516), refuted several published spectral graph-theory conjectures with the deep
cross-entropy method and NO LLM AT ALL. A weak model plus a good search loop is sufficient here —
which is exactly the shape this engine is built for.

The conjectures themselves (with citations, checked open/closed status, and the proven special cases
that tell you where a counterexample CANNOT be) live in `conjectures.py`.

CANDIDATE = a graph. A program's `build()` returns one, or a list of them, in any of these forms:
    * a `networkx.Graph`
    * an adjacency matrix: a square, symmetric, 0/1, zero-diagonal nested list or numpy array
    * an edge list: [(0,1), (1,2), ...]
    * a dict: {"n": 19, "edges": [(0,1), ...]}   <- unambiguous; use this when a graph has isolated
      vertices, since an edge list alone cannot express them

SCORE = the violation margin, lhs - rhs. margin > 0 IS a counterexample. `best_known()` is therefore
0.0: any positive margin is new mathematics (on a live conjecture), and there is no partial credit.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import networkx as nx
import numpy as np

from ..problem import Candidate, Verdict
from . import conjectures as conj
from .conjectures import MARGIN_EPS, Conjecture, Status

# A mutated program can emit anything. Decode defensively, and NEVER build a graph big enough to
# stall the verifier: the class check would reject an oversized graph anyway, so refuse it before
# paying for it. (This is a decoding guard, not the mathematical class bound — that is GraphClass.)
MAX_DECODE_NODES = 400
MAX_DECODE_EDGES = 20_000


class _BadCandidate(ValueError):
    """The candidate could not be decoded into a simple undirected graph."""


# ------------------------------------------------------------------------------------------------
# Candidate decoding — total, defensive, never raises out of `verify`.
# ------------------------------------------------------------------------------------------------
def _looks_like_adjacency_matrix(rows: Sequence[Any]) -> bool:
    """Square + symmetric + 0/1 + zero diagonal. Note the one ambiguous input, [[0,1],[1,0]], is
    read as the adjacency matrix of K_2 — which is the same graph the edge-list reading would give,
    so the ambiguity is harmless."""
    n = len(rows)
    if n == 0:
        return False
    for r in rows:
        if not isinstance(r, (list, tuple, np.ndarray)) or len(r) != n:
            return False
    for i in range(n):
        for j in range(n):
            v = rows[i][j]
            if isinstance(v, (bool, np.bool_)):
                v = int(v)
            if not isinstance(v, (int, float, np.integer, np.floating)):
                return False
            if float(v) not in (0.0, 1.0):
                return False
            if i == j and float(v) != 0.0:
                return False
            if float(v) != float(rows[j][i]):
                return False
    return True


def _from_matrix(rows: Sequence[Any]) -> nx.Graph:
    n = len(rows)
    if n > MAX_DECODE_NODES:
        raise _BadCandidate(f"adjacency matrix too large: n={n} > {MAX_DECODE_NODES}")
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if float(rows[i][j]) == 1.0:
                g.add_edge(i, j)
    return g


def _from_edges(edges: Iterable[Any], n: int | None = None) -> nx.Graph:
    g = nx.Graph()
    if n is not None:
        if not isinstance(n, (int, np.integer)) or n < 0 or n > MAX_DECODE_NODES:
            raise _BadCandidate(f"bad vertex count n={n!r}")
        g.add_nodes_from(range(int(n)))
    count = 0
    for e in edges:
        count += 1
        if count > MAX_DECODE_EDGES:
            raise _BadCandidate(f"too many edges (> {MAX_DECODE_EDGES})")
        if not isinstance(e, (list, tuple, np.ndarray)) or len(e) != 2:
            raise _BadCandidate(f"edge {e!r} is not a pair")
        u, v = e[0], e[1]
        for w in (u, v):
            if isinstance(w, (bool, np.bool_)) or not isinstance(w, (int, np.integer)):
                raise _BadCandidate(f"vertex label {w!r} is not an int")
            if w < 0 or w >= MAX_DECODE_NODES:
                raise _BadCandidate(f"vertex label {w!r} out of range [0, {MAX_DECODE_NODES})")
        if int(u) == int(v):
            raise _BadCandidate(f"self-loop at {u!r} (graphs must be simple)")
        g.add_edge(int(u), int(v))   # nx.Graph silently dedupes parallel edges
    return g


def _canonical(g: nx.Graph) -> nx.Graph:
    """Relabel to vertices 0..n-1, sorted.

    Not cosmetic — the witness must ROUND-TRIP. An edge list like [(0,5),(5,7)] yields the vertex set
    {0,5,7}, so the witness would record n=3 alongside labels up to 7, and `recheck_witness` would
    rebuild a completely different graph (range(3) plus edges to 5 and 7 => five vertices). Every
    decode path therefore ends here, and the margins are label-invariant anyway.
    """
    return nx.convert_node_labels_to_integers(g, ordering="sorted")


def decode_graph(candidate: Candidate) -> nx.Graph:
    """Coerce whatever a program emitted into a simple undirected graph on vertices 0..n-1.
    Raises `_BadCandidate` — callers must catch."""
    if isinstance(candidate, (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        g = nx.Graph()
        g.add_nodes_from(candidate.nodes())
        g.add_edges_from((u, v) for u, v in candidate.edges() if u != v)
        if g.number_of_nodes() > MAX_DECODE_NODES:
            raise _BadCandidate(f"graph too large: n={g.number_of_nodes()}")
        return _canonical(g)

    if isinstance(candidate, nx.Graph):
        g = nx.Graph()
        g.add_nodes_from(candidate.nodes())
        g.add_edges_from((u, v) for u, v in candidate.edges() if u != v)
        if g.number_of_nodes() > MAX_DECODE_NODES:
            raise _BadCandidate(f"graph too large: n={g.number_of_nodes()}")
        return _canonical(g)

    if isinstance(candidate, dict):
        if "edges" not in candidate:
            raise _BadCandidate("dict candidate must have an 'edges' key")
        return _canonical(_from_edges(candidate["edges"], candidate.get("n")))

    if isinstance(candidate, np.ndarray):
        if candidate.ndim != 2:
            raise _BadCandidate(f"numpy candidate must be 2-D, got ndim={candidate.ndim}")
        if not np.isfinite(candidate).all():
            raise _BadCandidate("adjacency matrix contains NaN/inf")
        rows = candidate.tolist()
        if not _looks_like_adjacency_matrix(rows):
            raise _BadCandidate("numpy candidate is not a symmetric 0/1 zero-diagonal matrix")
        return _canonical(_from_matrix(rows))

    if isinstance(candidate, (list, tuple)):
        rows = list(candidate)
        if not rows:
            raise _BadCandidate("empty candidate")
        if _looks_like_adjacency_matrix(rows):
            return _canonical(_from_matrix(rows))
        return _canonical(_from_edges(rows))

    raise _BadCandidate(f"cannot decode candidate of type {type(candidate).__name__}")


# ------------------------------------------------------------------------------------------------
# The witness: everything a third party needs to re-check the result WITHOUT running our code.
# ------------------------------------------------------------------------------------------------
def _witness(cjt: Conjecture, g: nx.Graph, sides: dict[str, float]) -> dict[str, Any]:
    edges = sorted((min(u, v), max(u, v)) for u, v in g.edges())
    return {
        "conjecture": cjt.key,
        "conjecture_status": cjt.status.value,
        "conjecture_statement": cjt.statement,
        "source": cjt.source,
        # --- the object itself: these two lines ARE the counterexample ---
        "n": g.number_of_nodes(),
        "edges": [list(e) for e in edges],
        "graph6": nx.to_graph6_bytes(g, header=False).decode().strip(),
        # --- the arithmetic, so a referee can see both sides without recomputing ---
        "lhs": float(sides["lhs"]),
        "rhs": float(sides["rhs"]),
        "margin": float(sides["margin"]),
        "invariants": {
            k: float(v) for k, v in sides.items() if k not in ("lhs", "rhs", "margin")
        },
        "refutes": bool(sides["margin"] > MARGIN_EPS),
        "margin_eps": MARGIN_EPS,
    }


def recheck_witness(detail: dict[str, Any]) -> Verdict:
    """Re-verify a claimed counterexample FROM THE WITNESS ALONE.

    This is the independent-recheck path: it reads only `conjecture`, `n` and `edges` out of the
    witness, rebuilds the graph from scratch, and re-runs the conjecture. It deliberately ignores
    the recorded `lhs`/`rhs`/`margin`, so a corrupted or fabricated verdict cannot survive it.
    A result this function does not confirm is NOT a result.
    """
    try:
        cjt = conj.get(str(detail["conjecture"]))
        g = _from_edges([tuple(e) for e in detail["edges"]], int(detail["n"]))
    except (KeyError, TypeError, ValueError, _BadCandidate) as exc:
        return Verdict(valid=False, score=-math.inf, detail={"reason": f"bad witness: {exc}"})

    reason = cjt.graph_class.violation(g)
    if reason is not None:
        return Verdict(valid=False, score=-math.inf, detail={"reason": f"not in class: {reason}"})

    sides = cjt.evaluate(g)
    return Verdict(valid=True, score=float(sides["margin"]), detail=_witness(cjt, g, sides))


def known_counterexample() -> nx.Graph:
    """The PUBLISHED counterexample to the positive control (`aouchiche_hansen_lam1_matching`).

    NOT a search result and never to be reported as one — it exists so the test suite can assert
    that the verifier detects a counterexample that is known to exist.

    Construction (n = 19): two stars K_{1,8} whose centres u and v are joined through a single shared
    middle vertex w, i.e. the path u - w - v with 8 leaves hung on u and 8 on v.
      * matching number mu = 2  (a vertex cover {u, v} of size 2 covers every edge)
      * lambda_1 = sqrt(10)     (by symmetry the eigenvector equation reduces to lambda^2 = 10)
      * lambda_1 + mu = 5.1623  <  sqrt(18) + 1 = 5.2426        => margin = +0.0804 > 0

    Wagner (arXiv:2104.14516) found a 19-vertex counterexample with reinforcement learning; the
    conjecture is tight at the star (lambda_1 = sqrt(n-1), mu = 1, margin exactly 0), and this graph
    beats the star by trading 1 extra unit of matching number for a fall of more than 1 in lambda_1.
    """
    g = nx.Graph()
    u, v, w = 0, 1, 2
    g.add_edge(u, w)
    g.add_edge(w, v)
    for i in range(8):
        g.add_edge(u, 3 + i)
        g.add_edge(v, 11 + i)
    return g


def double_star_counterexample(a: int = 14) -> nx.Graph:
    """The BALANCED DOUBLE STAR S(a,a): two adjacent centres with `a` leaves each, n = 2a+2.

    An infinite family of counterexamples to the positive control, and the one the search actually
    rediscovers (it falls straight out of the seed sweep over double stars). Closed form:

        mu(S(a,a))      = 2                       (the two centres are a vertex cover)
        lambda_1(S(a,a)) = (1 + sqrt(1+4a)) / 2   (from lambda^2 - lambda - a = 0)
        margin(a)        = sqrt(2a+1) + 1 - (1 + sqrt(1+4a))/2 - 2

    margin is exactly 0 at a = 12 (n = 26, where lambda_1 = 4 exactly) and strictly POSITIVE for
    every a >= 13 — so every balanced double star on n >= 28 vertices refutes the conjecture. The
    test suite checks the numerics against this closed form to 1e-9.

    Wagner's 19-vertex graph is the SMALLEST known counterexample; this family is the simplest. Both
    are published results — never report either as a discovery.
    """
    if a < 1:
        raise ValueError("a must be >= 1")
    g = nx.Graph()
    g.add_edge(0, 1)
    for i in range(a):
        g.add_edge(0, 2 + i)
        g.add_edge(1, 2 + a + i)
    return g


# ------------------------------------------------------------------------------------------------
# The Problem
# ------------------------------------------------------------------------------------------------
class GraphConjectureProblem:
    """Hunt a counterexample to one graph conjecture. Implements the `Problem` protocol.

    `allow_non_live` is a deliberate safety catch, not a formality. The library holds a refuted
    conjecture (the positive control) and two theorems (the negative controls); pointing a campaign
    at one of those and "discovering" a counterexample would be a fake win of exactly the kind that
    has already dissolved twice on this project. So constructing a Problem over a non-OPEN
    conjecture requires saying so out loud.
    """

    def __init__(
        self,
        conjecture: str | Conjecture = "elphick_linz_wocjan",
        *,
        allow_non_live: bool = False,
    ) -> None:
        cjt = conj.get(conjecture) if isinstance(conjecture, str) else conjecture
        if not cjt.live and not allow_non_live:
            raise ValueError(
                f"conjecture {cjt.key!r} has status {cjt.status.value.upper()}, not OPEN, so it is "
                f"not a legitimate hunting target: {cjt.status_note} "
                f"Pass allow_non_live=True only to use it as a control. "
                f"Live targets: {[c.key for c in conj.live_conjectures()]}"
            )
        self.conjecture = cjt
        self.name = f"graph_conj:{cjt.key}"

    # -- Problem ---------------------------------------------------------------------------------
    def verify(self, candidate: Candidate) -> Verdict:
        """Cheap, exact, deterministic, and safe on garbage. Never raises."""
        cjt = self.conjecture
        try:
            g = decode_graph(candidate)
        except _BadCandidate as exc:
            return Verdict(valid=False, score=-math.inf, detail={"reason": f"undecodable: {exc}"})
        except Exception as exc:  # a mutated program can emit truly anything
            return Verdict(
                valid=False,
                score=-math.inf,
                detail={"reason": f"decode failed: {type(exc).__name__}: {exc}"},
            )

        try:
            reason = cjt.graph_class.violation(g)
            if reason is not None:
                return Verdict(
                    valid=False,
                    score=-math.inf,
                    detail={"reason": f"not in class: {reason}", "n": g.number_of_nodes()},
                )
            sides = cjt.evaluate(g)
        except Exception as exc:
            return Verdict(
                valid=False,
                score=-math.inf,
                detail={"reason": f"evaluation failed: {type(exc).__name__}: {exc}"},
            )

        if not math.isfinite(sides["margin"]):
            return Verdict(valid=False, score=-math.inf, detail={"reason": "non-finite margin"})

        return Verdict(valid=True, score=float(sides["margin"]), detail=_witness(cjt, g, sides))

    def best_known(self) -> float:
        """0.0 — the conjecture asserts the margin never exceeds it. There is no partial credit and
        no record to inch towards: any margin > 0 settles the question outright."""
        return 0.0

    def is_improvement(self, verdict: Verdict) -> bool:
        """True only for a GENUINE counterexample, independently re-checked from the witness alone.

        Three gates, all required:
          1. the candidate is a legal graph in the conjecture's class;
          2. the margin clears MARGIN_EPS (1e-6) — a >= 6-order-of-magnitude cushion over the ~2e-12
             floating-point noise seen on the families where these conjectures are exactly tight, so
             an equality case can never be mistaken for a refutation;
          3. rebuilding the graph FROM THE WITNESS ALONE reproduces the same positive margin. If the
             witness does not re-check, there is no result — whatever the verdict claims.
        """
        if not verdict.valid or not math.isfinite(verdict.score):
            return False
        if verdict.score <= MARGIN_EPS:
            return False
        rechecked = recheck_witness(verdict.detail)
        if not rechecked.valid or rechecked.score <= MARGIN_EPS:
            return False
        return math.isclose(rechecked.score, verdict.score, rel_tol=1e-6, abs_tol=1e-9)

    def is_discovery(self, verdict: Verdict) -> bool:
        """`is_improvement` AND the conjecture is still OPEN — i.e. this is NEW mathematics.

        The distinction is the anti-self-deception gate. A counterexample to the positive control is
        a real counterexample (so `is_improvement` is True) but it was published in 2021, so it is a
        rediscovery, not a discovery. Only `is_discovery` may be banked as a result.
        """
        return self.conjecture.live and self.is_improvement(verdict)

    def describe(self) -> str:
        """The prompt surface shown to the LLM mutator."""
        c = self.conjecture
        gc = c.graph_class
        floor = (
            f"\nAlready verified exhaustively for all graphs on n <= {c.verified_up_to_n} vertices, "
            f"so DO NOT WASTE MOVES BELOW n = {c.verified_up_to_n + 1}."
            if c.verified_up_to_n
            else ""
        )
        return f"""\
GOAL: find a COUNTEREXAMPLE to an open conjecture in spectral graph theory.

CONJECTURE ({c.title}):
  {c.statement}

SOURCE: {c.source}
STATUS: {c.status.value.upper()} (checked {c.status_checked}) — {c.status_note}

GRAPH CLASS (a candidate outside this class scores nothing):
  {gc.describe()}{floor}

OBJECTIVE — maximise the violation margin:
      margin(G) = lhs(G) - rhs(G)
  where the conjecture asserts lhs(G) <= rhs(G) for every G in the class.
  margin <= 0  ->  G obeys the conjecture (the margin still tells you how CLOSE you are: drive it up)
  margin  > 0  ->  G REFUTES THE CONJECTURE. That is the whole objective. It is a finite object that
                   settles a published open problem outright; there is no partial credit beyond it.

WHERE NOT TO LOOK (these cases are PROVEN — a counterexample cannot exist there, so a program that
generates them is wasted compute):
  {c.hunting_notes}

CANDIDATE FORMAT — `build()` returns ONE graph, or a LIST of graphs (a list is strongly preferred:
emit a whole parameterised family per program and let the verifier score all of them). Any of:
  * a `networkx.Graph`
  * an adjacency matrix (square, symmetric, 0/1, zero diagonal) as a nested list or numpy array
  * an edge list, e.g. [(0,1), (1,2), (2,0)]
  * a dict {{"n": 19, "edges": [(0,1), ...]}}   <- use this if the graph has isolated vertices

STRATEGY THAT WORKS (this is program search, not object search — the payoff is in the GENERATOR):
  * Write generators for PARAMETERISED FAMILIES, not single graphs. A `build()` that sweeps two or
    three structural parameters and returns the whole sweep gets many shots per program.
  * The conjecture's EQUALITY CASES are the boundary — start from a family that is exactly tight and
    perturb it. A counterexample, if it exists, is almost certainly a small perturbation of a tight
    family, not a random graph.
  * Recombine the best parents: take the family that scored highest and vary its structure
    (subdivide an edge, hang a pendant, join two copies through a shared vertex, blow up a vertex
    into an independent set, add a chord).
  * Random graphs are a poor bet here — several of these conjectures are PROVEN for almost all
    graphs. The counterexample is structured, sparse and slightly irregular.
"""

    def seed_programs(self) -> list[str]:
        """Starting generators: classic graph families, each PARAMETERISED so evolution has knobs.

        Deliberately pure Python (edge lists, zero imports) so they run under the strictest sandbox.
        Deliberately GENERIC: no seed encodes the known counterexample to the positive control — the
        search has to find its parameters, or the positive control would be testing lookup, not
        search.
        """
        return [
            # --- paths, cycles, and cycles with pendants (near-tree, sparse, odd girth) ---
            '''\
def build():
    """Paths and cycles: the sparse, low-spectral-radius extreme."""
    out = []
    for n in range(11, 31):
        out.append({"n": n, "edges": [(i, i + 1) for i in range(n - 1)]})            # path P_n
        out.append({"n": n, "edges": [(i, (i + 1) % n) for i in range(n)]})          # cycle C_n
    return out
''',
            # --- stars, double stars, brooms: the tight family for several of these bounds ---
            '''\
def build():
    """Stars and double stars. The star K_{1,n-1} is an EXACT equality case for more than one
    conjecture in this library, so it is the boundary to perturb."""
    out = []
    for n in range(11, 31):
        out.append({"n": n, "edges": [(0, i) for i in range(1, n)]})                 # star K_{1,n-1}
    for a in range(2, 15):                                                            # double star:
        for b in range(2, 15):                                                        # two adjacent
            edges = [(0, 1)]                                                          # centres, with
            edges += [(0, 2 + i) for i in range(a)]                                   # a and b leaves
            edges += [(1, 2 + a + i) for i in range(b)]
            out.append({"n": 2 + a + b, "edges": edges})
    return out
''',
            # --- caterpillars: a spine with leaves. A broad, cheap, highly structured family. ---
            '''\
def build():
    """Caterpillars: a path spine with leaves hung on each spine vertex. Sweep spine length and the
    leaf distribution -- irregular trees live here, and trees sit exactly on the EFGW boundary."""
    out = []
    for spine in range(2, 7):
        for leaves in ([0] * spine, [1] * spine, list(range(spine)), [spine - i for i in range(spine)]):
            edges = [(i, i + 1) for i in range(spine - 1)]
            nxt = spine
            for i, k in enumerate(leaves):
                for _ in range(k):
                    edges.append((i, nxt))
                    nxt += 1
            if nxt >= 11:
                out.append({"n": nxt, "edges": edges})
    # asymmetric brooms: a path with a bunch of leaves at one end
    for path_len in range(2, 10):
        for k in range(2, 16):
            edges = [(i, i + 1) for i in range(path_len - 1)]
            edges += [(path_len - 1, path_len + i) for i in range(k)]
            out.append({"n": path_len + k, "edges": edges})
    return out
''',
            # --- unicyclic / bicyclic: sparse + NON-bipartite (odd girth). EFGW's hunting ground. ---
            '''\
def build():
    """Unicyclic and bicyclic graphs: a tree plus one or two extra edges. Sparse, irregular and --
    when the cycle is odd -- NON-bipartite, which is required to escape the proven cases."""
    out = []
    for n in range(11, 31):
        for g in (3, 5, 7, 9):                                   # odd girth => non-bipartite
            if g > n:
                continue
            edges = [(i, (i + 1) % g) for i in range(g)]         # a cycle C_g ...
            for j in range(g, n):                                # ... with a path/leaves hung off it
                edges.append((j % g if j < 2 * g else j - g, j))
            out.append({"n": n, "edges": edges})
    # a triangle with three paths of tunable length attached (a "spider" on an odd cycle)
    for a in range(1, 9):
        for b in range(1, 9):
            edges = [(0, 1), (1, 2), (2, 0)]
            nxt = 3
            for anchor, ln in ((0, a), (1, b)):
                prev = anchor
                for _ in range(ln):
                    edges.append((prev, nxt))
                    prev, nxt = nxt, nxt + 1
            if nxt >= 11:
                out.append({"n": nxt, "edges": edges})
    return out
''',
            # --- complete bipartite / multipartite (Turan): the tight family for BN and ELW ---
            '''\
def build():
    """Complete multipartite graphs -- the EXACT equality case for Bollobas-Nikiforov, hence the
    boundary to perturb for BN and ELW. (Proven, so they cannot themselves be counterexamples: the
    value is in what a small perturbation of them does.)"""
    out = []
    for a in range(1, 16):                                        # complete bipartite K_{a,b}
        for b in range(1, 16):
            if 11 <= a + b <= 30:
                out.append({"n": a + b, "edges": [(i, a + j) for i in range(a) for j in range(b)]})
    for parts in ([3, 3, 3, 3], [4, 4, 4], [2, 2, 2, 2, 2, 2], [5, 5, 2], [3, 3, 3, 3, 3], [6, 4, 3]):
        sizes, edges, base, blocks = parts, [], 0, []
        for s in sizes:
            blocks.append(list(range(base, base + s)))
            base += s
        for i, bi in enumerate(blocks):
            for bj in blocks[i + 1:]:
                edges += [(u, v) for u in bi for v in bj]
        out.append({"n": base, "edges": edges})
    return out
''',
            # --- perturbations of the tight families: the highest-value move ---
            '''\
def build():
    """The single highest-value move: take a family that sits EXACTLY on the boundary and nudge it
    off. Here, complete multipartite graphs (tight for BN/ELW) with one edge added or removed."""
    out = []
    for parts in ([3, 3, 3], [4, 4, 4], [3, 3, 3, 3], [5, 5, 5], [4, 4, 4, 4], [2, 3, 4, 5]):
        base, blocks = 0, []
        for s in parts:
            blocks.append(list(range(base, base + s)))
            base += s
        edges = []
        for i, bi in enumerate(blocks):
            for bj in blocks[i + 1:]:
                edges += [(u, v) for u in bi for v in bj]
        if base < 11:
            continue
        out.append({"n": base, "edges": edges})
        # add a chord INSIDE a part (creates an irregularity and raises omega)
        for bi in blocks:
            if len(bi) >= 2:
                out.append({"n": base, "edges": edges + [(bi[0], bi[1])]})
        # delete one cross edge (breaks regularity without breaking connectivity)
        for k in range(0, len(edges), max(1, len(edges) // 6)):
            out.append({"n": base, "edges": edges[:k] + edges[k + 1:]})
    return out
''',
            # --- kites, wheels, barbells, cones: dense cores + sparse tails => strong irregularity --
            '''\
def build():
    """Irregular graphs built from a dense core plus a sparse tail: kites (clique + path), wheels,
    barbells, and cones (a vertex joined to everything). Irregularity is mandatory -- these
    conjectures are PROVEN for regular graphs."""
    out = []
    for k in range(3, 10):                                        # kite: K_k with a path glued on
        for tail in range(1, 16):
            edges = [(i, j) for i in range(k) for j in range(i + 1, k)]
            prev = 0
            for t in range(tail):
                edges.append((prev, k + t))
                prev = k + t
            if 11 <= k + tail <= 30:
                out.append({"n": k + tail, "edges": edges})
    for n in range(11, 31):                                       # wheel: hub + cycle
        out.append({"n": n, "edges": [(0, i) for i in range(1, n)]
                    + [(i, i % (n - 1) + 1) for i in range(1, n)]})
    for k in range(3, 9):                                         # barbell: two cliques + a path
        for bar in range(1, 10):
            e = [(i, j) for i in range(k) for j in range(i + 1, k)]
            e += [(k + bar + i, k + bar + j) for i in range(k) for j in range(i + 1, k)]
            prev = 0
            for t in range(bar):
                e.append((prev, k + t))
                prev = k + t
            e.append((prev, k + bar))
            if 11 <= 2 * k + bar <= 30:
                out.append({"n": 2 * k + bar, "edges": e})
    return out
''',
            # --- named graphs + deterministic pseudo-random structured graphs ---
            '''\
def build():
    """Named small graphs (Petersen and its cousins) plus a DETERMINISTIC pseudo-random sweep.
    Random graphs are a weak bet -- several of these conjectures are proven for almost all graphs --
    so this is a baseline to beat, not the plan."""
    out = []
    # Petersen graph
    pet = [(i, (i + 1) % 5) for i in range(5)]
    pet += [(i, 5 + i) for i in range(5)]
    pet += [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    out.append({"n": 10, "edges": pet})
    # Petersen with a pendant vertex (breaks regularity, n = 11)
    out.append({"n": 11, "edges": pet + [(0, 10)]})
    # deterministic pseudo-random graphs (a simple LCG -- no imports, fully reproducible)
    seed = 12345
    for n in (12, 16, 20, 24):
        for dens in (12, 25, 40):
            edges, s = [], seed + n * 100 + dens
            for i in range(n):
                for j in range(i + 1, n):
                    s = (1103515245 * s + 12345) % 2147483648
                    if s % 100 < dens:
                        edges.append((i, j))
            edges += [(i, i + 1) for i in range(n - 1)]     # force connectivity
            out.append({"n": n, "edges": edges})
    return out
''',
        ]


def problems(*, include_controls: bool = False) -> list[GraphConjectureProblem]:
    """Every live target, one Problem each. With `include_controls`, also the positive/negative
    controls (constructed with `allow_non_live=True`) — for the test suite and engine smoke-checks."""
    out = [GraphConjectureProblem(c) for c in conj.live_conjectures()]
    if include_controls:
        for role in ("positive", "negative"):
            for c in conj.controls()[role]:
                out.append(GraphConjectureProblem(c, allow_non_live=True))
    return out


__all__ = [
    "GraphConjectureProblem",
    "decode_graph",
    "double_star_counterexample",
    "known_counterexample",
    "problems",
    "recheck_witness",
    "MARGIN_EPS",
    "Status",
]
