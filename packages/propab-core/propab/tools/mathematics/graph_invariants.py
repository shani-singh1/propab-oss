"""Graph invariants (M3, combinatorics / graph-theory cluster) via networkx.

A worker composes this to compute the standard structural invariants of a graph:
connectivity, girth, diameter, spectrum, planarity, degree sequence, triangle count,
automorphism count, isomorphism, and the three NP-hard invariants (chromatic number,
clique number, independence number).

HONESTY BY CONSTRUCTION
-----------------------
* Polynomial-time invariants are computed exactly (``exact=True``).
* The NP-hard invariants (chromatic / clique / independence) are computed with
  EXACT algorithms (exact backtracking colouring; Bron-Kerbosch maximal cliques on
  the graph / its complement). They are capped by node count. ABOVE THE CAP the
  tool returns ``status="unknown"`` — it NEVER returns a heuristic value dressed up
  as the exact answer. A cheap heuristic BOUND may be attached, but it is explicitly
  labelled (``bound.is_exact=False``, ``bound.kind`` in {upper_bound, lower_bound})
  so it can never be mistaken for the invariant.
* Every result reports ``method`` and ``exact`` so the caller knows exactly what
  was computed.

Never raises: bad input → ``ToolResult(success=False, ...)``; valid-but-oversized
NP-hard input → ``success=True`` with ``status="unknown"``.
"""
from __future__ import annotations

import math
from typing import Any

from propab.tools.types import ToolError, ToolResult

# --- caps -------------------------------------------------------------------
NP_HARD_MAX_NODES = 30          # chromatic / clique / independence exact cap
AUTOMORPHISM_MAX_NODES = 10     # exact automorphism enumeration cap

_OPS = {
    "chromatic_number", "clique_number", "independence_number", "is_connected",
    "girth", "diameter", "spectrum", "is_planar", "num_automorphisms",
    "is_isomorphic", "degree_sequence", "num_triangles",
}
_NP_HARD = {"chromatic_number", "clique_number", "independence_number"}


def _err(msg: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="validation_error", message=msg))


TOOL_SPEC = {
    "name": "graph_invariants",
    "domain": "mathematics",
    "audience": "worker",
    "description": (
        "Compute graph invariants with networkx. op in {chromatic_number, clique_number, "
        "independence_number, is_connected, girth, diameter, spectrum, is_planar, "
        "num_automorphisms, is_isomorphic, degree_sequence, num_triangles}. Graph given "
        "as 'edges' (list of [u,v]) with optional 'nodes', or 'adjacency' (0/1 matrix). "
        "is_isomorphic takes a second graph (edges2/adjacency2). NP-hard invariants use "
        "EXACT algorithms capped by node count; above the cap they return status='unknown' "
        "(never a heuristic value mislabelled as exact). Reports method + exact flag."
    ),
    "params": {
        "op": {"type": "str", "required": True, "description": f"One of {sorted(_OPS)}."},
        "edges": {"type": "list[list]", "required": False, "description": "Edge list, e.g. [[0,1],[1,2]]."},
        "nodes": {"type": "list", "required": False, "description": "Optional explicit node list (adds isolated nodes)."},
        "adjacency": {"type": "list[list[int]]", "required": False, "description": "Square 0/1 adjacency matrix (alternative to edges)."},
        "edges2": {"type": "list[list]", "required": False, "description": "Second graph edge list (is_isomorphic)."},
        "nodes2": {"type": "list", "required": False, "description": "Second graph node list (is_isomorphic)."},
        "adjacency2": {"type": "list[list[int]]", "required": False, "description": "Second graph adjacency matrix (is_isomorphic)."},
    },
    "output": {
        "op": "str — echoed op",
        "status": "str — 'ok' or 'unknown' (valid-but-oversized NP-hard)",
        "value": "int|bool|list|None — the invariant (None when unknown)",
        "exact": "bool — True when value is exact",
        "method": "str — algorithm used",
        "n_nodes": "int", "n_edges": "int",
        "bound": "dict|None — labelled heuristic bound when status='unknown' (never the exact value)",
        "note": "str — clarifying note (e.g. infinite diameter/girth)",
        "reason": "str — present when status='unknown'",
    },
    "example": {
        "params": {"op": "chromatic_number", "edges": [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]]},
        "output": {"op": "chromatic_number", "status": "ok", "value": 4, "exact": True},
    },
}


# --------------------------------------------------------------------------- #
# graph construction                                                          #
# --------------------------------------------------------------------------- #
def _build_graph(edges: Any, adjacency: Any, nodes: Any):
    """Return (nx.Graph, error_msg). Self-loops and duplicate edges collapse to a
    simple undirected graph (the invariants here are simple-graph invariants)."""
    import networkx as nx

    if adjacency is not None:
        if not isinstance(adjacency, (list, tuple)) or any(
            not isinstance(row, (list, tuple)) for row in adjacency
        ):
            return None, "'adjacency' must be a list of lists (square 0/1 matrix)."
        m = len(adjacency)
        if any(len(row) != m for row in adjacency):
            return None, "'adjacency' must be a square matrix."
        G = nx.Graph()
        G.add_nodes_from(range(m))
        for i in range(m):
            for j in range(i + 1, m):
                if adjacency[i][j]:
                    G.add_edge(i, j)
        return G, None

    if edges is None and nodes is None:
        return None, "provide a graph via 'edges' (and/or 'nodes') or 'adjacency'."
    if edges is not None and not isinstance(edges, (list, tuple)):
        return None, "'edges' must be a list of [u, v] pairs."
    G = nx.Graph()
    if nodes is not None:
        if not isinstance(nodes, (list, tuple)):
            return None, "'nodes' must be a list."
        G.add_nodes_from(nodes)
    for e in (edges or []):
        if not isinstance(e, (list, tuple)) or len(e) < 2:
            return None, f"each edge must be a [u, v] pair, got {e!r}."
        u, v = e[0], e[1]
        if u == v:
            continue  # ignore self-loop (simple graph)
        G.add_edge(u, v)
    return G, None


# --------------------------------------------------------------------------- #
# exact NP-hard invariants                                                     #
# --------------------------------------------------------------------------- #
def _greedy_upper_bound_colors(G) -> int:
    import networkx as nx
    if G.number_of_nodes() == 0:
        return 0
    coloring = nx.greedy_color(G, strategy="largest_first")
    return (max(coloring.values()) + 1) if coloring else 0


def _k_colorable(G, k: int) -> bool:
    """Exact backtracking test: is G properly k-colorable?"""
    nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
    adj = {v: set(G.neighbors(v)) for v in G}
    color: dict = {}

    def bt(i: int, max_used: int) -> bool:
        if i == len(nodes):
            return True
        v = nodes[i]
        used = {color[u] for u in adj[v] if u in color}
        limit = min(max_used + 1, k - 1)  # symmetry break: no gap in color indices
        for c in range(0, limit + 1):
            if c not in used:
                color[v] = c
                if bt(i + 1, max(max_used, c)):
                    return True
                del color[v]
        return False

    return bt(0, -1)


def _chromatic_number_exact(G) -> int:
    if G.number_of_nodes() == 0:
        return 0
    if G.number_of_edges() == 0:
        return 1
    ub = _greedy_upper_bound_colors(G)
    for k in range(1, ub + 1):
        if _k_colorable(G, k):
            return k
    return ub  # unreachable (ub is always colorable)


def _clique_number_exact(G) -> int:
    import networkx as nx
    return max((len(c) for c in nx.find_cliques(G)), default=0)


def _independence_number_exact(G) -> int:
    import networkx as nx
    comp = nx.complement(G)
    return max((len(c) for c in nx.find_cliques(comp)), default=0)


# cheap heuristic bounds (used ONLY in the oversized 'unknown' branch, clearly labelled)
def _greedy_clique_size(G) -> int:
    best = 0
    for v in sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True):
        clique = [v]
        for u in sorted(G.neighbors(v), key=lambda x: G.degree(x), reverse=True):
            if all(G.has_edge(u, w) for w in clique):
                clique.append(u)
        best = max(best, len(clique))
    return best


def _greedy_independent_set_size(G) -> int:
    chosen: set = set()
    banned: set = set()
    for v in sorted(G.nodes(), key=lambda x: G.degree(x)):  # low degree first
        if v not in banned:
            chosen.add(v)
            banned.add(v)
            banned.update(G.neighbors(v))
    return len(chosen)


def _lower_bound_chromatic(G) -> int:
    import networkx as nx
    if G.number_of_edges() == 0:
        return 1 if G.number_of_nodes() else 0
    if any(nx.triangles(G).values()):
        return 3
    return 2


def _np_hard_unknown(op: str, G, n: int) -> ToolResult:
    """Return status='unknown' plus a clearly-labelled heuristic BOUND."""
    reason = f"graph has {n} nodes, above the exact NP-hard cap {NP_HARD_MAX_NODES}."
    if op == "chromatic_number":
        bound = {
            "kind": "upper_bound",
            "value": _greedy_upper_bound_colors(G),
            "is_exact": False,
            "method": "networkx.greedy_color (largest_first) — a heuristic UPPER BOUND, NOT the chromatic number",
            "also_lower_bound": _lower_bound_chromatic(G),
        }
    elif op == "clique_number":
        bound = {
            "kind": "lower_bound",
            "value": _greedy_clique_size(G),
            "is_exact": False,
            "method": "greedy maximal clique — a LOWER BOUND, NOT the clique number",
        }
    else:  # independence_number
        bound = {
            "kind": "lower_bound",
            "value": _greedy_independent_set_size(G),
            "is_exact": False,
            "method": "greedy independent set — a LOWER BOUND, NOT the independence number",
        }
    return ToolResult(success=True, output={
        "op": op, "status": "unknown", "value": None, "exact": False,
        "reason": reason, "cap": NP_HARD_MAX_NODES, "bound": bound,
        "n_nodes": n, "n_edges": G.number_of_edges(),
    })


# --------------------------------------------------------------------------- #
# main entry                                                                   #
# --------------------------------------------------------------------------- #
def graph_invariants(
    op: str | None = None,
    edges: Any = None,
    nodes: Any = None,
    adjacency: Any = None,
    edges2: Any = None,
    nodes2: Any = None,
    adjacency2: Any = None,
) -> ToolResult:
    if op is None:
        return _err("Parameter 'op' is required.")
    if op not in _OPS:
        return _err(f"unknown op {op!r}; allowed: {sorted(_OPS)}.")
    try:
        import networkx as nx

        G, e = _build_graph(edges, adjacency, nodes)
        if e:
            return _err(e)
        n = G.number_of_nodes()
        base = {"op": op, "n_nodes": n, "n_edges": G.number_of_edges()}

        # ---- NP-hard (exact, capped) -------------------------------------- #
        if op in _NP_HARD:
            if n > NP_HARD_MAX_NODES:
                return _np_hard_unknown(op, G, n)
            if op == "chromatic_number":
                val = _chromatic_number_exact(G)
                method = "exact backtracking (min k-colorability)"
            elif op == "clique_number":
                val = _clique_number_exact(G)
                method = "Bron-Kerbosch maximal cliques (exact)"
            else:
                val = _independence_number_exact(G)
                method = "max clique of complement, Bron-Kerbosch (exact)"
            return ToolResult(success=True, output={
                **base, "status": "ok", "value": int(val), "exact": True, "method": method})

        # ---- polynomial-time invariants ----------------------------------- #
        if op == "is_connected":
            if n == 0:
                return _err("is_connected is undefined for a graph with no nodes.")
            return ToolResult(success=True, output={
                **base, "status": "ok", "value": bool(nx.is_connected(G)), "exact": True,
                "method": "networkx.is_connected"})

        if op == "girth":
            g = nx.girth(G)
            if g == 0 or math.isinf(g):
                return ToolResult(success=True, output={
                    **base, "status": "ok", "value": None, "exact": True,
                    "method": "networkx.girth", "is_infinite": True,
                    "note": "acyclic (no cycle) — girth is infinite"})
            return ToolResult(success=True, output={
                **base, "status": "ok", "value": int(g), "exact": True, "method": "networkx.girth"})

        if op == "diameter":
            if n == 0:
                return _err("diameter is undefined for a graph with no nodes.")
            if not nx.is_connected(G):
                return ToolResult(success=True, output={
                    **base, "status": "ok", "value": None, "exact": True,
                    "method": "networkx.diameter", "is_infinite": True,
                    "note": "disconnected — diameter is infinite"})
            return ToolResult(success=True, output={
                **base, "status": "ok", "value": int(nx.diameter(G)), "exact": True,
                "method": "networkx.diameter (BFS eccentricities)"})

        if op == "spectrum":
            import numpy as np
            if n == 0:
                spec: list = []
            else:
                A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
                spec = [round(float(x), 6) for x in sorted(np.linalg.eigvalsh(A))]
            return ToolResult(success=True, output={
                **base, "status": "ok", "value": spec, "exact": True,
                "method": "eigenvalues of the adjacency matrix (numpy.eigvalsh)"})

        if op == "is_planar":
            return ToolResult(success=True, output={
                **base, "status": "ok", "value": bool(nx.check_planarity(G)[0]), "exact": True,
                "method": "networkx.check_planarity (Boyer-Myrvold)"})

        if op == "degree_sequence":
            seq = sorted((int(d) for _, d in G.degree()), reverse=True)
            return ToolResult(success=True, output={
                **base, "status": "ok", "value": seq, "exact": True,
                "method": "sorted node degrees (descending)"})

        if op == "num_triangles":
            tri = sum(nx.triangles(G).values()) // 3
            return ToolResult(success=True, output={
                **base, "status": "ok", "value": int(tri), "exact": True,
                "method": "networkx.triangles (sum / 3)"})

        if op == "num_automorphisms":
            if n > AUTOMORPHISM_MAX_NODES:
                return ToolResult(success=True, output={
                    **base, "status": "unknown", "value": None, "exact": False,
                    "reason": f"graph has {n} nodes, above the automorphism cap {AUTOMORPHISM_MAX_NODES}.",
                    "cap": AUTOMORPHISM_MAX_NODES})
            from networkx.algorithms.isomorphism import GraphMatcher
            count = sum(1 for _ in GraphMatcher(G, G).isomorphisms_iter())
            return ToolResult(success=True, output={
                **base, "status": "ok", "value": int(count), "exact": True,
                "method": "VF2 automorphism enumeration (exact)"})

        if op == "is_isomorphic":
            if edges2 is None and adjacency2 is None and nodes2 is None:
                return _err("is_isomorphic requires a second graph via 'edges2'/'adjacency2'.")
            G2, e2 = _build_graph(edges2, adjacency2, nodes2)
            if e2:
                return _err(f"second graph: {e2}")
            return ToolResult(success=True, output={
                **base, "status": "ok", "value": bool(nx.is_isomorphic(G, G2)), "exact": True,
                "method": "VF2 (networkx.is_isomorphic)",
                "n_nodes2": G2.number_of_nodes(), "n_edges2": G2.number_of_edges()})

        return _err(f"unhandled op {op!r}.")
    except Exception as exc:  # noqa: BLE001 — never raise to the caller
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
