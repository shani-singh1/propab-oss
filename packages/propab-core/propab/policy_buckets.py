"""Budget and domain buckets — policy fitness is bucket-local only."""
from __future__ import annotations

BUDGET_BUCKETS = ("1h", "3h", "8h")
DOMAIN_BUCKETS = ("graphs", "algorithms", "math", "biology", "general")

_GRAPH_KEYWORDS = (
    "network", "graph", "contagion", "diffusion", "epidemic", "percolation",
    "modularity", "spectral", "cascade", "sis", "barab", "erdős", "er dos",
    "community", "outbreak", "spreading",
)
_ALGO_KEYWORDS = (
    "algorithm", "cache", "scheduling", "complexity", "benchmark", "optimizer",
    "gradient", "sorting", "search tree", "bfs", "dfs", "heuristic",
)
_MATH_KEYWORDS = (
    "theorem", "proof", "prime", "collatz", "erdos", "egyptian", "fraction",
    "modulo", "residue", "topology", "eigen", "matrix", "integral",
)
_BIO_KEYWORDS = (
    "gene", "protein", "ppi", "pathway", "knockdown", "expression", "essentiality",
    "metabolic", "evolution", "regulatory",
)


def budget_bucket(compute_budget_seconds: int) -> str:
    """Map compute budget to 1h / 3h / 8h bucket (no cross-bucket stats)."""
    hours = max(0.1, float(compute_budget_seconds) / 3600.0)
    if hours <= 1.5:
        return "1h"
    if hours <= 5.0:
        return "3h"
    return "8h"


def _keyword_hit(q: str, keyword: str) -> bool:
    """Substring match with padding to avoid false positives (e.g. ppi in stopping)."""
    k = keyword.strip().lower()
    if len(k) <= 3:
        parts = q.replace("-", " ").replace("/", " ").split()
        return k in parts
    return k in q


def domain_bucket(question: str, session_domain: str = "") -> str:
    """Map question + session domain hint to policy domain bucket."""
    q = (question or "").lower()
    sd = (session_domain or "").lower()

    if any(_keyword_hit(q, k) for k in _GRAPH_KEYWORDS) or "network" in sd:
        return "graphs"
    if any(_keyword_hit(q, k) for k in _BIO_KEYWORDS):
        return "biology"
    if any(_keyword_hit(q, k) for k in _MATH_KEYWORDS) or sd in ("mathematics", "statistics"):
        return "math"
    if any(_keyword_hit(q, k) for k in _ALGO_KEYWORDS) or sd in (
        "algorithm_optimization", "ml_research", "deep_learning", "data_analysis",
    ):
        return "algorithms"
    return "general"


def bucket_key(domain_bucket_name: str, budget_bucket_name: str) -> str:
    return f"{domain_bucket_name}:{budget_bucket_name}"
