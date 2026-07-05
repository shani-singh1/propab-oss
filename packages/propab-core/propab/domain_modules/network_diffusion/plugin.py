"""
Network diffusion / contagion DomainPlugin.

Scope-only: it owns the OOD-scope template historically hardcoded in
``scoped_claim.py`` for contagion/diffusion questions. It has no dedicated
worker verification path (``matches`` returns False so campaign routing is
unaffected), so contagion campaigns still use the generic verification path
exactly as before — the only thing relocated here is the domain-specific scope
text.
"""
from __future__ import annotations

from typing import Any

from propab.domain_modules.base import DomainPlugin


class NetworkDiffusionPlugin(DomainPlugin):
    domain_id = "network_diffusion"
    display_name = "Network diffusion / contagion (graph topology families)"
    version = "1.0"
    scope_question_markers = (
        "contagion",
        "diffusion",
        "spreading",
        "sis",
        "sir",
        "network",
    )
    # Vocabulary that marks a claim as network/graph-topology for artifact-model
    # selection in core's artifact gate. Owned here (previously the hardcoded
    # ``_NETWORK_MARKERS`` tuple in ``artifact_verification.py``); the set is kept
    # identical so artifact-gate behaviour is unchanged.
    artifact_question_markers = (
        "network",
        "graph",
        "contagion",
        "sis",
        "sir",
        "topology",
        "modular",
        "scale-free",
        "barab",
        "erdős",
        "erdos",
        "k-shell",
        "k-core",
        "diffusion",
    )
    theme_rules = (
        ("spectral", ("spectral gap", "eigenvalue", "laplacian", "adjacency matrix", "algebraic connectivity", "λ₂", "lambda_2", "spectral norm")),
        ("diffusion_dynamics", ("contagion", "diffusion", " sis ", " sir ", "transmission", "outbreak", "epidemic", "spreading", "infection")),
        ("normalization", ("pre-normalization", "post-normalization", "normalization", "k_source", "k_target")),
        ("assortativity", ("assortativity", "degree correlation", "rich-club")),
        ("clustering", ("clustering coefficient", "transitivity", "local clustering")),
        ("centrality", ("betweenness", "centrality", "eigenvector centrality", "degree-based removal")),
        ("degree_structure", ("gini", "degree distribution", "degree variance", "average degree", "k-core", "degree heterogeneity")),
        ("scale_free", ("barab", "scale-free", "scale free", "preferential attachment")),
        ("small_world", ("watts-strogatz", "watts strogatz", "small-world", "small world", "rewiring probability")),
        ("random_graph", ("erdős", "erdos", "erdős-rényi", "g(n,p)", "random graph")),
        ("percolation", ("percolation", "giant component", "critical threshold", "pc", "percolation threshold")),
        ("targeted_removal", ("targeted removal", "targeted attack", "node removal", "immunization")),
        ("sparse_regime", ("sparse graph", "average degree <", "fragmentation")),
    )
    theme_fallbacks = (
        ("diffusion_dynamics", ("network", "graph", "node", "edge"), 0.45),
        ("spectral", ("matrix", "rank", "eigen"), 0.40),
        ("percolation", ("threshold", "component", "redundancy"), 0.38),
    )

    def matches(self, *, question: str = "", payload: dict[str, Any] | None = None) -> bool:
        # No dedicated worker verification path — do not claim campaign routing.
        return False

    def available_features(self) -> list[str]:
        return []

    def scope_template(self) -> dict[str, str]:
        return {
            "population": "N=300–5000 node graphs, 30+ instances per topology family",
            "distribution": "Barabási–Albert and stochastic block model ensembles; avg degree 6–12",
            "claimed_generalization": "Effect should transfer to Watts–Strogatz graphs with matched average degree",
            "expected_failure_modes": "Fails on ER graphs or when modularity Q<0.2; breaks if seed set >5% of nodes",
            "ood_test": "Train/evaluate on BA+SBM; hold out WS family; require LOFO R²>0 on WS or refute",
        }

    def literature_profile(self) -> dict[str, Any]:
        return {
            "seed_papers": [
                {
                    "title": "Epidemic Spreading in Scale-Free Networks",
                    "authors": "R. Pastor-Satorras, A. Vespignani",
                    "year": 2001,
                    "doi": "10.1103/PhysRevLett.86.3200",
                },
            ],
            "search_terms": [
                "epidemic spreading", "contagion", "SIS model", "SIR model", "percolation threshold",
                "scale-free network", "targeted immunization", "network robustness",
                "Barabasi-Albert", "Watts-Strogatz",
            ],
            "source_priorities": ["arxiv", "semantic_scholar", "mathoverflow"],
            "classification_codes": {
                "arxiv": ["cs.SI", "physics.soc-ph", "nlin.AO"],
            },
            "open_problem_sources": [],
            "tabulation_sources": [],
            "canonical_surveys": [
                {"title": "Epidemic Spreading in Scale-Free Networks", "doi": "10.1103/PhysRevLett.86.3200"},
            ],
            "novelty_criteria": (
                "A finding is novel if it establishes a diffusion/contagion effect that "
                "transfers from the training topology family (BA/SBM) to a held-out family "
                "(WS) and is not already predicted by the vanishing-epidemic-threshold result "
                "for scale-free networks (Pastor-Satorras & Vespignani, 2001)."
            ),
        }
