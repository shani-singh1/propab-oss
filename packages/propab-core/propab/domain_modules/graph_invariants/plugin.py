"""Graph invariants DomainPlugin — deterministic SNAP-style verification."""
from __future__ import annotations

import time
from typing import Any

from propab.domain_modules.base import DomainPlugin, PreflightResult
from propab.domain_modules.graph_invariants.adapter import KNOWN_INVARIANTS, GraphInvariantSpec, GraphInvariantsAdapter
from propab.domain_modules.graph_invariants.verifier import classify_graph_verdict, run_graph_invariant_check


class GraphInvariantsPlugin(DomainPlugin):
    domain_id = "graph_invariants"
    display_name = "Graph invariants (real SNAP networks: collaboration, communication)"
    version = "1.0"
    scope_question_markers = (
        "graph invariant",
        "spectral gap",
        "clustering coefficient",
        "network family",
        "snap",
        "modularity",
        "algebraic connectivity",
    )
    artifact_question_markers = (
        "graph invariant",
        "spectral",
        "clustering",
        "network family",
        "snap",
        "modularity",
    )
    theme_rules = (
        ("spectral", ("spectral gap", "algebraic connectivity", "eigenvalue", "laplacian")),
        ("clustering", ("clustering coefficient", "transitivity", "local clustering")),
        ("degree_structure", ("average degree", "degree distribution")),
        ("scale_free", ("barab", "scale-free", "preferential attachment")),
        ("small_world", ("watts-strogatz", "small-world", "rewiring")),
        ("random_graph", ("erdős", "erdos", "erdos-renyi", "random graph")),
    )

    def matches(self, *, question: str = "", payload: dict[str, Any] | None = None) -> bool:
        if payload and str(payload.get("domain") or payload.get("domain_profile") or "") == "graph_invariants":
            return True
        q = (question or "").lower()
        hits = sum(1 for m in self.scope_question_markers if m in q)
        return hits >= 2 or "[domain_profile:graph_invariants]" in q

    def match_score(self, *, question: str = "", payload: dict[str, Any] | None = None) -> float:
        # Score = number of distinct graph-invariant markers present. Lets the
        # registry prefer this domain over a colliding one only when more of its
        # own specific vocabulary appears.
        if payload and str(payload.get("domain") or payload.get("domain_profile") or "") == "graph_invariants":
            return float(len(self.scope_question_markers))
        q = (question or "").lower()
        if "[domain_profile:graph_invariants]" in q:
            return float(len(self.scope_question_markers))
        return float(sum(1 for m in self.scope_question_markers if m in q))

    def scope_template(self) -> dict[str, str]:
        return {
            "population": (
                "Real SNAP networks: connected subgraphs of ca-GrQc (collaboration) "
                "and email-Eu-core (communication), 30 subgraphs × 2 real families"
            ),
            "distribution": "Leave-one-network-family-out holdout",
            "claimed_generalization": "Invariant relationship survives held-out real network family",
            "expected_failure_modes": "Family-specific topology masks invariant; small-n correlation noise",
            "ood_test": "Deterministic correlation/inequality on held-out real network category",
        }

    def available_features(self) -> list[str]:
        return list(KNOWN_INVARIANTS)

    def uses_synthetic_data(self) -> bool:
        # The invariant frame is now built from REAL SNAP networks (ca-GrQc
        # collaboration, email-Eu-core communication) via connected-subgraph
        # sampling — adapter meta records ``synthetic: False`` / provenance "real"
        # (see adapter.REAL_NETWORKS and data/graph_invariants/PROVENANCE.md).
        # Findings are real-data results, so no synthetic label is stamped (DOM2).
        return False

    def confirmation_criteria(self) -> dict[str, Any]:
        return {
            "min_metric_steps_for_confirm": 1,
            "requires_holdout": True,
            "holdout_type": "leave_network_family_out",
            "null_test": "held_out_family_replication",
            "verification_type": "deterministic",
        }

    def run_verification(
        self,
        hypothesis: dict[str, Any],
        evidence: dict[str, Any] | None = None,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        _ = evidence, features
        return run_graph_invariant_check(GraphInvariantSpec.from_hypothesis(hypothesis))

    def classify_verdict(self, hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
        return classify_graph_verdict(hypothesis_text, result)

    def preflight(self) -> PreflightResult:
        try:
            t0 = time.time()
            df = GraphInvariantsAdapter().load_frame()
            run_graph_invariant_check(GraphInvariantSpec(source_invariant="spectral_gap", target_invariant="clustering_coefficient"))
            elapsed = time.time() - t0
            if elapsed > 30:
                return PreflightResult(False, f"graph invariant check too slow: {elapsed:.1f}s", {})
            return PreflightResult(
                True,
                "SNAP subset loaded, invariant check ok",
                {"n_graphs": len(df), "n_families": df["network_family"].nunique(), "elapsed_sec": round(elapsed, 2)},
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(False, f"graph invariants preflight failed: {exc}")

    def domain_profile(self):
        from propab.domain_profiles.graph_invariants import GRAPH_INVARIANTS_PROFILE

        return GRAPH_INVARIANTS_PROFILE

    def literature_profile(self) -> dict[str, Any]:
        return {
            "seed_papers": [
                {
                    "title": "The Structure and Function of Complex Networks",
                    "authors": "M. E. J. Newman",
                    "year": 2003,
                    "doi": "10.1137/S003614450342480",
                },
            ],
            "search_terms": [
                "spectral gap", "algebraic connectivity", "clustering coefficient",
                "graph invariant", "network family", "modularity", "Watts-Strogatz",
                "Barabasi-Albert", "Erdos-Renyi random graph",
            ],
            "source_priorities": ["arxiv", "semantic_scholar", "mathoverflow"],
            "classification_codes": {
                "arxiv": ["cs.SI", "math.CO", "physics.soc-ph"],
            },
            "open_problem_sources": [],
            "tabulation_sources": [],
            "canonical_surveys": [
                {"title": "The Structure and Function of Complex Networks", "doi": "10.1137/S003614450342480"},
            ],
            "novelty_criteria": (
                "A finding is novel if it establishes a relationship between graph invariants "
                "(spectral, clustering, degree-structure) that survives leave-one-network-"
                "family-out holdout and is not a restatement of the well-known family-specific "
                "behavior surveyed in Newman (2003) (e.g. small-world clustering vs. random-graph "
                "clustering, scale-free degree heterogeneity)."
            ),
        }

    def hypothesis_on_topic(self, text: str, methodology: str | None = None) -> bool:
        combined = f"{text} {methodology or ''}".lower()
        if any(x in combined for x in ("sidon", "cap-set", "gene expression", "kcat")):
            return False
        return any(m in combined for m in self.scope_question_markers)
