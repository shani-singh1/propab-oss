"""
Network diffusion / contagion DomainPlugin.

A real discovery domain: it simulates diffusion (SIR / independent cascade) on
REAL SNAP networks (``ca-GrQc`` collaboration, ``email-Eu-core`` email) and tests
whether a structural feature (degree heterogeneity, clustering, assortativity)
predicts a diffusion outcome (final outbreak size, outbreak probability). A claim
is "confirmed" only if the structure->outcome correlation replicates on a
held-out real topology family, survives a within-family label-shuffle null, and
is robust to the choice of simulator.

Data is real (``uses_synthetic_data`` -> False); topology comes from SNAP edge
lists on disk, never fabricated. See ``adapter.py`` for provenance and
``verifier.py`` for the holdout/null design.
"""
from __future__ import annotations

import time
from typing import Any

from propab.domain_modules.base import DomainPlugin, PreflightResult
from propab.domain_modules.network_diffusion.adapter import (
    REAL_NETWORKS,
    STRUCTURAL_FEATURES,
    DiffusionSpec,
    network_adjacency,
    sample_subgraphs,
    structural_features,
)
from propab.domain_modules.network_diffusion.simulator import simulate
from propab.domain_modules.network_diffusion.verifier import (
    classify_diffusion_verdict,
    run_diffusion_experiment,
)


class NetworkDiffusionPlugin(DomainPlugin):
    domain_id = "network_diffusion"
    display_name = "Network diffusion / contagion (real SNAP topology families)"
    version = "2.0"
    scope_question_markers = (
        "contagion",
        "diffusion",
        "spreading",
        "sis",
        "sir",
        "epidemic",
        "outbreak",
        "cascade",
    )
    # Vocabulary that marks a claim as network/graph-topology for artifact-model
    # selection in core's artifact gate. Kept identical to the historical set so
    # artifact-gate behaviour is unchanged.
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

    # Words that mark a contagion/diffusion question specifically (not just any
    # graph question — graph_invariants owns static-invariant questions).
    _DIFFUSION_MARKERS = (
        "contagion",
        "diffusion",
        "epidemic",
        "outbreak",
        "spreading",
        "sir model",
        "sis model",
        "independent cascade",
        "cascade",
        "infection",
        "transmission",
        "immunization",
        "epidemic threshold",
    )

    def matches(self, *, question: str = "", payload: dict[str, Any] | None = None) -> bool:
        if payload and str(payload.get("domain") or payload.get("domain_profile") or "") == "network_diffusion":
            return True
        q = (question or "").lower()
        if "[domain_profile:network_diffusion]" in q:
            return True
        # Discriminate contagion/diffusion questions from static graph-invariant
        # questions: require at least one strong diffusion marker AND a graph/network
        # context. This keeps routing distinct from graph_invariants.
        hits = sum(1 for m in self._DIFFUSION_MARKERS if m in q)
        has_graph_ctx = any(w in q for w in ("network", "graph", "node", "edge", "topology"))
        return hits >= 2 or (hits >= 1 and has_graph_ctx)

    def available_features(self) -> list[str]:
        return list(STRUCTURAL_FEATURES)

    def uses_synthetic_data(self) -> bool:
        return False

    def scope_template(self) -> dict[str, str]:
        return {
            "population": "Real SNAP networks: ca-GrQc (collaboration) + email-Eu-core (email); ~36 induced subgraphs/family, 100 nodes each",
            "distribution": "Leave-one-real-network-family-out holdout (collaboration vs email topology)",
            "claimed_generalization": "Structure->diffusion correlation replicates on the held-out real network family",
            "expected_failure_modes": "Effect is simulator-specific (SIR vs cascade); outcome tracks network scale not topology; correlation vanishes on holdout",
            "ood_test": "Held-out real family Spearman correlation vs within-family shuffle null (p95 + permutation p<0.05) + alternate-simulator robustness",
        }

    def confirmation_criteria(self) -> dict[str, Any]:
        return {
            "min_metric_steps_for_confirm": 1,
            "requires_holdout": True,
            "holdout_type": "leave_network_family_out",
            "null_test": "within_family_outcome_shuffle",
            "verification_type": "simulation_statistical",
            "robustness_test": "alternate_simulator",
        }

    def objective_spec(self) -> dict[str, Any]:
        """Diffusion is scored by a held-out correlation, not a trained ML metric.

        The verifier emits ``metric_name="cross_family_diffusion_correlation"``
        (``network_diffusion/verifier.py``): the Spearman correlation between a
        structural feature and a simulated diffusion outcome, measured on a
        *held-out* real topology family and gated by a within-family shuffle null
        plus alternate-simulator robustness. Simulation + statistics, not ML
        training.

        This override is doubly required here: the domain's own vocabulary
        ("network", "diffusion") includes ``"network"``, which is an ML *question*
        token — so returning ``None`` would let ``_is_ml_campaign`` fall through to
        the keyword heuristic and mis-classify a diffusion campaign as ML. With
        ``is_ml=False`` core short-circuits that heuristic to False. There is no
        external best-known table for this held-out correlation, so the baseline is
        ``"measured"``.
        """
        return {
            "metric_name": "cross_family_diffusion_correlation",
            "direction": "higher_is_better",
            "is_ml": False,
            "baseline_kind": "measured",
        }

    def run_verification(
        self,
        hypothesis: dict[str, Any],
        evidence: dict[str, Any] | None = None,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        _ = evidence
        spec = DiffusionSpec.from_hypothesis(hypothesis)
        if features:
            usable = [f for f in features if f in STRUCTURAL_FEATURES]
            if usable:
                spec = DiffusionSpec(
                    structural_feature=usable[0],
                    outcome=spec.outcome,
                    simulator=spec.simulator,
                    claim_sign=spec.claim_sign,
                    held_out_family=spec.held_out_family,
                )
        return run_diffusion_experiment(spec)

    def classify_verdict(self, hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
        return classify_diffusion_verdict(hypothesis_text, result)

    def hypothesis_on_topic(self, text: str, methodology: str | None = None) -> bool:
        combined = f"{text} {methodology or ''}".lower()
        if any(x in combined for x in ("sidon", "cap-set", "kcat", "gene expression", "enzyme", "perovskite")):
            return False
        return any(m in combined for m in self._DIFFUSION_MARKERS)

    def preflight(self) -> PreflightResult:
        try:
            t0 = time.time()
            details: dict[str, Any] = {}
            # 1. Every real network loads.
            for fam in REAL_NETWORKS:
                adj = network_adjacency(fam)
                details[f"{fam}_nodes"] = len(adj)
            # 2. Subgraph sampling + a real simulation runs end to end.
            import numpy as np

            rng = np.random.default_rng(0)
            probe_fam = next(iter(REAL_NETWORKS))
            subs = sample_subgraphs(probe_fam, n_samples=2, target_size=80, rng=rng)
            if not subs:
                return PreflightResult(False, f"could not sample subgraphs from {probe_fam}")
            feats = structural_features(subs[0])
            final = simulate(subs[0], simulator="sir", outcome="final_size",
                             beta=0.12, gamma=0.5, n_runs=5, rng=rng)
            details["probe_feature_k2_over_k1"] = round(feats["k2_over_k1"], 3)
            details["probe_sir_final_size"] = round(final, 3)
            elapsed = time.time() - t0
            details["elapsed_sec"] = round(elapsed, 2)
            if elapsed > 60:
                return PreflightResult(False, f"diffusion preflight too slow: {elapsed:.1f}s", details)
            return PreflightResult(True, "real SNAP networks loaded, SIR simulation ok", details)
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(False, f"network diffusion preflight failed: {exc}")

    def domain_profile(self):
        from propab.domain_profiles.network_diffusion import NETWORK_DIFFUSION_PROFILE

        return NETWORK_DIFFUSION_PROFILE

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
                "Barabasi-Albert", "Watts-Strogatz", "independent cascade", "degree heterogeneity",
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
                "replicates from a training real topology family (e.g. collaboration) to a "
                "held-out family (e.g. email), survives a within-family shuffle null, and is "
                "not already predicted by the vanishing-epidemic-threshold result for "
                "heterogeneous networks (Pastor-Satorras & Vespignani, 2001)."
            ),
        }
