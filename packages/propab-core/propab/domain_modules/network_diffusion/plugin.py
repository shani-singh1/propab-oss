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
