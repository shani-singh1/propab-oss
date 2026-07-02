"""Domain-specific search intents for the V1 literature prior experiment."""
from __future__ import annotations

DOMAIN_QUERIES: dict[str, dict[str, object]] = {
    "materials": {
        "display_name": "Materials — dielectric constant prediction",
        "research_question": (
            "[domain_profile:materials] Discover which structural or compositional "
            "descriptors predict dielectric constant across crystal-system families. "
            "Require cross-family holdout (LOFO); report relationships that generalize."
        ),
        "arxiv_queries": [
            "all:dielectric+constant+prediction+inorganic+materials",
            "all:matbench+dielectric+machine+learning",
            "all:composition+descriptors+dielectric+constant+materials",
            "all:LOFO+leave-one-family-out+materials+property",
        ],
        "semantic_scholar_queries": [
            "dielectric constant prediction inorganic materials machine learning",
            "matbench dielectric composition descriptors",
            "crystal system generalization dielectric constant",
        ],
        "extraction_focus": (
            "Which compositional or structural descriptors (e.g. ionic polarizability, "
            "band gap, density, number of sites, Magpie-style features) are claimed to "
            "predict dielectric constant; whether each claim is said to generalize across "
            "crystal systems or composition families; reported LOFO / cross-family holdout "
            "results if any."
        ),
    },
}

DEFAULT_DOMAIN = "materials"
