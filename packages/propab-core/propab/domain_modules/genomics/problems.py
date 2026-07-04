"""Literature prior and open problems for genomics domain."""
from __future__ import annotations

from typing import Any


def get_literature_prior(question: str = "") -> dict[str, Any]:
    _ = question
    return {
        "established_facts": [
            "Many eQTLs replicate across tissues (GTEx Consortium, Nature 2015).",
            "Housekeeping genes show constitutive expression across tissues.",
            "Tissue specificity (tau index) separates housekeeping from tissue-enriched genes.",
        ],
        "open_questions": [
            "Which non-housekeeping genes show partial cross-tissue expression conservation?",
            "Do genes with low tau index share metabolic or stress-response functional annotations?",
            "Can expression variance alone predict cross-tissue LOFO generalization?",
        ],
        "domain": "genomics",
    }
