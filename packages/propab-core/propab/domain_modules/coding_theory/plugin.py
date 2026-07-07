"""Coding-theory DomainPlugin — real binary linear code construction + verified minimum distance."""
from __future__ import annotations

import re
from typing import Any

from propab.domain_modules.base import DomainPlugin, PreflightResult
from propab.domain_modules.coding_theory.problems import get_literature_prior
from propab.domain_modules.coding_theory.verifier import run_coding_experiment

_OFF_TOPIC_RE = re.compile(
    r"|".join(
        (
            r"\bsystem\s+path\b",
            r"\bpath\s+for\s+binar",
            r"\bsearch\s+(?:the\s+)?(?:system\s+)?path\b",
            r"\bfind\s+binar(?:y|ies)\b",
            r"\bpropab-submit\b",
            r"\bsubmit-results\b",
            r"\breport-metrics\b",
            r"\bfile\s*system\b",
            r"\bfilesystem\b",
            r"\bos\.environ\b",
            r"\bsubprocess\b",
            r"\bshell\s+command",
            r"\boperating\s+system\b",
        )
    ),
    re.IGNORECASE,
)

# On-topic markers for binary linear coding theory. Note: "binary code" here means
# error-correcting code; the off-topic filter above removes OS/binary-executable
# senses first.
_TOPIC_RE = re.compile(
    r"|".join(
        (
            r"\blinear\s+code\b",
            r"\berror[\s-]?correct",
            r"\bminimum\s+distance\b",
            r"\bhamming\s+(?:code|distance|weight)\b",
            r"\bgenerator\s+matrix\b",
            r"\bparity[\s-]?check\b",
            r"\bcodeword\b",
            r"\[\s*\d+\s*,\s*\d+(?:\s*,\s*\d+)?\s*\]",   # [n,k] or [n,k,d]
            r"\bgf\(2\)\b",
            r"\bbch\s+code\b",
            r"\breed[\s-]?muller\b",
            r"\bldpc\b",
            r"\bsimplex\s+code\b",
            r"\bgolay\b",
            r"\bgriesmer\b",
            r"\bsingleton\s+bound\b",
            r"\brepetition\s+code\b",
            r"\bcoding\s+theory\b",
        )
    ),
    re.IGNORECASE,
)

# Methodologies this verifier does NOT implement (would be an unverifiable claim).
_UNIMPLEMENTED_METHODOLOGY_RE = re.compile(
    r"|".join(
        (
            r"simulated\s+anneal",
            r"neural\s+network",
            r"deep\s+learning",
            r"genetic\s+algorithm",
            r"evolutionary\s+algorithm",
            r"\bsat\s+(?:solver|based|encoding)\b",
            r"\bz3\b",
            r"\bmilp\b",
            r"integer\s+linear\s+program",
            r"reinforcement\s+learning",
            r"\bmcmc\b",
            r"markov\s+chain",
            r"tabu\s+search",
            r"belief\s+propagation\s+decoder",  # decoding perf, not distance
            r"monte\s+carlo\s+decod",
        )
    ),
    re.IGNORECASE,
)

# Methodologies that map to implemented, verifiable computation.
_VERIFIABLE_METHODOLOGY_RE = re.compile(
    r"|".join(
        (
            r"exhaustive\s+enumeration",
            r"exhaustive\s+search",
            r"minimum[\s-]?distance\s+computation",
            r"generator\s+matrix",
            r"hamming",
            r"simplex",
            r"reed[\s-]?muller",
            r"repetition",
            r"parity[\s-]?check",
            r"algebraic\s+construction",
            r"explicit\s+construction",
            r"codeword\s+enumeration",
            r"combinatorial\s+computation",
        )
    ),
    re.IGNORECASE,
)


class CodingTheoryPlugin(DomainPlugin):
    domain_id = "coding_theory"
    display_name = "Binary Linear Error-Correcting Codes"
    version = "0.1.0"

    scope_question_markers = (
        "domain_profile:coding_theory",
        "linear code",
        "error-correcting code",
        "minimum distance",
        "hamming code",
        "generator matrix",
        "parity-check",
        "coding theory",
    )

    artifact_question_markers = (
        "coding theory",
        "linear code",
        "error-correcting",
        "minimum distance",
        "generator matrix",
        "hamming",
        "codeword",
        "bch",
    )

    theme_rules = (
        ("algebraic_construction", ("hamming", "simplex", "reed-muller", "bch", "golay", "algebraic")),
        ("bound_comparison", ("singleton", "griesmer", "sphere-packing", "upper bound", "best-known")),
        ("modified_code", ("shorten", "extend", "punctur", "augment")),
        ("exhaustive_distance", ("exhaustive", "enumerate", "codeword weight", "minimum weight")),
    )

    # --- Topic / methodology gating ----------------------------------------
    def hypothesis_on_topic(self, text: str, methodology: str | None = None) -> bool:
        """Reject OS/filesystem hypotheses and unimplemented methodologies."""
        t = (text or "").strip()
        if not t:
            return False
        if _OFF_TOPIC_RE.search(t):
            return False
        if not _TOPIC_RE.search(t):
            return False
        combined = f"{t}\n{methodology or ''}"
        if _UNIMPLEMENTED_METHODOLOGY_RE.search(combined):
            return False
        if _VERIFIABLE_METHODOLOGY_RE.search(combined):
            return True
        meth = (methodology or "").strip()
        if meth.startswith("{"):  # scope-enriched JSON — accept on-topic text
            return True
        if meth:
            return False
        return True

    def scope_template(self) -> dict[str, str]:
        return {
            "population": "Binary linear [n, k] codes over GF(2)",
            "distribution": "All k x n generator matrices of full row rank over GF(2)",
            "claimed_generalization": (
                "The constructed code attains the stated minimum distance d for the given [n, k]"
            ),
            "expected_failure_modes": (
                "Distance reported from a table instead of a computed witness; witness "
                "codeword does not achieve the claimed weight; k too large to enumerate honestly"
            ),
            "ood_test": (
                "Independent recomputation of the minimum distance on the actual generator "
                "matrix reproduces the claimed d and witness weight"
            ),
        }

    # --- Detection ----------------------------------------------------------
    def matches(self, *, question: str = "", payload: dict[str, Any] | None = None) -> bool:
        q = (question or "").lower()
        if "domain_profile:coding_theory" in q:
            return True
        if payload:
            if str(payload.get("domain_profile") or "") == "coding_theory":
                return True
            if str(payload.get("domain") or "") == "coding_theory":
                return True
        markers = (
            "linear code",
            "error-correcting code",
            "error correcting code",
            "minimum distance",
            "hamming code",
            "generator matrix",
            "parity-check",
            "coding theory",
            "reed-muller",
            "bch code",
        )
        return any(m in q for m in markers)

    def available_features(self) -> list[str]:
        return [
            "code_minimum_distance",
            "generator_matrix_rank",
            "codeword_weight_enumeration",
            "best_known_distance_gap",
        ]

    def uses_synthetic_data(self) -> bool:
        """Real GF(2) computation on explicit generator matrices — not synthetic data."""
        return False

    # --- Verification -------------------------------------------------------
    def run_verification(
        self,
        hypothesis: dict[str, Any],
        evidence: dict[str, Any] | None = None,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        _ = evidence
        return run_coding_experiment(hypothesis, features)

    def classify_verdict(
        self, hypothesis_text: str, result: dict[str, Any]
    ) -> tuple[str, str, float]:
        _ = hypothesis_text
        # A witness that failed independent recheck is never a verdict.
        if result.get("witness_recheck_ok") is False:
            return (
                "inconclusive",
                result.get("notes") or "witness failed independent recomputation",
                0.30,
            )
        if result.get("trivial_rediscovery"):
            return (
                "inconclusive",
                result.get("notes") or "reproduces best-known table value; not novel",
                0.40,
            )
        vf = int(result.get("verified_false_steps") or 0)
        vt = int(result.get("verified_true_steps") or 0)
        if vf > 0:
            return (
                "refuted",
                result.get("notes") or "computed distance refutes the claim",
                0.95,
            )
        if vt > 0 and result.get("discovery_worthy"):
            return (
                "confirmed",
                result.get("notes") or "real code beats best-known lower bound; witness re-checked",
                0.92,
            )
        if vt > 0:
            return (
                "inconclusive",
                result.get("notes") or "computation succeeded but does not beat known bound",
                0.50,
            )
        return (
            "inconclusive",
            result.get("notes") or "no definitive coding-theory result",
            0.50,
        )

    def artifact_models(
        self,
        evidence: dict[str, Any] | None = None,
        hypothesis: dict[str, Any] | None = None,
    ) -> list[Any]:
        from propab.artifact_verification import ArtifactModel

        _ = evidence, hypothesis
        return [
            ArtifactModel(
                artifact_id="table_value_leak",
                description=(
                    "Reported distance may echo a best-known table value rather than the "
                    "true distance of the constructed generator matrix."
                ),
                plausibility_score=0.3,
                why_plausible="Table-lookup shortcuts are the classic coding-theory false positive.",
                affected_components=["min_distance_computation"],
                proposed_test=(
                    "Independently recompute the minimum Hamming weight over all 2^k-1 "
                    "nonzero codewords of the actual generator and compare to the witness."
                ),
                artifact_rank=1,
            ),
        ]

    def confirmation_criteria(self) -> dict[str, Any]:
        return {
            "min_metric_steps_for_confirm": 1,
            "min_confidence": 0.90,
            "requires_holdout": False,
            "holdout_type": "none",
            "null_test": "independent_witness_recomputation",
            "verification_type": "deterministic",
        }

    # --- Preflight ----------------------------------------------------------
    def preflight(self) -> PreflightResult:
        try:
            import time

            from propab.domain_modules.coding_theory.constructors import (
                compute_min_distance,
                hamming_code,
                recompute_distance_of_witness,
            )

            start = time.time()
            g = hamming_code(3)  # [7,4,3]
            dist = compute_min_distance(g)
            if dist.get("min_distance") != 3:
                return PreflightResult(
                    False,
                    f"[7,4] Hamming code computed d={dist.get('min_distance')} (expected 3)",
                )
            # Independent witness recheck must agree.
            recheck = recompute_distance_of_witness(
                dist["generator_matrix"], dist["witness_message"],
            )
            if recheck.get("weight") != 3:
                return PreflightResult(
                    False,
                    f"Witness recomputation gave weight {recheck.get('weight')} (expected 3)",
                )
            elapsed = time.time() - start
            return PreflightResult(
                passed=True,
                reason=(
                    "GF(2) code construction + exhaustive min-distance works. "
                    f"[7,4] Hamming d=3 verified with re-checked witness. Time: {elapsed:.3f}s"
                ),
                details={"hamming_7_4_distance": 3, "elapsed_seconds": elapsed},
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(passed=False, reason=f"Preflight computation failed: {exc}")

    # --- Literature ---------------------------------------------------------
    def literature_prior(self, question: str) -> dict[str, Any]:
        return get_literature_prior(question)

    def literature_profile(self) -> dict[str, Any]:
        return {
            "seed_papers": [
                {
                    "title": "Error detecting and error correcting codes",
                    "authors": "R. W. Hamming",
                    "year": 1950,
                },
                {
                    "title": "Bounds for error-correcting codes",
                    "authors": "J. H. Griesmer",
                    "year": 1960,
                },
                {
                    "title": "Maximum distance separable codes and the Singleton bound",
                    "authors": "R. C. Singleton",
                    "year": 1964,
                },
                {
                    "title": "Bounds on the minimum distance of linear codes (server / tables)",
                    "authors": "M. Grassl",
                    "year": 2007,
                    "url": "http://www.codetables.de",
                },
            ],
            "search_terms": [
                "binary linear code", "minimum distance", "generator matrix",
                "parity-check matrix", "Hamming code", "BCH code", "Reed-Muller code",
                "simplex code", "Griesmer bound", "Singleton bound", "sphere-packing bound",
                "best-known linear codes", "optimal linear code", "code tables",
            ],
            "source_priorities": ["arxiv", "semantic_scholar", "zbmath", "oeis"],
            "classification_codes": {
                "arxiv": ["cs.IT", "math.CO", "math.IT"],
                "zbmath": ["94B05", "94B15", "94B25", "94B65"],
            },
            "open_problem_sources": [
                {
                    "name": "Bounds on the minimum distance of linear codes (codetables.de)",
                    "url": "http://www.codetables.de/",
                },
            ],
            "tabulation_sources": [
                {
                    "name": "codetables.de (Grassl)",
                    "url": "http://www.codetables.de/",
                    "description": "Best-known lower and upper bounds on d for [n,k]_q codes",
                },
                {
                    "name": "Brouwer's tables of bounds",
                    "description": "Classic tables of best-known binary linear code distances",
                },
                {
                    "name": "OEIS",
                    "identifiers": ["A005864", "A005865", "A005866"],
                    "description": "Maximum size of binary codes with given length and distance",
                },
            ],
            "canonical_surveys": [
                {
                    "title": "The Theory of Error-Correcting Codes",
                    "authors": "F. J. MacWilliams, N. J. A. Sloane",
                    "year": 1977,
                },
            ],
            "novelty_criteria": (
                "A finding is novel only if it exhibits an explicit binary linear [n, k] "
                "generator matrix over GF(2) whose independently recomputed minimum distance "
                "STRICTLY EXCEEDS the best-known lower bound tabulated at codetables.de / the "
                "Brouwer/Grassl tables. Reproducing a tabulated distance, or reporting a "
                "distance without an achieving witness codeword, is a rediscovery, not a discovery."
            ),
        }

    def belief_promotion_threshold(self) -> dict[str, Any]:
        return {
            "requires_supporting_nodes": 1,
            "max_supporting_nodes": 12,
            "requires_confidence": "weak",
            "allow_trend_promotion": True,
            "trend_definition": (
                "3+ confirmed codes whose computed minimum distance meets/beats the "
                "best-known bound across a family of [n,k] parameters"
            ),
        }

    def implementable_methodologies(self) -> list[str]:
        return [
            "exhaustive enumeration",
            "exhaustive search",
            "minimum distance computation",
            "codeword enumeration",
            "generator matrix",
            "hamming",
            "extended hamming",
            "simplex",
            "reed-muller",
            "repetition",
            "parity-check",
            "algebraic construction",
            "explicit construction",
            "combinatorial computation",
        ]
