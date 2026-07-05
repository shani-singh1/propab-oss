"""Math combinatorics DomainPlugin — deterministic Sidon/cap-set/AP-free verification."""
from __future__ import annotations

import re
from typing import Any

from propab.domain_modules.base import DomainPlugin, PreflightResult
from propab.domain_modules.math_combinatorics.problems import get_literature_prior
from propab.domain_modules.math_combinatorics.verifier import run_combinatorics_experiment

_OFF_TOPIC_RE = re.compile(
    r"|".join(
        (
            r"\bsystem\s+path\b",
            r"\bpath\s+for\s+binar",
            r"\bsearch\s+(?:the\s+)?(?:system\s+)?path\b",
            r"\bfind\s+binar",
            r"\bbinar(?:y|ies)\s+related",
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

_TOPIC_RE = re.compile(
    r"|".join(
        (
            r"\bsidon\b",
            r"\bcap[\s-]?set\b",
            r"\bsumset\b",
            r"\bap[\s-]?free\b",
            r"\barithmetic\s+progression",
            r"\badditive\s+combinator",
            r"\bextremal\b",
            r"\|A\+A\|",
            r"\bF_3\b",
            r"\bf_3\b",
            r"\|\{1,\.\.\.,n\}\|",
            r"\{1,\.\.\.,n\}",
        )
    ),
    re.IGNORECASE,
)

_UNIMPLEMENTED_METHODOLOGY_RE = re.compile(
    r"|".join(
        (
            r"simulated\s+anneal",
            r"fourier\s+analysis",
            r"\bfourier\b",
            r"\bgowers\b",
            r"\bsat\s+(?:solver|based|encoding|search)\b",
            r"\bz3\b",
            r"\bmilp\b",
            r"integer\s+linear\s+program",
            r"tensor\s+decomposition",
            r"neural\s+network",
            r"genetic\s+algorithm",
            r"branch[\s-]?and[\s-]?bound.*F_3\^8",
            r"exhaustive.*F_3\^8",
            r"F_3\^8.*exhaustive",
            r"\btabu\b",
            r"tabu\s+search",
            r"\bmcmc\b",
            r"markov\s+chain",
            r"hill[\s-]?climbing",
            r"stochastic\s+hill",
            r"stochastic\s+search",
            r"stochastic\s+optim",
            r"evolutionary\s+algorithm",
            r"chi[\s-]?squared",
            r"kolmogorov[\s-]?smirnov",
            r"\bks[\s-]?test\b",
            r"poisson",
            r"decile",
            r"spectral\s+peak",
        )
    ),
    re.IGNORECASE,
)

_VERIFIABLE_METHODOLOGY_RE = re.compile(
    r"|".join(
        (
            r"\bgreedy\b",
            r"bose[\s-]?chowla",
            r"exhaustive\s+search",
            r"multi[\s-]?n",
            r"\bsweep\b",
            r"cap[\s-]?set",
            r"ap[\s-]?free",
            r"best[\s-]?known",
            r"combinatorial\s+computation",
            r"product\s+construction",
            r"algebraic\s+construction",
            r"greedy\s+construction",
            r"sidon",
        )
    ),
    re.IGNORECASE,
)


class MathCombinatoricsPlugin(DomainPlugin):
    domain_id = "math_combinatorics"
    display_name = "Additive Combinatorics and Extremal Set Theory"
    version = "0.1.0"

    scope_question_markers = (
        "domain_profile:math_combinatorics",
        "sidon",
        "cap set",
        "sumset",
        "additive combinatorics",
        "arithmetic progression",
        "ap-free",
        "extremal set",
    )

    artifact_question_markers = (
        "combinatorics",
        "sidon",
        "cap set",
        "sumset",
        "additive",
        "extremal",
        "arithmetic progression",
        "ramsey",
    )
    theme_rules = (
        ("residue_class", (" mod ", " modulo ", "residue", "congruen")),
        ("parametric_family", ("parametric", "family", "closed-form")),
        ("finite_verification", ("exhaust", "scan", "up to n", "for all n", "counterexample")),
        ("unit_fraction", ("unit fraction", "egyptian", "1/n")),
        ("cache_policy", ("cache", "lru", "miss rate", "replacement policy", "belady")),
        ("scheduling", ("scheduling", "waiting time", "round-robin", "srpt", "queueing", "m/m/1")),
        ("auction", ("auction", "second-price", "bidder", "revenue equivalence")),
        ("collatz", ("collatz", "3n+1", "stopping time")),
        ("prime_gaps", ("prime gap", "cramér", "twin prime")),
    )

    def hypothesis_on_topic(self, text: str, methodology: str | None = None) -> bool:
        """Reject OS/filesystem hypotheses and unimplemented test methodologies."""
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
        # Scope-enriched methodology is JSON; accept when hypothesis text is on-topic.
        if meth.startswith("{"):
            return True
        if meth:
            return False
        return True

    def scope_template(self) -> dict[str, str]:
        return {
            "population": "Integers {1,...,n} or vector space F_3^n",
            "distribution": "All admissible combinatorial structures in the domain",
            "claimed_generalization": (
                "Structural/density property holds for the stated n or dimension range"
            ),
            "expected_failure_modes": (
                "Greedy construction artifact; bound valid only for tested n range"
            ),
            "ood_test": (
                "Independent search strategy or larger n replicates the claimed pattern"
            ),
        }

    def matches(self, *, question: str = "", payload: dict[str, Any] | None = None) -> bool:
        q = (question or "").lower()
        if "domain_profile:math_combinatorics" in q:
            return True
        if payload:
            if str(payload.get("domain_profile") or "") == "math_combinatorics":
                return True
            if str(payload.get("domain") or "") == "math_combinatorics":
                return True
        markers = ("sidon", "cap set", "sumset", "additive combinator", "ap-free")
        return any(m in q for m in markers)

    def available_features(self) -> list[str]:
        return [
            "sidon_set_density",
            "cap_set_size",
            "sumset_growth",
            "arithmetic_progression_free_density",
            "b2_plus_set_density",
        ]

    def run_verification(
        self,
        hypothesis: dict[str, Any],
        evidence: dict[str, Any] | None = None,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        feats = list(features or hypothesis.get("feature_subset") or [])
        if not feats:
            from propab.domain_modules.math_combinatorics.constructors import (
                extract_claim_text,
                is_cap_set_hypothesis,
            )

            text = str(hypothesis.get("text") or hypothesis.get("statement") or "")
            methodology = str(hypothesis.get("test_methodology") or "")
            claim = extract_claim_text(text, test_methodology=methodology, full_text=text)
            if is_cap_set_hypothesis(text, methodology, full_text=text):
                feats = ["cap_set_size"]
            elif "sumset" in claim.lower():
                feats = ["sumset_growth"]
            elif "ap-free" in claim.lower() or "arithmetic progression" in claim.lower():
                feats = ["arithmetic_progression_free_density"]
            else:
                feats = ["sidon_set_density"]
        return run_combinatorics_experiment(hypothesis, feats)

    def classify_verdict(
        self, hypothesis_text: str, result: dict[str, Any]
    ) -> tuple[str, str, float]:
        _ = hypothesis_text
        if result.get("trivial_rediscovery"):
            return (
                "inconclusive",
                result.get("notes") or "recomputes known result; not open-problem evidence",
                0.40,
            )
        vf = int(result.get("verified_false_steps") or 0)
        vt = int(result.get("verified_true_steps") or 0)
        if vf > 0:
            return (
                "refuted",
                result.get("notes") or "counterexample found",
                0.95,
            )
        if vt > 0 and result.get("discovery_worthy"):
            return (
                "confirmed",
                result.get("notes") or "open-problem computational evidence",
                0.92,
            )
        if vt > 0:
            return (
                "inconclusive",
                result.get("notes") or "computation succeeded but insufficient for open problem",
                0.50,
            )
        return "inconclusive", "no definitive combinatorial result", 0.5

    def artifact_models(
        self,
        evidence: dict[str, Any] | None = None,
        hypothesis: dict[str, Any] | None = None,
    ) -> list[Any]:
        from propab.artifact_verification import ArtifactModel

        _ = evidence, hypothesis
        return [
            ArtifactModel(
                artifact_id="algorithm_specific",
                description=(
                    "Pattern may be specific to the greedy search algorithm, "
                    "not a general structural property"
                ),
                plausibility_score=0.3,
                why_plausible="Greedy Sidon/cap-set constructions are not proven optimal.",
                affected_components=["search_strategy"],
                proposed_test="Rerun with a different search strategy (exhaustive vs greedy)",
                artifact_rank=1,
            ),
        ]

    def confirmation_criteria(self) -> dict[str, Any]:
        return {
            "min_metric_steps_for_confirm": 1,
            "min_confidence": 0.90,
            "requires_holdout": False,
            "holdout_type": "none",
            "null_test": "exhaustive_counterexample_search",
            "verification_type": "deterministic",
        }

    def preflight(self) -> PreflightResult:
        try:
            def is_sidon(s: set[int]) -> bool:
                lst = sorted(s)
                sums: set[int] = set()
                for i in range(len(lst)):
                    for j in range(i + 1, len(lst)):
                        pair_sum = lst[i] + lst[j]
                        if pair_sum in sums:
                            return False
                        sums.add(pair_sum)
                return True

            test_set = {1, 2, 5}
            if not is_sidon(test_set):
                return PreflightResult(False, "Known Sidon set failed check")

            import time

            start = time.time()
            max_size = 0
            for n in range(10, 201):
                current = [1]
                sums: set[int] = set()
                for x in range(2, n + 1):
                    new_sums = {x + y for y in current}
                    if not new_sums & sums:
                        sums |= new_sums
                        current.append(x)
                max_size = max(max_size, len(current))
            elapsed = time.time() - start

            if elapsed > 30:
                return PreflightResult(
                    passed=False,
                    reason=f"Combinatorial search too slow: {elapsed:.1f}s for n≤200",
                )

            return PreflightResult(
                passed=True,
                reason=(
                    f"Combinatorial computation works. Max Sidon size in {{1..200}}: "
                    f"{max_size}. Search time: {elapsed:.2f}s"
                ),
                details={"max_sidon_size": max_size, "elapsed_seconds": elapsed},
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(passed=False, reason=f"Preflight computation failed: {exc}")

    def literature_prior(self, question: str) -> dict[str, Any]:
        return get_literature_prior(question)

    def literature_profile(self) -> dict[str, Any]:
        return {
            "seed_papers": [
                {
                    "title": "On a problem of Sidon in additive number theory, and on some related problems",
                    "authors": "P. Erdos, P. Turan",
                    "year": 1941,
                },
                {
                    "title": "Progression-free sets in Z_4^n are exponentially small",
                    "authors": "E. Croot, V. Lev, P. Pach",
                    "year": 2016,
                    "arxiv_id": "1605.01506",
                },
                {
                    "title": "On large subsets of F_q^n with no three-term arithmetic progression",
                    "authors": "J. S. Ellenberg, D. Gijswijt",
                    "year": 2016,
                    "arxiv_id": "1605.09223",
                },
                {
                    "title": "On sets of integers which contain no three terms in arithmetical progression",
                    "authors": "F. A. Behrend",
                    "year": 1946,
                },
            ],
            "search_terms": [
                "Sidon set", "cap set", "sumset", "AP-free", "arithmetic progression free",
                "additive combinatorics", "extremal set theory", "Behrend construction",
                "Croot-Lev-Pach", "Ellenberg-Gijswijt", "Bose-Chowla",
            ],
            "source_priorities": ["arxiv", "oeis", "zbmath", "semantic_scholar", "mathoverflow"],
            "classification_codes": {
                "arxiv": ["math.CO", "math.NT"],
                "zbmath": ["05B10", "11B13", "11B25", "05D99"],
            },
            "open_problem_sources": [
                {"name": "Erdos problems list", "url": "https://www.erdosproblems.com/"},
            ],
            "tabulation_sources": [
                {
                    "name": "OEIS",
                    "identifiers": ["A005282", "A090245", "A003002"],
                    # A005282: Mian-Chowla (greedy Sidon/B2) sequence.
                    # A090245: max "no SET" card group per n attributes == cap-set size in F_3^n.
                    # A003002: r3(n), largest 3-AP-free subset of {1..n} (Behrend/Szemeredi regime).
                },
            ],
            "canonical_surveys": [
                {"title": "Additive Combinatorics", "doi": "10.1017/CBO9780511755149"},
            ],
            "novelty_criteria": (
                "A finding is novel if it establishes a Sidon/cap-set/AP-free bound or exact "
                "value not present in the OEIS tabulations above (A005282, A090245, A003002) "
                "and not directly implied by the Erdos-Turan/Lindstrom Sidon bounds, the "
                "Croot-Lev-Pach/Ellenberg-Gijswijt cap-set bound, or the Behrend/Szemeredi "
                "AP-free bounds listed in established_facts."
            ),
        }

    def belief_promotion_threshold(self) -> dict[str, Any]:
        return {
            "requires_supporting_nodes": 3,
            "max_supporting_nodes": 12,
            "requires_confidence": "weak",
            "allow_trend_promotion": True,
            "trend_definition": (
                "3+ confirmed nodes whose metric_value shows consistent "
                "directional movement across increasing parameter values"
            ),
        }

    def implementable_methodologies(self) -> list[str]:
        return [
            "greedy construction",
            "greedy search",
            "greedy sidon",
            "bose-chowla",
            "bose chowla",
            "exhaustive search",
            "cap set table",
            "cap-set lookup",
            "threshold sweep",
            "ratio sweep",
            "band validation",
            "counterexample search",
            "table lookup",
            "ap-free",
            "sumset",
        ]

    def extract_numerical_seeds(self, confirmed_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from propab.numerical_seeds import extract_math_combinatorics_seeds

        return extract_math_combinatorics_seeds(confirmed_nodes)

    def domain_profile(self):
        from propab.domain_profiles.math_combinatorics import MATH_COMBINATORICS_PROFILE

        return MATH_COMBINATORICS_PROFILE
