from __future__ import annotations

import unittest

from services.orchestrator.literature_cache import query_hash
from services.orchestrator.literature_quality import (
    build_search_intents,
    classify_evidence_status,
    gate_corpus_quality,
    insufficient_prior,
)
from services.orchestrator.schemas import Prior


class LiteratureCacheTests(unittest.TestCase):
    def test_query_hash_normalizes_whitespace(self) -> None:
        a = query_hash("  What is   X?  ")
        b = query_hash("what is x?")
        self.assertEqual(a, b)


class LiteratureQualityTests(unittest.TestCase):
    def test_build_search_intents_dedupes(self) -> None:
        intents = build_search_intents(
            "Main question?",
            {"rephrasings": ["Main question?", "Alt phrasing"], "concepts": ["concept A"]},
        )
        self.assertEqual(intents[0], "Main question?")
        self.assertEqual(len(intents), 3)

    def test_gate_fails_on_empty_corpus(self) -> None:
        passed, reasons = gate_corpus_quality(
            papers_kept=[],
            chunk_count=0,
            evidence_coverage=0.0,
        )
        self.assertFalse(passed)
        self.assertTrue(any("too_few" in r for r in reasons))

    def test_gate_passes_with_sufficient_corpus(self) -> None:
        passed, reasons = gate_corpus_quality(
            papers_kept=[{"id": "1"}, {"id": "2"}],
            chunk_count=5,
            evidence_coverage=0.9,
        )
        self.assertTrue(passed)
        self.assertEqual(reasons, [])

    def test_insufficient_prior_status(self) -> None:
        prior = insufficient_prior("Q?", {"evidence_coverage": 0.1})
        self.assertEqual(prior.evidence_status, "INSUFFICIENT_EVIDENCE")
        self.assertIsNotNone(prior.retrieval_diagnostics)

    def test_classify_missing_data_as_insufficient(self) -> None:
        prior = Prior(
            established_facts=[],
            contested_claims=[],
            open_gaps=[{"text": "no papers", "source_paper": "x", "gap_type": "missing_data"}],
            dead_ends=[],
            key_papers=[],
        )
        status = classify_evidence_status(prior, evidence_coverage=0.8, gate_passed=True)
        self.assertEqual(status, "INSUFFICIENT_EVIDENCE")

    def test_classify_conflicting(self) -> None:
        prior = Prior(
            established_facts=[],
            contested_claims=[{"text": "a"}, {"text": "b"}],
            open_gaps=[],
            dead_ends=[],
            key_papers=[],
        )
        status = classify_evidence_status(prior, evidence_coverage=0.8, gate_passed=True)
        self.assertEqual(status, "CONFLICTING_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
