"""Phase 2 — hybrid retrieval RRF (ARCHITECTURE.md §5.3)."""

from __future__ import annotations

from services.orchestrator.rrf_util import rrf_merge


def test_rrf_prefers_items_ranked_high_in_multiple_lists() -> None:
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    merged = rrf_merge([a, b])
    assert len(merged) >= 3
    top_ids = [cid for cid, _ in merged[:2]]
    assert "x" in top_ids and "y" in top_ids


def test_rrf_single_list() -> None:
    merged = rrf_merge([["a", "b"]])
    assert [cid for cid, _ in merged] == ["a", "b"]
