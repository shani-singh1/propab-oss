"""DISC1: novelty-gating of confirmed findings in the paper/finalize path.

Covers:
  1. ``apply_novelty_demotion`` moves a "known" confirmed finding into a
     ``rediscovery`` bucket and drops it from the headline ``counts["confirmed"]``;
     a "novel" finding is kept as a discovery.
  2. ``_demote_known_findings`` (the paper wiring) novelty-checks each confirmed
     finding and applies the demotion — with the literature service configured.
  3. ``literature_service_url=""`` → novelty check skipped, findings unchanged
     (backward compatible, no crash).
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from propab.paper_compiler import apply_novelty_demotion


def _findings(confirmed: list[dict]) -> dict:
    return {
        "counts": {
            "confirmed": len(confirmed),
            "refuted": 0,
            "inconclusive": 0,
            "unexecuted": 0,
            "tested": len(confirmed),
        },
        "confirmed": confirmed,
        "refuted": [],
        "inconclusive": [],
    }


def test_apply_novelty_demotion_known_becomes_rediscovery() -> None:
    findings = _findings(
        [
            {"id": "a", "text": "cap-set value c(6)=112", "confidence": 0.9},
            {"id": "b", "text": "a brand new bound", "confidence": 0.8},
        ]
    )
    novelty = {
        "a": {"verdict": "known", "confidence": 0.95, "explanation": "OEIS A090245",
              "matching_sources": [{"source_title": "Cap Set Tables"}]},
        "b": {"verdict": "novel", "confidence": 0.9, "matching_sources": []},
    }

    out = apply_novelty_demotion(findings, novelty)

    # Known one demoted out of the headline discovery bucket …
    assert [f["id"] for f in out["confirmed"]] == ["b"]
    assert out["counts"]["confirmed"] == 1
    # … into a rediscovery bucket, flagged for the paper.
    assert [f["id"] for f in out["rediscovery"]] == ["a"]
    assert out["counts"]["rediscovery"] == 1
    redis = out["rediscovery"][0]
    assert redis["is_rediscovery"] is True
    assert redis["novelty"] == "known"
    assert redis["node_role"] == "REDISCOVERY"
    # The novel finding is untouched (kept a discovery), labelled novel.
    kept = out["confirmed"][0]
    assert kept.get("is_rediscovery") is not True
    assert kept["novelty"] == "novel"
    # ``tested`` still counts the rediscovery (it WAS tested & confirmed).
    assert out["counts"]["tested"] == 2


def test_apply_novelty_demotion_uncertain_and_unavailable_kept() -> None:
    findings = _findings(
        [
            {"id": "a", "text": "finding a", "confidence": 0.9},
            {"id": "b", "text": "finding b", "confidence": 0.8},
        ]
    )
    novelty = {
        "a": {"verdict": "uncertain", "confidence": 0.5},
        "b": {"verdict": "uncertain", "source": "novelty_unavailable"},
    }
    out = apply_novelty_demotion(findings, novelty)
    # Nothing demoted: an uncertain / unavailable check never removes a discovery.
    assert out["counts"]["confirmed"] == 2
    assert out["counts"]["rediscovery"] == 0
    assert out["rediscovery"] == []


@pytest.mark.asyncio
async def test_demote_known_findings_wires_novelty_check() -> None:
    """The paper wiring calls check_finding_novelty per confirmed finding & demotes."""
    from services.orchestrator import paper as paper_mod

    findings = _findings(
        [
            {"id": "a", "text": "known result", "key_finding": "known result", "confidence": 0.9, "stats": {}},
            {"id": "b", "text": "novel result", "key_finding": "novel result", "confidence": 0.8, "stats": {}},
        ]
    )
    emitter = MagicMock()
    emitter.emit = AsyncMock()

    async def fake_novelty(claim, evidence, *, domain_plugin, session_id, emitter):
        if "known" in claim:
            return {"verdict": "known", "confidence": 0.95, "matching_sources": []}
        return {"verdict": "novel", "confidence": 0.9, "matching_sources": []}

    with (
        patch.object(paper_mod, "logger"),
        patch("propab.config.settings") as cfg,
        patch(
            "services.orchestrator.literature_client.check_finding_novelty",
            side_effect=fake_novelty,
        ),
        patch("propab.domain_modules.registry.resolve_domain_plugin", return_value=None),
    ):
        cfg.literature_service_url = "http://literature:8000"
        await paper_mod._demote_known_findings(
            findings,
            session_id=str(uuid.uuid4()),
            emitter=emitter,
            question="cap-set problem",
            synthesis={"domain_id": "math_combinatorics"},
        )

    assert [f["id"] for f in findings["confirmed"]] == ["b"]
    assert findings["counts"]["confirmed"] == 1
    assert [f["id"] for f in findings["rediscovery"]] == ["a"]
    assert findings["counts"]["rediscovery"] == 1


@pytest.mark.asyncio
async def test_demote_known_findings_skipped_when_url_empty() -> None:
    """literature_service_url="" → no novelty check, findings unchanged, no crash."""
    from services.orchestrator import paper as paper_mod

    findings = _findings(
        [{"id": "a", "text": "some finding", "key_finding": "some finding", "confidence": 0.9, "stats": {}}]
    )
    emitter = MagicMock()
    emitter.emit = AsyncMock()

    with (
        patch("propab.config.settings") as cfg,
        patch(
            "services.orchestrator.literature_client.check_finding_novelty",
            new_callable=AsyncMock,
        ) as chk,
    ):
        cfg.literature_service_url = ""
        await paper_mod._demote_known_findings(
            findings,
            session_id=str(uuid.uuid4()),
            emitter=emitter,
            question="q",
            synthesis={},
        )

    chk.assert_not_awaited()
    assert findings["counts"]["confirmed"] == 1
    assert "rediscovery" not in findings or findings.get("rediscovery") in (None, [])
