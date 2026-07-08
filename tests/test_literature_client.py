"""Mocked tests for build_prior_via_service — no live literature service required.

Covers: (1) PriorResponse → Prior field mapping (contradictions land in
contested_claims, tabulated_values preserved, citation_verification_rate surfaced),
(2) honest fallback to the OLD build_prior on service error with a recorded
``literature_service_fallback`` diagnostic.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from services.orchestrator.intake import ParsedQuestion
from services.orchestrator.literature_client import (
    build_prior_via_service,
    check_finding_novelty,
    map_prior_response,
)
from services.orchestrator.schemas import Prior


# A populated PriorResponse payload (as the literature service /prior returns).
PRIOR_RESPONSE = {
    "established_facts": [
        {
            "text": "The chromatic number is bounded by max degree + 1.",
            "claim_type": "theorem",
            "status": "proven",
            "verbatim": "chi(G) <= Delta(G) + 1",
            "source": "arxiv",
            "source_doi": "10.1000/abc",
            "source_title": "Graph Coloring Bounds",
            "source_year": 2019,
            "source_url": "https://arxiv.org/abs/1901.00001",
        }
    ],
    "open_gaps": [
        {
            "description": "Tightness of the bound for sparse graphs is open.",
            "what_is_known": "Holds for dense graphs.",
            "what_is_open": "Sparse regime.",
            "best_known_bound": "Delta+1",
            "computationally_approachable": True,
            "approachable_angle": "enumerate small sparse graphs",
        }
    ],
    "contradictions": [
        {
            "claim_a": {
                "text": "Bound is tight.",
                "claim_type": "claim",
                "status": "conjectured",
                "verbatim": "tight",
                "source_doi": "10.1000/aaa",
                "source_url": "https://arxiv.org/abs/1901.00002",
            },
            "claim_b": {
                "text": "Bound is not tight for sparse graphs.",
                "claim_type": "claim",
                "status": "proven",
                "verbatim": "not tight",
                "source_doi": "10.1000/bbb",
                "source_url": "https://arxiv.org/abs/1901.00003",
            },
            "contradiction_type": "direct",
            "resolution": None,
            "requires_investigation": True,
        }
    ],
    "dead_ends": [
        {
            "text": "Greedy coloring alone does not achieve the bound.",
            "claim_type": "observation",
            "status": "disproven",
            "verbatim": "greedy fails",
            "source_doi": "10.1000/ccc",
            "source_title": "Greedy Limits",
        }
    ],
    "tabulated_values": [
        {
            "description": "chromatic numbers of small graphs",
            "index_variable": "n",
            "value_variable": "chi",
            "values": {"1": 1, "2": 2, "3": 3},
            "source_doi": "10.1000/abc",
        }
    ],
    "novelty_bar": {
        "criteria": "beats Delta+1",
        "tabulated_ceiling": {"n": 3},
        "established_bounds": ["Delta+1"],
    },
    "sources_consulted": ["10.1000/abc", "10.1000/aaa", "10.1000/bbb"],
    "papers_indexed": 5,
    "citation_verification_rate": 0.8,
}


def _parsed(q: str = "What bounds the chromatic number of a graph?") -> ParsedQuestion:
    return ParsedQuestion(text=q, domain="graph_invariants", sub_questions=[q])


@pytest.fixture
def emitter() -> MagicMock:
    em = MagicMock()
    em.emit = AsyncMock()
    return em


@pytest.fixture
def llm() -> MagicMock:
    return MagicMock()


@pytest.fixture
def domain_plugin() -> MagicMock:
    plugin = MagicMock()
    plugin.domain_id = "graph_invariants"
    plugin.literature_profile.return_value = {"search_terms": ["chromatic number"]}
    return plugin


def test_map_prior_response_field_mapping() -> None:
    prior = map_prior_response(PRIOR_RESPONSE)

    assert isinstance(prior, Prior)
    # established_facts → established_facts
    assert len(prior.established_facts) == 1
    assert "chromatic number" in prior.established_facts[0]["text"]
    # contradictions → contested_claims (NOT open_gaps / not dropped)
    assert len(prior.contested_claims) == 1
    assert "CONTESTED" in prior.contested_claims[0]["text"]
    assert prior.open_gaps and "sparse" in prior.open_gaps[0]["text"]
    # dead_ends → dead_ends
    assert len(prior.dead_ends) == 1
    # sources_consulted → key_papers
    assert {kp["paper_id"] for kp in prior.key_papers} == {
        "10.1000/abc",
        "10.1000/aaa",
        "10.1000/bbb",
    }
    # citation_verification_rate → evidence_coverage
    assert prior.evidence_coverage == pytest.approx(0.8)
    # tabulated_values preserved (numerical seeds not dropped)
    assert prior.retrieval_diagnostics is not None
    tv = prior.retrieval_diagnostics["tabulated_values"]
    assert tv and tv[0]["values"] == {"1": 1, "2": 2, "3": 3}
    assert prior.retrieval_diagnostics["source"] == "literature_service"
    assert prior.evidence_status == "READY"


def test_map_prior_response_empty_is_insufficient() -> None:
    prior = map_prior_response(
        {
            "established_facts": [],
            "open_gaps": [],
            "contradictions": [],
            "dead_ends": [],
            "tabulated_values": [],
            "novelty_bar": {"criteria": ""},
            "sources_consulted": [],
            "papers_indexed": 0,
            "citation_verification_rate": None,
        }
    )
    assert prior.evidence_status == "INSUFFICIENT_EVIDENCE"
    assert prior.evidence_coverage == 0.0


def test_map_prior_response_consulted_only_is_insufficient() -> None:
    """Sources were searched but NOTHING was extracted → INSUFFICIENT_EVIDENCE.

    Regression: ``key_papers`` is derived purely from ``sources_consulted``
    (source names/ids that were merely searched, not evidence). A prior with
    zero facts/gaps/contradictions/dead_ends/tabulations must NOT be labelled
    READY just because some sources were consulted — that would feed an empty
    prior to generation as if it were real evidence.
    """
    prior = map_prior_response(
        {
            "established_facts": [],
            "open_gaps": [],
            "contradictions": [],
            "dead_ends": [],
            "tabulated_values": [],
            "novelty_bar": {"criteria": ""},
            "sources_consulted": ["arxiv", "pubmed", "semantic_scholar"],
            "papers_indexed": 7,
            "citation_verification_rate": None,
        }
    )
    # key_papers is non-empty (one per consulted source) but carries no evidence.
    assert prior.key_papers  # still surfaced for provenance/UI
    assert prior.evidence_status == "INSUFFICIENT_EVIDENCE"


def test_map_prior_response_tabulations_only_is_ready() -> None:
    """Numerical seeds are real evidence → READY even with no prose facts."""
    prior = map_prior_response(
        {
            "established_facts": [],
            "open_gaps": [],
            "contradictions": [],
            "dead_ends": [],
            "tabulated_values": [
                {"description": "seq", "values": {"1": 1, "2": 2}, "source_doi": "10.1/x"}
            ],
            "novelty_bar": {"criteria": ""},
            "sources_consulted": ["oeis"],
            "papers_indexed": 1,
            "citation_verification_rate": None,
        }
    )
    assert prior.evidence_status == "READY"


@pytest.mark.asyncio
async def test_build_prior_via_service_maps_response(emitter, llm, domain_plugin) -> None:
    """Mocked /prior returns a populated PriorResponse → correctly mapped Prior."""
    with patch(
        "services.orchestrator.literature_client._post_prior",
        new_callable=AsyncMock,
        return_value=PRIOR_RESPONSE,
    ):
        prior = await build_prior_via_service(
            _parsed(),
            session_id=str(uuid.uuid4()),
            emitter=emitter,
            session_factory=MagicMock(),
            llm=llm,
            domain_plugin=domain_plugin,
        )

    domain_plugin.literature_profile.assert_called_once()
    assert prior.contested_claims and "CONTESTED" in prior.contested_claims[0]["text"]
    assert prior.evidence_coverage == pytest.approx(0.8)
    assert prior.retrieval_diagnostics["tabulated_values"][0]["values"]["3"] == 3
    assert "literature_service_fallback" not in (prior.retrieval_diagnostics or {})
    # LIT_PRIOR_BUILT emitted on success.
    steps = [c.kwargs.get("step") for c in emitter.emit.await_args_list]
    assert "literature.prior_build" in steps


@pytest.mark.asyncio
async def test_build_prior_via_service_falls_back_on_error(emitter, llm, domain_plugin) -> None:
    """Mocked service error → falls back to OLD build_prior and records the reason."""
    fallback_prior = Prior(
        established_facts=[{"text": "fallback fact", "confidence": 0.5, "paper_ids": []}],
        contested_claims=[],
        open_gaps=[],
        dead_ends=[],
        key_papers=[],
        evidence_status="READY",
        retrieval_diagnostics={"source": "embedded_build_prior"},
    )

    with (
        patch(
            "services.orchestrator.literature_client._post_prior",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("service unreachable"),
        ),
        patch(
            "services.orchestrator.literature_client.build_prior",
            new_callable=AsyncMock,
            return_value=fallback_prior,
        ) as old_build,
    ):
        prior = await build_prior_via_service(
            _parsed(),
            session_id=str(uuid.uuid4()),
            emitter=emitter,
            session_factory=MagicMock(),
            llm=llm,
            domain_plugin=domain_plugin,
        )

    old_build.assert_awaited_once()
    assert prior.established_facts[0]["text"] == "fallback fact"
    # Fallback recorded, never silent.
    assert prior.retrieval_diagnostics is not None
    reason = prior.retrieval_diagnostics.get("literature_service_fallback")
    assert reason and "ConnectError" in reason


@pytest.mark.asyncio
async def test_campaign_prior_backward_compatible_uses_old_path(emitter, llm) -> None:
    """literature_service_url="" → campaign uses OLD build_prior unchanged."""
    from services.orchestrator import campaign_loop

    old_prior = Prior(
        established_facts=[],
        contested_claims=[],
        open_gaps=[],
        dead_ends=[],
        key_papers=[],
        evidence_status="READY",
    )
    campaign = MagicMock()
    campaign.id = str(uuid.uuid4())
    campaign.question = "graph question"
    campaign.domain_profile = None

    with (
        patch.object(campaign_loop.settings, "literature_service_url", ""),
        patch.object(
            campaign_loop, "build_prior", new_callable=AsyncMock, return_value=old_prior
        ) as old_build,
        patch.object(
            campaign_loop, "build_prior_via_service", new_callable=AsyncMock
        ) as new_build,
    ):
        prior = await campaign_loop._build_campaign_prior(
            _parsed(),
            campaign=campaign,
            emitter=emitter,
            session_factory=MagicMock(),
            llm=llm,
        )

    old_build.assert_awaited_once()
    new_build.assert_not_awaited()
    assert prior is old_prior


@pytest.mark.asyncio
async def test_campaign_prior_uses_service_when_url_set(emitter, llm) -> None:
    """literature_service_url set → campaign routes through build_prior_via_service."""
    from services.orchestrator import campaign_loop

    svc_prior = Prior(
        established_facts=[],
        contested_claims=[],
        open_gaps=[],
        dead_ends=[],
        key_papers=[],
        evidence_status="READY",
    )
    campaign = MagicMock()
    campaign.id = str(uuid.uuid4())
    campaign.question = "graph question"
    campaign.domain_profile = None

    with (
        patch.object(campaign_loop.settings, "literature_service_url", "http://literature:8000"),
        patch.object(campaign_loop, "build_prior", new_callable=AsyncMock) as old_build,
        patch.object(
            campaign_loop, "build_prior_via_service", new_callable=AsyncMock, return_value=svc_prior
        ) as new_build,
        patch(
            "propab.domain_modules.registry.resolve_domain_plugin", return_value=None
        ),
    ):
        prior = await campaign_loop._build_campaign_prior(
            _parsed(),
            campaign=campaign,
            emitter=emitter,
            session_factory=MagicMock(),
            llm=llm,
        )

    new_build.assert_awaited_once()
    old_build.assert_not_awaited()
    assert prior is svc_prior


# ── DISC1: novelty client ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_finding_novelty_known(emitter, domain_plugin) -> None:
    """A {"verdict":"known"} NoveltyResponse is returned verbatim for demotion."""
    from services.orchestrator import literature_client as lc

    resp = {
        "verdict": "known",
        "confidence": 0.93,
        "explanation": "matches OEIS A003023",
        "matching_sources": [{"source_doi": "10.1/x", "source_title": "Known Values"}],
        "recommendation": "cite existing result",
    }
    with (
        patch.object(lc.settings, "literature_service_url", "http://literature:8000"),
        patch.object(lc, "_post_novelty", new_callable=AsyncMock, return_value=resp) as post,
    ):
        out = await check_finding_novelty(
            "F(6) = 6 for the cap-set problem",
            {"index": 6, "value": 6},
            domain_plugin=domain_plugin,
            session_id=str(uuid.uuid4()),
            emitter=emitter,
        )

    assert out["verdict"] == "known"
    assert out["confidence"] == pytest.approx(0.93)
    # request body carried the finding + the plugin's literature_profile.
    body = post.await_args.args[0]
    assert body["finding"]["claim"].startswith("F(6)")
    assert body["finding"]["evidence"] == {"index": 6, "value": 6}
    assert body["finding"]["domain_id"] == "graph_invariants"
    assert body["literature_profile"] == {"search_terms": ["chromatic number"]}


@pytest.mark.asyncio
async def test_check_finding_novelty_novel_not_demoted(emitter, domain_plugin) -> None:
    """A {"verdict":"novel"} response is passed through unchanged (kept a discovery)."""
    from services.orchestrator import literature_client as lc

    resp = {"verdict": "novel", "confidence": 0.9, "explanation": "", "matching_sources": []}
    with (
        patch.object(lc.settings, "literature_service_url", "http://literature:8000"),
        patch.object(lc, "_post_novelty", new_callable=AsyncMock, return_value=resp),
    ):
        out = await check_finding_novelty(
            "A brand new bound",
            {},
            domain_plugin=domain_plugin,
            session_id=str(uuid.uuid4()),
            emitter=emitter,
        )
    assert out["verdict"] == "novel"
    assert out.get("source") != "novelty_unavailable"


@pytest.mark.asyncio
async def test_check_finding_novelty_service_error_is_uncertain(emitter, domain_plugin) -> None:
    """Service error → safe default uncertain/novelty_unavailable, never raises."""
    from services.orchestrator import literature_client as lc

    with (
        patch.object(lc.settings, "literature_service_url", "http://literature:8000"),
        patch.object(
            lc, "_post_novelty", new_callable=AsyncMock,
            side_effect=httpx.ConnectError("service unreachable"),
        ),
    ):
        out = await check_finding_novelty(
            "some finding",
            {},
            domain_plugin=domain_plugin,
            session_id=str(uuid.uuid4()),
            emitter=emitter,
        )
    assert out == {"verdict": "uncertain", "source": "novelty_unavailable"}


@pytest.mark.asyncio
async def test_check_finding_novelty_empty_url_skips(emitter, domain_plugin) -> None:
    """literature_service_url="" → check skipped (no HTTP), backward compatible."""
    from services.orchestrator import literature_client as lc

    with (
        patch.object(lc.settings, "literature_service_url", ""),
        patch.object(lc, "_post_novelty", new_callable=AsyncMock) as post,
    ):
        out = await check_finding_novelty(
            "some finding",
            {},
            domain_plugin=domain_plugin,
            session_id=str(uuid.uuid4()),
            emitter=emitter,
        )
    post.assert_not_awaited()
    assert out == {"verdict": "uncertain", "source": "novelty_unavailable"}


@pytest.mark.asyncio
async def test_build_prior_via_service_none_plugin_uses_generic(emitter, llm) -> None:
    """domain_plugin=None → empty profile / generic id, no crash."""
    captured: dict = {}

    async def fake_post(body):
        captured.update(body)
        return PRIOR_RESPONSE

    with patch(
        "services.orchestrator.literature_client._post_prior",
        side_effect=fake_post,
    ):
        prior = await build_prior_via_service(
            _parsed(),
            session_id=str(uuid.uuid4()),
            emitter=emitter,
            session_factory=MagicMock(),
            llm=llm,
            domain_plugin=None,
        )

    assert captured["domain_id"] == ""
    assert captured["literature_profile"] == {}
    assert isinstance(prior, Prior)
