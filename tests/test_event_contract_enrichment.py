"""Frontend cockpit event-contract enrichment (BE-1).

Backward-compatible additive fields/events that let the cockpit render real cost,
progress, and discovery data:

- ``llm.response`` carries ``tokens_in`` / ``tokens_out`` / ``duration_ms`` and a
  ``call_id`` that pairs it with its ``llm.prompt`` (no more FIFO guessing);
- worker ``agent.*`` events carry an authoritative ``round``;
- a periodic ``agent.progress`` heartbeat means "running" = "recently alive";
- SSE frames carry an ``id:`` line for ``Last-Event-ID`` replay;
- a first-class ``finding.certified`` event carries the witness + certification
  booleans + metric-vs-best-known.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from propab.llm import LLMClient
from propab.types import EventType
from services.worker.sub_agent_loop import _agent_heartbeat, _round_of


class _FakeEmitter:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def emit(self, **kwargs):
        self.events.append(kwargs)


def _events_of(emitter: _FakeEmitter, event_type: EventType) -> list[dict]:
    return [e for e in emitter.events if e.get("event_type") == event_type]


# ── 1. llm.response tokens / duration / call_id + prompt<->response pairing ────


def test_llm_response_carries_tokens_duration_and_paired_call_id() -> None:
    emitter = _FakeEmitter()
    c = LLMClient.__new__(LLMClient)  # bypass __init__ side effects
    c.provider = "openai"
    c.model = "test-model"
    c.api_key = "k"
    c.emitter = emitter
    c.session_factory = None

    async def fake_once(prompt: str, *, usage_out: dict[str, Any] | None = None) -> str:
        if usage_out is not None:
            usage_out["tokens_in"] = 11
            usage_out["tokens_out"] = 7
        return "the response"

    async def noop_persist(**_kw):
        return None

    c._call_provider_once = fake_once  # type: ignore[assignment]
    c._persist_call = noop_persist  # type: ignore[assignment]

    out = asyncio.run(c.call(prompt="hi", purpose="unit", session_id="s1"))
    assert out == "the response"

    prompts = _events_of(emitter, EventType.LLM_PROMPT)
    responses = _events_of(emitter, EventType.LLM_RESPONSE)
    assert len(prompts) == 1 and len(responses) == 1

    p_payload = prompts[0]["payload"]
    r_payload = responses[0]["payload"]

    # call_id present on both and identical → exact pairing.
    assert p_payload["call_id"]
    assert p_payload["call_id"] == r_payload["call_id"]

    # Response carries cost/latency fields.
    assert r_payload["tokens_in"] == 11
    assert r_payload["tokens_out"] == 7
    assert isinstance(r_payload["duration_ms"], int)
    assert r_payload["duration_ms"] >= 0


def test_llm_response_tokens_none_when_provider_reports_none() -> None:
    emitter = _FakeEmitter()
    c = LLMClient.__new__(LLMClient)
    c.provider = "openai"
    c.model = "m"
    c.api_key = "k"
    c.emitter = emitter
    c.session_factory = None

    async def fake_once(prompt: str, *, usage_out: dict[str, Any] | None = None) -> str:
        return "resp"  # provider reported no usage

    async def noop_persist(**_kw):
        return None

    c._call_provider_once = fake_once  # type: ignore[assignment]
    c._persist_call = noop_persist  # type: ignore[assignment]

    asyncio.run(c.call(prompt="hi", purpose="unit", session_id="s1"))
    r_payload = _events_of(emitter, EventType.LLM_RESPONSE)[0]["payload"]
    assert r_payload["tokens_in"] is None
    assert r_payload["tokens_out"] is None
    assert isinstance(r_payload["duration_ms"], int)


# ── 2. round attribution helper ────────────────────────────────────────────────


def test_round_of_reads_authoritative_int_only() -> None:
    assert _round_of({"round": 3}) == 3
    assert _round_of({"round": 0}) == 0
    assert _round_of({}) is None
    assert _round_of({"round": None}) is None
    assert _round_of({"round": True}) is None  # bool is not a round
    assert _round_of({"round": "2"}) is None


# ── 3. agent.progress heartbeat carries round + liveness ────────────────────────


def test_agent_heartbeat_emits_progress_with_round() -> None:
    emitter = _FakeEmitter()

    async def drive() -> None:
        task = asyncio.create_task(
            _agent_heartbeat(
                emitter,
                session_id="s1",
                hypothesis_id="h1",
                started=0.0,
                round_no=4,
                interval=0.01,
            )
        )
        await asyncio.sleep(0.035)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(drive())

    beats = _events_of(emitter, EventType.AGENT_PROGRESS)
    assert beats, "expected at least one agent.progress heartbeat"
    payload = beats[0]["payload"]
    assert payload["round"] == 4
    assert payload["alive"] is True
    assert payload["hypothesis_id"] == "h1"
    assert payload["heartbeat_seq"] >= 1


# ── 4. SSE id: line for Last-Event-ID replay ────────────────────────────────────


def test_sse_frame_prefixes_event_id_line() -> None:
    from services.api.app.routes.stream import _extract_event_id, _sse_frame

    payload = json.dumps({"event_id": "abc-123", "event_type": "llm.response", "payload": {}})
    frame = _sse_frame(payload)
    assert frame.startswith("id: abc-123\n")
    assert "data: " in frame
    assert frame.endswith("\n\n")
    assert _extract_event_id(payload) == "abc-123"

    # A frame without an event_id still yields a valid data-only frame.
    no_id = _sse_frame(json.dumps({"foo": "bar"}))
    assert no_id.startswith("data: ")
    assert "id:" not in no_id


# ── 5. finding.certified event with witness + certification booleans ────────────


class _FakeDB:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, *a, **k):
        return None

    async def commit(self):
        return None


def _fake_session_factory():
    return _FakeDB()


def _make_certifying_plugin():
    from propab.domain_modules.base import DomainPlugin
    from propab.domain_modules.math_combinatorics.discovery import certify_b3_record

    # A genuine B_3 set in {0,1}^3 that strictly beats a tiny published best, so
    # certify_b3_record returns certified=True with all four checks.
    witness_set = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    cert = certify_b3_record({"n": 3, "set": [list(v) for v in witness_set]}, published_best=3)

    class _CertPlugin(DomainPlugin):
        domain_id = "b3demo"
        display_name = "b3demo"

        def matches(self, *, question: str = "", payload=None) -> bool:
            return False

        def available_features(self) -> list[str]:
            return []

        def hypothesis_on_topic(self, text: str, methodology: str | None = None) -> bool:
            return True

        def run_verification(self, hypothesis, evidence=None, features=None):
            return {
                "metric_value": float(cert["size"]),
                "metric_name": "b3_binary_cube_size",
                "best_known_size": cert["published_best"],
                "vs_best_known": "exceeds_best_known",
                "record_witness_json": {"n": 3, "set": [list(v) for v in witness_set]},
                "certification": cert,
                "discovery_worthy": True,
                "record_status": "open",
                "deterministic": True,
                "verification_method": "combinatorial_computation",
                "verified_true_steps": 1,
            }

        def classify_verdict(self, hypothesis_text, result):
            return "confirmed", "certified record", 0.99

        def confirmation_criteria(self) -> dict[str, Any]:
            return {"min_metric_steps_for_confirm": 1}

        def uses_synthetic_data(self) -> bool:
            return False

    return _CertPlugin(), cert


def test_plugin_path_emits_finding_certified_with_expected_shape() -> None:
    from services.worker.sub_agent_loop import _plugin_verification_path

    plugin, cert = _make_certifying_plugin()
    emitter = _FakeEmitter()

    asyncio.run(
        _plugin_verification_path(
            payload={"round": 5},
            hypothesis={"text": "maximize B_3 set in {0,1}^3", "test_methodology": "m"},
            hypothesis_id="h1",
            campaign_node_id=None,
            session_id="s1",
            question="q",
            session_factory=_fake_session_factory,
            emitter=emitter,
            registry=None,
            trace_pointer="tp",
            started=0.0,
            baseline={},
            domain_plugin=plugin,
        )
    )

    certs = _events_of(emitter, EventType.FINDING_CERTIFIED)
    assert len(certs) == 1, "expected exactly one finding.certified event"
    payload = certs[0]["payload"]

    assert payload["certified"] is True
    assert payload["round"] == 5
    assert payload["domain"] == "b3demo"
    assert payload["metric_name"] == "b3_binary_cube_size"
    assert payload["vs_best_known"] == "exceeds_best_known"
    assert payload["witness"] == {"n": 3, "set": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    # Certification booleans passed through verbatim for the breakthrough card.
    checks = payload["checks"]
    assert set(checks) == {
        "in_binary_cube",
        "distinct_vectors",
        "strictly_beats_published",
        "is_b3",
    }
    assert all(checks.values()) is True


def test_agent_completed_carries_round_in_plugin_path() -> None:
    from services.worker.sub_agent_loop import _plugin_verification_path

    plugin, _ = _make_certifying_plugin()
    emitter = _FakeEmitter()

    asyncio.run(
        _plugin_verification_path(
            payload={"round": 9},
            hypothesis={"text": "maximize B_3 set in {0,1}^3", "test_methodology": "m"},
            hypothesis_id="h1",
            campaign_node_id=None,
            session_id="s1",
            question="q",
            session_factory=_fake_session_factory,
            emitter=emitter,
            registry=None,
            trace_pointer="tp",
            started=0.0,
            baseline={},
            domain_plugin=plugin,
        )
    )

    completed = _events_of(emitter, EventType.AGENT_COMPLETED)
    assert completed, "expected an agent.completed event"
    assert completed[-1]["payload"]["round"] == 9
