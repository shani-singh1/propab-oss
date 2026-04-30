from __future__ import annotations

"""
Cross-agent peer finding broadcast.

During a research round, when sub-agent A completes its hypothesis, the orchestrator
broadcasts A's partial result to all still-running sub-agents B, C, D via Redis.
Each agent polls its personal channel between think-act steps and injects peer
findings into its working context.

Channel naming: propab:peer:{hypothesis_id}
The orchestrator publishes to all channel IDs it has running; each sub-agent
subscribes only to its own hypothesis_id channel.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_CHANNEL_PREFIX = "propab:peer:"


def _channel(hypothesis_id: str) -> str:
    return f"{_CHANNEL_PREFIX}{hypothesis_id}"


async def publish_peer_finding(redis: Any, *, target_hypothesis_ids: list[str], finding: dict) -> int:
    """
    Publish a peer finding to all target agent channels.
    Returns the number of channels successfully published to.
    """
    payload = json.dumps(finding, ensure_ascii=False)
    published = 0
    for hid in target_hypothesis_ids:
        try:
            await redis.rpush(_channel(hid), payload)
            published += 1
        except Exception as exc:
            logger.warning("Failed to publish peer finding to %s: %s", hid, exc)
    return published


async def poll_peer_findings(redis: Any, *, hypothesis_id: str, max_findings: int = 10) -> list[dict]:
    """
    Non-blocking poll: drain the peer channel for this agent, up to max_findings.
    Returns a list of finding dicts (may be empty).
    """
    channel = _channel(hypothesis_id)
    findings: list[dict] = []
    for _ in range(max_findings):
        try:
            raw = await redis.lpop(channel)
        except Exception as exc:
            logger.debug("Peer channel poll error for %s: %s", hypothesis_id, exc)
            break
        if raw is None:
            break
        try:
            finding = json.loads(raw)
            if isinstance(finding, dict):
                findings.append(finding)
        except (json.JSONDecodeError, TypeError):
            pass
    return findings


def build_peer_finding_payload(result: dict) -> dict:
    """
    Build a compact peer finding dict from an ExperimentResult for broadcast.
    Only carries what other agents need — not the full trace.
    """
    return {
        "hypothesis_id": result.get("hypothesis_id"),
        "verdict": result.get("verdict"),
        "confidence": result.get("confidence"),
        "key_finding": (result.get("key_finding") or "")[:300],
        "learned": (result.get("learned") or "")[:300],
        "failure_reason": result.get("failure_reason"),
    }
