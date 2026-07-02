"""HTTP client for Propab campaign API (AstaBench solver backend)."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

# Legacy campaign.status values that mean the run finished.
_TERMINAL_STATUSES = frozenset({"breakthrough", "budget_exhausted", "failed", "completed"})

# Explicit stop_reason enums (fixes.md Step 2 — wait for these, not budget+10m wall).
_TERMINAL_STOP_REASONS = frozenset({
    "BREAKTHROUGH",
    "TIME_BUDGET_EXHAUSTED",
    "HYPOTHESIS_CAP_REACHED",
    "FRONTIER_EXHAUSTED",
    "ALL_BRANCHES_EXHAUSTED",
    "NO_DISPATCHABLE_NODES",
    "NO_BOOTSTRAP_SEEDS",
    "SYNTHESIS_EMPTY",
    "NO_SEEDS_GENERATED",
    "FRONTIER_REFILL_FAILED",
    "SALVAGED_AFTER_ERROR",
})


def _campaign_terminal(campaign: dict[str, Any]) -> bool:
    status = str(campaign.get("status") or "")
    stop = str(campaign.get("stop_reason") or "")
    if status in _TERMINAL_STATUSES:
        return True
    if stop in _TERMINAL_STOP_REASONS:
        return True
    return False


def default_max_wait_sec(budget_hours: float) -> float:
    """Hard safety cap: budget + 25% orchestrator drain + 30 min API grace."""
    return float(budget_hours) * 3600.0 * 1.25 + 1800.0


def launch_campaign(
    *,
    api_base: str,
    question: str,
    budget_hours: float,
    max_hypotheses: int | None = 80,
    timeout_sec: float = 120.0,
) -> str:
    body: dict[str, Any] = {
        "question": question,
        "compute_budget_hours": budget_hours,
        "policy_mode": "accepted",
        "breakthrough_criteria": {
            "metric_name": "discovery_score",
            "improvement_threshold": 0.05,
            "direction": "higher_is_better",
            "min_confidence": 0.85,
            "min_replications": 1,
        },
    }
    if max_hypotheses is not None:
        body["max_hypotheses"] = max_hypotheses

    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/campaigns",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        data = json.loads(resp.read())
    return str(data["campaign_id"])


def fetch_campaign(
    api_base: str,
    campaign_id: str,
    *,
    timeout_sec: float = 120.0,
    retries: int = 5,
) -> dict[str, Any]:
    req = urllib.request.Request(f"{api_base.rstrip('/')}/campaigns/{campaign_id}", method="GET")
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(min(30.0, 2.0 ** attempt))
    raise last_err  # type: ignore[misc]


def wait_for_campaign(
    api_base: str,
    campaign_id: str,
    *,
    poll_sec: float = 15.0,
    max_wait_sec: float | None = None,
    fetch_timeout_sec: float = 120.0,
    budget_hours: float | None = None,
) -> dict[str, Any]:
    """
    Poll until the campaign reaches a terminal status or explicit stop_reason.

    Does **not** return early while status is still ``active`` merely because
    the nominal budget elapsed (fixes.md Step 2).
    """
    if max_wait_sec is None and budget_hours is not None:
        max_wait_sec = default_max_wait_sec(budget_hours)
    elif max_wait_sec is None:
        max_wait_sec = 6 * 3600.0

    started = time.monotonic()
    last_payload: dict[str, Any] | None = None

    while True:
        elapsed = time.monotonic() - started
        try:
            payload = fetch_campaign(api_base, campaign_id, timeout_sec=fetch_timeout_sec)
            last_payload = payload
        except (urllib.error.URLError, TimeoutError, OSError):
            if elapsed > max_wait_sec:
                return {
                    "campaign": {"status": "failed", "campaign_id": campaign_id},
                    "_wait_timeout": True,
                    "_poll_error": True,
                }
            time.sleep(poll_sec)
            continue

        campaign = payload.get("campaign") or {}
        if _campaign_terminal(campaign):
            return payload

        if elapsed > max_wait_sec:
            payload["_wait_timeout"] = True
            payload["_non_terminal_at_timeout"] = True
            return payload

        time.sleep(poll_sec)


def health_check(api_base: str, *, timeout_sec: float = 5.0) -> bool:
    try:
        req = urllib.request.Request(f"{api_base.rstrip('/')}/health", method="GET")
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
