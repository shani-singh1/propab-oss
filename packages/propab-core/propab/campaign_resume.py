"""Resume helpers — backfill belief state and validate warm checkpoints."""
from __future__ import annotations

import json
from typing import Any

from propab.belief_state import CampaignBeliefState

# Re-export for API routes and tests (canonical definitions live on MandrakePlugin).
from propab.domain_modules.mandrake.plugin import (  # noqa: F401
    CONTRARIAN_BELIEF_FAMILY_SPECIFIC,
    CONTRARIAN_BELIEF_REDUNDANCY_ARTIFACT,
    CONTRARIAN_ORCHESTRATOR_DIRECTIVE,
    CONTRARIAN_QUESTION,
)


def _payload(ev: dict[str, Any]) -> dict[str, Any]:
    raw = ev.get("payload_json") or ev.get("payload") or {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def belief_state_from_synthesis_events(events: list[dict[str, Any]]) -> CampaignBeliefState | None:
    """Reconstruct belief state from the last campaign.synthesis event (DB backfill)."""
    synth_events = [e for e in events if (e.get("step") or "") == "campaign.synthesis"]
    if not synth_events:
        return None

    state = CampaignBeliefState()
    for ev in synth_events:
        p = _payload(ev)
        raw_beliefs = p.get("active_beliefs") or []
        if raw_beliefs:
            state.apply_synthesis_beliefs(raw_beliefs, allow_ungrounded=True)
        if p.get("branch_exhausted") is not None:
            state.branch_exhausted = bool(p.get("branch_exhausted"))
        if p.get("exhaustion_rounds") is not None:
            state.exhaustion_rounds = int(p.get("exhaustion_rounds") or 0)
        crit = p.get("critical_experiment")
        if isinstance(crit, dict) and crit.get("title"):
            state.recent_activity_summary = str(crit.get("title") or "")

    if not state.active_beliefs and not state.closed_beliefs:
        return None
    return state


def merge_completed_node_ids(tree_nodes: dict[str, Any]) -> list[str]:
    return [
        nid
        for nid, n in tree_nodes.items()
        if isinstance(n, dict) and n.get("verdict") in ("confirmed", "refuted", "inconclusive")
    ]


def backfill_belief_state_if_empty(
    belief_state: CampaignBeliefState,
    *,
    events: list[dict[str, Any]],
    tree_nodes: dict[str, Any],
) -> tuple[CampaignBeliefState, bool]:
    """Return (state, changed). Restores beliefs from events when DB meta was missing."""
    if belief_state.active_beliefs or belief_state.closed_beliefs:
        return belief_state, False
    restored = belief_state_from_synthesis_events(events)
    if restored is None:
        return belief_state, False
    restored.last_synthesis_node_ids = merge_completed_node_ids(tree_nodes)
    restored.results_since_last_synthesis = 0
    return restored, True


def apply_contrarian_belief_reset(
    belief_state: CampaignBeliefState,
    *,
    orchestrator_directive: str | None = None,
    close_prior_reason: str = "superseded by contrarian reframing (fixes.md)",
) -> CampaignBeliefState:
    """Delegate to MandrakePlugin — contrarian seeds are domain-owned."""
    from propab.domain_modules.registry import get_domain_plugin

    plugin = get_domain_plugin("mandrake")
    if plugin is None:
        raise RuntimeError("mandrake plugin not registered")
    return plugin.apply_contrarian_belief_reset(
        belief_state,
        orchestrator_directive=orchestrator_directive,
        close_prior_reason=close_prior_reason,
    )


def validate_resume_readiness(
    campaign: dict[str, Any],
    *,
    events: list[dict[str, Any]],
    launch_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pre-resume checklist (fixes.md): beliefs, cap, stop reason, session sync."""
    tree = campaign.get("hypothesis_tree") or {}
    nodes = tree.get("nodes") or {}
    bs = campaign.get("belief_state") or {}
    active = bs.get("active_beliefs") or []

    restored, backfilled = backfill_belief_state_if_empty(
        CampaignBeliefState.from_dict(bs),
        events=events,
        tree_nodes=nodes,
    )
    synth_n = sum(1 for e in events if (e.get("step") or "") == "campaign.synthesis")
    last_synth = None
    for e in reversed(events):
        if (e.get("step") or "") == "campaign.synthesis":
            last_synth = _payload(e)
            break

    issues: list[str] = []
    if synth_n > 0 and not active and not restored.active_beliefs:
        issues.append("belief_state_empty_despite_synthesis_events")
    if campaign.get("max_hypotheses_cap") is None and launch_meta and launch_meta.get("max_hypotheses"):
        issues.append("max_hypotheses_cap_not_persisted")
    if campaign.get("status") == "active" and any(
        (e.get("step") or "") in ("campaign.complete_salvaged", "campaign.complete")
        for e in events
    ):
        issues.append("campaign_status_stale_vs_events")

    return {
        "campaign_id": campaign.get("id"),
        "status": campaign.get("status"),
        "stop_reason": campaign.get("stop_reason"),
        "tree_nodes": len(nodes),
        "total_hypotheses": campaign.get("total_hypotheses"),
        "synthesis_events": synth_n,
        "belief_active_count": len(restored.active_beliefs),
        "belief_backfill_available": backfilled,
        "beliefs_preview": [b.statement[:80] for b in restored.active_beliefs[:3]],
        "last_critical_experiment": (last_synth or {}).get("critical_experiment"),
        "max_hypotheses_cap": campaign.get("max_hypotheses_cap"),
        "launch_max_hypotheses": (launch_meta or {}).get("max_hypotheses"),
        "resume_ready": len(issues) == 0 or (len(issues) == 1 and issues[0] == "max_hypotheses_cap_not_persisted"),
        "issues": issues,
    }
