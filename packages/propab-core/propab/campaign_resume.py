"""Resume helpers — backfill belief state and validate warm checkpoints."""
from __future__ import annotations

import json
from typing import Any

from propab.belief_state import BeliefObject, CampaignBeliefState, ClosedBelief


CONTRARIAN_QUESTION = (
    "Is RT activity, as measured across these 7 evolutionary families, one shared biophysical "
    "mechanism currently confounded by family-correlated nuisance variables — or are there "
    "genuinely distinct, family-specific activity mechanisms, such that no single feature set "
    "could ever predict activity across families, because \"activity\" does not refer to the same "
    "underlying physical process in each family?"
)

CONTRARIAN_BELIEF_FAMILY_SPECIFIC = (
    "RT activity in each family is governed by mechanisms specific to that family's evolutionary "
    "and structural context. Predictive signals exist within families even when sequence redundancy "
    "(nearest-neighbor effects) is controlled via clustered splitting."
)

CONTRARIAN_BELIEF_REDUNDANCY_ARTIFACT = (
    "Observed intra-family predictive signals are artifacts of sequence redundancy; model performance "
    "will collapse (R2 < 0) when the test set is restricted to sequences with <50% identity to the "
    "training set."
)

CONTRARIAN_ORCHESTRATOR_DIRECTIVE = (
    "Primary critical-experiment criterion: choose the next test because its result would move "
    "Belief 1 (family-specific signal under clustered split) and Belief 2 (sequence-redundancy "
    "artifact under low-identity holdout) in opposite directions — not because it refines either belief in isolation. "
    "Prioritize within-family models that discriminate between these rivals over further "
    "cross-family LOFO feature-combination searches, which have already run exhaustively under "
    "the prior framing. Do not silently revert to cross-family LOFO search as a fallback. "
    "Belief 2 must clear the same artifact-verification bar as Belief 1."
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
            state.apply_synthesis_beliefs(raw_beliefs)
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
    """Data-preserving resume: close prior active beliefs, seed two rival beliefs."""
    for belief in list(belief_state.active_beliefs):
        belief_state.abandon_belief(belief, close_prior_reason)

    belief_state.apply_synthesis_beliefs([
        {
            "statement": CONTRARIAN_BELIEF_FAMILY_SPECIFIC,
            "confidence": "weak",
            "status": "active",
            "supporting_nodes": [],
            "contradicting_nodes": [],
        },
        {
            "statement": CONTRARIAN_BELIEF_REDUNDANCY_ARTIFACT,
            "confidence": "weak",
            "status": "active",
            "supporting_nodes": [],
            "contradicting_nodes": [],
        },
    ])
    belief_state.exhaustion_rounds = 0
    belief_state.branch_exhausted = False
    belief_state.rival_exhaustion_mode = True
    belief_state.results_since_last_synthesis = 0
    belief_state.recent_activity_summary = (
        "Contrarian reframing: discriminate unified-mechanism vs family-specific-mechanism rivals."
    )
    directive = (orchestrator_directive or CONTRARIAN_ORCHESTRATOR_DIRECTIVE).strip()
    if directive:
        belief_state.add_human_message(directive)
    return belief_state


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
