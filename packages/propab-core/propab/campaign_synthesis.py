"""Campaign-level synthesis pass — sole generation-conditioning step (fixes.md §3)."""
from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any

from propab.belief_state import CampaignBeliefState, ClosedBelief
from propab.evidence_binding import BindingMetrics
from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.prompt_composer import compose_synthesis_prompt
from propab.research_quality import extract_theme_vector, infer_node_role
from propab.scoped_claim import enrich_entry_with_scope, parse_scope_from_methodology, validate_scoped_claim


def _question_relevance(question: str, snippets: list[str], core: str) -> float:
    try:
        from services.orchestrator.hypothesis_ranking import compute_question_relevance_score_lexical  # noqa: WPS433
        return float(compute_question_relevance_score_lexical(question, snippets, core))
    except ImportError:
        return 1.0

logger = logging.getLogger(__name__)

FRONTIER_DEDUP_SIMILARITY = 0.85


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").lower().strip())


def _dedup_key(text: str) -> str:
    """First-line claim title — stable under scope enrichment."""
    line = next((ln.strip() for ln in (text or "").splitlines() if ln.strip()), "")
    return _norm_text(line or text)


def text_similarity(a: str, b: str) -> float:
    """Normalized text similarity for frontier dedup (fixes.md P2)."""
    full = SequenceMatcher(None, _norm_text(a), _norm_text(b)).ratio()
    core = SequenceMatcher(None, _dedup_key(a), _dedup_key(b)).ratio()
    return max(full, core)


def _is_duplicate_frontier_candidate(
    text: str,
    tree: HypothesisTree,
    belief_state: CampaignBeliefState,
    *,
    same_round_texts: list[str],
    threshold: float = FRONTIER_DEDUP_SIMILARITY,
) -> tuple[bool, str]:
    for existing in tree.nodes.values():
        if text_similarity(text, existing.text) >= threshold:
            return True, f"tree_node:{existing.id}"
    for sibling in same_round_texts:
        if text_similarity(text, sibling) >= threshold:
            return True, "same_round_sibling"
    for belief in belief_state.closed_beliefs:
        if text_similarity(text, belief.statement) >= threshold:
            return True, f"closed_belief:{belief.statement[:40]}"
    return False, ""


def parse_synthesis_response(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {"_parse_error": True, "raw": raw[:2000]}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"_parse_error": True, "raw": raw[:2000]}
    return data if isinstance(data, dict) else {"_parse_error": True}


def _candidate_dicts(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in ("frontier_candidates",):
        for item in parsed.get(key) or []:
            if isinstance(item, dict) and item.get("text"):
                out.append(item)
    expl = parsed.get("exploratory_candidate")
    if isinstance(expl, dict) and expl.get("text"):
        out.append(expl)
    return out


def apply_synthesis_to_frontier(
    tree: HypothesisTree,
    belief_state: CampaignBeliefState,
    parsed: dict[str, Any],
    *,
    question: str,
    generation: int,
    relevance_threshold: float = 0.35,
    prior_snippets: list[str] | None = None,
) -> tuple[list[HypothesisNode], dict[str, Any]]:
    """
    Apply synthesis JSON to belief state and tree frontier.
    Returns (added_nodes, metrics).
    """
    metrics: dict[str, Any] = {
        "n_candidates_raw": len(_candidate_dicts(parsed)),
        "n_added": 0,
        "n_rejected_duplicate": 0,
        "n_rejected_off_topic": 0,
        "binding_rejected_count": 0,
        "binding_accepted_count": 0,
        "falsifiability_rejected_count": 0,
        "belief_cap_rejected_count": 0,
        "direction_exhausted": bool(parsed.get("direction_exhausted")),
    }

    binding_metrics = BindingMetrics()

    if parsed.get("_parse_error"):
        metrics["parse_error"] = True
        return [], metrics

    # Beliefs — Evidence Binding + falsifiability + rival cap at write time
    node_dicts = {
        nid: (n.to_dict() if hasattr(n, "to_dict") else n)
        for nid, n in tree.nodes.items()
    }
    belief_state.apply_synthesis_beliefs(
        parsed.get("beliefs") or [],
        tree_nodes=node_dicts,
        metrics=binding_metrics,
    )
    metrics.update(binding_metrics.to_dict())
    for item in parsed.get("closed_beliefs_append") or []:
        if isinstance(item, dict) and item.get("statement"):
            belief_state.closed_beliefs.append(ClosedBelief(
                statement=str(item["statement"]),
                reason=str(item.get("reason") or "closed by synthesis"),
            ))

    summary = str(parsed.get("recent_activity_summary") or "").strip()
    if summary:
        belief_state.recent_activity_summary = summary

    belief_state.record_synthesis_exhaustion(bool(parsed.get("direction_exhausted")))

    # Frontier candidates — dedup before add_seeds (fixes.md P2)
    from propab.domain_modules.registry import hypothesis_is_on_topic

    seed_dicts: list[dict[str, Any]] = []
    same_round_texts: list[str] = []
    for item in _candidate_dicts(parsed):
        raw_text = str(item.get("text") or "")
        if not hypothesis_is_on_topic(raw_text, question=question):
            metrics["n_rejected_off_topic"] = int(metrics.get("n_rejected_off_topic") or 0) + 1
            logger.debug("Rejected off-topic frontier candidate: %s", raw_text[:80])
            continue
        dup, dup_reason = _is_duplicate_frontier_candidate(
            raw_text,
            tree,
            belief_state,
            same_round_texts=same_round_texts,
        )
        if dup:
            metrics["n_rejected_duplicate"] = int(metrics.get("n_rejected_duplicate") or 0) + 1
            logger.debug("Rejected duplicate frontier candidate (%s): %s", dup_reason, raw_text[:80])
            continue
        entry = enrich_entry_with_scope(
            {
                "id": str(item.get("id") or f"syn_{len(seed_dicts)}"),
                "text": raw_text,
                "test_methodology": str(item.get("test_methodology") or "sub_agent"),
            },
            question,
        )
        scope = parse_scope_from_methodology(entry["text"], entry["test_methodology"])
        ok, _ = validate_scoped_claim(scope)
        if not ok:
            continue
        same_round_texts.append(raw_text)
        seed_dicts.append({
            "id": entry["id"],
            "text": entry["text"],
            "test_methodology": entry["test_methodology"],
            "claim_scope": entry.get("claim_scope"),
            "expansion_type": item.get("expansion_type") or "diagnostic",
            "expansion_reason": item.get("why_follows_from_beliefs") or "campaign_synthesis",
        })

    if not seed_dicts:
        return [], metrics

    added = tree.add_seeds(seed_dicts, generation=generation)
    snippets = list(prior_snippets or [])
    filtered: list[HypothesisNode] = []
    for node in added:
        primary, secondary, theme_conf = extract_theme_vector(node.text)
        node.primary_theme = primary
        node.secondary_themes = secondary
        node.theme_id = primary
        node.theme_confidence = theme_conf
        node.node_role = infer_node_role(node.text)
        try:
            from services.orchestrator.hypothesis_ranking import strip_question_suffix  # noqa: WPS433
        except ImportError:
            strip_question_suffix = lambda t: t  # type: ignore[misc, assignment]

        core = strip_question_suffix(node.text)
        node.question_relevance_score = _question_relevance(question, snippets, core)
        if (node.question_relevance_score or 0) >= relevance_threshold:
            filtered.append(node)

    if filtered:
        tree.add_to_frontier(filtered)
    metrics["n_added"] = len(filtered)
    metrics["critical_experiment"] = parsed.get("critical_experiment")
    return filtered, metrics


async def run_campaign_synthesis_pass(
    *,
    campaign_id: str,
    question: str,
    tree: HypothesisTree,
    belief_state: CampaignBeliefState,
    llm: Any,
    generation: int,
    prior_snippets: list[str] | None = None,
    emitter: Any | None = None,
    session_factory: Any | None = None,
) -> tuple[list[HypothesisNode], dict[str, Any]]:
    """Tier-2 batched synthesis: one LLM call per trigger."""
    completed_ids = {
        nid for nid, n in tree.nodes.items() if n.verdict in ("confirmed", "refuted", "inconclusive")
    }
    new_since = completed_ids - set(belief_state.last_synthesis_node_ids)

    prompt = compose_synthesis_prompt(
        question=question,
        belief_state=belief_state,
        tree=tree,
        since_node_ids=new_since if new_since else None,
    )

    raw = await llm.call(
        prompt=prompt,
        purpose="campaign.synthesis",
        session_id=campaign_id,
    )
    parsed = parse_synthesis_response(raw)
    added, metrics = apply_synthesis_to_frontier(
        tree,
        belief_state,
        parsed,
        question=question,
        generation=generation,
        prior_snippets=prior_snippets,
    )

    belief_state.results_since_last_synthesis = 0
    belief_state.last_synthesis_node_ids = list(completed_ids)

    # Ownership-contracts observability: persist this round's health metrics
    # (duplicate rate, evidence-binding rejection rate, citation integrity,
    # belief stability). Never blocks synthesis if logging fails.
    if session_factory is not None:
        from propab.health_metrics import log_synthesis_health

        await log_synthesis_health(
            session_factory,
            campaign_id=campaign_id,
            generation=generation,
            metrics=metrics,
            active_belief_statements=[b.statement for b in belief_state.active_beliefs],
        )

    if emitter is not None:
        from propab.types import EventType

        await emitter.emit(
            session_id=campaign_id,
            event_type=EventType.CAMPAIGN_PROGRESS,
            step="campaign.synthesis",
            payload={
                "n_candidates_added": metrics.get("n_added", 0),
                "n_rejected_duplicate": metrics.get("n_rejected_duplicate", 0),
                "n_candidates_raw": metrics.get("n_candidates_raw", 0),
                "binding_rejected_count": metrics.get("binding_rejected_count", 0),
                "binding_accepted_count": metrics.get("binding_accepted_count", 0),
                "falsifiability_rejected_count": metrics.get("falsifiability_rejected_count", 0),
                "belief_cap_rejected_count": metrics.get("belief_cap_rejected_count", 0),
                "active_beliefs": [b.to_dict() for b in belief_state.active_beliefs],
                "branch_exhausted": belief_state.branch_exhausted,
                "exhaustion_rounds": belief_state.exhaustion_rounds,
                "rival_exhaustion_mode": belief_state.rival_exhaustion_mode,
                "belief_exhaustion_rounds": [
                    {"statement": b.statement[:120], "exhaustion_rounds": b.exhaustion_rounds}
                    for b in belief_state.active_beliefs[:3]
                ],
                "critical_experiment": metrics.get("critical_experiment"),
                "direction_exhausted": metrics.get("direction_exhausted"),
            },
        )

    return added, metrics


def should_trigger_synthesis(
    belief_state: CampaignBeliefState,
    *,
    results_since: int,
    max_concurrent: int,
    queued_candidates: int,
    threshold_multiplier: float = 1.0,
) -> bool:
    """Tier-2 trigger: batch accumulated OR frontier fully dry (not merely below cap)."""
    if belief_state.branch_exhausted:
        return False
    batch_threshold = max(1, int(max_concurrent * threshold_multiplier))
    if results_since >= batch_threshold:
        return True
    # Look-ahead: only when nothing dispatchable remains and we have partial batch evidence.
    if queued_candidates == 0 and results_since >= max(1, batch_threshold // 2):
        return True
    return False
