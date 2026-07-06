"""Campaign-level synthesis pass — sole generation-conditioning step (fixes.md §3)."""
from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any
from uuid import uuid4

from propab.belief_state import CampaignBeliefState, ClosedBelief
from propab.evidence_binding import BindingMetrics
from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.prompt_composer import compose_synthesis_prompt
from propab.research_quality import extract_theme_vector, infer_node_role
from propab.scoped_claim import (
    compute_scope_delta,
    enrich_entry_with_scope,
    parse_scope_from_methodology,
    validate_scoped_claim,
)


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
    allowed_parent_id: str | None = None,
    threshold: float = FRONTIER_DEDUP_SIMILARITY,
) -> tuple[bool, str]:
    for existing in tree.nodes.values():
        existing_text = existing.text if hasattr(existing, "text") else str(existing.get("text", ""))
        existing_id = existing.id if hasattr(existing, "id") else str(existing.get("id", "?"))
        similarity = text_similarity(text, existing_text)
        if allowed_parent_id and existing_id == allowed_parent_id and similarity < 0.98:
            continue
        if similarity >= threshold:
            return True, f"tree_node:{existing_id}"
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


def _node_child_count(node: HypothesisNode) -> int:
    return len(getattr(node, "children", None) or [])


def _eligible_expansion_parents(tree: HypothesisTree) -> list[HypothesisNode]:
    parents: list[HypothesisNode] = []
    for node in tree.nodes.values():
        if not isinstance(node, HypothesisNode):
            continue
        if node.verdict not in ("confirmed", "refuted", "inconclusive"):
            continue
        if node.id in tree.exhausted:
            continue
        if node.depth >= 8:
            continue
        if _node_child_count(node) >= 5:
            continue
        parents.append(node)
    return parents


def _candidate_parent_ids(item: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    scalar_keys = (
        "parent_id",
        "parent_node_id",
        "expand_parent_id",
        "refinement_of",
        "source_node_id",
        "target_node_id",
    )
    list_keys = (
        "discriminates_node_ids",
        "related_node_ids",
        "based_on_node_ids",
        "supporting_nodes",
        "contradicting_nodes",
        "node_ids",
    )
    for key in scalar_keys:
        val = item.get(key)
        if val:
            ids.append(str(val))
    for key in list_keys:
        val = item.get(key)
        if isinstance(val, list):
            ids.extend(str(x) for x in val if x)
        elif val:
            ids.append(str(val))
    seen: set[str] = set()
    out: list[str] = []
    for node_id in ids:
        if node_id not in seen:
            seen.add(node_id)
            out.append(node_id)
    return out


def _preferred_parent_verdicts(item: dict[str, Any]) -> tuple[str, ...]:
    expansion_type = str(item.get("expansion_type") or "").lower()
    if expansion_type == "retest":
        return ("inconclusive", "refuted", "confirmed")
    if expansion_type == "alternative":
        return ("refuted", "inconclusive", "confirmed")
    if expansion_type in {"boundary", "mechanistic", "generalization"}:
        return ("confirmed", "inconclusive", "refuted")
    if expansion_type == "diagnostic":
        return ("inconclusive", "refuted", "confirmed")
    if item.get("implements_critical_experiment"):
        return ("confirmed", "inconclusive", "refuted")
    return ("confirmed", "refuted", "inconclusive")


def _resolve_synthesis_parent(
    item: dict[str, Any],
    raw_text: str,
    tree: HypothesisTree,
    eligible_parents: list[HypothesisNode],
) -> tuple[HypothesisNode | None, str]:
    if not eligible_parents:
        return None, "no_completed_parent"

    eligible_by_id = {node.id: node for node in eligible_parents}
    for node_id in _candidate_parent_ids(item):
        parent = eligible_by_id.get(node_id)
        if parent is not None:
            return parent, "explicit"

    preferred = _preferred_parent_verdicts(item)
    ranked: list[tuple[float, int, int, int, HypothesisNode]] = []
    for parent in eligible_parents:
        try:
            verdict_rank = preferred.index(parent.verdict)
        except ValueError:
            verdict_rank = len(preferred)
        similarity = text_similarity(raw_text, parent.text)
        ranked.append((
            -float(verdict_rank),
            int(parent.generation or 0),
            int(parent.depth or 0),
            -_node_child_count(parent),
            parent,
        ))
        ranked[-1] = (
            ranked[-1][0] + similarity,
            ranked[-1][1],
            ranked[-1][2],
            ranked[-1][3],
            ranked[-1][4],
        )
    if not ranked:
        return None, "no_eligible_parent"
    ranked.sort(reverse=True, key=lambda x: (x[0], x[1], x[2], x[3]))
    return ranked[0][4], "inferred"


def _child_node_from_synthesis(
    *,
    parent: HypothesisNode,
    entry: dict[str, Any],
    item: dict[str, Any],
    generation: int,
    question_relevance_score: float | None = None,
) -> HypothesisNode:
    primary, secondary, theme_conf = extract_theme_vector(entry["text"])
    parent_scope = parse_scope_from_methodology(parent.text, parent.test_methodology)
    child_scope = parse_scope_from_methodology(entry["text"], entry.get("test_methodology"))
    scope_delta = compute_scope_delta(parent_scope, child_scope)
    return HypothesisNode(
        id=str(uuid4()),
        text=entry["text"],
        parent_id=parent.id,
        depth=parent.depth + 1,
        verdict="pending",
        generation=generation,
        expansion_type=item.get("expansion_type") or "diagnostic",
        expansion_reason=item.get("why_follows_from_beliefs") or "campaign_synthesis",
        node_role=infer_node_role(entry["text"]),
        primary_theme=primary,
        secondary_themes=secondary,
        theme_id=item.get("theme_id") or primary,
        theme_confidence=theme_conf,
        lineage_length=int(parent.lineage_length or (parent.depth + 1)) + 1,
        question_relevance_score=question_relevance_score,
        test_methodology=entry.get("test_methodology"),
        feature_subset=list(item.get("feature_subset") or []),
        mechanism_id=item.get("mechanism_id"),
        claim_scope=entry.get("claim_scope"),
        scope_delta=entry.get("scope_delta") or scope_delta,
    )


def apply_synthesis_to_frontier(
    tree: HypothesisTree,
    belief_state: CampaignBeliefState,
    parsed: dict[str, Any],
    *,
    question: str,
    generation: int,
    relevance_threshold: float = 0.35,
    prior_snippets: list[str] | None = None,
    domain_id: str | None = None,
    synthesis_history_buckets: list[dict[str, str]] | None = None,
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
        "n_rejected_unimplementable": 0,
        "n_rejected_diversity": 0,
        "beliefs_promoted_by_trend": 0,
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

    from propab.domain_modules.registry import get_domain_plugin
    from propab.belief_promotion import apply_trend_promotion_to_beliefs

    plugin = get_domain_plugin(domain_id) if domain_id else None
    if plugin is not None:
        promoted = apply_trend_promotion_to_beliefs(
            belief_state,
            node_dicts,
            plugin.belief_promotion_threshold(),
        )
        metrics["beliefs_promoted_by_trend"] = promoted
    if plugin is not None:
        from propab.belief_promotion import refresh_active_belief_trend_support

        refresh_active_belief_trend_support(
            belief_state,
            node_dicts,
            plugin.belief_promotion_threshold(),
        )
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

    # Frontier candidates: bind to completed parents before inserting roots/children.
    from propab.domain_modules.registry import hypothesis_is_on_topic
    from propab.numerical_seeds import classify_hypothesis_bucket, claim_has_numeric_falsifier
    from propab.synthesis_diversity import (
        diversity_requirement_prompt,
        methodology_implementable,
        resolve_forced_problem_type,
    )

    implementable: list[str] = []
    if plugin is not None:
        implementable = plugin.implementable_methodologies()
    from propab.synthesis_diversity import tree_problem_counts_from_nodes

    tree_counts = tree_problem_counts_from_nodes(node_dicts)
    forced_type = resolve_forced_problem_type(
        synthesis_history_buckets or [],
        [b.statement for b in belief_state.active_beliefs],
        streak=3,
        tree_problem_counts=tree_counts,
    )

    candidate_dicts: list[dict[str, Any]] = []
    same_round_texts: list[str] = []
    eligible_parents = _eligible_expansion_parents(tree)
    critical_raw: list[dict[str, Any]] = []
    other_raw: list[dict[str, Any]] = []
    for item in _candidate_dicts(parsed):
        if item.get("implements_critical_experiment"):
            critical_raw.append(item)
        else:
            other_raw.append(item)
    candidate_items = critical_raw + other_raw

    for item in candidate_items:
        raw_text = str(item.get("text") or "")
        methodology = str(item.get("test_methodology") or "")
        if implementable and not methodology_implementable(raw_text, methodology, implementable):
            metrics["n_rejected_unimplementable"] = int(metrics.get("n_rejected_unimplementable") or 0) + 1
            logger.debug("Rejected unimplementable methodology: %s", raw_text[:80])
            continue
        if domain_id == "math_combinatorics" and not claim_has_numeric_falsifier(raw_text, methodology):
            metrics["n_rejected_unimplementable"] = int(metrics.get("n_rejected_unimplementable") or 0) + 1
            logger.debug("Rejected non-numeric structural claim: %s", raw_text[:80])
            continue
        bucket = classify_hypothesis_bucket(raw_text, methodology)
        problem_type = bucket.get("problem_type")
        if forced_type and problem_type != forced_type:
            metrics["n_rejected_diversity"] = int(metrics.get("n_rejected_diversity") or 0) + 1
            logger.debug(
                "Rejected non-forced problem type %s (need %s): %s",
                problem_type,
                forced_type,
                raw_text[:80],
            )
            continue
        if not hypothesis_is_on_topic(
            raw_text,
            question=question,
            test_methodology=str(item.get("test_methodology") or ""),
        ):
            metrics["n_rejected_off_topic"] = int(metrics.get("n_rejected_off_topic") or 0) + 1
            logger.debug("Rejected off-topic frontier candidate: %s", raw_text[:80])
            continue
        parent, parent_mode = _resolve_synthesis_parent(item, raw_text, tree, eligible_parents)
        dup, dup_reason = _is_duplicate_frontier_candidate(
            raw_text,
            tree,
            belief_state,
            same_round_texts=same_round_texts,
            allowed_parent_id=parent.id if parent else None,
        )
        if dup:
            metrics["n_rejected_duplicate"] = int(metrics.get("n_rejected_duplicate") or 0) + 1
            logger.debug("Rejected duplicate frontier candidate (%s): %s", dup_reason, raw_text[:80])
            continue
        entry = enrich_entry_with_scope(
            {
                "id": str(item.get("id") or f"syn_{len(candidate_dicts)}"),
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
        entry_dict: dict[str, Any] = {
            "id": entry["id"],
            "text": entry["text"],
            "test_methodology": entry["test_methodology"],
            "claim_scope": entry.get("claim_scope"),
            "expansion_type": item.get("expansion_type") or "diagnostic",
            "expansion_reason": item.get("why_follows_from_beliefs") or "campaign_synthesis",
            "_raw_item": item,
        }
        if item.get("implements_critical_experiment"):
            entry_dict["implements_critical_experiment"] = True
        if parent is not None:
            entry_dict["parent_id"] = parent.id
            entry_dict["_parent_mode"] = parent_mode
        elif eligible_parents:
            metrics["n_rejected_missing_parent"] = int(metrics.get("n_rejected_missing_parent") or 0) + 1
            continue
        else:
            entry_dict["_parent_mode"] = parent_mode
        candidate_dicts.append(entry_dict)

    if not candidate_dicts and forced_type:
        from propab.synthesis_diversity import fallback_synthesis_seeds

        fallback_items = fallback_synthesis_seeds(
            forced_type,
            generation=generation,
            question=question,
        )
        metrics["diversity_fallback_injected"] = len(fallback_items)
        for item in fallback_items:
            raw_text = str(item.get("text") or "")
            parent, parent_mode = _resolve_synthesis_parent(item, raw_text, tree, eligible_parents)
            dup, dup_reason = _is_duplicate_frontier_candidate(
                raw_text,
                tree,
                belief_state,
                same_round_texts=same_round_texts,
                allowed_parent_id=parent.id if parent else None,
            )
            if dup:
                metrics["n_rejected_duplicate"] = int(metrics.get("n_rejected_duplicate") or 0) + 1
                logger.debug("Rejected duplicate fallback candidate (%s): %s", dup_reason, raw_text[:80])
                continue
            entry = enrich_entry_with_scope(
                {
                    "id": str(item.get("id") or f"fallback_{forced_type}_{len(candidate_dicts)}"),
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
            entry_dict = {
                "id": entry["id"],
                "text": entry["text"],
                "test_methodology": entry["test_methodology"],
                "claim_scope": entry.get("claim_scope"),
                "expansion_type": item.get("expansion_type") or "diagnostic",
                "expansion_reason": item.get("expansion_reason") or f"diversity_fallback_{forced_type}",
                "_raw_item": item,
            }
            if parent is not None:
                entry_dict["parent_id"] = parent.id
                entry_dict["_parent_mode"] = parent_mode
            elif not eligible_parents:
                entry_dict["_parent_mode"] = parent_mode
            else:
                metrics["n_rejected_missing_parent"] = int(metrics.get("n_rejected_missing_parent") or 0) + 1
                continue
            candidate_dicts.append(entry_dict)

    if not candidate_dicts:
        return [], metrics

    snippets = list(prior_snippets or [])
    root_dicts = [entry for entry in candidate_dicts if not entry.get("parent_id")]
    child_dicts = [entry for entry in candidate_dicts if entry.get("parent_id")]
    added: list[HypothesisNode] = []
    if root_dicts:
        added.extend(tree.add_seeds(root_dicts, generation=generation))
        metrics["n_added_as_roots"] = len(root_dicts)
    child_nodes: list[HypothesisNode] = []
    for entry in child_dicts:
        parent = tree.nodes.get(str(entry.get("parent_id")))
        if not isinstance(parent, HypothesisNode):
            metrics["n_rejected_missing_parent"] = int(metrics.get("n_rejected_missing_parent") or 0) + 1
            continue
        item = entry.get("_raw_item") if isinstance(entry.get("_raw_item"), dict) else entry
        child_nodes.append(_child_node_from_synthesis(
            parent=parent,
            entry=entry,
            item=item,
            generation=generation,
        ))
    added.extend(child_nodes)
    if child_nodes:
        metrics["n_added_as_children"] = len(child_nodes)

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
        if node.parent_id:
            node.lineage_length = tree.lineage_length(node.parent_id) + 1
        if (node.question_relevance_score or 0) >= relevance_threshold:
            filtered.append(node)

    if filtered:
        tree.add_to_frontier(filtered)
    metrics["n_added"] = len(filtered)
    metrics["critical_experiment"] = parsed.get("critical_experiment")

    # Lineage-derivation quality (§3.2): of the candidates that became children,
    # how many named an explicit parent (the LLM genuinely derived the lineage)
    # vs. fell back to similarity-inference. A low rate means the tree has
    # structural depth but arbitrary parent→child edges — depth without real
    # refinement. Surfaced so a campaign whose lineage is mostly inferred is
    # visible, not silent.
    child_modes = [e.get("_parent_mode") for e in child_dicts]
    n_explicit = sum(1 for m in child_modes if m == "explicit")
    n_inferred = sum(1 for m in child_modes if m == "inferred")
    metrics["n_lineage_explicit"] = n_explicit
    metrics["n_lineage_inferred"] = n_inferred
    metrics["lineage_derivation_rate"] = (
        round(n_explicit / (n_explicit + n_inferred), 3) if (n_explicit + n_inferred) else None
    )
    # Convergence health (§3.3): mean confirmed-ancestry depth after this round.
    # Rises when the search deepens real findings; flat ⇒ shallow-breadth failure.
    metrics["confirmed_lineage_depth"] = round(tree.confirmed_lineage_depth(), 3)
    return filtered, metrics


def apply_diversity_fallback_seeds(
    tree: HypothesisTree,
    *,
    forced_type: str,
    generation: int,
    question: str = "",
    prior_snippets: list[str] | None = None,
    relevance_threshold: float = 0.35,
    belief_state: CampaignBeliefState | None = None,
) -> list[HypothesisNode]:
    """Inject deterministic seeds when LLM synthesis ignores diversity reset."""
    from propab.synthesis_diversity import fallback_synthesis_seeds

    seed_dicts = fallback_synthesis_seeds(
        forced_type,
        generation=generation,
        question=question,
    )
    if not seed_dicts:
        return []
    added, _metrics = apply_synthesis_to_frontier(
        tree,
        belief_state or CampaignBeliefState(),
        {"beliefs": [], "frontier_candidates": seed_dicts, "direction_exhausted": False},
        question=question,
        generation=generation,
        # Deterministic fallback seeds are already the escape hatch after an empty
        # synthesis pass; keep the historical behavior that they are not dropped
        # by lexical relevance scoring.
        relevance_threshold=0.0,
        prior_snippets=prior_snippets,
    )
    for node in added:
        node.question_relevance_score = 1.0
    return added


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
    domain_id: str | None = None,
    synthesis_history_buckets: list[dict[str, str]] | None = None,
    diversity_reset_instruction: str | None = None,
    lifetime_context: str = "",
) -> tuple[list[HypothesisNode], dict[str, Any]]:
    """Tier-2 batched synthesis: one LLM call per trigger."""
    completed_ids = {
        nid for nid, n in tree.nodes.items() if n.verdict in ("confirmed", "refuted", "inconclusive")
    }
    new_since = completed_ids - set(belief_state.last_synthesis_node_ids)

    from propab.synthesis_diversity import (
        diversity_requirement_prompt,
        history_problem_counts,
        methodology_implementable,
        resolve_forced_problem_type,
    )

    from propab.synthesis_diversity import tree_problem_counts_from_nodes

    node_dicts = {
        nid: (n.to_dict() if hasattr(n, "to_dict") else n)
        for nid, n in tree.nodes.items()
    }
    tree_counts = tree_problem_counts_from_nodes(node_dicts)
    forced_type = resolve_forced_problem_type(
        synthesis_history_buckets or [],
        [b.statement for b in belief_state.active_beliefs],
        streak=3,
        tree_problem_counts=tree_counts,
    )
    prompt = compose_synthesis_prompt(
        question=question,
        belief_state=belief_state,
        tree=tree,
        since_node_ids=new_since if new_since else None,
        lifetime_context=lifetime_context,
    )
    if forced_type:
        dominant = None
        if synthesis_history_buckets:
            counts = history_problem_counts(synthesis_history_buckets[-5:])
            if counts:
                dominant = counts.most_common(1)[0][0]
        prompt = f"{prompt}\n\n{diversity_requirement_prompt(forced_type, avoid_type=dominant)}"
    if diversity_reset_instruction:
        prompt = f"{prompt}\n\n{diversity_reset_instruction}"

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
        domain_id=domain_id,
        synthesis_history_buckets=synthesis_history_buckets,
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
                "n_rejected_missing_parent": metrics.get("n_rejected_missing_parent", 0),
                "n_candidates_raw": metrics.get("n_candidates_raw", 0),
                "n_added_as_children": metrics.get("n_added_as_children", 0),
                "n_added_as_roots": metrics.get("n_added_as_roots", 0),
                "n_lineage_explicit": metrics.get("n_lineage_explicit", 0),
                "n_lineage_inferred": metrics.get("n_lineage_inferred", 0),
                "lineage_derivation_rate": metrics.get("lineage_derivation_rate"),
                "confirmed_lineage_depth": metrics.get("confirmed_lineage_depth", 0.0),
                "binding_rejected_count": metrics.get("binding_rejected_count", 0),
                "binding_accepted_count": metrics.get("binding_accepted_count", 0),
                "falsifiability_rejected_count": metrics.get("falsifiability_rejected_count", 0),
                "belief_cap_rejected_count": metrics.get("belief_cap_rejected_count", 0),
                "n_rejected_diversity": metrics.get("n_rejected_diversity", 0),
                "beliefs_promoted_by_trend": metrics.get("beliefs_promoted_by_trend", 0),
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
