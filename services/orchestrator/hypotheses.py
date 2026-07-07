from __future__ import annotations

import json
import re
from typing import Any

from propab.config import settings
from propab.events import EventEmitter
from propab.llm import LLMClient
from propab.types import EventType
from services.orchestrator.campaign_diagnostics import infer_hypothesis_theme
from services.orchestrator.hypothesis_ranking import (
    apply_architecture_ranking,
    compute_question_relevance_scores,
    strip_question_suffix,
)
from services.orchestrator.intake import ParsedQuestion
from services.orchestrator.schemas import Prior, RankedHypothesis


# Tokens that carry no domain signal — never used as the subject of a claim.
_STOPWORDS = frozenset(
    {
        "which", "what", "where", "when", "whether", "does", "doing", "about",
        "there", "their", "these", "those", "would", "could", "should", "using",
        "study", "studies", "research", "question", "investigate", "analyze",
        "compare", "measure", "determine", "understand", "consider", "given",
        "under", "above", "below", "between", "across", "within", "domain",
        "profile", "value", "values", "effect", "effects", "result", "results",
        "based", "than", "then", "with", "from", "into", "over", "more", "most",
        "less", "some", "many", "such", "each", "both", "same", "other", "this",
        "that", "have", "will", "been", "being",
    }
)


def _salient_terms(question: str, *, prior: Prior | None = None, limit: int = 8) -> list[str]:
    """Content words that characterize the question's own domain shape.

    Domain-agnostic: pulls the question's rarer nouns (length >= 4, not a
    stopword) plus terms borrowed from the literature prior's open gaps and key
    papers, so an arbitrary question still yields concrete subject matter.
    """
    text = strip_question_suffix(question or "")
    # Drop any [domain_profile:...] tag so it never leaks into a claim.
    text = re.sub(r"\[domain_profile:[a-z0-9_]+\]", " ", text, flags=re.I)
    seen: list[str] = []

    def _add(raw: str) -> None:
        for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", raw or ""):
            wl = w.lower()
            if wl in _STOPWORDS or wl in {s.lower() for s in seen}:
                continue
            seen.append(w)

    _add(text)
    if prior is not None:
        for gap in (prior.open_gaps or [])[:3]:
            _add(str(gap.get("text") or gap.get("gap") or ""))
        for kp in (prior.key_papers or [])[:3]:
            _add(str(kp.get("title") or ""))
    return seen[:limit]


def _open_gap_lines(prior: Prior | None, *, limit: int = 6) -> list[str]:
    """Human-readable open-gap statements from the literature prior, richest first.

    Feeds the generation prompt the literature's genuinely OPEN questions (with
    any stated best-known bound and computationally-approachable angle) so the
    model targets what is unknown, rather than re-deriving tabulated constants.
    """
    lines: list[str] = []
    for gap in (prior.open_gaps or []) if prior else []:
        if not isinstance(gap, dict):
            continue
        text = str(gap.get("what_is_open") or gap.get("text") or gap.get("gap") or "").strip()
        if not text:
            continue
        bound = str(gap.get("best_known_bound") or "").strip()
        angle = str(gap.get("approachable_angle") or "").strip()
        extra = []
        if bound:
            extra.append(f"best-known bound: {bound}")
        if angle:
            extra.append(f"approachable angle: {angle}")
        lines.append(text + (f" ({'; '.join(extra)})" if extra else ""))
        if len(lines) >= limit:
            break
    return lines


def _ensure_null_hypothesis(hypotheses: list[RankedHypothesis], question: str) -> list[RankedHypothesis]:
    if not hypotheses:
        return hypotheses
    for h in hypotheses:
        t = (h.text or "").lower()
        if any(k in t for k in ("null hypothesis", "no significant effect", "no effect", "not significantly")):
            return hypotheses
    # Force one null hypothesis for scientific falsification.
    target = hypotheses[-1]
    target.text = (
        f"Null hypothesis: No falsifiable pattern in the research question holds beyond "
        f"what random variation would produce under the same verification procedure. "
        f"(Question: {question})"
    )
    if not (target.test_methodology or "").strip():
        target.test_methodology = "Test against baseline and verify p-value >= 0.05 under repeated runs."
    return hypotheses


def _parse_hypothesis_json(raw: str) -> list[dict[str, Any]]:
    """Parse LLM hypothesis array; tolerate markdown fences and trailing prose."""
    text = (raw or "").strip()
    if not text:
        return []
    for marker in ("```json", "```JSON", "```"):
        if marker in text:
            inner = text.split(marker, 1)[1].split("```", 1)[0].strip()
            if inner.startswith("["):
                try:
                    data = json.loads(inner)
                    return data if isinstance(data, list) else []
                except json.JSONDecodeError:
                    pass
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            pass
    return []


def _verification_budget_block(question: str) -> str:
    """Advisory computable-parameter ceilings from the owning domain plugin.

    Domain-general: resolves whichever plugin owns the question and asks for its
    ``verification_budget_hint()``. A domain with no hint (the default ``None``)
    yields an empty string, so non-computational domains are unaffected. This is
    guidance in the prompt, never a hard clamp — proposals outside the ceilings
    are discouraged (they usually cannot be verified in the per-node budget), not
    forbidden.
    """
    try:
        from propab.domain_modules.registry import resolve_domain_plugin

        plugin = resolve_domain_plugin(question=question or "")
        hint = plugin.verification_budget_hint() if plugin is not None else None
    except Exception:  # noqa: BLE001 — advisory only; never break generation
        hint = None
    if not isinstance(hint, dict) or not hint:
        return ""
    note = str(hint.get("note") or "").strip()
    ceilings = {k: v for k, v in hint.items() if k != "note"}
    ceiling_lines = "\n".join(f"- {k}: {v}" for k, v in ceilings.items())
    body = (
        "\nCOMPUTABLE-PARAMETER GUIDANCE (advisory — proposals beyond these ceilings "
        "usually cannot be verified within the per-node compute budget and return no "
        "signal, so size instances at or below them):\n"
        + ceiling_lines
    )
    if note:
        body += f"\n{note}"
    return body + "\n"


def _build_hypothesis_prompt(
    parsed: ParsedQuestion,
    prior: Prior,
    max_hypotheses: int,
    prior_round_findings: str = "",
    *,
    novelty_nudge: bool = False,
    skills_block: str = "",
) -> str:
    prior_block = (
        f"\nResults from previous research rounds:\n{prior_round_findings}\n"
        if prior_round_findings.strip()
        else ""
    )
    # Domain flavor comes ENTIRELY from the injected literature gaps (already
    # specific to this research area's own literature) — never from hardcoded
    # domain keywords. An empty gaps list simply yields a general open-question
    # instruction; no fabricated domain content is inserted.
    gap_lines = _open_gap_lines(prior)
    if gap_lines:
        gaps_block = "Open questions in this research area (from its literature — TARGET THESE):\n" + "\n".join(
            f"- {g}" for g in gap_lines
        )
    else:
        gaps_block = (
            "No pre-catalogued open questions were retrieved. Infer where the frontier of "
            "THIS research question genuinely lies and target that, not settled ground."
        )

    nudge_block = ""
    if novelty_nudge:
        nudge_block = (
            "\nRETRY — the previous attempt produced no usable, on-topic hypothesis. Propose "
            "conceptually DISTINCT, genuinely novel directions this time. Do not restate the "
            "question back, and do not hedge into unfalsifiable generalities.\n"
        )

    # Research-skills injection: the skills the AGENT selected from the catalog (see
    # _select_and_read_skills) are passed in and injected here — no auto-load by
    # domain/question. The agent inferred its domain and pulled what it needs.
    skills_section = f"\n{skills_block}\n" if skills_block else ""

    # Feasible-parameter guidance from the owning domain plugin (default: empty for
    # non-computational domains). Keeps proposals in the verifiable regime so a
    # good idea is not wasted on an instance that times out with no signal.
    budget_block = _verification_budget_block(parsed.text)

    return f"""
You are a research hypothesis generator whose job is to ADVANCE what is known — not to
re-measure or re-derive it. Propose hypotheses whose answers are currently UNKNOWN and,
if resolved, would change what this research area accepts as established.
{skills_section}
Research question: {parsed.text}

Prior established facts (already known — do NOT propose hypotheses whose answer these settle):
{json.dumps(prior.established_facts)}

{gaps_block}

Prior open gaps (structured):
{json.dumps(prior.open_gaps)}

Prior dead ends (do not repeat these):
{json.dumps(prior.dead_ends)}
{prior_block}{nudge_block}{budget_block}
Generate exactly {max_hypotheses} hypotheses.

NOVELTY REQUIREMENTS (these override everything else):
- Each hypothesis must target an OPEN question grounded in the gaps above — something whose
  answer is not already tabulated, catalogued, or settled in the established facts.
- Do NOT propose a claim whose answer is a known or tabulated value, a re-derivation of an
  established result, or a measurement of an already-characterized method/procedure.
- Do NOT restate a previously-tested hypothesis with different parameters (no re-tests of a
  known result under a new size/dimension/threshold). A parameter sweep of a settled finding
  is NOT a new hypothesis.
- Prefer hypotheses that could change what is known: challenge or improve a current
  best-known result, propose a genuinely new construction or mechanism, or state a
  falsifiable structural conjecture that the literature has not resolved.
- Cover CONCEPTUALLY DISTINCT directions across the set (different mechanisms, invariants, or
  questions) — not several variations of the same idea.

FORM REQUIREMENTS:
- Each hypothesis must be specific and falsifiable, NOT generic.
- Each must state its test methodology naming at least one specific analysis tool
  (e.g. statistical_significance, bootstrap_confidence, literature_baseline_compare).
- Do NOT repeat confirmed findings, refuted hypotheses, or dead ends from prior rounds.
- Do NOT use generic phrasing like "Hypothesis 1: ..." or "The intervention has an effect."
- One hypothesis should be a null hypothesis (no significant effect).
- EVERY hypothesis MUST include explicit scope boundaries:
  * population — who/what instances (size, family, regime)
  * distribution — the generating/sampling distribution (dataset, ensemble, simulator, corpus)
  * claimed_generalization — where you expect this to transfer (must differ from distribution)
  * expected_failure_modes — at least one regime where the claim should break
  * ood_test — concrete hold-out / transfer test run BEFORE confirmation
- BAD (too vague): "Factor X predicts outcome Y."
- GOOD (scoped, transferable): state the population/distribution it holds on, where it should
  transfer, where it should break, and the out-of-distribution test that decides it.
{f'- For non-round-1: hypotheses should be MORE targeted based on prior round results.' if prior_round_findings else ''}

Return JSON array only. Each item:
{{id, text, test_methodology, gap_reference, expected_result,
  population, distribution, claimed_generalization, expected_failure_modes, ood_test}}
"""


def _is_ml_template_hypothesis(text: str) -> bool:
    """Generic ML/intervention placeholders — not bare 'Hypothesis N:' prefixes (seed suite fix)."""
    core = strip_question_suffix(text).lower()
    markers = (
        "targeted intervention",
        "the intervention has no statistically significant effect",
        "baseline metric",
        "noise robustness",
        "measurably improves the primary metric",
        "a concrete, question-scoped claim about",
    )
    if any(m in core for m in markers):
        return True
    # Legacy rank-5 fallback only (discovery fallbacks no longer use this phrasing).
    if re.match(r"^hypothesis\s+\d+\s*:", core) and "null hypothesis" not in core:
        return "intervention" in core or "baseline" in core or "primary metric" in core
    return False


async def _select_and_read_skills(
    parsed: ParsedQuestion,
    prior: Prior,
    llm: LLMClient,
    session_id: str,
    emitter: EventEmitter,
) -> str:
    """Agentic skill selection (awareness -> on-demand read).

    Show the agent the full skill CATALOG (names + descriptions only — cheap), let IT
    decide which skills it needs after inferring its own domain, and read back ONLY the
    bodies it picked. No auto-load by domain/question, no injecting every skill body.
    Falls back to the core hypothesis-phase skills if selection fails, so generation
    always has methodology to work with.
    """
    try:
        from propab.skills import (
            catalog_skill_names,
            read_skills,
            render_catalog,
            render_skills_block,
            skills_catalog,
        )
    except Exception:  # noqa: BLE001 — skills are additive; never break generation
        return ""

    catalog = render_catalog()
    if not catalog:
        return ""

    def _core_fallback() -> str:
        return render_skills_block([s for s in skills_catalog("hypothesis") if s.scope == "core"])

    sel_prompt = (
        "You are a research agent about to generate hypotheses. First choose which "
        "methodology skills you need. Infer the domain yourself from the question — do "
        "NOT assume one is pre-selected.\n\n"
        f"Research question: {parsed.text}\n\n"
        f"{catalog}\n\n"
        "Return ONLY a JSON array of the exact skill names you will read (up to 8), e.g. "
        '["falsifiable-hypothesis-design", "divergent-hypothesis-ideation"]. Include the '
        "core methodology you need plus, if a domain clearly applies, that domain's "
        "technique skills. Do not invent names."
    )
    try:
        raw = await llm.call(prompt=sel_prompt, purpose="skill_selection", session_id=session_id)
    except Exception:  # noqa: BLE001 — best-effort; fall back to core methodology
        return _core_fallback()

    names: list[str] = []
    m = re.search(r"\[[\s\S]*?\]", raw or "")
    if m:
        try:
            data = json.loads(m.group(0))
            valid = catalog_skill_names()
            names = [str(x) for x in data if isinstance(x, str) and str(x) in valid]
        except (json.JSONDecodeError, TypeError, ValueError):
            names = []

    block = read_skills(names) if names else ""
    if not block:
        block = _core_fallback()
    try:
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.LLM_RESPONSE,
            step="skills.selected",
            payload={"selected_skills": names, "purpose": "skill_selection"},
        )
    except Exception:  # noqa: BLE001
        pass
    return block


async def generate_ranked_hypotheses(
    parsed: ParsedQuestion,
    prior: Prior,
    max_hypotheses: int,
    llm: LLMClient,
    session_id: str,
    emitter: EventEmitter,
    *,
    use_llm_ranking: bool = True,
    prior_round_findings: str = "",
    meta: dict | None = None,
) -> list[RankedHypothesis]:
    # Awareness -> on-demand read: the agent picks its own methodology skills from the
    # catalog before generating (no auto-load by domain). Reused for the retry so the
    # agent keeps the methodology it chose.
    skills_block = await _select_and_read_skills(parsed, prior, llm, session_id, emitter)
    prompt = _build_hypothesis_prompt(
        parsed, prior, max_hypotheses, prior_round_findings, skills_block=skills_block
    )
    raw = await llm.call(prompt=prompt, purpose="hypothesis_generation", session_id=session_id)
    generated = _parse_hypothesis_json(raw)
    if not generated:
        # Honest single retry with a diversity/novelty-nudged prompt. NO template
        # fabrication: if the model still yields nothing usable, we return empty
        # below and the caller stops the round honestly (frontier-exhausted).
        retry_prompt = _build_hypothesis_prompt(
            parsed, prior, max_hypotheses, prior_round_findings,
            novelty_nudge=True, skills_block=skills_block,
        )
        raw_retry = await llm.call(
            prompt=retry_prompt + "\n\nReturn ONLY a JSON array of exactly "
            f"{max_hypotheses} hypothesis objects. No markdown.",
            purpose="hypothesis_generation_retry",
            session_id=session_id,
        )
        generated = _parse_hypothesis_json(raw_retry)
    if meta is not None:
        raw_count = sum(1 for x in generated if isinstance(x, dict) and x.get("text"))
        meta["llm_empty"] = raw_count == 0
        meta["raw_llm_count"] = raw_count

    if isinstance(generated, list):
        themed = []
        for item in generated:
            if isinstance(item, dict):
                text = str(item.get("text") or "")
                themed.append({**item, "theme": infer_hypothesis_theme(text)})
            else:
                themed.append(item)
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.HYPO_GENERATED,
            step="hypothesis.generate",
            payload={"hypotheses": themed},
        )
    else:
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.HYPO_GENERATED,
            step="hypothesis.generate",
            payload={"hypotheses": [], "note": "Model returned non-array JSON; no usable hypotheses this round."},
        )

    # Build hypotheses ONLY from real LLM entries. Propab must REASON, never
    # expand a template: an entry the model did not produce, or one whose scope is
    # missing/invalid, is DROPPED (and counted) — it is never replaced by a
    # fabricated templated hypothesis.
    from propab.scoped_claim import (
        enrich_entry_with_scope,
        parse_scope_from_entry,
        validate_scoped_claim,
    )

    gen_list = [x for x in (generated if isinstance(generated, list) else []) if isinstance(x, dict)]
    hypotheses: list[RankedHypothesis] = []
    n_dropped_no_scope = 0
    n_dropped_template = 0
    for idx, raw_entry in enumerate(gen_list[:max_hypotheses]):
        rank = len(hypotheses) + 1
        composite = round(max(0.15, 1.0 - idx * 0.12), 3)

        scope = parse_scope_from_entry(raw_entry)
        if scope is None or not validate_scoped_claim(scope)[0]:
            # No valid scope from the model → drop honestly rather than template-fill.
            n_dropped_no_scope += 1
            continue
        entry = enrich_entry_with_scope(dict(raw_entry), parsed.text, allow_template_fill=False)

        raw_text = str(entry.get("text", ""))
        if not raw_text or _is_ml_template_hypothesis(raw_text):
            n_dropped_template += 1
            continue

        methodology = str(entry.get("test_methodology", ""))
        if not methodology.strip():
            methodology = "Test with statistical_significance or bootstrap_confidence against a matched baseline."

        hypotheses.append(
            RankedHypothesis(
                id=str(entry.get("id", f"h{rank}")),
                text=raw_text,
                test_methodology=methodology,
                scores={
                    "novelty": round(max(0.2, composite - 0.1), 3),
                    "testability": round(max(0.3, composite), 3),
                    "impact": round(max(0.25, composite - 0.05), 3),
                    "scope_fit": round(max(0.2, composite - 0.08), 3),
                    "composite": composite,
                    "scope_valid": 1.0 if entry.get("_scope_valid") else 0.0,
                    "scope_fallback": 0.0,
                    "is_fallback": 0.0,
                },
                rank=rank,
            )
        )
    if meta is not None:
        meta["n_dropped_no_scope"] = n_dropped_no_scope
        meta["n_dropped_template"] = n_dropped_template

    if not hypotheses:
        # No usable, scope-valid, on-topic LLM hypothesis this round. Return empty
        # and let the caller stop honestly — never fabricate a templated seed.
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.HYPO_REJECTED,
            step="hypothesis.no_usable_llm_output",
            payload={
                "n_raw_llm": len(gen_list),
                "n_dropped_no_scope": n_dropped_no_scope,
                "n_dropped_template": n_dropped_template,
                "note": "LLM produced no usable on-topic hypothesis; round returns empty (no template fill).",
            },
        )
        return []

    if use_llm_ranking and (
        settings.llm_provider.strip().lower() == "ollama" or settings.llm_api_secret.strip()
    ):
        hypotheses = await apply_architecture_ranking(
            hypotheses=hypotheses,
            prior=prior,
            question=parsed.text,
            llm=llm,
            session_id=session_id,
        )
    hypotheses = _ensure_null_hypothesis(hypotheses, parsed.text)

    # Question relevance gate (fixes.md P0.3) — reject off-topic / generic templates.
    threshold = float(getattr(settings, "hypothesis_relevance_threshold", 0.35))
    texts = [strip_question_suffix(h.text) for h in hypotheses]
    relevance_scores = await compute_question_relevance_scores(parsed.text, prior, texts)
    kept: list[RankedHypothesis] = []
    rejected: list[dict[str, str | float]] = []
    for h, rel, core_text in zip(hypotheses, relevance_scores, texts, strict=False):
        if _is_ml_template_hypothesis(h.text):
            rejected.append({"id": h.id, "text": core_text[:200], "question_relevance_score": rel, "reason": "ml_template"})
            continue
        from propab.scoped_claim import is_boilerplate_scope, parse_scope_from_methodology, validate_scoped_claim

        scope = parse_scope_from_methodology(h.text, h.test_methodology)
        scope_ok, scope_missing = validate_scoped_claim(scope)
        if not scope_ok:
            rejected.append({
                "id": h.id,
                "text": core_text[:200],
                "question_relevance_score": rel,
                "reason": "missing_scope",
                "missing": scope_missing,
            })
            continue
        if scope and is_boilerplate_scope(scope, parsed.text):
            # Every hypothesis here is genuine LLM output (no fabricated fallbacks);
            # a verbatim template scope is therefore a cosmetic gate bypass — reject.
            rejected.append({
                "id": h.id,
                "text": core_text[:200],
                "question_relevance_score": rel,
                "reason": "boilerplate_scope",
            })
            continue
        h.scores = dict(h.scores or {})
        h.scores["question_relevance"] = rel
        if rel >= threshold:
            kept.append(h)
        else:
            rejected.append({"id": h.id, "text": core_text[:200], "question_relevance_score": rel, "reason": "below_threshold"})
    if rejected:
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.HYPO_REJECTED,
            step="hypothesis.relevance_gate",
            payload={"threshold": threshold, "rejected_count": len(rejected), "rejected": rejected[:12]},
        )
    # Whatever survived the gate is what we return. If the gate rejected
    # everything, we return an empty list — the caller stops the round honestly.
    # We do NOT rebuild from templates to keep the round alive.
    hypotheses = kept

    n_generated = len(gen_list)
    scope_rejected = sum(1 for r in rejected if r.get("reason") in ("missing_scope", "boilerplate_scope"))
    from collections import Counter

    reason_counts = dict(Counter(r.get("reason", "?") for r in rejected))
    scope_metrics = {
        "session_id": session_id,
        "n_generated": n_generated,
        "n_scope_rejected": scope_rejected,
        "n_scope_passed": len(kept),
        "scope_rejection_rate": round(scope_rejected / max(1, n_generated), 4),
        "rejection_reasons": reason_counts,
    }
    if meta is not None:
        meta["scope_metrics"] = scope_metrics
    await emitter.emit(
        session_id=session_id,
        event_type=EventType.HYPO_REJECTED,
        step="hypothesis.scope_gate",
        payload=scope_metrics,
    )

    return hypotheses
