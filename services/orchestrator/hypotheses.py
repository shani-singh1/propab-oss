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


# Tokens that carry no domain signal — never used as the subject of a seed claim.
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


def _parametric_families(question: str) -> list[str]:
    """Parametric knobs literally present in the question (e.g. 'n', 'k', 'p<0.05', '10^6')."""
    text = strip_question_suffix(question or "")
    families: list[str] = []
    # single-letter parameters like n, k, p, m used as variables
    for m in re.findall(r"(?<![A-Za-z])([a-z])\s*(?:[=<>≤≥]|\b(?:up to|below|above)\b)", text, flags=re.I):
        if m.lower() not in {s.lower() for s in families}:
            families.append(m)
    # explicit numeric bounds / powers
    for m in re.findall(r"(\d+(?:\^\d+|e\d+|,\d{3})*)", text):
        if m not in families:
            families.append(m)
    return families[:4]


def _domain_shape_options(question: str, *, prior: Prior | None = None) -> list[str]:
    """Domain-GENERAL discovery fallbacks derived from the question's own structure.

    Produces, for ANY question with zero keyword matches, a set of concrete,
    falsifiable seed claims: a discovery claim, a competing-mechanism contrast,
    a parametric/boundary probe, and a finite-verification claim — each phrased
    from the question's salient terms and (when available) the literature prior.
    This replaces the former hardcoded per-topic keyword table.
    """
    terms = _salient_terms(question, prior=prior)
    subject = " ".join(terms[:3]) if terms else "the stated phenomenon"
    # Full-context phrase reuses the question's own salient vocabulary so the
    # domain-general seed still shares enough tokens with the question to clear
    # the lexical relevance gate (the embedding gate is primary in production).
    context = " ".join(terms) if terms else subject
    secondary = " ".join(terms[3:6]) if len(terms) > 3 else subject
    families = _parametric_families(question)
    fam = families[0] if families else None
    bound = next((f for f in families if any(c.isdigit() for c in f)), "10^4")

    if fam and fam.isalpha():
        boundary = (
            f"Within {context}, there is a boundary regime in parameter {fam} where the "
            f"effect claimed for {subject} weakens or reverses, locatable by sweeping {fam}."
        )
    elif families:
        boundary = (
            f"Within {context}, the claimed effect on {subject} has a threshold near "
            f"{families[0]} beyond which the sign or magnitude of the effect changes."
        )
    else:
        boundary = (
            f"Within {context}, a parametric regime exists in which the claimed effect on "
            f"{subject} weakens or reverses, identifiable by a systematic parameter sweep."
        )

    return [
        (
            f"Within {context}, a measurable structured pattern relating {subject} to "
            f"{secondary} is detectable by direct simulation or exact enumeration and "
            f"exceeds a matched noise-only null model."
        ),
        (
            f"For {context}, two competing mechanisms for {subject} make divergent, "
            f"testable predictions on held-out instance families, so their ranking can be "
            f"decided empirically rather than assumed."
        ),
        boundary,
        (
            f"Finite verification up to {bound} distinguishes the leading claim about "
            f"{subject} within {context} from a random baseline with a reproducible test statistic."
        ),
    ]


def _domain_fallback_options(question: str, *, prior: Prior | None = None) -> list[str]:
    """Domain-general discovery fallbacks for seed generation.

    PRIMARY behaviour is now domain-shape driven (``_domain_shape_options``): any
    question — including ones matching none of the historical demo topics — yields
    concrete, question-relevant seed claims. The canned demo hypotheses survive
    ONLY as an optional supplement, appended after the general seeds so the demo
    topics keep their tuned phrasing without gating the general path.
    """
    general = _domain_shape_options(question, prior=prior)
    extra = _demo_topic_supplement(question)
    if not extra:
        return general
    # General seeds first, demo phrasings appended, de-duplicated.
    merged: list[str] = list(general)
    seen = {g.strip().lower() for g in general}
    for opt in extra:
        key = opt.strip().lower()
        if key not in seen:
            merged.append(opt)
            seen.add(key)
    return merged


def _scoped_contagion_text(claim: str, *, ood: str) -> str:
    return (
        f"{claim}\n"
        "Population: N=300–5000 node graphs, ≥30 instances per topology family\n"
        "Distribution: Barabási–Albert (m=3–5) and stochastic block model, avg degree 6–12\n"
        f"Claimed generalization: Effect transfers to Watts–Strogatz with matched average degree\n"
        "Expected failure modes: Vanishes on ER graphs or when modularity Q<0.2; breaks if seed set >5%\n"
        f"OOD test: {ood}"
    )


def _demo_topic_supplement(question: str) -> list[str]:
    """OPTIONAL extra phrasings for the historical demo topics.

    These are NEVER the sole source of seeds (see ``_domain_fallback_options``);
    they only enrich the general seeds when a demo topic is recognized, so the
    tuned demo behaviour is preserved without making the engine look
    domain-general while being keyword-gated.
    """
    ql = (question or "").lower()
    if any(k in ql for k in ("egyptian", "unit fraction", "1/n", "erdős", "erdos", "strauss")):
        return [
            "Odd n ≡ 1 (mod 4) below 10,000 more often admit five-term odd-denominator representations than other residue classes.",
            "A finite scan up to n = 10,000 finds no counterexample to a fixed modular necessary condition for five-term sums.",
            "Density of representable odd n decreases as the smallest prime factor of n increases, testable by exact enumeration.",
        ]
    if any(k in ql for k in ("collatz", "3n+1", "stopping time")):
        return [
            "Residue class mod 8 predicts median Collatz stopping time among integers below 10^6.",
            "Numbers with many factors of 2 in their trajectory prefix have shorter stopping times on average.",
            "Stopping-time variance is higher for n ≡ 3 (mod 4) than for n ≡ 1 (mod 4).",
        ]
    if any(k in ql for k in ("prime gap", "cramér", "prime gaps")):
        return [
            "Local prime gaps above 10^6 follow a log-scaled distribution closer to Cramér heuristics than a constant baseline.",
            "Intervals with higher prime density show smaller normalized gaps than sparse intervals.",
            "Twin-prime-like small gaps occur more often than a naive random model predicts in [10^6, 10^6+10^5].",
        ]
    if any(k in ql for k in ("contagion", "epidemic", "diffusion", "spreading", "sir", "sis")):
        return [
            _scoped_contagion_text(
                "Epidemic peak time on scale-free networks is more sensitive to degree exponent "
                "γ than to average degree when avg degree is held at 8–12.",
                ood="Hold out Watts–Strogatz; LOFO R² on peak-time vs γ must exceed 0 on WS.",
            ),
            _scoped_contagion_text(
                "Competing SIS and IC diffusion models rank hub-removal impact differently on "
                "Barabási–Albert ensembles with m=3–5.",
                ood="Evaluate same ranking on SBM graphs; confirm or refute if ranking flips.",
            ),
            _scoped_contagion_text(
                "Assortative mixing raises outbreak final size under fixed R0 in configuration-model "
                "replicas with heavy-tailed degree sequences.",
                ood="Transfer test on ER graphs with matched mean degree; effect should weaken.",
            ),
        ]
    if any(k in ql for k in ("cache", "replacement", "lru", "miss rate")):
        return [
            "LRU-adversarial traces expose at least 2× higher miss rate for LRU than for LFU on fixed capacity.",
            "Zipf access with exponent s>1 favors adaptive policies over FIFO at capacity ≤ 256.",
            "Belady-optimal offline policy gap to LRU shrinks as trace length increases beyond 10^5 references.",
        ]
    return []


def _null_hypothesis_text(question: str) -> str:
    q = (question or "").strip()
    return (
        f"Null hypothesis: No falsifiable pattern in the research question holds beyond "
        f"what random variation would produce under the same verification procedure. "
        f"(Question: {q})"
    )


def _fallback_hypothesis_text(question: str, rank: int, *, prior: Prior | None = None) -> str:
    """Discovery or control fallback without generic ML template phrasing."""
    if rank >= 5:
        null = _null_hypothesis_text(question)
        from propab.scoped_claim import enrich_entry_with_scope
        return enrich_entry_with_scope({"text": null}, question)["text"]
    options = _domain_fallback_options(question, prior=prior)
    text = options[(rank - 1) % len(options)]
    from propab.scoped_claim import enrich_entry_with_scope
    return enrich_entry_with_scope({"text": text}, question)["text"]


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


def _build_hypothesis_prompt(
    parsed: ParsedQuestion,
    prior: Prior,
    max_hypotheses: int,
    prior_round_findings: str = "",
) -> str:
    prior_block = (
        f"\nResults from previous research rounds:\n{prior_round_findings}\n"
        if prior_round_findings.strip()
        else ""
    )
    return f"""
You are a research hypothesis generator.

Research question: {parsed.text}

Prior established facts:
{json.dumps(prior.established_facts)}

Prior open gaps:
{json.dumps(prior.open_gaps)}

Prior dead ends (do not repeat these):
{json.dumps(prior.dead_ends)}
{prior_block}
Generate exactly {max_hypotheses} hypotheses.

Requirements:
- Each hypothesis must be specific and falsifiable, NOT generic.
- Each must state its test methodology naming at least one specific statistical tool
  (e.g. statistical_significance, bootstrap_confidence, literature_baseline_compare).
- Do NOT repeat confirmed findings, refuted hypotheses, or dead ends from prior rounds.
- Do NOT use generic phrasing like "Hypothesis 1: ..." or "The intervention has an effect."
- One hypothesis should be a null hypothesis (no significant effect).
- EVERY hypothesis MUST include explicit scope boundaries (fixes.md Step 2):
  * population — who/what instances (size, family, regime)
  * distribution — training/generating distribution (graph family, dataset, simulator)
  * claimed_generalization — where you expect this to transfer (must differ from distribution)
  * expected_failure_modes — at least one regime where the claim should break
  * ood_test — concrete hold-out / LOFO / transfer test run BEFORE confirmation
- BAD: "k-shell predicts spreading."
- GOOD: "In BA and SBM graphs with avg degree 6–12, k-shell predicts spreading velocity;
  should transfer to WS graphs; OOD: train BA+SBM, evaluate WS LOFO R²."
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


def _inject_discovery_fallbacks(
    kept: list[RankedHypothesis],
    *,
    question: str,
    max_hypotheses: int,
    min_discovery: int = 3,
    prior: Prior | None = None,
) -> list[RankedHypothesis]:
    """Ensure at least ``min_discovery`` discovery hypotheses survive the gate.

    Injected seeds are marked ``is_fallback=1.0`` in their scores so downstream
    consumers can distinguish a synthesized seed from an LLM-validated,
    scope-checked hypothesis (G2: fallbacks must never masquerade as validated).
    """
    from propab.research_quality import infer_node_role

    discovery = [h for h in kept if infer_node_role(h.text) != "CONTROL"]
    if len(discovery) >= min_discovery:
        return kept
    existing_texts = {strip_question_suffix(h.text) for h in kept}
    options = _domain_fallback_options(question, prior=prior)
    rank_base = len(kept) + 1
    for opt in options:
        if len(discovery) >= min_discovery:
            break
        candidate = f"{opt} (Question: {question.strip()})"
        if strip_question_suffix(candidate) in existing_texts:
            continue
        if _is_ml_template_hypothesis(candidate):
            continue
        kept.append(
            RankedHypothesis(
                id=f"fallback_d{len(discovery)+1}",
                text=candidate,
                test_methodology=(
                    "Test with statistical_significance, bootstrap_confidence, or finite enumeration."
                ),
                scores={
                    "question_relevance": 0.5,
                    "composite": 0.4,
                    "scope_fit": 0.5,
                    "is_fallback": 1.0,
                    "scope_fallback": 1.0,
                },
                rank=rank_base,
            )
        )
        discovery.append(kept[-1])
        existing_texts.add(strip_question_suffix(candidate))
        rank_base += 1
    return kept[:max_hypotheses]


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
    prompt = _build_hypothesis_prompt(parsed, prior, max_hypotheses, prior_round_findings)
    raw = await llm.call(prompt=prompt, purpose="hypothesis_generation", session_id=session_id)
    generated = _parse_hypothesis_json(raw)
    if not generated:
        raw_retry = await llm.call(
            prompt=prompt + "\n\nReturn ONLY a JSON array of exactly "
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
            payload={"hypotheses": [], "note": "Model returned non-array JSON; falling back to templates."},
        )

    hypotheses: list[RankedHypothesis] = []
    for idx in range(max_hypotheses):
        rank = idx + 1
        composite = round(max(0.15, 1.0 - idx * 0.12), 3)
        gen_list = generated if isinstance(generated, list) else []
        raw_entry = gen_list[idx] if idx < len(gen_list) and isinstance(gen_list[idx], dict) else {}

        from propab.scoped_claim import enrich_entry_with_scope, parse_scope_from_entry, validate_scoped_claim

        scope = parse_scope_from_entry(raw_entry) if raw_entry else None
        used_fallback = False
        if scope is None or not validate_scoped_claim(scope)[0]:
            used_fallback = True
            entry = enrich_entry_with_scope(
                {"text": _fallback_hypothesis_text(parsed.text, rank, prior=prior), "id": raw_entry.get("id", f"h{rank}")},
                parsed.text,
                allow_template_fill=False,
            )
        else:
            entry = enrich_entry_with_scope(dict(raw_entry), parsed.text, allow_template_fill=False)

        raw_text = str(entry.get("text", ""))
        if not raw_text or _is_ml_template_hypothesis(raw_text):
            used_fallback = True
            entry = enrich_entry_with_scope(
                {"text": _fallback_hypothesis_text(parsed.text, rank, prior=prior), "id": f"h{rank}"},
                parsed.text,
                allow_template_fill=False,
            )
            raw_text = str(entry.get("text", ""))

        methodology = str(entry.get("test_methodology", ""))
        if not methodology.strip():
            methodology = (
                "Test with statistical_significance or bootstrap_confidence, "
                "comparing treatment vs baseline metric vectors."
            )

        scores_extra: dict[str, float] = {
            "scope_valid": 1.0 if entry.get("_scope_valid") else 0.0,
            "scope_fallback": 1.0 if used_fallback else 0.0,
            # G2: a fallback seed is never an LLM-validated, on-topic hypothesis;
            # mark it so downstream consumers cannot count it as validated.
            "is_fallback": 1.0 if used_fallback else 0.0,
        }

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
                    **scores_extra,
                },
                rank=rank,
            )
        )

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
        is_fallback = (h.scores or {}).get("scope_fallback") == 1.0 or (h.scores or {}).get("is_fallback") == 1.0
        if scope and is_boilerplate_scope(scope, parsed.text):
            if not is_fallback:
                # A non-fallback (i.e. purportedly LLM-validated) hypothesis with
                # verbatim template scope is a cosmetic gate bypass — reject it.
                rejected.append({
                    "id": h.id,
                    "text": core_text[:200],
                    "question_relevance_score": rel,
                    "reason": "boilerplate_scope",
                })
                continue
            # G2: a fallback with template scope may carry the campaign, but it
            # MUST stay flagged so it cannot be counted as a validated,
            # scope-checked, on-topic hypothesis downstream.
            h.scores = dict(h.scores or {})
            h.scores["is_fallback"] = 1.0
            h.scores["boilerplate_scope"] = 1.0
            h.scores["scope_valid"] = 0.0
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
    if kept:
        hypotheses = _inject_discovery_fallbacks(
            kept,
            question=parsed.text,
            max_hypotheses=max_hypotheses,
            min_discovery=min(3, max(1, max_hypotheses - 1)),
            prior=prior,
        )
    else:
        # Gate rejected everything — rebuild from domain-general fallbacks + null.
        hypotheses = []
        for idx in range(max_hypotheses):
            rank = idx + 1
            hypotheses.append(
                RankedHypothesis(
                    id=f"h{rank}",
                    text=_fallback_hypothesis_text(parsed.text, rank, prior=prior),
                    test_methodology="Test with statistical_significance or bootstrap_confidence.",
                    # G2: rebuilt seeds are fallbacks, flagged so they are never
                    # mistaken for validated LLM hypotheses.
                    scores={
                        "composite": 0.4,
                        "question_relevance": 0.45,
                        "is_fallback": 1.0,
                        "scope_fallback": 1.0,
                    },
                    rank=rank,
                )
            )
        hypotheses = _ensure_null_hypothesis(hypotheses, parsed.text)
        texts = [strip_question_suffix(h.text) for h in hypotheses]
        relevance_scores = await compute_question_relevance_scores(parsed.text, prior, texts)
        hypotheses = [
            h
            for h, rel in zip(hypotheses, relevance_scores, strict=False)
            if not _is_ml_template_hypothesis(h.text) and rel >= threshold
        ]
        hypotheses = _inject_discovery_fallbacks(
            hypotheses,
            question=parsed.text,
            max_hypotheses=max_hypotheses,
            min_discovery=min(3, max(1, max_hypotheses - 1)),
            prior=prior,
        )

    n_generated = max_hypotheses
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
