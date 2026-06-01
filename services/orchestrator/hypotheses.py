from __future__ import annotations

import json

import re

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


def _fallback_hypothesis_text(question: str, rank: int) -> str:
    q = (question or "").strip()
    ql = q.lower()
    if "activation" in ql and "transformer" in ql:
        options = [
            "GELU improves convergence stability over ReLU in transformer sequence classification.",
            "SiLU reduces early gradient noise compared with ReLU under identical optimizer settings.",
            "Activation choice changes time-to-target accuracy even when final loss is similar.",
            "Smooth activations reduce variance of validation loss across random seeds.",
            "Activation effects are stronger at higher learning rates than conservative schedules.",
        ]
    elif "warmup" in ql:
        options = [
            "Learning-rate warmup improves final generalization, not only early-step stability.",
            "Warmup benefit persists when early instability is controlled via gradient clipping.",
            "Warmup particularly improves adaptive optimizer behavior by stabilizing moment estimates.",
            "Delayed warmup underperforms immediate warmup on final validation metrics.",
            "Warmup has diminishing gains beyond a moderate ramp length.",
        ]
    elif "batch normalization" in ql or "pre-norm" in ql or "post-norm" in ql:
        options = [
            "Pre-norm improves gradient flow robustness in noisy MLP training.",
            "Post-norm converges faster initially but becomes less stable at high noise.",
            "Pre-norm permits larger stable learning rates than post-norm under equal width/depth.",
            "Norm placement interacts with depth more strongly than with width in noisy settings.",
            "Pre-norm reduces gradient-variance growth across deeper layers.",
        ]
    elif any(k in ql for k in ("optimizer", "sgd", "adam", "adamw", "rmsprop", "adagrad")):
        options = [
            "AdamW is the strongest overall optimizer across mixed loss-surface geometries.",
            "RMSProp performs best on plateaus due to adaptive scaling of sparse gradients.",
            "SGD with momentum gives better final loss on noisy landscapes than adaptive methods.",
            "Adagrad leads early convergence on sparse problems but degrades in long runs.",
            "Optimizer rankings invert between sharp ravines and flat minima regimes.",
        ]
    else:
        options = [
            "A targeted intervention measurably improves the primary metric against a baseline.",
            "The intervention improves stability under higher noise or stronger perturbations.",
            "The intervention trades speed for better final quality in controlled runs.",
            "Performance gains depend on model scale and are not uniform across settings.",
            "Observed gains remain after controlling for parameter count and compute budget.",
        ]
    if rank >= 5:
        return f"Hypothesis {rank}: Null hypothesis — the intervention has no statistically significant effect beyond noise-level variation. (Question: {q})"
    text = options[(rank - 1) % len(options)]
    return f"Hypothesis {rank}: {text} (Question: {q})"


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
{f'- For non-round-1: hypotheses should be MORE targeted based on prior round results.' if prior_round_findings else ''}

Return JSON array only. Each item: {{id, text, test_methodology, gap_reference, expected_result}}
"""


def _is_ml_template_hypothesis(text: str) -> bool:
    """Generic ML/intervention placeholders that must never enter the tree (fixes.md P0.3)."""
    core = strip_question_suffix(text).lower()
    if re.match(r"^hypothesis\s+\d+\s*:", core):
        return True
    markers = (
        "targeted intervention",
        "the intervention has no statistically significant effect",
        "baseline metric",
        "effect size",
        "noise robustness",
        "measurably improves the primary metric",
    )
    return any(m in core for m in markers)


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
) -> list[RankedHypothesis]:
    prompt = _build_hypothesis_prompt(parsed, prior, max_hypotheses, prior_round_findings)
    raw = await llm.call(prompt=prompt, purpose="hypothesis_generation", session_id=session_id)
    try:
        generated = json.loads(raw)
    except json.JSONDecodeError:
        generated = []

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
        entry = gen_list[idx] if idx < len(gen_list) and isinstance(gen_list[idx], dict) else {}
        raw_text = str(entry.get("text", ""))

        # Reject generic fallback phrasing; use domain-specific fallback instead
        if not raw_text or _is_ml_template_hypothesis(raw_text):
            raw_text = _fallback_hypothesis_text(parsed.text, rank)
        if _is_ml_template_hypothesis(raw_text):
            raw_text = (
                f"Hypothesis {rank}: A concrete, question-scoped claim about "
                f"{parsed.text[:120]}..."
            )

        methodology = str(entry.get("test_methodology", ""))
        if not methodology.strip():
            methodology = (
                "Test with statistical_significance or bootstrap_confidence, "
                "comparing treatment vs baseline metric vectors."
            )

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
        hypotheses = kept

    return hypotheses
