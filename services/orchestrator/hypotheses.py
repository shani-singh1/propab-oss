from __future__ import annotations

import json

from propab.config import settings
from propab.events import EventEmitter
from propab.llm import LLMClient
from propab.types import EventType
from services.orchestrator.hypothesis_ranking import apply_architecture_ranking
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
        f"Null hypothesis: The intervention has no statistically significant effect for this question beyond noise-level differences. "
        f"(Question: {question})"
    )
    if not (target.test_methodology or "").strip():
        target.test_methodology = "Test against baseline and verify p-value >= 0.05 under repeated runs."
    return hypotheses


def _build_hypothesis_prompt(parsed: ParsedQuestion, prior: Prior, max_hypotheses: int) -> str:
    return f"""
You are a research hypothesis generator.

Research question: {parsed.text}

Prior established facts:
{json.dumps(prior.established_facts)}

Prior open gaps:
{json.dumps(prior.open_gaps)}

Prior dead ends:
{json.dumps(prior.dead_ends)}

Generate exactly {max_hypotheses} hypotheses.
Return JSON array only with fields: id, text, test_methodology, gap_reference, expected_result.
One of the hypotheses must be a null hypothesis predicting no significant effect.
"""


async def generate_ranked_hypotheses(
    parsed: ParsedQuestion,
    prior: Prior,
    max_hypotheses: int,
    llm: LLMClient,
    session_id: str,
    emitter: EventEmitter,
    *,
    use_llm_ranking: bool = True,
) -> list[RankedHypothesis]:
    prompt = _build_hypothesis_prompt(parsed, prior, max_hypotheses)
    raw = await llm.call(prompt=prompt, purpose="hypothesis_generation", session_id=session_id)
    try:
        generated = json.loads(raw)
    except json.JSONDecodeError:
        generated = []

    if isinstance(generated, list):
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.HYPO_GENERATED,
            step="hypothesis.generate",
            payload={"hypotheses": generated},
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
        hypotheses.append(
            RankedHypothesis(
                id=str(entry.get("id", f"h{rank}")),
                text=str(entry.get("text", _fallback_hypothesis_text(parsed.text, rank))),
                test_methodology=str(
                    entry.get(
                        "test_methodology",
                        "Run a controlled experiment and compare baseline vs intervention metrics.",
                    )
                ),
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
    return hypotheses
