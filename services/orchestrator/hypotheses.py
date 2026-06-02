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


def _domain_fallback_options(question: str) -> list[str]:
    """Domain-specific discovery fallbacks (no generic ML intervention phrasing)."""
    ql = (question or "").lower()
    if any(k in ql for k in ("egyptian", "unit fraction", "1/n", "erdős", "erdos", "strauss")):
        return [
            "Odd n ≡ 1 (mod 4) below 10,000 more often admit five-term odd-denominator representations than other residue classes.",
            "A finite scan up to n = 10,000 finds no counterexample to a fixed modular necessary condition for five-term sums.",
            "Density of representable odd n decreases as the smallest prime factor of n increases, testable by exact enumeration.",
            "Parametric families n = pq with distinct odd primes p,q show higher representation rate than prime n.",
        ]
    if any(k in ql for k in ("collatz", "3n+1", "stopping time")):
        return [
            "Residue class mod 8 predicts median Collatz stopping time among integers below 10^6.",
            "Numbers with many factors of 2 in their trajectory prefix have shorter stopping times on average.",
            "No residue class below mod 16 exhibits stopping times exceeding 5× the global median.",
            "Stopping-time variance is higher for n ≡ 3 (mod 4) than for n ≡ 1 (mod 4).",
        ]
    if any(k in ql for k in ("prime gap", "cramér", "prime gaps")):
        return [
            "Local prime gaps above 10^6 follow a log-scaled distribution closer to Cramér heuristics than a constant baseline.",
            "Intervals with higher prime density show smaller normalized gaps than sparse intervals.",
            "Twin-prime-like small gaps occur more often than a naive random model predicts in [10^6, 10^6+10^5].",
            "Maximum gap in sliding windows grows sublinearly with window center up to 10^7.",
        ]
    if any(k in ql for k in ("contagion", "epidemic", "diffusion", "spreading", "sir", "sis")):
        return [
            "Epidemic peak time on scale-free networks is more sensitive to degree exponent γ than to average degree.",
            "Competing SIS and IC diffusion models rank hub-removal impact differently on the same graph ensemble.",
            "Assortative mixing raises outbreak final size under fixed R0 in configuration-model replicas.",
            "Algebraic connectivity correlates with time-to-50% infected across Barabási–Albert and ER families.",
        ]
    if any(k in ql for k in ("resilience", "targeted removal", "robustness", "node removal", "percolation")):
        return [
            "Targeted highest-degree removal fragments BA graphs faster than random removal at equal fraction removed.",
            "Percolation threshold on heavy-tailed configuration models shifts left when the tail exponent decreases.",
            "Betweenness-centrality removal predicts giant-component collapse better than degree on spatial networks.",
            "Assortativity protects against fragmentation under targeted attack up to a critical removal fraction.",
        ]
    if any(k in ql for k in ("community", "modularity", "clustering", "block model")):
        return [
            "Modularity optimization recovers planted blocks in SBM down to edge probability p = c/n log n.",
            "Spectral clustering outperforms modularity when inter-block edge density is below 0.02.",
            "Normalized cut objective aligns with planted partitions on balanced two-block models.",
            "Greedy modularity fails to detect small communities below 5% graph size.",
        ]
    if any(k in ql for k in ("cache", "replacement", "lru", "miss rate")):
        return [
            "LRU-adversarial traces expose at least 2× higher miss rate for LRU than for LFU on fixed capacity.",
            "Zipf access with exponent s>1 favors adaptive policies over FIFO at capacity ≤ 256.",
            "Belady-optimal offline policy gap to LRU shrinks as trace length increases beyond 10^5 references.",
            "Random replacement is within 15% of LRU miss rate only on nearly uniform access patterns.",
        ]
    if any(k in ql for k in ("scheduling", "waiting time", "round-robin", "queue", "m/m/1")):
        return [
            "SRPT reduces mean waiting time versus round-robin on Pareto job sizes with α < 2.",
            "M/M/1 waiting-time variance at ρ=0.9 is within 10% of heavy-traffic approximation.",
            "Weighted fair queueing bounds 99th-percentile delay for low-priority flows vs strict priority.",
            "Bursty arrivals inflate tail latency more under FCFS than under shortest-job-first.",
        ]
    if any(k in ql for k in ("coloring", "graph color", "chromatic")):
        return [
            "Smallest-last ordering uses fewer colors than random order on random geometric graphs.",
            "Greedy coloring exceeds chromatic number by at most one on trees with ≥ 10^4 vertices.",
            "DSatur heuristic beats largest-first on Erdős–Rényi G(n,p) at p = 0.05.",
            "Edge density above 0.2 forces greedy palette size ≥ ω(G) + 2 on some instances.",
        ]
    if any(k in ql for k in ("gene", "regulatory", "knockdown", "expression")):
        return [
            "Hub gene knockdown increases expression variance of downstream targets more than leaf knockdown.",
            "Feedback loops in published GRNs amplify perturbation effects within three hops.",
            "Essential genes cluster in high in-degree modules of the regulatory network.",
            "Knockdown of low-betweenness genes rarely changes global expression PCA axes.",
        ]
    if any(k in ql for k in ("protein", "ppi", "essentiality", "betweenness")):
        return [
            "High-betweenness PPI proteins are enriched for essentiality vs degree-matched controls.",
            "Peripheral PPI modules have lower essential-gene fraction than core modules.",
            "Clustering coefficient in PPI networks anti-correlates with essentiality at fixed degree.",
            "Bottleneck proteins bridge more shortest paths than expected in degree-preserving null models.",
        ]
    if any(k in ql for k in ("evolution", "selection", "frequency spectrum", "pathway")):
        return [
            "Selection strength from site-frequency spectra correlates with metabolic pathway centrality.",
            "Peripheral pathway genes show weaker selection signatures than hub enzymes.",
            "Purifying selection dominates nonsynonymous SNPs in conserved pathway cores.",
            "Positive-selection outliers map to pathway crosstalk nodes more than isolated reactions.",
        ]
    if any(k in ql for k in ("auction", "second-price", "private value", "revenue")):
        return [
            "Correlated private values reduce second-price revenue vs independent-value predictions.",
            "Reserve-price optimization shifts bidder surplus more than mechanism choice in thin markets.",
            "Revenue equivalence breaks when values share a common shock component.",
            "Winner's curse appears in affiliated-value auctions with n < 20 bidders.",
        ]
    if any(k in ql for k in ("interbank", "cascade", "exposure", "market", "adoption", "referral")):
        return [
            "Interbank exposure cascades show heavier depth tails under fractional contagion than threshold models.",
            "Referral-network adoption fits logistic growth better than exponential in early windows.",
            "Network effects raise adoption curvature when central influencers connect otherwise sparse clusters.",
            "Cascade depth scales with eigenvector centrality of the seed node in exposure graphs.",
        ]
    if any(k in ql for k in ("load balanc", "power-of-two", "latency", "tail latency")):
        return [
            "Power-of-two-choices assignment cuts p99 latency versus uniform random under bursty Poisson arrivals.",
            "Join-shortest-queue reduces mean delay when service times are heavy-tailed.",
            "Random assignment induces queue instability at ρ>0.9 before JSQ does on identical traces.",
            "Affinity routing increases tail latency when hot keys concentrate on few servers.",
        ]
    if any(k in ql for k in ("power grid", "cascade failure", "bridge", "line graph")):
        return [
            "Cascade failures in toy power grids concentrate on bridges in the line graph.",
            "High-degree buses trigger larger cascades than high-betweenness buses in IEEE test cases.",
            "Removing top 5% edges by effective resistance reduces cascade size more than random edge cuts.",
            "Islands after line outages correlate with cut-set size in the transmission graph.",
        ]
    if any(k in ql for k in ("tsp", "christofides", "approximation", "metric")):
        return [
            "2-approximation MST-doubling tour length exceeds Christofides by >10% on random Euclidean instances.",
            "Christofides is within 5% of optimal on n ≤ 200 planar point sets.",
            "Greedy insertion beats nearest-neighbor on uniform random points in the unit square.",
            "Tour length scales linearly with √n for random Euclidean instances in dimension 2.",
        ]
    if any(k in ql for k in ("kuramoto", "synchronization", "fiedler", "spectral")):
        return [
            "Fiedler value (algebraic connectivity) correlates with Kuramoto synchronization time on RGG.",
            "Higher spectral gap reduces phase dispersion at fixed coupling strength K.",
            "Synchronization failure occurs below a critical K proportional to 1/λ2.",
            "Degree-heterogeneous Kuramoto networks desynchronize when hub phases drift first.",
        ]
    if any(k in ql for k in ("bfs", "bidirectional", "shortest path")):
        return [
            "Bidirectional BFS visits fewer edges than unidirectional BFS on sparse graphs with diameter < 20.",
            "Hub nodes inflate unidirectional frontier size on social-network-like graphs.",
            "Bidirectional search wins when start and goal degrees are balanced.",
            "Unidirectional BFS matches bidirectional cost only on expander-like graphs.",
        ]
    if "activation" in ql and "transformer" in ql:
        return [
            "GELU improves convergence stability over ReLU in transformer sequence classification.",
            "SiLU reduces early gradient noise compared with ReLU under identical optimizer settings.",
            "Activation choice changes time-to-target accuracy even when final loss is similar.",
            "Smooth activations reduce variance of validation loss across random seeds.",
        ]
    if "warmup" in ql:
        return [
            "Learning-rate warmup improves final generalization, not only early-step stability.",
            "Warmup benefit persists when early instability is controlled via gradient clipping.",
            "Warmup particularly improves adaptive optimizer behavior by stabilizing moment estimates.",
            "Delayed warmup underperforms immediate warmup on final validation metrics.",
        ]
    if "batch normalization" in ql or "pre-norm" in ql or "post-norm" in ql:
        return [
            "Pre-norm improves gradient flow robustness in noisy MLP training.",
            "Post-norm converges faster initially but becomes less stable at high noise.",
            "Pre-norm permits larger stable learning rates than post-norm under equal width/depth.",
            "Norm placement interacts with depth more strongly than with width in noisy settings.",
        ]
    if any(k in ql for k in ("optimizer", "sgd", "adam", "adamw", "rmsprop", "adagrad")):
        return [
            "AdamW is the strongest overall optimizer across mixed loss-surface geometries.",
            "RMSProp performs best on plateaus due to adaptive scaling of sparse gradients.",
            "SGD with momentum gives better final loss on noisy landscapes than adaptive methods.",
            "Adagrad leads early convergence on sparse problems but degrades in long runs.",
        ]
    # Last resort: anchor claims to question nouns (not ML templates).
    snippet = " ".join(w for w in re.findall(r"[a-zA-Z]{5,}", question)[:6])
    return [
        f"A measurable structural pattern in {snippet or 'the stated domain'} is detectable with simulation or exact enumeration.",
        f"Competing mechanisms for {snippet or 'the phenomenon'} make different predictions on held-out graph or number families.",
        f"A parametric family in the question admits a boundary regime where the claimed effect reverses.",
        f"Finite verification up to 10^4 distinguishes the main claim from a noise-only null model.",
    ]


def _null_hypothesis_text(question: str) -> str:
    q = (question or "").strip()
    return (
        f"Null hypothesis: No falsifiable pattern in the research question holds beyond "
        f"what random variation would produce under the same verification procedure. "
        f"(Question: {q})"
    )


def _fallback_hypothesis_text(question: str, rank: int) -> str:
    """Discovery or control fallback without generic ML template phrasing."""
    if rank >= 5:
        return _null_hypothesis_text(question)
    options = _domain_fallback_options(question)
    text = options[(rank - 1) % len(options)]
    return f"{text} (Question: {question.strip()})"


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
) -> list[RankedHypothesis]:
    """Ensure at least ``min_discovery`` discovery hypotheses survive the gate."""
    from propab.research_quality import infer_node_role

    discovery = [h for h in kept if infer_node_role(h.text) != "CONTROL"]
    if len(discovery) >= min_discovery:
        return kept
    existing_texts = {strip_question_suffix(h.text) for h in kept}
    options = _domain_fallback_options(question)
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
                scores={"question_relevance": 0.5, "composite": 0.4, "scope_fit": 0.5},
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
    try:
        generated = json.loads(raw)
    except json.JSONDecodeError:
        generated = []
    if meta is not None:
        if isinstance(generated, list):
            raw_count = sum(1 for x in generated if isinstance(x, dict) and x.get("text"))
            meta["llm_empty"] = raw_count == 0
            meta["raw_llm_count"] = raw_count
        else:
            meta["llm_empty"] = True
            meta["raw_llm_count"] = 0

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
            raw_text = _fallback_hypothesis_text(parsed.text, rank)

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
        hypotheses = _inject_discovery_fallbacks(
            kept,
            question=parsed.text,
            max_hypotheses=max_hypotheses,
            min_discovery=min(3, max(1, max_hypotheses - 1)),
        )
    else:
        # Gate rejected everything — rebuild from domain fallbacks + null.
        hypotheses = []
        for idx in range(max_hypotheses):
            rank = idx + 1
            hypotheses.append(
                RankedHypothesis(
                    id=f"h{rank}",
                    text=_fallback_hypothesis_text(parsed.text, rank),
                    test_methodology="Test with statistical_significance or bootstrap_confidence.",
                    scores={"composite": 0.4, "question_relevance": 0.45},
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
        )

    return hypotheses
