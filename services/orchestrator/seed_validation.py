"""Seed-generation validation suite (fixes.md Phase 1) — no sub-agents or sandbox."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from propab.config import settings
from propab.db import create_engine, create_session_factory
from propab.events import EventEmitter
from propab.llm import LLMClient
from propab.research_quality import extract_theme_vector, infer_node_role
from services.orchestrator.hypotheses import generate_ranked_hypotheses
from services.orchestrator.intake import ParsedQuestion, parse_question
from services.orchestrator.literature import build_prior
from services.orchestrator.schemas import Prior
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

# 25 questions across domains (fixes.md Phase 1)
SEED_VALIDATION_QUESTIONS: list[tuple[str, str]] = [
    ("math_erdos", "For which odd integers n does 1/n admit an Egyptian fraction decomposition using five distinct unit fractions? Test residue classes and report counterexamples up to n = 10,000."),
    ("math_collatz", "Do Collatz stopping times for integers in [1, 10^6] cluster by residue class modulo small primes? Test whether any mod-8 class has systematically longer trajectories."),
    ("math_prime_gaps", "Are local prime gap sizes above 10^6 better predicted by Cramér-type heuristics than by a naive log-density baseline? Compare gap distributions across intervals."),
    ("math_unit_fraction", "Which odd n below 5,000 admit five distinct odd denominators in 1/a+1/b+1/c+1/d+1/e = 1/n? Enumerate and test modular obstructions."),
    ("network_resilience", "Which graph structural metrics best predict robustness under targeted degree-based node removal in Barabási–Albert and Erdős–Rényi families?"),
    ("network_contagion", "Investigate which structural properties of complex networks most strongly determine the speed and extent of contagion spreading under competing diffusion models."),
    ("network_community", "Does modularity maximization recover planted communities in stochastic block models better than spectral clustering at low edge density?"),
    ("network_percolation", "At what average degree does the giant component emerge in configuration-model graphs with heavy-tailed degree sequences? Compare to Erdős–Rényi thresholds."),
    ("algo_cache", "Which cache replacement policies minimize miss rate on LRU-adversarial and Zipf-like access traces under fixed capacity?"),
    ("algo_scheduling", "Do shortest-remaining-time-first heuristics reduce mean waiting time versus round-robin on heavy-tailed job-size distributions?"),
    ("algo_coloring", "Does greedy graph coloring with smallest-last ordering use fewer colors than random vertex order on geometric random graphs?"),
    ("algo_graph_search", "Is bidirectional BFS faster than unidirectional BFS on sparse social graphs with known diameter under 20?"),
    ("bio_gene_reg", "Does knockdown of hub genes in a published gene-regulatory network disproportionately increase expression variance across targets?"),
    ("bio_ppi", "Are high-betweenness proteins in PPI networks enriched for essentiality compared to degree-matched controls?"),
    ("bio_evolution", "Does selection strength inferred from site-frequency spectra correlate with pathway centrality in metabolic networks?"),
    ("econ_contagion", "Do interbank exposure networks exhibit cascade depth distributions consistent with fractional contagion rather than simple threshold models?"),
    ("econ_network_effects", "Does user adoption growth on referral networks follow logistic dynamics more closely than exponential early growth?"),
    ("econ_auction", "In second-price auctions with correlated private values, does revenue differ systematically from independent-value theory predictions?"),
    ("systems_load_balance", "Does power-of-two-choices assignment reduce tail latency versus uniform random assignment under bursty arrivals?"),
    ("systems_scheduling", "Do weighted fair queueing disciplines bound delay jitter for low-priority flows compared to strict priority?"),
    ("systems_queueing", "Is M/M/1 waiting-time variance well approximated by heavy-traffic limits at utilization above 0.85?"),
    ("math_spectral", "Does algebraic connectivity (Fiedler value) predict synchronization time in Kuramoto models on random geometric graphs?"),
    ("network_assortativity", "Does degree assortativity change the epidemic threshold more than average degree in scale-free networks with γ < 3?"),
    ("algo_approximation", "Does a 2-approximation metric TSP heuristic beat Christofides on random Euclidean instances in dimension 2?"),
    ("systems_failure", "Do cascade failures in power-grid toy models concentrate on bridges in the line graph more than on high-degree buses?"),
]

PHASE2_CONTAGION_QUESTION = SEED_VALIDATION_QUESTIONS[5][1]


@dataclass
class SeedQuestionResult:
    question_id: str
    question: str
    ok: bool
    elapsed_sec: float
    error: str | None = None
    prior_status: str | None = None
    llm_empty_generation: bool = False
    raw_llm_count: int = 0
    surviving: int = 0
    discovery_count: int = 0
    control_count: int = 0
    themes: list[str] = field(default_factory=list)
    hypotheses_preview: list[str] = field(default_factory=list)


@dataclass
class SeedSuiteReport:
    results: list[SeedQuestionResult]
    empty_generation_rate: float
    mean_discovery_survivors: float
    mean_control_ratio: float
    questions_with_non_general_theme: int
    pass_empty_rate: bool
    pass_discovery_count: bool
    pass_control_ratio: bool
    pass_theme_diversity: bool
    passed: bool


class _NullEmitter(EventEmitter):
    """DB-only emitter for offline seed validation (no Redis)."""

    def __init__(self, session_factory: async_sessionmaker) -> None:
        self.session_factory = session_factory

    async def emit(self, **kwargs: Any) -> None:
        return None


def _apply_fast_literature_profile() -> None:
    """Tighter literature for seed checks; avoid embed rate-limit stalls."""
    settings.literature_fetch_per_intent = min(settings.literature_fetch_per_intent, 4)
    settings.literature_max_candidates = min(settings.literature_max_candidates, 12)
    settings.literature_expansion_rounds = 1
    settings.literature_skip_pdf = True
    # Skip question↔paper embedding filter (429-prone); retrieval still runs BM25-only.
    settings.literature_skip_relevance_embed = True


def analyze_hypotheses(hypotheses: list[Any], *, llm_empty: bool, raw_count: int) -> dict[str, Any]:
    discovery = 0
    control = 0
    themes: list[str] = []
    previews: list[str] = []
    for h in hypotheses:
        text = getattr(h, "text", "") or ""
        previews.append(text[:160])
        role = infer_node_role(text)
        if role == "CONTROL":
            control += 1
        else:
            discovery += 1
        primary, _ = extract_theme_vector(text)
        themes.append(primary)
    return {
        "llm_empty_generation": llm_empty,
        "raw_llm_count": raw_count,
        "surviving": len(hypotheses),
        "discovery_count": discovery,
        "control_count": control,
        "themes": themes,
        "hypotheses_preview": previews,
    }


def evaluate_suite(results: list[SeedQuestionResult]) -> SeedSuiteReport:
    ok_results = [r for r in results if r.ok]
    n_ok = max(1, len(ok_results))
    empty_n = sum(1 for r in ok_results if r.llm_empty_generation)
    empty_rate = empty_n / n_ok
    mean_disc = sum(r.discovery_count for r in ok_results) / n_ok
    control_ratios = [
        r.control_count / r.surviving for r in ok_results if r.surviving > 0
    ]
    mean_control = sum(control_ratios) / max(1, len(control_ratios))
    non_general_q = sum(
        1 for r in ok_results if r.themes and any(t != "general" for t in r.themes)
    )

    pass_empty = empty_rate < 0.05
    pass_discovery = len(ok_results) > 0 and all(r.discovery_count >= 3 for r in ok_results)
    pass_control = mean_control <= 0.20
    pass_themes = non_general_q >= max(1, int(0.8 * n_ok))
    pass_all_completed = len(ok_results) == len(results)

    return SeedSuiteReport(
        results=results,
        empty_generation_rate=empty_rate,
        mean_discovery_survivors=mean_disc,
        mean_control_ratio=mean_control,
        questions_with_non_general_theme=non_general_q,
        pass_empty_rate=pass_empty,
        pass_discovery_count=pass_discovery,
        pass_control_ratio=pass_control,
        pass_theme_diversity=pass_themes,
        passed=pass_empty and pass_discovery and pass_control and pass_themes and pass_all_completed,
    )


async def run_seed_pipeline_for_question(
    question_id: str,
    question: str,
    *,
    session_factory: async_sessionmaker,
    llm: LLMClient,
    prior_timeout_sec: int = 150,
    total_timeout_sec: int = 200,
    max_hypotheses: int | None = None,
) -> SeedQuestionResult:
    t0 = time.monotonic()
    session_id = str(uuid.uuid4())
    emitter = _NullEmitter(session_factory)
    max_h = max_hypotheses or int(settings.campaign_batch_size)

    async with session_factory() as session:
        await session.execute(
            text(
                """
                INSERT INTO research_sessions (id, question, status, stage)
                VALUES (CAST(:id AS uuid), :question, 'active', 'seed_validation')
                ON CONFLICT (id) DO NOTHING
                """
            ),
            {"id": session_id, "question": question[:2000]},
        )
        await session.commit()

    try:
        parsed = await parse_question(question)

        async def _prior() -> Prior:
            return await build_prior(
                parsed,
                session_id=session_id,
                emitter=emitter,
                session_factory=session_factory,
                paper_ttl_days=30,
                llm=llm,
            )

        try:
            if getattr(settings, "_seed_validation_skip_literature", False):
                prior = Prior(
                    established_facts=[],
                    contested_claims=[],
                    open_gaps=[{"text": "Literature skipped for seed-only validation.", "source_paper": "bootstrap", "gap_type": "missing_data"}],
                    dead_ends=[],
                    key_papers=[],
                    evidence_status="SEED_ONLY",
                )
                prior_status = "SEED_ONLY"
            else:
                prior = await asyncio.wait_for(_prior(), timeout=prior_timeout_sec)
                prior_status = getattr(prior, "evidence_status", None) or "OK"
        except TimeoutError:
            prior = Prior(
                established_facts=[],
                contested_claims=[],
                open_gaps=[],
                dead_ends=[],
                key_papers=[],
                evidence_status="PRIOR_TIMEOUT",
            )
            prior_status = "PRIOR_TIMEOUT"
        except Exception as exc:
            prior = Prior(
                established_facts=[],
                contested_claims=[],
                open_gaps=[{"text": f"Prior build failed: {exc}"[:200], "source_paper": "bootstrap", "gap_type": "missing_data"}],
                dead_ends=[],
                key_papers=[],
                evidence_status="PRIOR_ERROR",
            )
            prior_status = "PRIOR_ERROR"

        remaining = max(5.0, total_timeout_sec - (time.monotonic() - t0))

        from services.orchestrator.hypotheses import generate_ranked_hypotheses

        meta: dict[str, Any] = {}
        hyps = await asyncio.wait_for(
            generate_ranked_hypotheses(
                parsed,
                prior,
                max_hypotheses=max_h,
                llm=llm,
                session_id=session_id,
                emitter=emitter,
                meta=meta,
                use_llm_ranking=False,
            ),
            timeout=remaining,
        )
        stats = analyze_hypotheses(
            hyps,
            llm_empty=bool(meta.get("llm_empty")),
            raw_count=int(meta.get("raw_llm_count") or 0),
        )
        return SeedQuestionResult(
            question_id=question_id,
            question=question,
            ok=True,
            elapsed_sec=round(time.monotonic() - t0, 2),
            prior_status=prior_status,
            **stats,
        )
    except asyncio.TimeoutError:
        return SeedQuestionResult(
            question_id=question_id,
            question=question,
            ok=False,
            elapsed_sec=round(time.monotonic() - t0, 2),
            error="total_timeout",
        )
    except Exception as exc:
        return SeedQuestionResult(
            question_id=question_id,
            question=question,
            ok=False,
            elapsed_sec=round(time.monotonic() - t0, 2),
            error=str(exc)[:500] or type(exc).__name__,
        )
