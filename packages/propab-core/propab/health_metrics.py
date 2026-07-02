"""
Campaign health-metric logging (ownership-contracts observability).

Every component in ``propab_ownership_contracts.md`` has one health metric — the
number that tells you whether it is doing its job. This module computes those
metrics from data already produced by the pipeline and persists them to Postgres,
so debugging starts with "which component's number is out of range" instead of
"Propab failed".

Metrics and their tables (see the contracts doc):

| Metric                              | When            | Table                       |
|-------------------------------------|-----------------|-----------------------------|
| Hypothesis duplicate rate           | per synth round | campaign_synthesis_events   |
| Evidence-binding rejection rate     | per synth round | campaign_synthesis_events   |
| Belief citation integrity           | per synth round | campaign_synthesis_events   |
| Belief stability                    | per synth round | campaign_synthesis_events   |
| Literature citation verification    | per prior build | campaign_literature_priors  |
| Worker experiment success rate      | per campaign    | research_campaigns          |
| Worker utilization                  | per campaign    | research_campaigns          |
| Verification artifact-gate precision| per campaign    | campaign_audit_results      |

This module lives in core (no service imports) and never raises into the caller:
metric logging must never break a running campaign.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

logger = logging.getLogger(__name__)

# Target ranges from the ownership contracts (used only to emit warnings).
_DUPLICATE_RATE_WARN = 0.20            # generator recycling above this
_BINDING_REJECTION_WARN = 0.15         # synthesis attaching irrelevant citations
_BINDING_ZERO_MIN_CALLS = 50           # 0% rejection over this many calls is suspicious
_EXPERIMENT_SUCCESS_WARN = 0.50        # experiments failing before producing evidence
_UTILIZATION_WARN = 0.60               # workers idle / synthesis too slow

# Inconclusive reasons that indicate an execution failure (not a genuine null result).
_EXECUTION_FAILURE_REASONS = (
    "timeout", "tool_error", "tool error", "sandbox", "crash", "error",
    "exception", "failed", "no_output", "no output",
)


def _rate(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _norm_statement(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


# ── Per synthesis round ──────────────────────────────────────────────────────

async def log_synthesis_health(
    session_factory: async_sessionmaker,
    *,
    campaign_id: str,
    generation: int,
    metrics: dict[str, Any],
    active_belief_statements: list[str],
) -> dict[str, Any]:
    """Compute + persist the four per-round metrics; return the computed rates.

    ``belief_stability`` is computed against the previous round's persisted
    statements (Postgres holds the round history), so no in-memory tracking is
    needed and it is correct across restarts. Never raises.
    """
    raw = int(metrics.get("n_candidates_raw") or 0)
    dup = int(metrics.get("n_rejected_duplicate") or 0)
    rej = int(metrics.get("binding_rejected_count") or 0)
    acc = int(metrics.get("binding_accepted_count") or 0)

    duplicate_rate = _rate(dup, raw)
    binding_total = rej + acc
    binding_rejection_rate = _rate(rej, binding_total)
    citation_integrity = _rate(acc, binding_total)

    stmts = [_norm_statement(s) for s in (active_belief_statements or []) if str(s).strip()]
    belief_stability: float | None = None

    try:
        async with session_factory() as db:
            prev = (await db.execute(
                text("""
                    SELECT active_belief_statements
                    FROM campaign_synthesis_events
                    WHERE campaign_id = CAST(:cid AS uuid)
                    ORDER BY created_at DESC
                    LIMIT 1
                """),
                {"cid": campaign_id},
            )).scalar_one_or_none()

            prev_stmts: list[str] = []
            if prev:
                loaded = prev if isinstance(prev, list) else json.loads(prev)
                prev_stmts = [_norm_statement(s) for s in loaded]

            if prev_stmts and stmts:
                kept = sum(1 for s in stmts if s in set(prev_stmts))
                belief_stability = kept / len(stmts)
            elif not stmts:
                belief_stability = None

            await db.execute(
                text("""
                    INSERT INTO campaign_synthesis_events (
                        campaign_id, generation, n_candidates_raw, n_added,
                        n_rejected_duplicate, binding_rejected_count, binding_accepted_count,
                        falsifiability_rejected_count, belief_cap_rejected_count,
                        hypothesis_duplicate_rate, evidence_binding_rejection_rate,
                        belief_citation_integrity, belief_stability, active_belief_statements
                    ) VALUES (
                        CAST(:cid AS uuid), :gen, :raw, :added, :dup, :rej, :acc,
                        :fals, :cap, :dup_rate, :bind_rate, :integrity, :stability,
                        CAST(:stmts AS jsonb)
                    )
                """),
                {
                    "cid": campaign_id,
                    "gen": int(generation),
                    "raw": raw,
                    "added": int(metrics.get("n_added") or 0),
                    "dup": dup,
                    "rej": rej,
                    "acc": acc,
                    "fals": int(metrics.get("falsifiability_rejected_count") or 0),
                    "cap": int(metrics.get("belief_cap_rejected_count") or 0),
                    "dup_rate": duplicate_rate,
                    "bind_rate": binding_rejection_rate,
                    "integrity": citation_integrity,
                    "stability": belief_stability,
                    "stmts": json.dumps(active_belief_statements or []),
                },
            )
            await db.commit()

            # Suspicious-zero check: binding never rejecting over many calls may
            # mean the check is not actually running (ownership-contracts rule).
            totals = (await db.execute(
                text("""
                    SELECT COALESCE(SUM(binding_rejected_count), 0),
                           COALESCE(SUM(binding_accepted_count), 0)
                    FROM campaign_synthesis_events
                    WHERE campaign_id = CAST(:cid AS uuid)
                """),
                {"cid": campaign_id},
            )).one()
            tot_rej, tot_acc = int(totals[0]), int(totals[1])
            if (tot_rej + tot_acc) >= _BINDING_ZERO_MIN_CALLS and tot_rej == 0:
                logger.warning(
                    "[campaign %s] Evidence Binding called %d times with 0 rejections — "
                    "binding check may not be running.", campaign_id, tot_rej + tot_acc,
                )
    except Exception:  # noqa: BLE001 — metric logging must never break a campaign
        logger.exception("[campaign %s] log_synthesis_health failed", campaign_id)
        return {}

    if duplicate_rate is not None and duplicate_rate > _DUPLICATE_RATE_WARN:
        logger.warning("[campaign %s] hypothesis duplicate rate %.0f%% (>%.0f%% target)",
                       campaign_id, duplicate_rate * 100, _DUPLICATE_RATE_WARN * 100)
    if binding_rejection_rate is not None and binding_rejection_rate > _BINDING_REJECTION_WARN:
        logger.warning("[campaign %s] evidence-binding rejection rate %.0f%% (>%.0f%% target)",
                       campaign_id, binding_rejection_rate * 100, _BINDING_REJECTION_WARN * 100)

    return {
        "hypothesis_duplicate_rate": duplicate_rate,
        "evidence_binding_rejection_rate": binding_rejection_rate,
        "belief_citation_integrity": citation_integrity,
        "belief_stability": belief_stability,
    }


# ── Per literature prior build ───────────────────────────────────────────────

_CITATION_KEYS = ("citation", "citations", "doi", "paper", "papers", "source", "reference", "arxiv", "url")


def count_established_verified(established_facts: Any) -> tuple[int, int]:
    """(established_facts_count, verified_citation_count).

    "Verified" here means the fact carries a concrete, retrievable citation
    (a proxy for the contract's citation-verification rate — a fact with no
    citation cannot be verified and must not count as verified). Handles both
    dict-shaped and object-shaped facts defensively.
    """
    facts = list(established_facts or [])
    verified = 0
    for f in facts:
        get = f.get if isinstance(f, dict) else (lambda k, _f=f: getattr(_f, k, None))
        if any(get(k) for k in _CITATION_KEYS):
            verified += 1
    return len(facts), verified


async def log_literature_prior_health(
    session_factory: async_sessionmaker,
    *,
    campaign_id: str,
    established_facts_count: int,
    verified_citation_count: int,
) -> float | None:
    """Persist the literature citation verification rate. Never raises."""
    rate = _rate(verified_citation_count, established_facts_count)
    try:
        async with session_factory() as db:
            await db.execute(
                text("""
                    INSERT INTO campaign_literature_priors (
                        campaign_id, citation_verification_rate,
                        established_facts_count, verified_citation_count
                    ) VALUES (CAST(:cid AS uuid), :rate, :facts, :verified)
                """),
                {
                    "cid": campaign_id,
                    "rate": rate,
                    "facts": int(established_facts_count),
                    "verified": int(verified_citation_count),
                },
            )
            await db.commit()
    except Exception:  # noqa: BLE001
        logger.exception("[campaign %s] log_literature_prior_health failed", campaign_id)
    return rate


# ── Per campaign (end) ───────────────────────────────────────────────────────

def compute_experiment_success_rate(tree: Any) -> tuple[float | None, dict[str, int]]:
    """Fraction of tested hypotheses that reached a definitive verdict.

    Definitive = confirmed or refuted. The denominator is all non-pending nodes;
    genuine inconclusive results and execution-failure inconclusive results are
    both counted as non-definitive, and execution failures are reported separately.
    """
    definitive = 0
    tested = 0
    execution_failures = 0
    for node in getattr(tree, "nodes", {}).values():
        verdict = getattr(node, "verdict", "pending")
        if verdict == "pending":
            continue
        tested += 1
        if verdict in ("confirmed", "refuted"):
            definitive += 1
        elif verdict == "inconclusive":
            reason = str(getattr(node, "inconclusive_reason", "") or "").lower()
            if any(tok in reason for tok in _EXECUTION_FAILURE_REASONS):
                execution_failures += 1
    counts = {"definitive": definitive, "tested": tested, "execution_failures": execution_failures}
    return _rate(definitive, tested), counts


def compute_worker_utilization(
    total_agent_seconds: float,
    elapsed_seconds: float,
    max_concurrency: int,
) -> float | None:
    """Fraction of available worker-seconds that were spent running experiments.

    ``total_agent_seconds`` is the summed wall-clock of every dispatched sub-agent;
    capacity is ``elapsed_seconds * max_concurrency``. Capped at 1.0. This is a
    real utilization estimate for the whole campaign (prior/baseline phases where
    no experiments run legitimately lower it)."""
    capacity = float(elapsed_seconds) * max(1, int(max_concurrency))
    if capacity <= 0:
        return None
    return min(1.0, float(total_agent_seconds) / capacity)


async def log_campaign_end_health(
    session_factory: async_sessionmaker,
    *,
    campaign: Any,
    total_agent_seconds: float,
    max_concurrency: int,
) -> dict[str, Any]:
    """Persist per-campaign worker metrics onto research_campaigns. Never raises."""
    success_rate, counts = compute_experiment_success_rate(campaign.hypothesis_tree)
    utilization = compute_worker_utilization(
        total_agent_seconds, campaign.elapsed_seconds(), max_concurrency,
    )
    try:
        async with session_factory() as db:
            await db.execute(
                text("""
                    UPDATE research_campaigns
                    SET worker_experiment_success_rate = :succ,
                        worker_utilization = :util
                    WHERE id = CAST(:cid AS uuid)
                """),
                {"cid": campaign.id, "succ": success_rate, "util": utilization},
            )
            await db.commit()
    except Exception:  # noqa: BLE001
        logger.exception("[campaign %s] log_campaign_end_health failed", campaign.id)

    if success_rate is not None and success_rate < _EXPERIMENT_SUCCESS_WARN:
        logger.warning("[campaign %s] experiment success rate %.0f%% (<%.0f%% target) — %r",
                       campaign.id, success_rate * 100, _EXPERIMENT_SUCCESS_WARN * 100, counts)
    if utilization is not None and utilization < _UTILIZATION_WARN:
        logger.warning("[campaign %s] worker utilization %.0f%% (<%.0f%% target)",
                       campaign.id, utilization * 100, _UTILIZATION_WARN * 100)

    return {
        "worker_experiment_success_rate": success_rate,
        "worker_utilization": utilization,
        **counts,
    }


# Signals in a confirmed finding's evidence that an independent null/permutation
# test backed the confirmation (label-shuffle null, permutation p95, etc.).
_NULL_TEST_SIGNALS = (
    "label_shuffle", "label-shuffle", "permutation", "null_p95", "null p95",
    "beats null", "shuffle_null", "null distribution", "p95",
)


def compute_confirmed_audit_counts(tree: Any) -> tuple[int, int]:
    """(confirmed_findings_count, survived_audit_count) for the artifact-gate precision.

    Survived = confirmed findings whose recorded evidence shows an independent
    null/permutation test backed the confirmation. This is the in-pipeline audit
    signal; the offline permutation-audit tooling can overwrite with a full re-run.
    """
    confirmed = 0
    survived = 0
    for node in getattr(tree, "nodes", {}).values():
        if getattr(node, "verdict", "") != "confirmed":
            continue
        confirmed += 1
        blob = " ".join(
            str(x or "").lower()
            for x in (
                getattr(node, "evidence_summary", ""),
                json.dumps(getattr(node, "finding", None) or {}),
                getattr(node, "verification_method", ""),
                getattr(node, "replication_level", ""),
            )
        )
        if any(sig in blob for sig in _NULL_TEST_SIGNALS):
            survived += 1
    return confirmed, survived


async def log_campaign_audit(
    session_factory: async_sessionmaker,
    *,
    campaign_id: str,
    confirmed_findings_count: int,
    survived_audit_count: int,
) -> float | None:
    """Persist the verification artifact-gate precision (post-campaign audit). Never raises."""
    precision = _rate(survived_audit_count, confirmed_findings_count)
    try:
        async with session_factory() as db:
            await db.execute(
                text("""
                    INSERT INTO campaign_audit_results (
                        campaign_id, artifact_gate_precision,
                        confirmed_findings_count, survived_audit_count
                    ) VALUES (CAST(:cid AS uuid), :prec, :confirmed, :survived)
                """),
                {
                    "cid": campaign_id,
                    "prec": precision,
                    "confirmed": int(confirmed_findings_count),
                    "survived": int(survived_audit_count),
                },
            )
            await db.commit()
    except Exception:  # noqa: BLE001
        logger.exception("[campaign %s] log_campaign_audit failed", campaign_id)
    return precision
