"""Lifetime knowledge Postgres persistence (T1-001).

Replaces JSON last-writer-wins with per-entity upserts so concurrent campaign
finalization never loses claims, theories, or numerical seeds.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from propab.config import settings
from propab.knowledge_graph import (
    Claim,
    FailureRecord,
    KnowledgeGraph,
    MechanismRecord,
    ResearchQuestion,
    Theory,
)
from propab.meta_science import CampaignObservation, MetaScienceLedger
from propab.policy_fitness_ledger import FitnessRecord, PolicyFitnessLedger

logger = logging.getLogger(__name__)

_engine: Engine | None = None


def lifetime_postgres_enabled() -> bool:
    return str(getattr(settings, "lifetime_store_backend", "json")).lower() == "postgres"


def _sync_database_url() -> str:
    url = settings.database_url
    if "+asyncpg" in url:
        return url.replace("+asyncpg", "+psycopg")
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(_sync_database_url(), pool_pre_ping=True)
    return _engine


def _json(v: Any) -> str:
    return json.dumps(v, default=str)


# ── KnowledgeGraph ───────────────────────────────────────────────────────────

def upsert_claim(claim: Claim, *, domain: str = "general") -> None:
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO lifetime_knowledge_claims (
                    id, domain, campaign_id, claim_text, claim_type, confidence, claim_payload
                ) VALUES (
                    :id, :domain, CAST(:campaign_id AS uuid), :claim_text, :claim_type, :confidence,
                    CAST(:payload AS jsonb)
                )
                ON CONFLICT (domain, campaign_id, claim_text) DO UPDATE SET
                    confidence = EXCLUDED.confidence,
                    claim_type = EXCLUDED.claim_type,
                    claim_payload = EXCLUDED.claim_payload
            """),
            {
                "id": claim.id,
                "domain": domain,
                "campaign_id": claim.campaign_id or "00000000-0000-0000-0000-000000000000",
                "claim_text": claim.text,
                "claim_type": claim.claim_type,
                "confidence": claim.confidence,
                "payload": _json(claim.to_dict()),
            },
        )


def upsert_theory(theory: Theory, *, domain: str = "general") -> None:
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO lifetime_knowledge_theories (
                    id, domain, theory_text, supporting_campaign_ids, theory_payload
                ) VALUES (
                    :id, :domain, :theory_text, CAST(:campaign_ids AS uuid[]), CAST(:payload AS jsonb)
                )
                ON CONFLICT (id) DO UPDATE SET
                    theory_text = EXCLUDED.theory_text,
                    supporting_campaign_ids = EXCLUDED.supporting_campaign_ids,
                    theory_payload = EXCLUDED.theory_payload,
                    updated_at = NOW()
            """),
            {
                "id": theory.id,
                "domain": domain,
                "theory_text": theory.name,
                "campaign_ids": "{}",
                "payload": _json(theory.to_dict()),
            },
        )


def upsert_numerical_seed(
    *,
    domain: str,
    campaign_id: str,
    seed: dict[str, Any],
) -> None:
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO lifetime_numerical_seeds (
                    domain, campaign_id, finding_type, claim, parameters,
                    source_node_ids, next_hypotheses, seed_payload
                ) VALUES (
                    :domain, CAST(:campaign_id AS uuid), :finding_type, :claim,
                    CAST(:parameters AS jsonb), CAST(:source_node_ids AS uuid[]),
                    CAST(:next_hypotheses AS jsonb), CAST(:seed_payload AS jsonb)
                )
            """),
            {
                "domain": domain,
                "campaign_id": campaign_id,
                "finding_type": seed.get("finding_type"),
                "claim": str(seed.get("claim") or seed.get("text") or ""),
                "parameters": _json(seed.get("parameters") or {}),
                "source_node_ids": "{}",
                "next_hypotheses": _json(seed.get("next_hypotheses") or []),
                "seed_payload": _json(seed),
            },
        )


def save_graph_meta(graph: KnowledgeGraph) -> None:
    """Persist non-claim entities and graph metadata as upserted JSON blobs."""
    eng = get_engine()
    payload = {
        "version": graph.version,
        "mechanisms": {k: v.to_dict() for k, v in graph.mechanisms.items()},
        "failures": {k: v.to_dict() for k, v in graph.failures.items()},
        "questions": {k: v.to_dict() for k, v in graph.questions.items()},
        "links": graph.links,
        "campaign_ids": graph.campaign_ids,
        "diversity_by_domain": graph.diversity_by_domain,
    }
    with eng.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO lifetime_knowledge_meta (meta_key, meta_value)
                VALUES ('graph_aux', CAST(:payload AS jsonb))
                ON CONFLICT (meta_key) DO UPDATE SET
                    meta_value = EXCLUDED.meta_value,
                    updated_at = NOW()
            """),
            {"payload": _json(payload)},
        )


def save_knowledge_graph(graph: KnowledgeGraph, *, domain: str = "general") -> None:
    for claim in graph.claims.values():
        upsert_claim(claim, domain=domain)
    for theory in graph.theories.values():
        upsert_theory(theory, domain=domain)
    for dom, seeds in graph.numerical_seeds.items():
        for seed in seeds:
            cid = str(seed.get("campaign_id") or "00000000-0000-0000-0000-000000000000")
            upsert_numerical_seed(domain=dom, campaign_id=cid, seed=seed)
    save_graph_meta(graph)


def load_knowledge_graph() -> KnowledgeGraph:
    eng = get_engine()
    graph = KnowledgeGraph()
    with eng.connect() as conn:
        claim_rows = conn.execute(text("""
            SELECT claim_payload FROM lifetime_knowledge_claims ORDER BY created_at
        """)).mappings().all()
        for row in claim_rows:
            payload = row["claim_payload"] or {}
            if isinstance(payload, str):
                payload = json.loads(payload)
            cid = payload.get("id")
            if cid:
                graph.claims[cid] = Claim(**{k: v for k, v in payload.items() if k in Claim.__dataclass_fields__})

        theory_rows = conn.execute(text("""
            SELECT theory_payload FROM lifetime_knowledge_theories ORDER BY created_at
        """)).mappings().all()
        for row in theory_rows:
            payload = row["theory_payload"] or {}
            if isinstance(payload, str):
                payload = json.loads(payload)
            tid = payload.get("id")
            if tid:
                graph.theories[tid] = Theory(**{k: v for k, v in payload.items() if k in Theory.__dataclass_fields__})

        seed_rows = conn.execute(text("""
            SELECT domain, seed_payload FROM lifetime_numerical_seeds ORDER BY created_at
        """)).mappings().all()
        for row in seed_rows:
            dom = row["domain"]
            payload = row["seed_payload"] or {}
            if isinstance(payload, str):
                payload = json.loads(payload)
            graph.numerical_seeds.setdefault(dom, []).append(payload)

        meta_row = conn.execute(text("""
            SELECT meta_value FROM lifetime_knowledge_meta WHERE meta_key = 'graph_aux'
        """)).mappings().one_or_none()
        if meta_row:
            aux = meta_row["meta_value"] or {}
            if isinstance(aux, str):
                aux = json.loads(aux)
            graph.version = int(aux.get("version") or graph.version)
            for k, v in (aux.get("mechanisms") or {}).items():
                graph.mechanisms[k] = MechanismRecord(**v)
            for k, v in (aux.get("failures") or {}).items():
                graph.failures[k] = FailureRecord(**v)
            for k, v in (aux.get("questions") or {}).items():
                graph.questions[k] = ResearchQuestion(**v)
            graph.links = list(aux.get("links") or [])
            graph.campaign_ids = list(aux.get("campaign_ids") or [])
            graph.diversity_by_domain = dict(aux.get("diversity_by_domain") or {})
    return graph


# ── MetaScienceLedger ────────────────────────────────────────────────────────

def save_meta_ledger(meta: MetaScienceLedger) -> None:
    eng = get_engine()
    with eng.begin() as conn:
        for obs in meta.observations:
            conn.execute(
                text("""
                    INSERT INTO lifetime_meta_observations (campaign_id, observation_payload)
                    VALUES (CAST(:campaign_id AS uuid), CAST(:payload AS jsonb))
                    ON CONFLICT (campaign_id) DO UPDATE SET
                        observation_payload = EXCLUDED.observation_payload,
                        updated_at = NOW()
                """),
                {"campaign_id": obs.campaign_id, "payload": _json(obs.to_dict())},
            )


def load_meta_ledger() -> MetaScienceLedger:
    eng = get_engine()
    obs: list[CampaignObservation] = []
    with eng.connect() as conn:
        rows = conn.execute(text("""
            SELECT observation_payload FROM lifetime_meta_observations ORDER BY updated_at
        """)).mappings().all()
        for row in rows:
            payload = row["observation_payload"] or {}
            if isinstance(payload, str):
                payload = json.loads(payload)
            fields = {k: v for k, v in payload.items() if k in CampaignObservation.__dataclass_fields__}
            obs.append(CampaignObservation(**fields))
    return MetaScienceLedger(observations=obs)


# ── PolicyFitnessLedger ──────────────────────────────────────────────────────

def save_fitness_ledger(fitness: PolicyFitnessLedger) -> None:
    eng = get_engine()
    with eng.begin() as conn:
        for rec in fitness.records:
            conn.execute(
                text("""
                    INSERT INTO lifetime_policy_fitness (policy_id, campaign_id, record_payload)
                    VALUES (:policy_id, CAST(:campaign_id AS uuid), CAST(:payload AS jsonb))
                    ON CONFLICT (policy_id, campaign_id) DO UPDATE SET
                        record_payload = EXCLUDED.record_payload,
                        updated_at = NOW()
                """),
                {
                    "policy_id": rec.policy_id,
                    "campaign_id": rec.campaign_id,
                    "payload": _json(rec.to_dict()),
                },
            )


def load_fitness_ledger() -> PolicyFitnessLedger:
    eng = get_engine()
    recs: list[FitnessRecord] = []
    with eng.connect() as conn:
        rows = conn.execute(text("""
            SELECT record_payload FROM lifetime_policy_fitness ORDER BY updated_at
        """)).mappings().all()
        for row in rows:
            payload = row["record_payload"] or {}
            if isinstance(payload, str):
                payload = json.loads(payload)
            fields = {k: v for k, v in payload.items() if k in FitnessRecord.__dataclass_fields__}
            recs.append(FitnessRecord(**fields))
    return PolicyFitnessLedger(records=recs)


# ── PolicyStore ──────────────────────────────────────────────────────────────

def save_policy_store(store: Any) -> None:
    """Persist full policy store as a singleton meta row (upsert, not LWW file)."""
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO lifetime_knowledge_meta (meta_key, meta_value)
                VALUES ('policy_store', CAST(:payload AS jsonb))
                ON CONFLICT (meta_key) DO UPDATE SET
                    meta_value = EXCLUDED.meta_value,
                    updated_at = NOW()
            """),
            {"payload": _json(store.to_dict())},
        )


def load_policy_store_cls() -> Any:
    from propab.policy_store import PolicyStore

    eng = get_engine()
    with eng.connect() as conn:
        row = conn.execute(text("""
            SELECT meta_value FROM lifetime_knowledge_meta WHERE meta_key = 'policy_store'
        """)).mappings().one_or_none()
    if row is None:
        store = PolicyStore()
        store._migrate_legacy_search_policy()
        return store
    payload = row["meta_value"] or {}
    if isinstance(payload, str):
        payload = json.loads(payload)
    return PolicyStore.from_dict(payload)
