"""
Settings for the literature service. Standalone — does not import propab-core.
Every external dependency (Postgres, Qdrant, OpenAI) is optional: missing
config degrades to an in-memory / offline fallback rather than failing to
start, so the service is trivially runnable in dev and CI.
"""
from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "propab-literature"
    api_host: str = "0.0.0.0"
    api_port: int = 8020

    # Storage backends. "memory" needs nothing running and is the default so
    # the service works out of the box; "postgres"/"qdrant" are used in prod.
    postgres_backend: str = "memory"
    database_url: str = "postgresql://propab:propab@localhost:5432/propab_literature"
    qdrant_backend: str = "memory"
    qdrant_url: str = ""
    qdrant_collection: str = "literature_claims"

    # Embeddings. Default matches the rest of the Propab codebase (Gemini via
    # EMBED_PROVIDER=google in the root .env) so this service shares the same
    # embedding quality/cost profile rather than silently using a weaker one.
    # Falls back to a deterministic offline hashing embedding when no API key
    # is configured, so semantic dedup/retrieval still work (with lower
    # quality) in environments with no network access.
    embed_provider: str = "google"
    embed_model: str = "gemini-embedding-2"
    openai_api_key: str = ""
    # Unprefixed (no LITERATURE_ prefix) so this picks up the same
    # GOOGLE_API_KEY/GEMINI_API_KEY already set for the rest of Propab —
    # mirrors propab.config.Settings.google_api_key exactly.
    google_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    # gemini-embedding-2 returns 3072-dim vectors (measured live); only used
    # for Qdrant collection sizing (qdrant_backend=qdrant) — the in-memory
    # store (default) doesn't care.
    embed_dim: int = 3072

    # LLM used only by evaluator/litqa2_live.py to answer retrieved-evidence
    # questions (the core /prior, /novelty, /gaps pipeline has no LLM step —
    # this is eval-only). Matches the root .env's LLM_PROVIDER/LLM_MODEL so
    # this shares the same model as the rest of Propab rather than a
    # different one nobody asked for.
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3-flash-preview"
    llm_http_timeout_sec: float = 120.0
    # Frontier reasoning model used ONLY for the final answer step in the
    # LitQA2 eval. Cheap sub-tasks (query reformulation, sufficiency judging)
    # stay on the flash ``llm_model``. Rationale (measured): correct-when-
    # answered was capped at ~0.61 with flash answering — the AstaBench
    # LitQA2 baseline is Claude Opus 4.7 at xhigh reasoning, and you cannot
    # beat a frontier-reasoning baseline with a flash model regardless of
    # retrieval quality. gemini-3.1-pro-preview is the same weight class.
    answer_model: str = "gemini-3.1-pro-preview"

    # Cache directory for LaTeX source, extraction output, embeddings — keyed
    # by arxiv_id / doi. Never re-fetched once cached (see arxiv.py Step 7).
    cache_dir: str = "./data/literature_cache"

    # HTTP behavior against public APIs — polite defaults per each API's
    # published rate-limit guidance.
    http_timeout_sec: float = 30.0
    arxiv_min_interval_sec: float = 3.0
    oeis_min_interval_sec: float = 1.0
    semantic_scholar_min_interval_sec: float = 1.0
    stackexchange_min_interval_sec: float = 1.0
    zbmath_min_interval_sec: float = 1.0
    pubmed_min_interval_sec: float = 0.4
    biorxiv_min_interval_sec: float = 1.0
    europepmc_min_interval_sec: float = 0.5
    crossref_min_interval_sec: float = 1.0
    # Both unprefixed (no LITERATURE_ prefix) so they pick up the same keys
    # set for the rest of Propab rather than needing their own copies —
    # same reasoning as google_api_key above. Confirmed live: the public
    # (keyless) Semantic Scholar tier 429s under any real usage; a key
    # removes that rate limit almost entirely.
    semantic_scholar_api_key: str = Field(default="", validation_alias=AliasChoices("SEMANTIC_SCHOLAR_API_KEY"))
    ncbi_api_key: str = Field(default="", validation_alias=AliasChoices("NCBI_API_KEY"))
    user_agent: str = "propab-literature-service/0.1 (research; contact: oss@propab.dev)"

    arxiv_max_results_per_query: int = 200
    citation_depth_levels: int = 2
    semantic_scholar_max_citations_per_seed: int = 1000

    # Novelty check thresholds (agent3.md retriever spec).
    novelty_similarity_floor: float = 0.70
    novelty_top_k: int = 30
    novelty_confidence_verdict_floor: float = 0.85
    dedup_similarity_threshold: float = 0.95

    # Latency budgets, enforced as soft deadlines in pipeline.py.
    depth_timeout_sec: dict[str, int] = Field(
        default_factory=lambda: {"standard": 60, "deep": 300, "exhaustive": 900}
    )

    artifacts_dir: str = "./artifacts"

    model_config = SettingsConfigDict(
        env_prefix="LITERATURE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
