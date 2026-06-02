from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    propab_profile: str = "dev"
    database_url: str = "postgresql+asyncpg://propab:propab@localhost:5432/propab"
    redis_url: str = "redis://localhost:6379/0"
    openai_api_key: str = ""
    google_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        description="Google AI Studio / Gemini API key when LLM_PROVIDER=gemini",
    )
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    ollama_base_url: str = "http://127.0.0.1:11434"
    orchestrator_url: str = ""
    orchestrator_internal_token: str = ""
    sub_agent_plan_source: str = "llm"
    sub_agent_max_planned_steps: int = 6
    sub_agent_max_rounds: int = 4
    sub_agent_tools_per_round: int = 4
    # Paper: substantive (default) requires confirmed/refuted ledger or metric-like / deep tool trace.
    # strict_confirmed = only confirmed hypotheses; always = prior behavior (any non-empty trace).
    paper_policy: str = "substantive"
    paper_min_substantive_tools: int = 2
    embed_provider: str = "openai"
    embed_model: str = "text-embedding-3-small"
    qdrant_url: str = ""
    qdrant_collection: str = "propab_chunks"
    propab_data_dir: str = "./data"
    minio_endpoint: str = ""
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "propab"
    minio_secure: bool = False
    sandbox_timeout_sec: int = 120
    sandbox_memory_mb: int = 512
    # Image for the code-execution sandbox. Must contain the scientific Python stack the
    # agents rely on (numpy/scipy/torch). Defaults to the worker image so free-form code
    # has the same instruments as the tools; override with SANDBOX_IMAGE. Network is always
    # disabled in the sandbox, so code must not download datasets — use tools for that.
    sandbox_image: str = "propab-oss-worker:latest"
    # Max Docker executions per ``code.generated`` step (minimum 1). When 1, no second
    # run after timeout rewrite. See ``services.worker.sub_agent_loop.run_code_step``.
    sandbox_code_max_retries: int = 3
    sandbox_use_domain_timeout_floor: bool = True
    # After a sandbox wall-timeout on heavy code, one LLM rewrite + one extra execution (if LLM available).
    sandbox_after_timeout_llm_rewrite: bool = True
    agent_tool_n_steps_cap: int = 0
    literature_answer_similarity: float = 0.92
    # Literature retrieval (domain-agnostic): query-scoped cache, multi-intent fetch, quality gates.
    literature_query_cache_ttl_days: int = 14
    literature_query_cache_similarity: float = 0.82
    literature_fetch_per_intent: int = 10
    literature_max_candidates: int = 40
    literature_relevance_threshold: float = 0.4
    literature_relevance_threshold_floor: float = 0.28
    literature_citation_expand_max: int = 12
    literature_min_papers_kept: int = 2
    literature_min_retrieval_chunks: int = 3
    literature_min_evidence_coverage: float = 0.35
    literature_expansion_rounds: int = 2
    literature_pdf_parallelism: int = 4
    literature_skip_pdf: bool = False
    literature_skip_relevance_embed: bool = False
    literature_arxiv_min_interval_sec: float = 6.0
    literature_arxiv_max_retries: int = 5
    # Think-act agent budgets
    agent_max_steps: int = 12
    agent_min_steps: int = 4
    agent_max_seconds: int = 600
    # Hard cap on tool invocations per sub-agent hypothesis (0 = unlimited). Independent of agent_max_seconds.
    agent_max_tool_calls: int = 40
    # httpx read/connect timeout for OpenAI / external LLM HTTP (seconds)
    llm_http_timeout_sec: float = 180.0
    # Transient-error resilience for LLM calls (timeouts, 429, 5xx). Retries use
    # exponential backoff: base * 2**attempt. Keeps long campaigns alive.
    llm_max_retries: int = 3
    llm_retry_base_delay_sec: float = 2.0
    max_code_steps_per_hypothesis: int = 1
    n_steps_default: int = 150
    classification_default_dataset: str = "mnist"
    # Verification bar: a hypothesis may only be "confirmed" if its supporting metric was observed
    # in at least this many independent metric-bearing steps (replication/reproduction guard).
    # Set to 1 to allow single-shot confirmations. Domain-agnostic.
    min_metric_steps_for_confirm: int = 2
    # Multi-round orchestrator
    research_max_rounds: int = 4
    research_max_hours: float = 3.5
    research_max_seconds_per_round: int = 3000
    research_target_confirmed: int = 5
    research_max_hypotheses: int = 60
    research_min_marginal_return: float = 0.03
    research_max_stale_rounds: int = 3
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # Campaign mode — long-running research over days
    campaign_compute_budget_seconds: int = 14400    # 4 hours default
    campaign_max_hypotheses: int = 500              # hard cap on tree size
    campaign_batch_size: int = 5                    # hypotheses per campaign round
    campaign_checkpoint_every: int = 300            # checkpoint interval (seconds)
    campaign_breakthrough_threshold: float = 0.05   # 5% improvement to declare success
    campaign_min_confidence: float = 0.85           # min confidence for breakthrough
    campaign_min_replications: int = 3              # min confirmed replications
    campaign_expand_on_confirmed: bool = True       # expand tree on confirmed findings
    campaign_expand_on_refuted: bool = True         # generate alternatives on refuted
    campaign_expand_on_inconclusive: bool = True    # refine/decompose inconclusive (fixes.md P1.2)
    campaign_expansion_merit_novelty_min: float = 0.25
    campaign_expansion_merit_info_gain_min: float = 0.30
    campaign_theme_saturation_penalty: float = 0.15
    campaign_max_inconclusive_expansions: int = 2
    hypothesis_relevance_threshold: float = 0.35    # question↔hypothesis gate (fixes.md P0.3)
    # Max seconds to wait for in-flight (non-blocking) tree expansions to refill the
    # frontier before falling back to a blocking seed regeneration.
    campaign_expansion_drain_sec: int = 90
    campaign_max_tree_depth: int = 8                # max hypothesis depth in tree
    campaign_prior_timeout_sec: int = 180
    # If a batch has unfinished Celery sub-agents and no completions for this many seconds, revoke the oldest.
    campaign_frontier_evict_idle_sec: int = 900       # 0 disables eviction
    campaign_batch_max_wait_sec: int = 0
    # Celery ar.get timeout for the baseline sub-agent (full think-act trace — often 10–20+ min).
    campaign_baseline_worker_timeout_sec: int = 600
    # Cap train_model n_steps for baseline_measurement fallback (lower = faster baseline).
    campaign_baseline_max_train_steps: int = 150
    campaign_baseline_mode: str = "sub_agent"
    # Tighter think-act caps for the baseline-measurement sub-agent only (full agents use agent_max_*).
    campaign_baseline_agent_max_steps: int = 6
    campaign_baseline_agent_max_seconds: int = 480
    campaign_baseline_agent_max_tool_calls: int = 14
    # If >0, revoke a still-running Celery sub-agent after this many seconds since dispatch
    # (independent of sandbox completion — kills stuck timeout/LLM loops). 0 = disabled.
    campaign_sub_agent_max_wall_sec: int = 0
    # Pull next_batch(size = campaign_batch_size * multiplier) for more parallel Celery workers per wave.
    campaign_inflight_multiplier: int = 1
    # 0 = derive from campaign_batch_size * campaign_inflight_multiplier (pipelined dispatch).
    campaign_max_concurrent_sub_agents: int = 0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def llm_api_secret(self) -> str:
        """API secret passed into ``LLMClient`` (OpenAI, Gemini, or empty for Ollama)."""
        p = (self.llm_provider or "").strip().lower()
        if p == "gemini":
            return (self.google_api_key or "").strip()
        if p == "ollama":
            return ""
        return (self.openai_api_key or "").strip()

    @property
    def embed_api_secret(self) -> str:
        """API secret for embedding provider."""
        p = (self.embed_provider or "").strip().lower()
        if p in {"google", "gemini"}:
            return (self.google_api_key or "").strip()
        return (self.openai_api_key or "").strip()


settings = Settings()


def _apply_profile(s: Settings) -> None:
    profile = (s.propab_profile or "dev").strip().lower()
    # Legacy names → dev (single fast-debug profile).
    if profile in {"research", "deep", "campaign_dev"}:
        profile = "dev"

    profiles: dict[str, dict[str, float | int | str | bool]] = {
        # Fast local / CI debugging: short sandbox, heuristic sub-agent, small campaigns,
        # fast_tool baseline. Use ``propab agent`` + ``propab health`` (see README).
        "dev": {
            "sub_agent_plan_source": "heuristic",
            "sub_agent_max_planned_steps": 4,
            "sub_agent_max_rounds": 2,
            "sub_agent_tools_per_round": 3,
            "research_max_rounds": 2,
            "research_max_hypotheses": 12,
            "research_max_hours": 1.0,
            "research_max_seconds_per_round": 400,
            "agent_max_steps": 8,
            "agent_min_steps": 2,
            "agent_max_seconds": 240,
            "agent_max_tool_calls": 20,
            "agent_tool_n_steps_cap": 80,
            "sandbox_timeout_sec": 90,
            "sandbox_use_domain_timeout_floor": False,
            "sandbox_code_max_retries": 1,
            "sandbox_after_timeout_llm_rewrite": False,
            "n_steps_default": 80,
            "max_code_steps_per_hypothesis": 1,
            "classification_default_dataset": "mnist",
            "llm_http_timeout_sec": 120.0,
            "campaign_compute_budget_seconds": 1800,
            "campaign_batch_size": 2,
            "campaign_max_hypotheses": 24,
            "campaign_checkpoint_every": 60,
            "campaign_expand_on_confirmed": False,
            "campaign_expand_on_refuted": False,
            "campaign_expand_on_inconclusive": False,
            "campaign_frontier_evict_idle_sec": 45,
            "campaign_batch_max_wait_sec": 300,
            "campaign_prior_timeout_sec": 45,
            "campaign_baseline_mode": "fast_tool",
            "campaign_baseline_worker_timeout_sec": 120,
            "campaign_baseline_max_train_steps": 50,
            "campaign_min_replications": 1,
            "campaign_sub_agent_max_wall_sec": 600,
        },
        # Production campaign runs (e.g. Campaign v1): full budgets, LLM sub-agent, sub_agent baseline.
        "campaign": {
            "research_max_rounds": 50,
            "research_max_hypotheses": 500,
            "research_max_hours": 4.0,
            "research_max_seconds_per_round": 1800,
            "agent_max_steps": 14,
            "agent_min_steps": 5,
            # Must exceed worst-case sandbox wait (one Docker run + optional rewrite run); see sub_agent_loop.
            "agent_max_seconds": 1800,
            # Bounded wall per Docker run; domain floor is off so we do not inherit 480s floors.
            "sandbox_timeout_sec": 240,
            "sandbox_use_domain_timeout_floor": False,
            "sandbox_code_max_retries": 1,
            # One Docker run per generated block by default (no second run after rewrite).
            "sandbox_after_timeout_llm_rewrite": False,
            # train_model defaults must fit wall clock; agent_tool_n_steps_cap clamps LLM params.
            "n_steps_default": 120,
            "agent_tool_n_steps_cap": 96,
            "max_code_steps_per_hypothesis": 2,
            "classification_default_dataset": "mnist",
            "llm_http_timeout_sec": 300.0,
            "campaign_compute_budget_seconds": 14400,
            "campaign_batch_size": 5,
            "campaign_inflight_multiplier": 2,
            "campaign_max_concurrent_sub_agents": 0,
            "campaign_max_hypotheses": 500,
            "campaign_breakthrough_threshold": 0.05,
            "campaign_min_confidence": 0.85,
            "campaign_min_replications": 3,
            "campaign_frontier_evict_idle_sec": 420,
            "campaign_baseline_mode": "fast_tool",
            "campaign_baseline_worker_timeout_sec": 180,
            "campaign_baseline_max_train_steps": 50,
            "campaign_sub_agent_max_wall_sec": 2700,
            "campaign_baseline_agent_max_steps": 6,
            "campaign_baseline_agent_max_seconds": 420,
            "campaign_baseline_agent_max_tool_calls": 14,
            "agent_max_tool_calls": 20,
            "campaign_prior_timeout_sec": 900,
            "literature_skip_pdf": True,
            "literature_max_candidates": 24,
            "literature_fetch_per_intent": 6,
            "literature_expansion_rounds": 1,
        },
    }
    selected = profiles.get(profile, profiles["dev"])
    for key, value in selected.items():
        setattr(s, key, value)


_apply_profile(settings)
