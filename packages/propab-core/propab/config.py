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
    sandbox_code_max_retries: int = 3
    literature_answer_similarity: float = 0.92
    # Think-act agent budgets
    agent_max_steps: int = 12
    agent_min_steps: int = 4
    agent_max_seconds: int = 600
    # Hard cap on tool invocations per sub-agent hypothesis (0 = unlimited). Independent of agent_max_seconds.
    agent_max_tool_calls: int = 40
    # httpx read/connect timeout for OpenAI / external LLM HTTP (seconds)
    llm_http_timeout_sec: float = 180.0
    max_code_steps_per_hypothesis: int = 1
    n_steps_default: int = 150
    classification_default_dataset: str = "mnist"
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
    campaign_max_tree_depth: int = 8                # max hypothesis depth in tree
    # If a batch has unfinished Celery sub-agents and no completions for this many seconds, revoke the oldest.
    campaign_frontier_evict_idle_sec: int = 900       # 0 disables eviction

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
    profiles: dict[str, dict[str, float | int | str]] = {
        "dev": {
            "research_max_rounds": 2,
            "research_max_hypotheses": 15,
            "agent_max_steps": 10,
            "agent_min_steps": 3,
            # Match research wall budget so dev runs fewer surprise timeouts than 300s cap.
            "agent_max_seconds": 600,
            # MNIST / sandbox training needs far more than 60s wall time in Docker.
            "sandbox_timeout_sec": 420,
            "n_steps_default": 180,
            "max_code_steps_per_hypothesis": 2,
            "classification_default_dataset": "mnist",
            "llm_http_timeout_sec": 240.0,
        },
        "research": {
            "research_max_rounds": 4,
            "research_max_hypotheses": 60,
            "agent_max_steps": 12,
            "agent_min_steps": 4,
            "agent_max_seconds": 600,
            "sandbox_timeout_sec": 420,
            "n_steps_default": 300,
            "max_code_steps_per_hypothesis": 1,
            "classification_default_dataset": "mnist",
            "llm_http_timeout_sec": 300.0,
        },
        "deep": {
            "research_max_rounds": 6,
            "research_max_hypotheses": 80,
            "agent_max_steps": 16,
            "agent_min_steps": 5,
            "agent_max_seconds": 1200,
            "sandbox_timeout_sec": 600,
            "n_steps_default": 500,
            "max_code_steps_per_hypothesis": 2,
            "classification_default_dataset": "mnist",
            "llm_http_timeout_sec": 420.0,
        },
        # Campaign v1: 4-hour budget, hypothesis tree, breakthrough detection.
        # Designed for a single hard question (e.g. optimal MLP for MNIST).
        "campaign": {
            "research_max_rounds": 50,
            "research_max_hypotheses": 500,
            "research_max_hours": 4.0,
            "research_max_seconds_per_round": 1800,
            "agent_max_steps": 14,
            "agent_min_steps": 5,
            # Must exceed worst-case sandbox wait (sandbox_timeout × retries); see fixes.md §agent vs sandbox.
            "agent_max_seconds": 1800,
            "sandbox_timeout_sec": 480,
            "sandbox_code_max_retries": 1,
            "n_steps_default": 300,
            "max_code_steps_per_hypothesis": 2,
            "classification_default_dataset": "mnist",
            "llm_http_timeout_sec": 300.0,
            "campaign_compute_budget_seconds": 14400,
            "campaign_batch_size": 5,
            "campaign_max_hypotheses": 500,
            "campaign_breakthrough_threshold": 0.05,
            "campaign_min_confidence": 0.85,
            "campaign_min_replications": 3,
            "campaign_frontier_evict_idle_sec": 600,
            "agent_max_tool_calls": 48,
        },
    }
    selected = profiles.get(profile, profiles["dev"])
    for key, value in selected.items():
        setattr(s, key, value)


_apply_profile(settings)
