from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
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
    sub_agent_plan_source: str = "heuristic"
    sub_agent_max_planned_steps: int = 4
    embed_model: str = "text-embedding-3-small"
    qdrant_url: str = ""
    qdrant_collection: str = "propab_chunks"
    propab_data_dir: str = "./data"
    minio_endpoint: str = ""
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "propab"
    minio_secure: bool = False
    sandbox_timeout_sec: int = 30
    sandbox_memory_mb: int = 512
    sandbox_code_max_retries: int = 3
    literature_answer_similarity: float = 0.92
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

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


settings = Settings()
