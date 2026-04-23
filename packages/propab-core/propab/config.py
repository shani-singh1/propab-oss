from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://propab:propab@localhost:5432/propab"
    redis_url: str = "redis://localhost:6379/0"
    openai_api_key: str = ""
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
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
    )


settings = Settings()
