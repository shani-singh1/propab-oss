from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://propab:propab@localhost:5432/propab"
    redis_url: str = "redis://localhost:6379/0"
    openai_api_key: str = ""
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
