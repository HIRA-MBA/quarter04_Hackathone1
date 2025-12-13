from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "physical-ai-backend"
    debug: bool = False
    secret_key: str = Field(default="change-me-in-production")

    # Database
    database_url: str = Field(default="postgresql+asyncpg://localhost/physical_ai")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = None
    qdrant_collection: str = "book_content"

    # OpenAI
    openai_api_key: str = Field(default="")
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    # CORS
    cors_origins: list[str] = Field(default=["http://localhost:3000"])

    # Authentication
    auth_enabled: bool = False
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
