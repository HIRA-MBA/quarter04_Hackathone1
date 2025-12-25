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

    # Anthropic (Claude)
    anthropic_api_key: str = Field(default="")
    claude_model: str = "claude-sonnet-4-20250514"
    translation_provider: str = Field(default="openai")  # "openai" or "claude"

    # CORS
    # Note: FastAPI CORSMiddleware does NOT support wildcards like "https://*.vercel.app"
    # Use exact origins or set CORS_ORIGINS env var with comma-separated values
    # Set CORS_ORIGINS="*" to allow all origins (for development/testing)
    cors_origins: list[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:5173",
            "https://quarter04-hackathone1.vercel.app",
        ]
    )
    # Allow all origins flag - set CORS_ALLOW_ALL=true for permissive CORS
    cors_allow_all: bool = Field(default=True)

    # Authentication
    auth_enabled: bool = False
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # OAuth - Google
    google_client_id: str = Field(default="")
    google_client_secret: str = Field(default="")
    google_redirect_uri: str = Field(default="http://localhost:8000/api/auth/oauth/google/callback")

    # OAuth - GitHub
    github_client_id: str = Field(default="")
    github_client_secret: str = Field(default="")
    github_redirect_uri: str = Field(default="http://localhost:8000/api/auth/oauth/github/callback")

    # Email (SMTP)
    smtp_host: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=587)
    smtp_user: str = Field(default="")
    smtp_password: str = Field(default="")
    smtp_from_email: str = Field(default="noreply@physical-ai-textbook.com")
    smtp_from_name: str = Field(default="Physical AI Textbook")

    # Frontend URL (for email links)
    frontend_url: str = Field(default="http://localhost:3000")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
