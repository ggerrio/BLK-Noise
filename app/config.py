"""Application configuration loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings, read from environment or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str | None = Field(
        default=None,
        description="Google Gemini API key.",
    )
    gemini_model: str = Field(
        default="gemini-3.8-flash",
        description="Gemini model id used for generation.",
    )
    db_path: Path = Field(
        default=PROJECT_ROOT / "data" / "listings.db",
        description="Path to the SQLite database file.",
    )
    api_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the FastAPI service used by the UI.",
    )
    top_n_default: int = Field(
        default=5,
        ge=1,
        description="Default number of results when the caller does not specify.",
    )

    def require_gemini_api_key(self) -> str:
        """Return the API key, raising if it was never configured."""
        if not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return self.gemini_api_key


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()


settings = get_settings()
