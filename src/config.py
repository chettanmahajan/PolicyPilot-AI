"""Application settings, loaded from the environment / .env file."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets - no defaults, so a missing value fails loudly instead of
    # silently signing tokens with a guessable key.
    gemini_api_key: str = Field(default="", description="Gemini API key")
    jwt_secret: str = Field(min_length=32, description="HMAC key used to sign JWTs")

    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    database_url: str = f"sqlite:///{PROJECT_ROOT / 'policypilot.db'}"
    gemini_model: str = "gemini-3.5-flash"
    embedding_model: str = "gemini-embedding-001"
    api_base_url: str = "http://127.0.0.1:8000"

    # Retrieval
    knowledge_base_dir: Path = PROJECT_ROOT / "knowledge_base"
    index_path: Path = PROJECT_ROOT / "src" / "index" / "policy_index.npz"
    retrieval_top_k: int = 6

    # Customer photo evidence. Kept in a private folder that is never served
    # statically; the only way to read a photo is the owner-checked endpoint.
    uploads_dir: Path = PROJECT_ROOT / "uploads"
    max_photo_bytes: int = 5 * 1024 * 1024
    max_photos_per_upload: int = 5


settings = Settings()  # type: ignore[call-arg]
