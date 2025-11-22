from pathlib import Path

from pydantic import  AnyHttpUrl, BaseConfig
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Django services
    AUTH_VERIFY_URL: str = "http://localhost:8000/api/v1/token/verify/"  # POST {token}
    DATA_BASE_URL: str = "http://localhost:8000/api/v1/"
    AUTH_STUB_ENABLED: bool = True
    AUTH_STUB_USER_ID: int = 1
    AUTH_DEFAULT_USER_ID: int = 1

    # Optional direct JWT validation (in addition to AUTH_VERIFY_URL)
    JWT_ISSUER: Optional[str] = None
    JWT_AUDIENCE: Optional[str] = None
    JWT_JWKS_URL: Optional[AnyHttpUrl] = None  # e.g., http://django:8000/.well-known/jwks.json
    SQL_USER: str = "stub"
    SQL_PASSWD: str = "stub"
    SQL_HOST: str = "localhost"
    SQL_PORT: str = "5432"
    SQL_NAME: str = "stub"

    # Postgres
    PG_DSN: str | None = None
    STUB_DB_PATH: Path = Path(__file__).parent.parent / "local.sqlite3"
    STUB_DB_ENABLED: bool = True

    # LLM (multi-provider)
    DEFAULT_MODEL: str = "dummy"
    DEFAULT_PROVIDER: str = "dummy"

    # default OpenAI-compatible
    LLM_BASE_URL: Optional[AnyHttpUrl] = None
    LLM_API_KEY: Optional[str] = None

    # Ollama (local)
    OLLAMA_BASE_URL: Optional[AnyHttpUrl] = "http://localhost:1234/v1"
    OLLAMA_API_KEY: Optional[str] = 'lm-studio'

    # Runtime
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: str = "http://localhost:3000,https://your-frontend"

    # Rate limit (optional)
    RATE_LIMIT_RPS: int = 3
    RATE_LIMIT_BURST: int = 10

    SYSTEM_PROMPT: str = "You are a helpful assistant. Use supplied context when relevant."

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if not self.PG_DSN:
            self.PG_DSN = f"sqlite:///{self.STUB_DB_PATH}"

    class Config:
        env_file = Path(__file__).parent.parent / ".env"
        extra = "allow"


settings = Settings()
