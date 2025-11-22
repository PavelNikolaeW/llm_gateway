from pathlib import Path

from pydantic import  AnyHttpUrl, BaseConfig
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Django services
    AUTH_VERIFY_URL: str = "http://localhost:8000/api/v1/token/verify/"  # POST {token}
    DATA_BASE_URL: str = "http://localhost:8000/api/v1/"

    # Optional direct JWT validation (in addition to AUTH_VERIFY_URL)
    JWT_ISSUER: Optional[str] = None
    JWT_AUDIENCE: Optional[str] = None
    JWT_JWKS_URL: Optional[AnyHttpUrl] = None  # e.g., http://django:8000/.well-known/jwks.json
    SQL_USER: str
    SQL_PASSWD: str
    SQL_HOST: str
    SQL_PORT: str
    SQL_NAME: str

    # Postgres
    PG_DSN: str

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

    class Config:
        env_file = Path(__file__).parent.parent / ".env"
        extra = "allow"


settings = Settings()
