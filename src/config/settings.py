"""Application configuration settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database Configuration
    sql_user: str
    sql_passwd: str
    sql_name: str
    sql_host: str = "localhost"
    sql_port: int = 5432

    # Authentication
    auth_verify_url: str
    django_secret_key: str
    jwt_token: str | None = None

    # CORS
    cors_origins: str = "http://localhost:8080"

    # LLM Providers (optional for now)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Redis (optional)
    redis_url: str | None = None

    # Application
    debug: bool = False
    log_level: str = "INFO"

    @property
    def database_url(self) -> str:
        """Build PostgreSQL connection URL."""
        return f"postgresql+asyncpg://{self.sql_user}:{self.sql_passwd}@{self.sql_host}:{self.sql_port}/{self.sql_name}"

    @property
    def sync_database_url(self) -> str:
        """Build synchronous PostgreSQL connection URL for Alembic."""
        return f"postgresql://{self.sql_user}:{self.sql_passwd}@{self.sql_host}:{self.sql_port}/{self.sql_name}"


# Global settings instance
settings = Settings()
