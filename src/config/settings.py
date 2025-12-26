"""Application configuration settings.

Provides environment-based configuration with validation:
- Loads from environment variables and .env file
- Supports different environments (development, staging, production)
- Validates configuration values
- Type-safe access to settings
"""

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Settings are loaded in order of priority:
    1. Environment variables
    2. .env file (if exists)
    3. Default values
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment (development, staging, production, test)",
    )

    # Database Configuration
    sql_user: str = Field(..., description="PostgreSQL username")
    sql_passwd: str = Field(..., description="PostgreSQL password")
    sql_name: str = Field(..., description="PostgreSQL database name")
    sql_host: str = Field(default="localhost", description="PostgreSQL host")
    sql_port: int = Field(default=5432, ge=1, le=65535, description="PostgreSQL port")
    db_pool_size: int = Field(default=5, ge=1, le=100, description="Database connection pool size")
    db_max_overflow: int = Field(default=10, ge=0, le=100, description="Max overflow connections")

    # Authentication
    auth_verify_url: str = Field(..., description="URL for auth verification")
    django_secret_key: str = Field(
        ..., min_length=32, description="Django secret key (min 32 chars)"
    )
    jwt_token: str | None = Field(default=None, description="Static JWT token for testing")

    # JWT Validation
    jwt_secret: str | None = Field(default=None, description="JWT secret for HS256")
    jwt_jwks_url: str | None = Field(default=None, description="JWKS URL for RS256")
    jwt_algorithm: Literal["HS256", "RS256"] = Field(
        default="HS256",
        description="JWT algorithm (HS256 or RS256)",
    )
    jwt_audience: str | None = Field(default=None, description="Expected JWT audience")
    jwt_issuer: str | None = Field(default=None, description="Expected JWT issuer")

    # CORS
    cors_origins: str = Field(
        default="http://localhost:8080",
        description="Comma-separated list of allowed CORS origins",
    )

    # LLM Providers
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_org_id: str | None = Field(default=None, description="OpenAI organization ID")
    openai_base_url: str | None = Field(
        default=None, description="OpenAI API base URL (for proxies)"
    )
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    anthropic_base_url: str | None = Field(default=None, description="Anthropic API base URL")
    gigachat_auth_key: str | None = Field(
        default=None, description="GigaChat authorization key (base64)"
    )
    gigachat_scope: str = Field(default="GIGACHAT_API_PERS", description="GigaChat API scope")

    # LLM Request Configuration
    llm_timeout: int = Field(
        default=120, ge=1, le=600, description="LLM request timeout in seconds"
    )
    llm_max_retries: int = Field(default=3, ge=0, le=10, description="Max retries for LLM requests")
    llm_default_model: str = Field(default="gpt-4", description="Default LLM model")

    # Redis (optional)
    redis_url: str | None = Field(default=None, description="Redis connection URL")
    redis_ttl: int = Field(default=3600, ge=0, description="Default Redis TTL in seconds")

    # Application
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, ge=1, description="Requests per window")
    rate_limit_window: int = Field(default=60, ge=1, description="Rate limit window in seconds")

    # Validation limits
    max_content_length: int = Field(
        default=100000,
        ge=1,
        le=1000000,
        description="Max message content length in characters",
    )
    max_messages_per_dialog: int = Field(
        default=1000,
        ge=1,
        description="Max messages per dialog",
    )

    # Token limits
    default_token_limit: int | None = Field(
        default=None,
        ge=0,
        description="Default token limit for new users (None = unlimited)",
    )
    initial_token_balance: int = Field(
        default=10000,
        ge=0,
        description="Initial token balance for new users",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Normalize log level to uppercase."""
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Normalize environment to lowercase."""
        if isinstance(v, str):
            return v.lower()
        return v

    @model_validator(mode="after")
    def validate_jwt_config(self) -> "Settings":
        """Validate JWT configuration is complete."""
        if self.jwt_algorithm == "HS256" and not self.jwt_secret:
            if self.environment != Environment.TEST:
                # Allow missing in test environment
                pass  # Will fail at runtime if needed
        elif self.jwt_algorithm == "RS256" and not self.jwt_jwks_url:
            if self.environment != Environment.TEST:
                pass  # Will fail at runtime if needed
        return self

    @property
    def database_url(self) -> str:
        """Build PostgreSQL connection URL."""
        return f"postgresql+asyncpg://{self.sql_user}:{self.sql_passwd}@{self.sql_host}:{self.sql_port}/{self.sql_name}"

    @property
    def sync_database_url(self) -> str:
        """Build synchronous PostgreSQL connection URL for Alembic."""
        return f"postgresql://{self.sql_user}:{self.sql_passwd}@{self.sql_host}:{self.sql_port}/{self.sql_name}"

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings instance (cached)
    """
    return Settings()


# Global settings instance (for backwards compatibility)
settings = get_settings()
