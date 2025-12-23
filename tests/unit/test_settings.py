"""Unit tests for application settings."""
import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config.settings import Environment, Settings, get_settings


# Base required env vars for Settings instantiation
BASE_ENV = {
    "SQL_USER": "testuser",
    "SQL_PASSWD": "testpass",
    "SQL_NAME": "testdb",
    "SQL_HOST": "localhost",
    "SQL_PORT": "5432",
    "AUTH_VERIFY_URL": "http://auth.example.com/verify",
    "DJANGO_SECRET_KEY": "a" * 32,  # Min 32 chars
    "JWT_SECRET": "testsecret",
}


class TestEnvironment:
    """Tests for Environment enum."""

    def test_environment_values(self):
        """Test environment enum values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"
        assert Environment.TEST.value == "test"


class TestSettingsValidation:
    """Tests for settings validation."""

    def test_missing_required_fields(self):
        """Test missing required fields raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError):
                Settings(_env_file=None)  # Disable .env file loading

    def test_valid_minimal_settings(self):
        """Test valid minimal settings."""
        with patch.dict(os.environ, BASE_ENV, clear=True):
            settings = Settings()
            assert settings.sql_user == "testuser"
            assert settings.sql_name == "testdb"

    def test_sql_port_validation(self):
        """Test SQL port must be valid."""
        env = {**BASE_ENV, "SQL_PORT": "70000"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "sql_port" in str(exc_info.value).lower()

    def test_django_secret_key_min_length(self):
        """Test Django secret key must be at least 32 chars."""
        env = {**BASE_ENV, "DJANGO_SECRET_KEY": "short"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "django_secret_key" in str(exc_info.value).lower()

    def test_log_level_normalized(self):
        """Test log level is normalized to uppercase."""
        env = {**BASE_ENV, "LOG_LEVEL": "debug"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
            assert settings.log_level == "DEBUG"

    def test_environment_normalized(self):
        """Test environment is normalized to lowercase."""
        env = {**BASE_ENV, "ENVIRONMENT": "PRODUCTION"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
            assert settings.environment == Environment.PRODUCTION

    def test_max_content_length_validation(self):
        """Test max content length has bounds."""
        # Too small
        env = {**BASE_ENV, "MAX_CONTENT_LENGTH": "0"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError):
                Settings()

        # Too large
        env = {**BASE_ENV, "MAX_CONTENT_LENGTH": "10000000"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError):
                Settings()

    def test_llm_timeout_validation(self):
        """Test LLM timeout has bounds."""
        env = {**BASE_ENV, "LLM_TIMEOUT": "1000"}  # > 600
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError):
                Settings()

    def test_jwt_algorithm_validation(self):
        """Test JWT algorithm must be HS256 or RS256."""
        env = {**BASE_ENV, "JWT_ALGORITHM": "INVALID"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError):
                Settings()


class TestSettingsProperties:
    """Tests for settings properties."""

    def test_database_url(self):
        """Test database URL is built correctly."""
        with patch.dict(os.environ, BASE_ENV, clear=True):
            settings = Settings()
            expected = "postgresql+asyncpg://testuser:testpass@localhost:5432/testdb"
            assert settings.database_url == expected

    def test_sync_database_url(self):
        """Test sync database URL is built correctly."""
        with patch.dict(os.environ, BASE_ENV, clear=True):
            settings = Settings()
            expected = "postgresql://testuser:testpass@localhost:5432/testdb"
            assert settings.sync_database_url == expected

    def test_is_production(self):
        """Test is_production property."""
        env = {**BASE_ENV, "ENVIRONMENT": "production"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
            assert settings.is_production is True
            assert settings.is_development is False

    def test_is_development(self):
        """Test is_development property."""
        env = {**BASE_ENV, "ENVIRONMENT": "development"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
            assert settings.is_development is True
            assert settings.is_production is False

    def test_cors_origins_list(self):
        """Test CORS origins list parsing."""
        env = {**BASE_ENV, "CORS_ORIGINS": "http://localhost:3000, http://example.com, https://api.example.com"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
            assert settings.cors_origins_list == [
                "http://localhost:3000",
                "http://example.com",
                "https://api.example.com",
            ]


class TestSettingsDefaults:
    """Tests for settings default values."""

    def test_default_environment(self):
        """Test default environment is development."""
        with patch.dict(os.environ, BASE_ENV, clear=True):
            settings = Settings(_env_file=None)
            assert settings.environment == Environment.DEVELOPMENT

    def test_default_debug(self):
        """Test default debug is False."""
        with patch.dict(os.environ, BASE_ENV, clear=True):
            settings = Settings(_env_file=None)
            assert settings.debug is False

    def test_default_log_level(self):
        """Test default log level is INFO."""
        with patch.dict(os.environ, BASE_ENV, clear=True):
            settings = Settings(_env_file=None)
            assert settings.log_level == "INFO"

    def test_default_llm_timeout(self):
        """Test default LLM timeout."""
        with patch.dict(os.environ, BASE_ENV, clear=True):
            settings = Settings(_env_file=None)
            assert settings.llm_timeout == 120

    def test_default_rate_limit(self):
        """Test default rate limiting settings."""
        with patch.dict(os.environ, BASE_ENV, clear=True):
            settings = Settings(_env_file=None)
            assert settings.rate_limit_enabled is True
            assert settings.rate_limit_requests == 100
            assert settings.rate_limit_window == 60


class TestGetSettings:
    """Tests for get_settings function."""

    def test_returns_settings_instance(self):
        """Test get_settings returns Settings instance."""
        with patch.dict(os.environ, BASE_ENV, clear=True):
            # Clear cache to force new instance
            get_settings.cache_clear()
            settings = get_settings()
            assert isinstance(settings, Settings)

    def test_caches_settings(self):
        """Test get_settings returns cached instance."""
        with patch.dict(os.environ, BASE_ENV, clear=True):
            get_settings.cache_clear()
            settings1 = get_settings()
            settings2 = get_settings()
            assert settings1 is settings2
