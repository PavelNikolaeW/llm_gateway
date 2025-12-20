"""Tests for health check functionality."""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

from src.api.health import (
    HealthStatus,
    ComponentHealth,
    SystemHealth,
    check_database,
    check_redis,
    check_system_health,
    is_ready,
    is_alive,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_statuses(self):
        """Test all health status values."""
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.UNHEALTHY == "unhealthy"


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_component_health_minimal(self):
        """Test ComponentHealth with minimal fields."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
        )
        assert health.name == "test"
        assert health.status == HealthStatus.HEALTHY
        assert health.latency_ms is None
        assert health.message is None

    def test_component_health_full(self):
        """Test ComponentHealth with all fields."""
        health = ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            latency_ms=5.5,
            message="Connection OK",
            details={"version": "15.0"},
        )
        assert health.latency_ms == 5.5
        assert health.message == "Connection OK"
        assert health.details["version"] == "15.0"


class TestSystemHealth:
    """Tests for SystemHealth dataclass."""

    def test_system_health_to_dict(self):
        """Test SystemHealth serialization."""
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            version="1.0.0",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            components=[
                ComponentHealth(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    latency_ms=5.0,
                ),
                ComponentHealth(
                    name="redis",
                    status=HealthStatus.DEGRADED,
                    message="Connection slow",
                ),
            ],
        )

        result = health.to_dict()

        assert result["status"] == "healthy"
        assert result["version"] == "1.0.0"
        assert result["timestamp"] == "2024-01-01T12:00:00+00:00"
        assert len(result["components"]) == 2
        assert result["components"][0]["name"] == "database"
        assert result["components"][0]["latency_ms"] == 5.0
        assert result["components"][1]["status"] == "degraded"


class TestCheckDatabase:
    """Tests for check_database function."""

    @pytest.mark.asyncio
    async def test_database_healthy(self):
        """Test database check when healthy."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(return_value=AsyncMock())
        mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.connect.return_value.__aexit__ = AsyncMock()

        with patch("src.api.health.get_engine", return_value=mock_engine):
            result = await check_database()

        assert result.name == "database"
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms is not None

    @pytest.mark.asyncio
    async def test_database_unhealthy(self):
        """Test database check when unhealthy."""
        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(side_effect=Exception("Connection refused"))

        with patch("src.api.health.get_engine", return_value=mock_engine):
            result = await check_database()

        assert result.name == "database"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Connection refused" in result.message


class TestCheckRedis:
    """Tests for check_redis function."""

    @pytest.mark.asyncio
    async def test_redis_not_configured(self):
        """Test Redis check when not configured."""
        with patch("src.api.health.settings") as mock_settings:
            mock_settings.redis_url = None

            result = await check_redis()

        assert result.name == "redis"
        assert result.status == HealthStatus.HEALTHY
        assert "not configured" in result.message

    @pytest.mark.asyncio
    async def test_redis_healthy(self):
        """Test Redis check when healthy."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        with patch("src.api.health.settings") as mock_settings:
            mock_settings.redis_url = "redis://localhost:6379"

            with patch("src.api.health.get_redis", return_value=mock_redis):
                result = await check_redis()

        assert result.name == "redis"
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_redis_unavailable(self):
        """Test Redis check when unavailable."""
        with patch("src.api.health.settings") as mock_settings:
            mock_settings.redis_url = "redis://localhost:6379"

            with patch("src.api.health.get_redis", return_value=None):
                result = await check_redis()

        assert result.name == "redis"
        assert result.status == HealthStatus.DEGRADED


class TestIsReady:
    """Tests for is_ready function."""

    @pytest.mark.asyncio
    async def test_ready_when_database_healthy(self):
        """Test ready check when database is healthy."""
        mock_health = SystemHealth(
            status=HealthStatus.HEALTHY,
            version="1.0.0",
            timestamp=datetime.now(timezone.utc),
            components=[
                ComponentHealth(name="database", status=HealthStatus.HEALTHY),
                ComponentHealth(name="redis", status=HealthStatus.DEGRADED),
            ],
        )

        with patch("src.api.health.check_system_health", return_value=mock_health):
            result = await is_ready()

        assert result is True

    @pytest.mark.asyncio
    async def test_not_ready_when_database_unhealthy(self):
        """Test ready check when database is unhealthy."""
        mock_health = SystemHealth(
            status=HealthStatus.UNHEALTHY,
            version="1.0.0",
            timestamp=datetime.now(timezone.utc),
            components=[
                ComponentHealth(name="database", status=HealthStatus.UNHEALTHY),
                ComponentHealth(name="redis", status=HealthStatus.HEALTHY),
            ],
        )

        with patch("src.api.health.check_system_health", return_value=mock_health):
            result = await is_ready()

        assert result is False


class TestIsAlive:
    """Tests for is_alive function."""

    @pytest.mark.asyncio
    async def test_always_alive(self):
        """Test liveness check always returns True."""
        result = await is_alive()
        assert result is True
