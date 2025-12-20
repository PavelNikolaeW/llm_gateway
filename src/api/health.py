"""Health check endpoints and utilities.

Provides comprehensive health checks for all dependencies:
- Database (PostgreSQL)
- Cache (Redis)
- LLM Providers (optional)
"""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.config.settings import settings
from src.data.cache import get_redis
from src.data.database import get_engine

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a component."""

    name: str
    status: HealthStatus
    latency_ms: float | None = None
    message: str | None = None
    details: dict[str, Any] | None = None


@dataclass
class SystemHealth:
    """Overall system health."""

    status: HealthStatus
    version: str
    timestamp: datetime
    components: list[ComponentHealth]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON response."""
        return {
            "status": self.status.value,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "latency_ms": c.latency_ms,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.components
            ],
        }


async def check_database() -> ComponentHealth:
    """Check database connectivity."""
    import time
    from sqlalchemy import text

    start = time.time()
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        latency = (time.time() - start) * 1000

        return ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2),
            message="PostgreSQL connection OK",
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"Database health check failed: {e}")
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 2),
            message=f"Database connection failed: {str(e)}",
        )


async def check_redis() -> ComponentHealth:
    """Check Redis connectivity."""
    import time

    if not settings.redis_url:
        return ComponentHealth(
            name="redis",
            status=HealthStatus.HEALTHY,
            message="Redis not configured (optional)",
        )

    start = time.time()
    try:
        redis = await get_redis()
        if redis is None:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.DEGRADED,
                message="Redis unavailable (graceful degradation)",
            )

        await redis.ping()
        latency = (time.time() - start) * 1000

        return ComponentHealth(
            name="redis",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2),
            message="Redis connection OK",
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.warning(f"Redis health check failed: {e}")
        return ComponentHealth(
            name="redis",
            status=HealthStatus.DEGRADED,
            latency_ms=round(latency, 2),
            message=f"Redis connection failed: {str(e)}",
        )


async def check_system_health() -> SystemHealth:
    """Check health of all system components."""
    # Run health checks concurrently
    db_health, redis_health = await asyncio.gather(
        check_database(),
        check_redis(),
    )

    components = [db_health, redis_health]

    # Determine overall status
    statuses = [c.status for c in components]
    if HealthStatus.UNHEALTHY in statuses:
        overall_status = HealthStatus.UNHEALTHY
    elif HealthStatus.DEGRADED in statuses:
        overall_status = HealthStatus.DEGRADED
    else:
        overall_status = HealthStatus.HEALTHY

    return SystemHealth(
        status=overall_status,
        version="1.0.0",
        timestamp=datetime.now(timezone.utc),
        components=components,
    )


async def is_ready() -> bool:
    """Check if the application is ready to receive traffic.

    Returns:
        True if all critical components are healthy
    """
    health = await check_system_health()
    # Database is critical, Redis is optional
    db_status = next((c for c in health.components if c.name == "database"), None)
    return db_status is not None and db_status.status == HealthStatus.HEALTHY


async def is_alive() -> bool:
    """Check if the application is alive (simple liveness check).

    Returns:
        True if the application process is running
    """
    return True
