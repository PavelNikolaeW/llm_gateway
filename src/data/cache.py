"""Redis cache client and operations."""

import json
import logging
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Global Redis client
_redis_client: Redis | None = None


async def get_redis() -> Redis | None:
    """Get or create Redis client.

    Returns None if Redis is not configured or unavailable.
    Implements graceful degradation.
    """
    global _redis_client

    if not settings.redis_url:
        return None

    if _redis_client is None:
        try:
            _redis_client = Redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            await _redis_client.ping()
            logger.info("Redis connection established")
        except RedisError as e:
            logger.warning(f"Redis unavailable, using DB fallback: {e}")
            _redis_client = None

    return _redis_client


async def close_redis() -> None:
    """Close Redis connection."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis connection closed")


class CacheService:
    """Service for caching operations with automatic fallback to database.

    Cache keys:
    - user:{user_id}:balance - token balance (TTL: 5 min)
    - model:{name} - model metadata (TTL: 1 hour)
    - jwks:keys - JWT public keys (TTL: 1 hour)
    """

    # TTL in seconds
    TTL_BALANCE = 300  # 5 minutes
    TTL_MODEL = 3600  # 1 hour
    TTL_JWKS = 3600  # 1 hour

    @staticmethod
    async def get(key: str) -> Any | None:
        """Get value from cache.

        Returns None on cache miss or if Redis unavailable.
        """
        redis = await get_redis()
        if redis is None:
            return None

        try:
            value = await redis.get(key)
            if value:
                return json.loads(value)
            return None
        except RedisError as e:
            logger.warning(f"Cache get failed for {key}: {e}")
            return None

    @staticmethod
    async def set(key: str, value: Any, ttl: int) -> bool:
        """Set value in cache with TTL.

        Returns True on success, False on failure.
        """
        redis = await get_redis()
        if redis is None:
            return False

        try:
            serialized = json.dumps(value)
            await redis.setex(key, ttl, serialized)
            return True
        except RedisError as e:
            logger.warning(f"Cache set failed for {key}: {e}")
            return False

    @staticmethod
    async def delete(key: str) -> bool:
        """Delete key from cache (invalidation).

        Returns True on success, False on failure.
        """
        redis = await get_redis()
        if redis is None:
            return False

        try:
            await redis.delete(key)
            return True
        except RedisError as e:
            logger.warning(f"Cache delete failed for {key}: {e}")
            return False

    @staticmethod
    def balance_key(user_id: int) -> str:
        """Generate cache key for user balance."""
        return f"user:{user_id}:balance"

    @staticmethod
    def model_key(model_name: str) -> str:
        """Generate cache key for model metadata."""
        return f"model:{model_name}"

    @staticmethod
    def jwks_key() -> str:
        """Generate cache key for JWKS."""
        return "jwks:keys"

    async def get_balance(self, user_id: int) -> dict | None:
        """Get cached user balance."""
        key = self.balance_key(user_id)
        return await self.get(key)

    async def set_balance(self, user_id: int, balance_data: dict) -> bool:
        """Cache user balance with 5 min TTL."""
        key = self.balance_key(user_id)
        return await self.set(key, balance_data, self.TTL_BALANCE)

    async def invalidate_balance(self, user_id: int) -> bool:
        """Invalidate cached balance after update."""
        key = self.balance_key(user_id)
        return await self.delete(key)

    async def get_model(self, model_name: str) -> dict | None:
        """Get cached model metadata."""
        key = self.model_key(model_name)
        return await self.get(key)

    async def set_model(self, model_name: str, model_data: dict) -> bool:
        """Cache model metadata with 1 hour TTL."""
        key = self.model_key(model_name)
        return await self.set(key, model_data, self.TTL_MODEL)

    async def get_jwks(self) -> dict | None:
        """Get cached JWKS."""
        key = self.jwks_key()
        return await self.get(key)

    async def set_jwks(self, jwks_data: dict) -> bool:
        """Cache JWKS with 1 hour TTL."""
        key = self.jwks_key()
        return await self.set(key, jwks_data, self.TTL_JWKS)


# Global cache service instance
cache_service = CacheService()
