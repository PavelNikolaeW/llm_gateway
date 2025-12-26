"""Rate limiter using Redis sliding window algorithm.

Implements per-user rate limiting with configurable requests per window.
Falls back to allowing requests if Redis is unavailable.
"""

import logging
import time
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError

from src.config.settings import settings
from src.data.cache import get_redis

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int
    reset_at: int  # Unix timestamp when window resets
    limit: int
    window: int


class RateLimiter:
    """Rate limiter using Redis sliding window.

    Uses a sorted set to track requests within the window.
    Key format: rate_limit:{user_id}
    """

    def __init__(
        self,
        requests_per_window: int | None = None,
        window_seconds: int | None = None,
    ):
        """Initialize rate limiter.

        Args:
            requests_per_window: Max requests per window (default: from settings)
            window_seconds: Window size in seconds (default: from settings)
        """
        self.requests_per_window = requests_per_window or settings.rate_limit_requests
        self.window_seconds = window_seconds or settings.rate_limit_window

    def _key(self, identifier: str) -> str:
        """Generate Redis key for rate limit tracking."""
        return f"rate_limit:{identifier}"

    async def check(self, identifier: str) -> RateLimitResult:
        """Check if request is allowed under rate limit.

        Uses sliding window algorithm:
        1. Remove entries older than window
        2. Count remaining entries
        3. If under limit, add current request

        Args:
            identifier: User ID or IP address to rate limit

        Returns:
            RateLimitResult with allowed status and metadata
        """
        if not settings.rate_limit_enabled:
            return RateLimitResult(
                allowed=True,
                remaining=self.requests_per_window,
                reset_at=int(time.time()) + self.window_seconds,
                limit=self.requests_per_window,
                window=self.window_seconds,
            )

        redis = await get_redis()
        if redis is None:
            # Allow if Redis unavailable (graceful degradation)
            logger.warning("Rate limiter: Redis unavailable, allowing request")
            return RateLimitResult(
                allowed=True,
                remaining=self.requests_per_window,
                reset_at=int(time.time()) + self.window_seconds,
                limit=self.requests_per_window,
                window=self.window_seconds,
            )

        try:
            return await self._check_with_redis(redis, identifier)
        except RedisError as e:
            logger.warning(f"Rate limiter error: {e}, allowing request")
            return RateLimitResult(
                allowed=True,
                remaining=self.requests_per_window,
                reset_at=int(time.time()) + self.window_seconds,
                limit=self.requests_per_window,
                window=self.window_seconds,
            )

    async def _check_with_redis(self, redis: Redis, identifier: str) -> RateLimitResult:
        """Perform rate limit check using Redis.

        Uses sliding window with sorted set.
        """
        key = self._key(identifier)
        now = time.time()
        window_start = now - self.window_seconds
        reset_at = int(now) + self.window_seconds

        # Remove old entries outside window
        await redis.zremrangebyscore(key, 0, window_start)

        # Count current entries
        current_count = await redis.zcard(key)

        if current_count >= self.requests_per_window:
            # Over limit
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=reset_at,
                limit=self.requests_per_window,
                window=self.window_seconds,
            )

        # Under limit - add this request
        await redis.zadd(key, {str(now): now})
        # Set TTL to auto-cleanup
        await redis.expire(key, self.window_seconds + 1)

        remaining = self.requests_per_window - current_count - 1

        return RateLimitResult(
            allowed=True,
            remaining=max(0, remaining),
            reset_at=reset_at,
            limit=self.requests_per_window,
            window=self.window_seconds,
        )

    async def reset(self, identifier: str) -> bool:
        """Reset rate limit for an identifier.

        Useful for testing or admin operations.
        """
        redis = await get_redis()
        if redis is None:
            return False

        try:
            key = self._key(identifier)
            await redis.delete(key)
            return True
        except RedisError as e:
            logger.warning(f"Rate limiter reset error: {e}")
            return False


# Global rate limiter instance
rate_limiter = RateLimiter()
