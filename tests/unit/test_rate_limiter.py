"""Tests for rate limiter."""
import pytest
import time
from unittest.mock import AsyncMock, patch, MagicMock

from src.api.rate_limiter import RateLimiter, RateLimitResult


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_rate_limit_result_dataclass(self):
        """Test RateLimitResult dataclass."""
        result = RateLimitResult(
            allowed=True,
            remaining=5,
            reset_at=1234567890,
            limit=10,
            window=60,
        )
        assert result.allowed is True
        assert result.remaining == 5
        assert result.reset_at == 1234567890
        assert result.limit == 10
        assert result.window == 60

    def test_limiter_init_defaults(self):
        """Test RateLimiter default initialization."""
        with patch("src.api.rate_limiter.settings") as mock_settings:
            mock_settings.rate_limit_requests = 100
            mock_settings.rate_limit_window = 60
            limiter = RateLimiter()
            assert limiter.requests_per_window == 100
            assert limiter.window_seconds == 60

    def test_limiter_init_custom(self):
        """Test RateLimiter with custom values."""
        limiter = RateLimiter(requests_per_window=50, window_seconds=30)
        assert limiter.requests_per_window == 50
        assert limiter.window_seconds == 30

    def test_key_generation(self):
        """Test Redis key generation."""
        limiter = RateLimiter()
        assert limiter._key("user:123") == "rate_limit:user:123"
        assert limiter._key("ip:192.168.1.1") == "rate_limit:ip:192.168.1.1"

    @pytest.mark.asyncio
    async def test_check_disabled(self):
        """Test check when rate limiting is disabled."""
        with patch("src.api.rate_limiter.settings") as mock_settings:
            mock_settings.rate_limit_enabled = False
            mock_settings.rate_limit_requests = 100
            mock_settings.rate_limit_window = 60
            limiter = RateLimiter()

            result = await limiter.check("user:123")

            assert result.allowed is True
            assert result.remaining == 100
            assert result.limit == 100
            assert result.window == 60

    @pytest.mark.asyncio
    async def test_check_redis_unavailable(self):
        """Test check when Redis is unavailable."""
        with patch("src.api.rate_limiter.settings") as mock_settings:
            mock_settings.rate_limit_enabled = True
            mock_settings.rate_limit_requests = 100
            mock_settings.rate_limit_window = 60

            with patch("src.api.rate_limiter.get_redis", return_value=None):
                limiter = RateLimiter()
                result = await limiter.check("user:123")

                assert result.allowed is True
                assert result.remaining == 100

    @pytest.mark.asyncio
    async def test_check_under_limit(self):
        """Test check when under rate limit."""
        mock_redis = AsyncMock()
        mock_redis.zremrangebyscore = AsyncMock(return_value=0)
        mock_redis.zcard = AsyncMock(return_value=5)  # 5 requests made
        mock_redis.zadd = AsyncMock()
        mock_redis.expire = AsyncMock()

        with patch("src.api.rate_limiter.settings") as mock_settings:
            mock_settings.rate_limit_enabled = True
            mock_settings.rate_limit_requests = 100
            mock_settings.rate_limit_window = 60

            with patch("src.api.rate_limiter.get_redis", return_value=mock_redis):
                limiter = RateLimiter(requests_per_window=100, window_seconds=60)
                result = await limiter.check("user:123")

                assert result.allowed is True
                assert result.remaining == 94  # 100 - 5 - 1

    @pytest.mark.asyncio
    async def test_check_over_limit(self):
        """Test check when over rate limit."""
        mock_redis = AsyncMock()
        mock_redis.zremrangebyscore = AsyncMock(return_value=0)
        mock_redis.zcard = AsyncMock(return_value=100)  # 100 requests made

        with patch("src.api.rate_limiter.settings") as mock_settings:
            mock_settings.rate_limit_enabled = True
            mock_settings.rate_limit_requests = 100
            mock_settings.rate_limit_window = 60

            with patch("src.api.rate_limiter.get_redis", return_value=mock_redis):
                limiter = RateLimiter(requests_per_window=100, window_seconds=60)
                result = await limiter.check("user:123")

                assert result.allowed is False
                assert result.remaining == 0

    @pytest.mark.asyncio
    async def test_check_redis_error(self):
        """Test check handles Redis errors gracefully."""
        from redis.exceptions import RedisError

        mock_redis = AsyncMock()
        mock_redis.zremrangebyscore = AsyncMock(side_effect=RedisError("Connection failed"))

        with patch("src.api.rate_limiter.settings") as mock_settings:
            mock_settings.rate_limit_enabled = True
            mock_settings.rate_limit_requests = 100
            mock_settings.rate_limit_window = 60

            with patch("src.api.rate_limiter.get_redis", return_value=mock_redis):
                limiter = RateLimiter()
                result = await limiter.check("user:123")

                # Should allow on error (graceful degradation)
                assert result.allowed is True

    @pytest.mark.asyncio
    async def test_reset_success(self):
        """Test reset clears rate limit."""
        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock()

        with patch("src.api.rate_limiter.get_redis", return_value=mock_redis):
            limiter = RateLimiter()
            result = await limiter.reset("user:123")

            assert result is True
            mock_redis.delete.assert_called_once_with("rate_limit:user:123")

    @pytest.mark.asyncio
    async def test_reset_redis_unavailable(self):
        """Test reset when Redis unavailable."""
        with patch("src.api.rate_limiter.get_redis", return_value=None):
            limiter = RateLimiter()
            result = await limiter.reset("user:123")

            assert result is False

    @pytest.mark.asyncio
    async def test_reset_redis_error(self):
        """Test reset handles Redis errors."""
        from redis.exceptions import RedisError

        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock(side_effect=RedisError("Error"))

        with patch("src.api.rate_limiter.get_redis", return_value=mock_redis):
            limiter = RateLimiter()
            result = await limiter.reset("user:123")

            assert result is False
