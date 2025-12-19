"""Unit tests for Pydantic schema validation."""
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.shared.schemas import (
    AgentConfig,
    DialogCreate,
    MessageCreate,
    SetLimitRequest,
    TokenDeductRequest,
    TopUpTokensRequest,
)


class TestMessageCreate:
    """Tests for MessageCreate schema."""

    def test_valid_content(self):
        """Test valid message content."""
        msg = MessageCreate(content="Hello, world!")
        assert msg.content == "Hello, world!"

    def test_empty_content_rejected(self):
        """Test empty content is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MessageCreate(content="")
        assert "string_too_short" in str(exc_info.value)

    def test_content_exceeds_max_length(self):
        """Test content exceeding max length is rejected."""
        # Patch settings to use a small max length for testing
        with patch("src.config.settings.settings") as mock_settings:
            mock_settings.max_content_length = 100

            # Content that exceeds the limit
            long_content = "x" * 150
            with pytest.raises(ValidationError) as exc_info:
                MessageCreate(content=long_content)

            assert "maximum length" in str(exc_info.value).lower()

    def test_content_at_max_length(self):
        """Test content exactly at max length is accepted."""
        with patch("src.config.settings.settings") as mock_settings:
            mock_settings.max_content_length = 100

            # Content exactly at limit
            content = "x" * 100
            msg = MessageCreate(content=content)
            assert len(msg.content) == 100


class TestAgentConfig:
    """Tests for AgentConfig schema."""

    def test_valid_temperature(self):
        """Test valid temperature values."""
        config = AgentConfig(temperature=0.7)
        assert config.temperature == 0.7

    def test_temperature_at_boundaries(self):
        """Test temperature at boundaries."""
        config_min = AgentConfig(temperature=0.0)
        assert config_min.temperature == 0.0

        config_max = AgentConfig(temperature=1.0)
        assert config_max.temperature == 1.0

    def test_temperature_out_of_range(self):
        """Test temperature outside valid range."""
        with pytest.raises(ValidationError):
            AgentConfig(temperature=1.5)

        with pytest.raises(ValidationError):
            AgentConfig(temperature=-0.1)

    def test_valid_max_tokens(self):
        """Test valid max_tokens values."""
        config = AgentConfig(max_tokens=100)
        assert config.max_tokens == 100

    def test_max_tokens_zero_or_negative_rejected(self):
        """Test max_tokens zero or negative is rejected."""
        with pytest.raises(ValidationError):
            AgentConfig(max_tokens=0)

        with pytest.raises(ValidationError):
            AgentConfig(max_tokens=-1)

    def test_valid_top_p(self):
        """Test valid top_p values."""
        config = AgentConfig(top_p=0.9)
        assert config.top_p == 0.9

    def test_top_p_out_of_range(self):
        """Test top_p outside valid range."""
        with pytest.raises(ValidationError):
            AgentConfig(top_p=1.5)

        with pytest.raises(ValidationError):
            AgentConfig(top_p=-0.1)

    def test_valid_presence_penalty(self):
        """Test valid presence_penalty values."""
        config = AgentConfig(presence_penalty=0.5)
        assert config.presence_penalty == 0.5

    def test_presence_penalty_at_boundaries(self):
        """Test presence_penalty at boundaries."""
        config_min = AgentConfig(presence_penalty=-2.0)
        assert config_min.presence_penalty == -2.0

        config_max = AgentConfig(presence_penalty=2.0)
        assert config_max.presence_penalty == 2.0

    def test_presence_penalty_out_of_range(self):
        """Test presence_penalty outside valid range."""
        with pytest.raises(ValidationError):
            AgentConfig(presence_penalty=2.5)

        with pytest.raises(ValidationError):
            AgentConfig(presence_penalty=-2.5)


class TestDialogCreate:
    """Tests for DialogCreate schema."""

    def test_valid_dialog(self):
        """Test valid dialog creation."""
        dialog = DialogCreate(title="Test Dialog", model_name="gpt-4")
        assert dialog.title == "Test Dialog"

    def test_title_max_length(self):
        """Test title max length is enforced."""
        long_title = "x" * 256  # 256 chars, max is 255
        with pytest.raises(ValidationError):
            DialogCreate(title=long_title)

    def test_title_at_max_length(self):
        """Test title at max length is accepted."""
        title = "x" * 255  # Exactly 255 chars
        dialog = DialogCreate(title=title)
        assert len(dialog.title) == 255


class TestTokenDeductRequest:
    """Tests for TokenDeductRequest schema."""

    def test_valid_amount(self):
        """Test valid positive amount."""
        req = TokenDeductRequest(
            amount=100,
            dialog_id="00000000-0000-0000-0000-000000000001",
            message_id="00000000-0000-0000-0000-000000000002",
        )
        assert req.amount == 100

    def test_zero_amount_rejected(self):
        """Test zero amount is rejected."""
        with pytest.raises(ValidationError):
            TokenDeductRequest(
                amount=0,
                dialog_id="00000000-0000-0000-0000-000000000001",
                message_id="00000000-0000-0000-0000-000000000002",
            )

    def test_negative_amount_rejected(self):
        """Test negative amount is rejected."""
        with pytest.raises(ValidationError):
            TokenDeductRequest(
                amount=-100,
                dialog_id="00000000-0000-0000-0000-000000000001",
                message_id="00000000-0000-0000-0000-000000000002",
            )


class TestSetLimitRequest:
    """Tests for SetLimitRequest schema."""

    def test_valid_limit(self):
        """Test valid positive limit."""
        req = SetLimitRequest(limit=10000)
        assert req.limit == 10000

    def test_zero_limit_allowed(self):
        """Test zero limit is allowed (effectively no tokens)."""
        req = SetLimitRequest(limit=0)
        assert req.limit == 0

    def test_null_limit_allowed(self):
        """Test null limit is allowed (unlimited)."""
        req = SetLimitRequest(limit=None)
        assert req.limit is None

    def test_negative_limit_rejected(self):
        """Test negative limit is rejected."""
        with pytest.raises(ValidationError):
            SetLimitRequest(limit=-100)


class TestTopUpTokensRequest:
    """Tests for TopUpTokensRequest schema."""

    def test_positive_amount(self):
        """Test positive amount for top-up."""
        req = TopUpTokensRequest(amount=1000)
        assert req.amount == 1000

    def test_negative_amount_for_deduct(self):
        """Test negative amount for deduction."""
        req = TopUpTokensRequest(amount=-500)
        assert req.amount == -500

    def test_zero_amount_allowed(self):
        """Test zero amount is allowed (no-op)."""
        req = TopUpTokensRequest(amount=0)
        assert req.amount == 0
