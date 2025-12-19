"""Unit tests for AgentConfigurator."""
import pytest

from src.domain.agent_configurator import AgentConfigurator
from src.shared.exceptions import ValidationError
from src.shared.schemas import AgentConfig


@pytest.fixture
def configurator():
    """Create AgentConfigurator instance."""
    return AgentConfigurator()


# Agent Type Tests


def test_get_agent_type_default(configurator):
    """Test getting default agent type."""
    agent = configurator.get_agent_type("default")

    assert agent.name == "default"
    assert agent.config.temperature == 0.7
    assert agent.description == "Balanced assistant for general tasks"


def test_get_agent_type_code_assistant(configurator):
    """Test getting code_assistant agent type."""
    agent = configurator.get_agent_type("code_assistant")

    assert agent.name == "code_assistant"
    assert agent.config.temperature == 0.2
    assert agent.config.max_tokens == 4096


def test_get_agent_type_creative_writer(configurator):
    """Test getting creative_writer agent type."""
    agent = configurator.get_agent_type("creative_writer")

    assert agent.name == "creative_writer"
    assert agent.config.temperature == 0.9


def test_get_agent_type_unknown_raises_error(configurator):
    """Test getting unknown agent type raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        configurator.get_agent_type("unknown_agent")

    assert "Unknown agent type" in exc_info.value.message
    assert "unknown_agent" in exc_info.value.message
    assert "Available types:" in exc_info.value.message


def test_get_all_agent_types(configurator):
    """Test getting all agent types."""
    types = configurator.get_all_agent_types()

    assert len(types) == 3
    names = [t.name for t in types]
    assert "default" in names
    assert "code_assistant" in names
    assert "creative_writer" in names


def test_agent_type_exists(configurator):
    """Test checking if agent type exists."""
    assert configurator.agent_type_exists("default") is True
    assert configurator.agent_type_exists("code_assistant") is True
    assert configurator.agent_type_exists("unknown") is False


# Config Validation Tests


def test_validate_config_valid(configurator):
    """Test validating a valid config."""
    config = AgentConfig(temperature=0.5, max_tokens=1000)
    validated = configurator.validate_config(config)

    assert validated.temperature == 0.5
    assert validated.max_tokens == 1000


def test_validate_config_from_dict(configurator):
    """Test validating config from dict."""
    config_dict = {"temperature": 0.8, "max_tokens": 2000}
    validated = configurator.validate_config(config_dict)

    assert validated.temperature == 0.8
    assert validated.max_tokens == 2000


def test_validate_config_temperature_too_low(configurator):
    """Test validation fails for temperature < 0."""
    config = {"temperature": -0.1}

    with pytest.raises(ValidationError) as exc_info:
        configurator.validate_config(config)

    assert "temperature" in exc_info.value.message.lower()


def test_validate_config_temperature_too_high(configurator):
    """Test validation fails for temperature > 1."""
    config = {"temperature": 1.5}

    with pytest.raises(ValidationError) as exc_info:
        configurator.validate_config(config)

    assert "temperature" in exc_info.value.message.lower()


def test_validate_config_temperature_boundary_values(configurator):
    """Test temperature boundary values are valid."""
    # Temperature = 0 should be valid
    config_zero = configurator.validate_config({"temperature": 0.0})
    assert config_zero.temperature == 0.0

    # Temperature = 1 should be valid
    config_one = configurator.validate_config({"temperature": 1.0})
    assert config_one.temperature == 1.0


def test_validate_config_max_tokens_zero(configurator):
    """Test validation fails for max_tokens = 0."""
    config = {"max_tokens": 0}

    with pytest.raises(ValidationError) as exc_info:
        configurator.validate_config(config)

    assert "max_tokens" in exc_info.value.message.lower()


def test_validate_config_max_tokens_negative(configurator):
    """Test validation fails for max_tokens < 0."""
    config = {"max_tokens": -100}

    with pytest.raises(ValidationError) as exc_info:
        configurator.validate_config(config)

    assert "max_tokens" in exc_info.value.message.lower()


def test_validate_config_max_tokens_exceeds_context_window(configurator):
    """Test validation fails when max_tokens > context_window."""
    config = {"max_tokens": 10000}

    with pytest.raises(ValidationError) as exc_info:
        configurator.validate_config(config, context_window=4096)

    assert "max_tokens" in exc_info.value.message.lower()
    assert "context window" in exc_info.value.message.lower()


def test_validate_config_max_tokens_within_context_window(configurator):
    """Test max_tokens within context window is valid."""
    config = {"max_tokens": 4096}
    validated = configurator.validate_config(config, context_window=8000)

    assert validated.max_tokens == 4096


def test_validate_config_top_p_invalid(configurator):
    """Test validation fails for invalid top_p."""
    config = {"top_p": 1.5}

    with pytest.raises(ValidationError) as exc_info:
        configurator.validate_config(config)

    assert "top_p" in exc_info.value.message.lower()


def test_validate_config_presence_penalty_invalid(configurator):
    """Test validation fails for invalid presence_penalty."""
    config = {"presence_penalty": 3.0}

    with pytest.raises(ValidationError) as exc_info:
        configurator.validate_config(config)

    assert "presence_penalty" in exc_info.value.message.lower()


def test_validate_config_frequency_penalty_invalid(configurator):
    """Test validation fails for invalid frequency_penalty."""
    config = {"frequency_penalty": -3.0}

    with pytest.raises(ValidationError) as exc_info:
        configurator.validate_config(config)

    assert "frequency_penalty" in exc_info.value.message.lower()


# Merge Configs Tests


def test_merge_configs_agent_type_only(configurator):
    """Test merging with only agent type."""
    merged = configurator.merge_configs(agent_type="code_assistant")

    assert merged.temperature == 0.2
    assert merged.max_tokens == 4096


def test_merge_configs_user_overrides_agent(configurator):
    """Test user config overrides agent type config."""
    user_config = {"temperature": 0.5}
    merged = configurator.merge_configs(
        agent_type="code_assistant",
        user_config=user_config,
    )

    # User's temperature overrides code_assistant's 0.2
    assert merged.temperature == 0.5
    # max_tokens from code_assistant is preserved
    assert merged.max_tokens == 4096


def test_merge_configs_user_overrides_all(configurator):
    """Test user config can override all values."""
    user_config = {"temperature": 0.8, "max_tokens": 2000}
    merged = configurator.merge_configs(
        agent_type="code_assistant",
        user_config=user_config,
    )

    assert merged.temperature == 0.8
    assert merged.max_tokens == 2000


def test_merge_configs_dialog_config(configurator):
    """Test dialog config is used as base."""
    dialog_config = {"temperature": 0.6, "top_p": 0.9}
    merged = configurator.merge_configs(dialog_config=dialog_config)

    assert merged.temperature == 0.6
    assert merged.top_p == 0.9


def test_merge_configs_precedence(configurator):
    """Test full precedence chain: user > agent > dialog."""
    dialog_config = {"temperature": 0.3, "top_p": 0.8, "max_tokens": 1000}
    user_config = {"temperature": 0.9}

    merged = configurator.merge_configs(
        agent_type="code_assistant",  # temp=0.2, max_tokens=4096
        user_config=user_config,
        dialog_config=dialog_config,
    )

    # temperature: user (0.9) > agent (0.2) > dialog (0.3)
    assert merged.temperature == 0.9
    # max_tokens: agent (4096) > dialog (1000)
    assert merged.max_tokens == 4096
    # top_p: dialog (0.8)
    assert merged.top_p == 0.8


def test_merge_configs_with_validation(configurator):
    """Test merged config is validated."""
    user_config = {"max_tokens": 10000}

    with pytest.raises(ValidationError):
        configurator.merge_configs(
            user_config=user_config,
            context_window=4096,
        )


def test_merge_configs_empty(configurator):
    """Test merging with no configs returns empty config."""
    merged = configurator.merge_configs()

    assert merged.temperature is None
    assert merged.max_tokens is None


def test_merge_configs_user_config_as_agent_config(configurator):
    """Test user config can be an AgentConfig object."""
    user_config = AgentConfig(temperature=0.75, max_tokens=3000)
    merged = configurator.merge_configs(user_config=user_config)

    assert merged.temperature == 0.75
    assert merged.max_tokens == 3000


# Get Effective Config Tests


def test_get_effective_config_default(configurator):
    """Test getting effective config for default agent."""
    config = configurator.get_effective_config("default")

    assert config.temperature == 0.7


def test_get_effective_config_with_context_window(configurator):
    """Test effective config respects context window."""
    # code_assistant has max_tokens=4096, which is valid for 8000 context
    config = configurator.get_effective_config("code_assistant", context_window=8000)

    assert config.max_tokens == 4096


def test_get_effective_config_invalid_agent(configurator):
    """Test getting config for invalid agent raises error."""
    with pytest.raises(ValidationError):
        configurator.get_effective_config("invalid_agent")
