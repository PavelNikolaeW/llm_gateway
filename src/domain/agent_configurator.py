"""Agent Configurator - define, validate, and apply agent configurations.

Merge Strategy (documented):
    When merging configs, the order of precedence is:
    1. User-provided config (highest priority)
    2. Agent type preset config
    3. Default values

    This allows users to override agent presets while still benefiting
    from sensible defaults.
"""
import logging
from typing import Any

from src.shared.exceptions import ValidationError
from src.shared.schemas import AgentConfig, AgentTypeInfo

logger = logging.getLogger(__name__)


# Predefined agent types with their default configurations
AGENT_TYPES: dict[str, AgentTypeInfo] = {
    "default": AgentTypeInfo(
        name="default",
        description="Balanced assistant for general tasks",
        config=AgentConfig(temperature=0.7),
    ),
    "code_assistant": AgentTypeInfo(
        name="code_assistant",
        description="Focused assistant for coding tasks with lower temperature for accuracy",
        config=AgentConfig(temperature=0.2, max_tokens=4096),
    ),
    "creative_writer": AgentTypeInfo(
        name="creative_writer",
        description="Creative assistant with higher temperature for varied outputs",
        config=AgentConfig(temperature=0.9),
    ),
}


class AgentConfigurator:
    """Service for managing agent configurations.

    Features:
    - Define agent types (default, code_assistant, creative_writer)
    - Validate agent config (temperature 0-1, max_tokens > 0 and <= context_window)
    - Merge agent config with dialog settings
    """

    def __init__(self):
        """Initialize with predefined agent types."""
        self._agent_types = AGENT_TYPES.copy()

    def get_agent_type(self, name: str) -> AgentTypeInfo:
        """Get agent type by name.

        Args:
            name: Agent type name

        Returns:
            AgentTypeInfo with config

        Raises:
            ValidationError: If agent type is unknown (HTTP 400)
        """
        if name not in self._agent_types:
            available = list(self._agent_types.keys())
            raise ValidationError(
                f"Unknown agent type '{name}'. Available types: {', '.join(available)}"
            )
        return self._agent_types[name]

    def get_all_agent_types(self) -> list[AgentTypeInfo]:
        """Get all available agent types."""
        return list(self._agent_types.values())

    def agent_type_exists(self, name: str) -> bool:
        """Check if an agent type exists."""
        return name in self._agent_types

    def validate_config(
        self,
        config: dict[str, Any] | AgentConfig,
        context_window: int | None = None,
    ) -> AgentConfig:
        """Validate agent configuration.

        Args:
            config: Configuration to validate (dict or AgentConfig)
            context_window: Model's context window for max_tokens validation

        Returns:
            Validated AgentConfig

        Raises:
            ValidationError: If config values are out of range (HTTP 400)
        """
        # Convert dict to AgentConfig if needed
        if isinstance(config, dict):
            try:
                config = AgentConfig(**config)
            except Exception as e:
                raise ValidationError(f"Invalid agent config: {e}")

        # Validate temperature range (0-1)
        if config.temperature is not None:
            if not 0 <= config.temperature <= 1:
                raise ValidationError(
                    f"Invalid temperature {config.temperature}. Must be between 0 and 1."
                )

        # Validate max_tokens (> 0 and <= context_window)
        if config.max_tokens is not None:
            if config.max_tokens <= 0:
                raise ValidationError(
                    f"Invalid max_tokens {config.max_tokens}. Must be greater than 0."
                )
            if context_window is not None and config.max_tokens > context_window:
                raise ValidationError(
                    f"Invalid max_tokens {config.max_tokens}. "
                    f"Must be <= model context window ({context_window})."
                )

        # Validate top_p range (0-1)
        if config.top_p is not None:
            if not 0 <= config.top_p <= 1:
                raise ValidationError(
                    f"Invalid top_p {config.top_p}. Must be between 0 and 1."
                )

        # Validate presence_penalty range (-2 to 2)
        if config.presence_penalty is not None:
            if not -2 <= config.presence_penalty <= 2:
                raise ValidationError(
                    f"Invalid presence_penalty {config.presence_penalty}. Must be between -2 and 2."
                )

        # Validate frequency_penalty range (-2 to 2)
        if config.frequency_penalty is not None:
            if not -2 <= config.frequency_penalty <= 2:
                raise ValidationError(
                    f"Invalid frequency_penalty {config.frequency_penalty}. Must be between -2 and 2."
                )

        return config

    def merge_configs(
        self,
        agent_type: str | None = None,
        user_config: dict[str, Any] | AgentConfig | None = None,
        dialog_config: dict[str, Any] | None = None,
        context_window: int | None = None,
    ) -> AgentConfig:
        """Merge configurations with proper precedence.

        Precedence (highest to lowest):
        1. User-provided config
        2. Agent type preset config
        3. Dialog config
        4. Default values

        Args:
            agent_type: Optional agent type name to use as base
            user_config: User-provided configuration (highest priority)
            dialog_config: Dialog-level configuration
            context_window: Model's context window for validation

        Returns:
            Merged and validated AgentConfig

        Raises:
            ValidationError: If agent type is unknown or config is invalid
        """
        # Start with empty config
        merged: dict[str, Any] = {}

        # Apply dialog config first (lowest priority)
        if dialog_config:
            for key, value in dialog_config.items():
                if value is not None:
                    merged[key] = value

        # Apply agent type preset
        if agent_type:
            preset = self.get_agent_type(agent_type)
            preset_dict = preset.config.model_dump(exclude_none=True)
            for key, value in preset_dict.items():
                if value is not None:
                    merged[key] = value

        # Apply user config (highest priority)
        if user_config:
            if isinstance(user_config, AgentConfig):
                user_dict = user_config.model_dump(exclude_none=True)
            else:
                user_dict = {k: v for k, v in user_config.items() if v is not None}
            for key, value in user_dict.items():
                if value is not None:
                    merged[key] = value

        # Convert to AgentConfig and validate
        config = AgentConfig(**merged)
        return self.validate_config(config, context_window)

    def get_effective_config(
        self,
        agent_type: str = "default",
        context_window: int | None = None,
    ) -> AgentConfig:
        """Get effective configuration for an agent type.

        Args:
            agent_type: Agent type name
            context_window: Optional context window for validation

        Returns:
            AgentConfig for the agent type
        """
        preset = self.get_agent_type(agent_type)
        return self.validate_config(preset.config, context_window)


# Global agent configurator instance
agent_configurator = AgentConfigurator()
