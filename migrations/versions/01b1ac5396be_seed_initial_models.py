"""Seed initial models

Revision ID: 01b1ac5396be
Revises: 4b12416971ad
Create Date: 2025-12-19 17:06:25.805771

"""

from typing import Sequence, Union
from datetime import datetime

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "01b1ac5396be"
down_revision: Union[str, Sequence[str], None] = "4b12416971ad"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Seed initial models with pricing and configuration."""
    models_table = sa.table(
        "models",
        sa.column("name", sa.String),
        sa.column("provider", sa.String),
        sa.column("cost_per_1k_prompt_tokens", sa.Numeric),
        sa.column("cost_per_1k_completion_tokens", sa.Numeric),
        sa.column("context_window", sa.Integer),
        sa.column("enabled", sa.Boolean),
        sa.column("created_at", sa.DateTime),
        sa.column("updated_at", sa.DateTime),
    )

    now = datetime.utcnow()

    op.bulk_insert(
        models_table,
        [
            # OpenAI Models
            {
                "name": "gpt-4-turbo",
                "provider": "openai",
                "cost_per_1k_prompt_tokens": 0.01,
                "cost_per_1k_completion_tokens": 0.03,
                "context_window": 128000,
                "enabled": True,
                "created_at": now,
                "updated_at": now,
            },
            {
                "name": "gpt-3.5-turbo",
                "provider": "openai",
                "cost_per_1k_prompt_tokens": 0.0005,
                "cost_per_1k_completion_tokens": 0.0015,
                "context_window": 16385,
                "enabled": True,
                "created_at": now,
                "updated_at": now,
            },
            # Anthropic Models
            {
                "name": "claude-3-opus",
                "provider": "anthropic",
                "cost_per_1k_prompt_tokens": 0.015,
                "cost_per_1k_completion_tokens": 0.075,
                "context_window": 200000,
                "enabled": True,
                "created_at": now,
                "updated_at": now,
            },
            {
                "name": "claude-3-sonnet",
                "provider": "anthropic",
                "cost_per_1k_prompt_tokens": 0.003,
                "cost_per_1k_completion_tokens": 0.015,
                "context_window": 200000,
                "enabled": True,
                "created_at": now,
                "updated_at": now,
            },
        ],
    )


def downgrade() -> None:
    """Remove seeded models."""
    op.execute(
        """
        DELETE FROM models
        WHERE name IN ('gpt-4-turbo', 'gpt-3.5-turbo', 'claude-3-opus', 'claude-3-sonnet')
        """
    )
