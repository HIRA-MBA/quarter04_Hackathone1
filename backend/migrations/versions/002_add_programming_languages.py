"""add programming_languages to user_preferences

Revision ID: 002_add_programming_languages
Revises: 001_initial_schema
Create Date: 2025-12-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = '002_add_programming_languages'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add programming_languages column to user_preferences table."""
    op.add_column(
        'user_preferences',
        sa.Column(
            'programming_languages',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            server_default=text("'{}'::jsonb"),
            comment='Programming language proficiency: {python: level, cpp: level, javascript: level}'
        )
    )


def downgrade() -> None:
    """Remove programming_languages column from user_preferences table."""
    op.drop_column('user_preferences', 'programming_languages')
