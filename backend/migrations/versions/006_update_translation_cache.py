"""Update translation_cache table with status and chapter_id fields.

Revision ID: 006
Revises: 005
Create Date: 2025-12-13

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Rename source_hash to content_hash
    op.alter_column(
        'translation_cache',
        'source_hash',
        new_column_name='content_hash'
    )

    # Rename chapter to chapter_id
    op.alter_column(
        'translation_cache',
        'chapter',
        new_column_name='chapter_id'
    )

    # Add status column
    op.add_column(
        'translation_cache',
        sa.Column('status', sa.String(20), server_default='pending', nullable=False)
    )

    # Add index on chapter_id for faster lookups
    op.create_index(
        'ix_translation_cache_chapter_id',
        'translation_cache',
        ['chapter_id']
    )


def downgrade() -> None:
    # Drop index
    op.drop_index('ix_translation_cache_chapter_id', table_name='translation_cache')

    # Remove status column
    op.drop_column('translation_cache', 'status')

    # Rename back
    op.alter_column(
        'translation_cache',
        'chapter_id',
        new_column_name='chapter'
    )

    op.alter_column(
        'translation_cache',
        'content_hash',
        new_column_name='source_hash'
    )
