import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool, create_engine
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.config import get_settings
from app.models import Base  # Import Base to get target_metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the database URL from settings
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

# Set target metadata from models for autogenerate support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with the given connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using synchronous engine.

    Alembic doesn't natively support async, so we use a sync engine
    by removing the +asyncpg driver suffix from the URL.
    """
    # Get database URL and convert to sync driver
    db_url = settings.database_url

    # Replace async driver with sync driver
    if "+asyncpg" in db_url:
        sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    else:
        sync_url = db_url

    connectable = create_engine(
        sync_url,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        do_run_migrations(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
