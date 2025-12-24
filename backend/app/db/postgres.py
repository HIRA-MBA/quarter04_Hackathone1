from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings


def get_engine():
    """Create async database engine."""
    settings = get_settings()
    db_url = settings.database_url

    # Auto-convert to asyncpg driver if needed
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)

    # Handle SSL parameter differences (asyncpg uses ssl= not sslmode=)
    db_url = db_url.replace("sslmode=require", "ssl=require")
    db_url = db_url.replace("&channel_binding=require", "")

    return create_async_engine(
        db_url,
        echo=settings.debug,
        pool_pre_ping=True,
    )


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Create async session maker."""
    engine = get_engine()
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@asynccontextmanager
async def get_db_session() -> AsyncSession:
    """Dependency for getting database sessions."""
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
