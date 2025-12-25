from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings

# Cache engine and session maker to avoid recreating on every request
_engine = None
_session_maker = None


def get_engine():
    """Create async database engine (cached)."""
    global _engine
    if _engine is None:
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

        _engine = create_async_engine(
            db_url,
            echo=settings.debug,
            pool_pre_ping=True,
        )
    return _engine


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Create async session maker (cached)."""
    global _session_maker
    if _session_maker is None:
        engine = get_engine()
        _session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_maker


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for getting database sessions."""
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
