"""API dependencies for authentication and database access."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.postgres import get_db_session
from app.models.user import User
from app.services.auth.auth import AuthService

settings = get_settings()
security = HTTPBearer(auto_error=False)


async def get_auth_service(
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> AuthService:
    """Get authentication service instance."""
    return AuthService(db)


async def get_current_user_optional(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> User | None:
    """Get current user if authenticated, None otherwise."""
    if not credentials:
        return None

    token = credentials.credentials
    payload = auth_service.decode_token(token)
    if not payload:
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None

    # Skip type check for refresh tokens
    if payload.get("type") in ("refresh", "reset"):
        return None

    user = await auth_service.get_user_by_id(user_id)
    if not user or not user.is_active:
        return None

    return user


async def get_current_user(
    user: Annotated[User | None, Depends(get_current_user_optional)],
) -> User:
    """Get current authenticated user (required)."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_current_verified_user(
    user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Get current authenticated and verified user."""
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified",
        )
    return user


# Rate limiting implementation
class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[float]] = {}

    async def check_rate_limit(self, key: str) -> bool:
        """Check if request is within rate limit."""
        import time

        current_time = time.time()
        window_start = current_time - 60

        if key not in self.requests:
            self.requests[key] = []

        # Clean old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key] if req_time > window_start
        ]

        if len(self.requests[key]) >= self.requests_per_minute:
            return False

        self.requests[key].append(current_time)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter(requests_per_minute=60)


async def check_rate_limit(request: Request) -> None:
    """Dependency to check rate limit."""
    client_ip = request.client.host if request.client else "unknown"
    if not await rate_limiter.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
        )
