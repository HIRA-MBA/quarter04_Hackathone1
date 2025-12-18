"""Authentication service for user management."""

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.session import Session
from app.models.user import User
from app.models.preference import UserPreference

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Service for authentication operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)

    @staticmethod
    def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=settings.access_token_expire_minutes
            )
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode, settings.secret_key, algorithm=settings.jwt_algorithm
        )
        return encoded_jwt

    @staticmethod
    def create_refresh_token(data: dict[str, Any]) -> str:
        """Create a JWT refresh token (7 days expiry)."""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(days=7)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(
            to_encode, settings.secret_key, algorithm=settings.jwt_algorithm
        )
        return encoded_jwt

    @staticmethod
    def decode_token(token: str) -> dict[str, Any] | None:
        """Decode and validate a JWT token."""
        try:
            payload = jwt.decode(
                token, settings.secret_key, algorithms=[settings.jwt_algorithm]
            )
            return payload
        except JWTError:
            return None

    async def get_user_by_email(self, email: str) -> User | None:
        """Get a user by email address."""
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_user_by_id(self, user_id: UUID) -> User | None:
        """Get a user by ID."""
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def create_user(
        self,
        email: str,
        password: str,
        full_name: str | None = None,
    ) -> User:
        """Create a new user with email/password."""
        hashed_password = self.get_password_hash(password)
        user = User(
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            is_active=True,
            is_verified=False,
        )
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)

        # Create default preferences
        preference = UserPreference(
            user_id=user.id,
            language="en",
            experience_level=None,
            completed_chapters={},
            bookmarks={},
        )
        self.db.add(preference)
        await self.db.commit()

        return user

    async def create_oauth_user(
        self,
        email: str,
        full_name: str | None,
        provider: str,
        oauth_id: str,
    ) -> User:
        """Create a new user from OAuth provider."""
        user = User(
            email=email,
            full_name=full_name,
            oauth_provider=provider,
            oauth_id=oauth_id,
            is_active=True,
            is_verified=True,  # OAuth users are pre-verified
        )
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)

        # Create default preferences
        preference = UserPreference(
            user_id=user.id,
            language="en",
            experience_level=None,
            completed_chapters={},
            bookmarks={},
        )
        self.db.add(preference)
        await self.db.commit()

        return user

    async def authenticate_user(self, email: str, password: str) -> User | None:
        """Authenticate a user with email and password."""
        user = await self.get_user_by_email(email)
        if not user:
            return None
        if not user.hashed_password:
            return None  # OAuth user, can't login with password
        if not self.verify_password(password, user.hashed_password):
            return None
        return user

    async def create_session(
        self,
        user_id: UUID,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> Session:
        """Create a new session for a user."""
        access_token = self.create_access_token({"sub": str(user_id)})
        expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=settings.access_token_expire_minutes
        )

        session = Session(
            user_id=user_id,
            token=access_token,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def invalidate_session(self, token: str) -> bool:
        """Invalidate a session by token."""
        result = await self.db.execute(select(Session).where(Session.token == token))
        session = result.scalar_one_or_none()
        if session:
            await self.db.delete(session)
            await self.db.commit()
            return True
        return False

    async def get_session_by_token(self, token: str) -> Session | None:
        """Get a session by token."""
        result = await self.db.execute(select(Session).where(Session.token == token))
        session = result.scalar_one_or_none()
        if session and session.expires_at > datetime.now(timezone.utc):
            return session
        return None

    async def refresh_session(self, refresh_token: str) -> tuple[str, str] | None:
        """Refresh an access token using a refresh token."""
        payload = self.decode_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None

        user_id = payload.get("sub")
        if not user_id:
            return None

        user = await self.get_user_by_id(UUID(user_id))
        if not user or not user.is_active:
            return None

        new_access_token = self.create_access_token({"sub": str(user_id)})
        new_refresh_token = self.create_refresh_token({"sub": str(user_id)})

        return new_access_token, new_refresh_token

    async def request_password_reset(self, email: str) -> str | None:
        """Generate a password reset token."""
        user = await self.get_user_by_email(email)
        if not user:
            return None

        reset_token = self.create_access_token(
            {"sub": str(user.id), "type": "reset"},
            expires_delta=timedelta(hours=1),
        )
        return reset_token

    async def reset_password(self, token: str, new_password: str) -> bool:
        """Reset a user's password using a reset token."""
        payload = self.decode_token(token)
        if not payload or payload.get("type") != "reset":
            return False

        user_id = payload.get("sub")
        if not user_id:
            return False

        user = await self.get_user_by_id(UUID(user_id))
        if not user:
            return False

        user.hashed_password = self.get_password_hash(new_password)
        await self.db.commit()
        return True

    def create_verification_token(self, user_id: UUID) -> str:
        """Create an email verification token (24 hours expiry)."""
        to_encode = {"sub": str(user_id), "type": "verify"}
        expire = datetime.now(timezone.utc) + timedelta(hours=24)
        to_encode["exp"] = expire
        encoded_jwt = jwt.encode(
            to_encode, settings.secret_key, algorithm=settings.jwt_algorithm
        )
        return encoded_jwt

    async def verify_email(self, token: str) -> User | None:
        """Verify a user's email using the verification token."""
        payload = self.decode_token(token)
        if not payload or payload.get("type") != "verify":
            return None

        user_id = payload.get("sub")
        if not user_id:
            return None

        user = await self.get_user_by_id(UUID(user_id))
        if not user:
            return None

        if user.is_verified:
            return user  # Already verified

        user.is_verified = True
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def resend_verification(self, email: str) -> str | None:
        """Generate a new verification token for an unverified user."""
        user = await self.get_user_by_email(email)
        if not user or user.is_verified:
            return None

        return self.create_verification_token(user.id)

    async def get_user_by_oauth(self, provider: str, oauth_id: str) -> User | None:
        """Get a user by OAuth provider and ID."""
        result = await self.db.execute(
            select(User).where(
                User.oauth_provider == provider,
                User.oauth_id == oauth_id,
            )
        )
        return result.scalar_one_or_none()

    async def link_oauth_to_user(
        self,
        user: User,
        provider: str,
        oauth_id: str,
    ) -> User:
        """Link an OAuth account to an existing user."""
        user.oauth_provider = provider
        user.oauth_id = oauth_id
        user.is_verified = True  # OAuth linking verifies email
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def get_or_create_oauth_user(
        self,
        email: str,
        full_name: str | None,
        provider: str,
        oauth_id: str,
    ) -> tuple[User, bool]:
        """
        Get or create a user from OAuth data.

        Returns:
            Tuple of (user, is_new_user)
        """
        # First check if OAuth account is already linked
        existing_oauth_user = await self.get_user_by_oauth(provider, oauth_id)
        if existing_oauth_user:
            return existing_oauth_user, False

        # Check if email exists (user may have signed up with password)
        existing_email_user = await self.get_user_by_email(email)
        if existing_email_user:
            # Link OAuth to existing account
            await self.link_oauth_to_user(existing_email_user, provider, oauth_id)
            return existing_email_user, False

        # Create new OAuth user
        user = await self.create_oauth_user(email, full_name, provider, oauth_id)
        return user, True
