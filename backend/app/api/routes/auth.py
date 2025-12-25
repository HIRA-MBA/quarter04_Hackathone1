"""Authentication API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field

from app.api.deps import (
    check_rate_limit,
    get_auth_service,
    get_current_user,
)
from app.config import get_settings
from app.models.user import User
from app.services.auth.auth import AuthService
from app.services.auth.email import EmailService

settings = get_settings()
router = APIRouter(prefix="/auth", tags=["auth"])

# In-memory state storage for OAuth (use Redis in production)
oauth_states: dict[str, str] = {}


# Request/Response schemas
class SignupRequest(BaseModel):
    """Signup request schema."""

    email: EmailStr
    password: str = Field(min_length=8, max_length=100)
    full_name: str | None = Field(None, max_length=255)


class SigninRequest(BaseModel):
    """Signin request schema."""

    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response schema."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    """Refresh token request schema."""

    refresh_token: str


class PasswordResetRequest(BaseModel):
    """Password reset request schema."""

    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation schema."""

    token: str
    new_password: str = Field(min_length=8, max_length=100)


class UserResponse(BaseModel):
    """User response schema."""

    id: str
    email: str
    full_name: str | None
    is_verified: bool
    oauth_provider: str | None

    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str


class VerifyEmailRequest(BaseModel):
    """Email verification request schema."""

    token: str


class ResendVerificationRequest(BaseModel):
    """Resend verification email request schema."""

    email: EmailStr


class OAuthUrlResponse(BaseModel):
    """OAuth URL response schema."""

    url: str
    state: str


class OAuthProvidersResponse(BaseModel):
    """Available OAuth providers response."""

    google: bool
    github: bool


# Routes
@router.post(
    "/signup",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_rate_limit)],
)
async def signup(
    request: Request,
    body: SignupRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenResponse:
    """Register a new user with email and password."""
    # Check if user already exists
    existing_user = await auth_service.get_user_by_email(body.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create user
    user = await auth_service.create_user(
        email=body.email,
        password=body.password,
        full_name=body.full_name,
    )

    # Send verification email (don't block signup if email fails)
    try:
        verification_token = auth_service.create_verification_token(user.id)
        await EmailService.send_verification_email(
            to_email=user.email,
            token=verification_token,
            full_name=user.full_name,
        )
    except Exception as e:
        # Log but don't fail signup - user can request verification email later
        print(f"Failed to send verification email: {e}")

    # Create tokens
    access_token = auth_service.create_access_token({"sub": str(user.id)})
    refresh_token = auth_service.create_refresh_token({"sub": str(user.id)})

    # Create session
    await auth_service.create_session(
        user_id=user.id,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=30 * 60,  # 30 minutes
    )


@router.post(
    "/signin",
    response_model=TokenResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def signin(
    request: Request,
    body: SigninRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenResponse:
    """Sign in with email and password."""
    user = await auth_service.authenticate_user(body.email, body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    # Create tokens
    access_token = auth_service.create_access_token({"sub": str(user.id)})
    refresh_token = auth_service.create_refresh_token({"sub": str(user.id)})

    # Create session
    await auth_service.create_session(
        user_id=user.id,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=30 * 60,
    )


@router.post("/signout", response_model=MessageResponse)
async def signout(
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> MessageResponse:
    """Sign out and invalidate the current session."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        await auth_service.invalidate_session(token)

    return MessageResponse(message="Successfully signed out")


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    body: RefreshRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenResponse:
    """Refresh access token using refresh token."""
    result = await auth_service.refresh_session(body.refresh_token)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    access_token, new_refresh_token = result
    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=30 * 60,
    )


@router.post(
    "/password-reset",
    response_model=MessageResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def request_password_reset(
    body: PasswordResetRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> MessageResponse:
    """Request a password reset email."""
    # Always return success to prevent email enumeration
    user = await auth_service.get_user_by_email(body.email)
    reset_token = await auth_service.request_password_reset(body.email)

    # Send password reset email
    if reset_token and user:
        await EmailService.send_password_reset_email(
            to_email=body.email,
            token=reset_token,
            full_name=user.full_name,
        )

    return MessageResponse(
        message="If an account exists with this email, a reset link has been sent"
    )


@router.post("/password-reset/confirm", response_model=MessageResponse)
async def confirm_password_reset(
    body: PasswordResetConfirm,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> MessageResponse:
    """Reset password using the reset token."""
    success = await auth_service.reset_password(body.token, body.new_password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    return MessageResponse(message="Password has been reset successfully")


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Annotated[User, Depends(get_current_user)],
) -> UserResponse:
    """Get current authenticated user's information."""
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        is_verified=current_user.is_verified,
        oauth_provider=current_user.oauth_provider,
    )


# OAuth routes (placeholders for Google/GitHub)
@router.get("/oauth/{provider}")
async def oauth_redirect(provider: str) -> MessageResponse:
    """Redirect to OAuth provider for authentication."""
    if provider not in ("google", "github"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported OAuth provider",
        )

    # In production, redirect to OAuth provider
    # This is a placeholder that would be implemented with actual OAuth flow
    return MessageResponse(message=f"OAuth with {provider} - redirect URL would be generated here")


@router.get("/status")
async def auth_status() -> dict:
    """Check auth service status and database connectivity."""
    from app.db.postgres import get_session_maker
    from sqlalchemy import text

    result = {
        "auth_service": "ok",
        "database": "unknown",
        "email_configured": EmailService.is_configured(),
    }

    try:
        session_maker = get_session_maker()
        async with session_maker() as session:
            await session.execute(text("SELECT 1"))
        result["database"] = "ok"
    except Exception as e:
        result["database"] = f"error: {type(e).__name__}: {str(e)}"

    return result


@router.get("/oauth/{provider}/callback")
async def oauth_callback(
    provider: str,
    code: str,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenResponse:
    """Handle OAuth callback from provider."""
    if provider not in ("google", "github"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported OAuth provider",
        )

    # In production, exchange code for tokens and get user info
    # This is a placeholder implementation
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="OAuth callback handling not yet implemented",
    )
