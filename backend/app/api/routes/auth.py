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


# OAuth routes
@router.get("/oauth/{provider}")
async def oauth_redirect(provider: str) -> OAuthUrlResponse:
    """Get OAuth redirect URL for the specified provider."""
    import secrets

    if provider not in ("google", "github"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported OAuth provider",
        )

    # Generate state for CSRF protection
    state = secrets.token_urlsafe(32)
    oauth_states[state] = provider

    if provider == "github":
        if not settings.github_client_id:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="GitHub OAuth not configured. Set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET.",
            )
        # GitHub OAuth URL
        redirect_uri = settings.github_redirect_uri
        url = (
            f"https://github.com/login/oauth/authorize"
            f"?client_id={settings.github_client_id}"
            f"&redirect_uri={redirect_uri}"
            f"&scope=user:email"
            f"&state={state}"
        )
    elif provider == "google":
        if not settings.google_client_id:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Google OAuth not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.",
            )
        # Google OAuth URL
        redirect_uri = settings.google_redirect_uri
        url = (
            f"https://accounts.google.com/o/oauth2/v2/auth"
            f"?client_id={settings.google_client_id}"
            f"&redirect_uri={redirect_uri}"
            f"&response_type=code"
            f"&scope=email%20profile"
            f"&state={state}"
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported provider")

    return OAuthUrlResponse(url=url, state=state)


@router.get("/status")
async def auth_status() -> dict:
    """Check auth service status and database connectivity."""
    from sqlalchemy import text

    from app.db.postgres import get_session_maker

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
    request: Request,
    provider: str,
    code: str,
    state: str | None = None,
    auth_service: Annotated[AuthService, Depends(get_auth_service)] = None,
):
    """Handle OAuth callback from provider."""
    import httpx

    if provider not in ("google", "github"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported OAuth provider",
        )

    # Verify state (CSRF protection)
    if state and state in oauth_states:
        del oauth_states[state]
    # Note: In production, you should strictly verify state

    if provider == "github":
        # Exchange code for access token
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://github.com/login/oauth/access_token",
                data={
                    "client_id": settings.github_client_id,
                    "client_secret": settings.github_client_secret,
                    "code": code,
                    "redirect_uri": settings.github_redirect_uri,
                },
                headers={"Accept": "application/json"},
            )

            if token_response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to exchange code for token",
                )

            token_data = token_response.json()
            access_token = token_data.get("access_token")

            if not access_token:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"No access token received: {token_data.get('error_description', 'Unknown error')}",
                )

            # Get user info from GitHub
            user_response = await client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
            )

            if user_response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get user info from GitHub",
                )

            github_user = user_response.json()

            # Get user's email (may need separate call if email is private)
            email = github_user.get("email")
            if not email:
                emails_response = await client.get(
                    "https://api.github.com/user/emails",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/json",
                    },
                )
                if emails_response.status_code == 200:
                    emails = emails_response.json()
                    # Get primary email
                    for e in emails:
                        if e.get("primary"):
                            email = e.get("email")
                            break
                    if not email and emails:
                        email = emails[0].get("email")

            if not email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Could not get email from GitHub. Please make sure your email is public or grant email permission.",
                )

            full_name = github_user.get("name") or github_user.get("login")
            oauth_id = str(github_user.get("id"))

    elif provider == "google":
        # Exchange code for access token
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": settings.google_client_id,
                    "client_secret": settings.google_client_secret,
                    "code": code,
                    "redirect_uri": settings.google_redirect_uri,
                    "grant_type": "authorization_code",
                },
            )

            if token_response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to exchange code for token",
                )

            token_data = token_response.json()
            access_token = token_data.get("access_token")

            # Get user info from Google
            user_response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )

            if user_response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get user info from Google",
                )

            google_user = user_response.json()
            email = google_user.get("email")
            full_name = google_user.get("name")
            oauth_id = google_user.get("id")

    else:
        raise HTTPException(status_code=400, detail="Unsupported provider")

    # Check if user exists
    existing_user = await auth_service.get_user_by_email(email)

    if existing_user:
        # User exists - log them in
        user = existing_user
        # Update OAuth info if not set
        if not existing_user.oauth_provider:
            existing_user.oauth_provider = provider
            existing_user.oauth_id = oauth_id
            await auth_service.db.commit()
    else:
        # Create new user
        user = await auth_service.create_oauth_user(
            email=email,
            full_name=full_name,
            oauth_provider=provider,
            oauth_id=oauth_id,
        )

    # Create tokens
    jwt_access_token = auth_service.create_access_token({"sub": str(user.id)})
    refresh_token = auth_service.create_refresh_token({"sub": str(user.id)})

    # Create session
    await auth_service.create_session(
        user_id=user.id,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    # Redirect to frontend with tokens
    from urllib.parse import urlencode

    from fastapi.responses import RedirectResponse

    frontend_url = settings.frontend_url
    params = urlencode({
        "access_token": jwt_access_token,
        "refresh_token": refresh_token,
    })
    redirect_url = f"{frontend_url}/auth/callback?{params}"

    return RedirectResponse(url=redirect_url)
