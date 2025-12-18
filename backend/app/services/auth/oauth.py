"""OAuth service for Google and GitHub authentication."""

import secrets
from typing import Any
from urllib.parse import urlencode

import httpx

from app.config import get_settings

settings = get_settings()


class OAuthService:
    """Service for OAuth authentication with Google and GitHub."""

    # Google OAuth endpoints
    GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
    GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

    # GitHub OAuth endpoints
    GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
    GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
    GITHUB_USERINFO_URL = "https://api.github.com/user"
    GITHUB_EMAILS_URL = "https://api.github.com/user/emails"

    @staticmethod
    def generate_state() -> str:
        """Generate a random state parameter for OAuth security."""
        return secrets.token_urlsafe(32)

    @classmethod
    def get_google_auth_url(cls, state: str) -> str:
        """Generate Google OAuth authorization URL."""
        params = {
            "client_id": settings.google_client_id,
            "redirect_uri": settings.google_redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }
        return f"{cls.GOOGLE_AUTH_URL}?{urlencode(params)}"

    @classmethod
    def get_github_auth_url(cls, state: str) -> str:
        """Generate GitHub OAuth authorization URL."""
        params = {
            "client_id": settings.github_client_id,
            "redirect_uri": settings.github_redirect_uri,
            "scope": "read:user user:email",
            "state": state,
        }
        return f"{cls.GITHUB_AUTH_URL}?{urlencode(params)}"

    @classmethod
    async def exchange_google_code(cls, code: str) -> dict[str, Any] | None:
        """Exchange Google authorization code for tokens."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    cls.GOOGLE_TOKEN_URL,
                    data={
                        "client_id": settings.google_client_id,
                        "client_secret": settings.google_client_secret,
                        "code": code,
                        "grant_type": "authorization_code",
                        "redirect_uri": settings.google_redirect_uri,
                    },
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError:
                return None

    @classmethod
    async def exchange_github_code(cls, code: str) -> dict[str, Any] | None:
        """Exchange GitHub authorization code for tokens."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    cls.GITHUB_TOKEN_URL,
                    data={
                        "client_id": settings.github_client_id,
                        "client_secret": settings.github_client_secret,
                        "code": code,
                        "redirect_uri": settings.github_redirect_uri,
                    },
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError:
                return None

    @classmethod
    async def get_google_user_info(cls, access_token: str) -> dict[str, Any] | None:
        """Get user info from Google using access token."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    cls.GOOGLE_USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError:
                return None

    @classmethod
    async def get_github_user_info(cls, access_token: str) -> dict[str, Any] | None:
        """Get user info from GitHub using access token."""
        async with httpx.AsyncClient() as client:
            try:
                # Get basic user info
                response = await client.get(
                    cls.GITHUB_USERINFO_URL,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/vnd.github+json",
                    },
                )
                response.raise_for_status()
                user_data = response.json()

                # Get user's email if not public
                if not user_data.get("email"):
                    emails_response = await client.get(
                        cls.GITHUB_EMAILS_URL,
                        headers={
                            "Authorization": f"Bearer {access_token}",
                            "Accept": "application/vnd.github+json",
                        },
                    )
                    emails_response.raise_for_status()
                    emails = emails_response.json()

                    # Find primary email
                    for email in emails:
                        if email.get("primary") and email.get("verified"):
                            user_data["email"] = email["email"]
                            break

                return user_data
            except httpx.HTTPError:
                return None

    @classmethod
    def is_google_configured(cls) -> bool:
        """Check if Google OAuth is properly configured."""
        return bool(settings.google_client_id and settings.google_client_secret)

    @classmethod
    def is_github_configured(cls) -> bool:
        """Check if GitHub OAuth is properly configured."""
        return bool(settings.github_client_id and settings.github_client_secret)
