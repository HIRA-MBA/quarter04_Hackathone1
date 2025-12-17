"""API endpoint tests for Physical AI Textbook backend."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestHealthEndpoints:
    """Test health and root endpoints."""

    def test_health_check(self):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data

    def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Physical AI" in data["message"]


class TestChatEndpoints:
    """Test chat/RAG endpoints."""

    def test_chat_stats(self):
        """Test chat stats endpoint."""
        response = client.get("/api/chat/stats")
        assert response.status_code == 200
        data = response.json()
        assert "collection_name" in data

    def test_chat_search(self):
        """Test chat search endpoint."""
        response = client.get("/api/chat/search?q=ROS%202")
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data

    def test_chat_message_validation(self):
        """Test chat message validation."""
        # Empty message should fail
        response = client.post("/api/chat", json={"message": ""})
        assert response.status_code == 422


class TestTranslationEndpoints:
    """Test translation endpoints."""

    def test_supported_languages(self):
        """Test supported languages endpoint."""
        response = client.get("/api/translation/languages")
        assert response.status_code == 200
        data = response.json()
        assert "source_languages" in data
        assert "target_languages" in data
        assert any(lang["code"] == "ur" for lang in data["target_languages"])


class TestAuthEndpoints:
    """Test authentication endpoints."""

    def test_signup_validation(self):
        """Test signup validates email format."""
        response = client.post(
            "/api/auth/signup",
            json={
                "email": "invalid-email",
                "password": "testpassword123",
            },
        )
        assert response.status_code == 422

    @pytest.mark.skip(reason="Requires database connection")
    def test_signin_invalid_credentials(self):
        """Test signin with invalid credentials."""
        response = client.post(
            "/api/auth/signin",
            json={
                "email": "nonexistent@example.com",
                "password": "wrongpassword",
            },
        )
        assert response.status_code == 401

    def test_me_unauthorized(self):
        """Test /me endpoint without auth."""
        response = client.get("/api/auth/me")
        assert response.status_code == 401


class TestUserEndpoints:
    """Test user endpoints."""

    def test_preferences_unauthorized(self):
        """Test preferences endpoint without auth."""
        response = client.get("/api/user/preferences")
        assert response.status_code == 401

    def test_progress_unauthorized(self):
        """Test progress endpoint without auth."""
        response = client.get("/api/user/progress")
        assert response.status_code == 401
