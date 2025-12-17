"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client fixture."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Create mock auth headers for testing."""
    return {"Authorization": "Bearer test_token"}
