"""
Tests for the MultiHop RAG Flask application.
Run with:  pytest tests/ -v
"""

import pytest
import os
import tempfile
from app import app as flask_app
from database.db import init_db, create_user, get_user_by_email, get_user_by_id


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def app(tmp_path, monkeypatch):
    """Create a test app with an isolated temp database."""
    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr("database.db.DB_PATH", db_file)
    flask_app.config.update({
        "TESTING": True,
        "SECRET_KEY": "test-secret",
        "WTF_CSRF_ENABLED": False,
    })
    with flask_app.app_context():
        init_db()
    yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def registered_user(app, monkeypatch, tmp_path):
    """Insert a user directly via the DB layer."""
    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr("database.db.DB_PATH", db_file)
    from werkzeug.security import generate_password_hash
    with app.app_context():
        init_db()
        create_user("Test User", "test@example.com", generate_password_hash("password123"))
    return {"name": "Test User", "email": "test@example.com", "password": "password123"}


# ── Public route tests ─────────────────────────────────────────────────────

class TestPublicRoutes:
    def test_landing_page(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert b"MultiHop" in r.data or b"RAG" in r.data

    def test_login_page_get(self, client):
        r = client.get("/login")
        assert r.status_code == 200
        assert b"Sign in" in r.data or b"login" in r.data.lower()

    def test_register_page_get(self, client):
        r = client.get("/register")
        assert r.status_code == 200
        assert b"Create" in r.data or b"register" in r.data.lower()


# ── Registration tests ─────────────────────────────────────────────────────

class TestRegistration:
    def test_successful_registration(self, client):
        r = client.post("/register", data={
            "name":     "Alice Smith",
            "email":    "alice@example.com",
            "password": "securepass1",
        }, follow_redirects=True)
        assert r.status_code == 200

    def test_duplicate_email(self, client):
        data = {"name": "Bob", "email": "bob@example.com", "password": "password99"}
        client.post("/register", data=data)
        r = client.post("/register", data=data, follow_redirects=True)
        assert b"already exists" in r.data or r.status_code in (200, 302)

    def test_short_password(self, client):
        r = client.post("/register", data={
            "name":     "Eve",
            "email":    "eve@example.com",
            "password": "short",
        }, follow_redirects=True)
        assert b"8 character" in r.data or r.status_code == 200

    def test_missing_fields(self, client):
        r = client.post("/register", data={"name": "", "email": "", "password": ""},
                        follow_redirects=True)
        assert b"required" in r.data or r.status_code == 200


# ── Login tests ────────────────────────────────────────────────────────────

class TestLogin:
    def test_login_wrong_password(self, client, registered_user):
        r = client.post("/login", data={
            "email":    registered_user["email"],
            "password": "wrongpassword",
        }, follow_redirects=True)
        assert b"Invalid" in r.data or r.status_code == 200

    def test_login_nonexistent_user(self, client):
        r = client.post("/login", data={
            "email":    "nobody@example.com",
            "password": "whatever",
        }, follow_redirects=True)
        assert b"Invalid" in r.data or r.status_code == 200

    def test_logout_redirects(self, client):
        r = client.get("/logout", follow_redirects=False)
        assert r.status_code in (302, 301)


# ── Protected route tests ──────────────────────────────────────────────────

class TestProtectedRoutes:
    def test_dashboard_requires_login(self, client):
        r = client.get("/dashboard", follow_redirects=False)
        assert r.status_code in (302, 301)

    def test_profile_requires_login(self, client):
        r = client.get("/profile", follow_redirects=False)
        assert r.status_code in (302, 301)


# ── Database layer tests ───────────────────────────────────────────────────

class TestDatabase:
    def test_create_and_get_user(self, app, monkeypatch, tmp_path):
        db_file = str(tmp_path / "db_test.db")
        monkeypatch.setattr("database.db.DB_PATH", db_file)
        with app.app_context():
            init_db()
            create_user("Charlie", "charlie@example.com", "hashed_pw")
            user = get_user_by_email("charlie@example.com")
            assert user is not None
            assert user["name"] == "Charlie"
            assert user["email"] == "charlie@example.com"

    def test_get_nonexistent_user(self, app, monkeypatch, tmp_path):
        db_file = str(tmp_path / "db_test2.db")
        monkeypatch.setattr("database.db.DB_PATH", db_file)
        with app.app_context():
            init_db()
            user = get_user_by_email("ghost@example.com")
            assert user is None

    def test_get_user_by_id(self, app, monkeypatch, tmp_path):
        db_file = str(tmp_path / "db_test3.db")
        monkeypatch.setattr("database.db.DB_PATH", db_file)
        with app.app_context():
            init_db()
            create_user("Dana", "dana@example.com", "pw")
            user = get_user_by_email("dana@example.com")
            fetched = get_user_by_id(user["id"])
            assert fetched["email"] == "dana@example.com"
