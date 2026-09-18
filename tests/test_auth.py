"""Registration, login, password hashing and JWT validation."""

import jwt
import pytest
from sqlalchemy import select

from src.auth import create_access_token, hash_password, verify_password
from src.config import settings
from src.models import User
from tests.conftest import auth, register_and_login

PASSWORD = "correct-horse-battery"


# --------------------------------------------------------------------------
# Password hashing
# --------------------------------------------------------------------------


def test_hash_is_not_plaintext_and_is_salted():
    first = hash_password(PASSWORD)
    second = hash_password(PASSWORD)

    assert PASSWORD not in first
    assert first.startswith("$2b$")
    assert first != second, "each hash must use a fresh salt"
    assert verify_password(PASSWORD, first)
    assert verify_password(PASSWORD, second)


def test_verify_rejects_wrong_password_and_malformed_hash():
    assert not verify_password("wrong", hash_password(PASSWORD))
    assert not verify_password(PASSWORD, "not-a-bcrypt-hash")


def test_password_hash_is_persisted_not_the_password(client, db_session):
    client.post("/register", json={"email": "a@example.com", "password": PASSWORD})

    user = db_session.scalar(select(User).where(User.email == "a@example.com"))
    assert user is not None
    assert user.password_hash != PASSWORD
    assert verify_password(PASSWORD, user.password_hash)


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def test_register_returns_user_without_password_hash(client):
    response = client.post("/register", json={"email": "New@Example.com", "password": PASSWORD})

    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "new@example.com", "email should be normalised to lowercase"
    assert "password" not in body
    assert "password_hash" not in body


def test_duplicate_email_is_rejected(client):
    client.post("/register", json={"email": "dupe@example.com", "password": PASSWORD})
    response = client.post("/register", json={"email": "dupe@example.com", "password": PASSWORD})

    assert response.status_code == 409


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "not-an-email", "password": PASSWORD},
        {"email": "short@example.com", "password": "abc"},
        {"email": "missing@example.com"},
    ],
)
def test_invalid_registration_payloads_are_rejected(client, payload):
    assert client.post("/register", json=payload).status_code == 422


def test_password_longer_than_bcrypt_limit_is_rejected(client):
    # bcrypt silently truncates past 72 bytes; the API must refuse instead.
    response = client.post(
        "/register", json={"email": "long@example.com", "password": "x" * 100}
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------
# Login
# --------------------------------------------------------------------------


def test_login_returns_a_usable_bearer_token(client):
    token = register_and_login(client, "login@example.com")

    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    assert "sub" in payload and "exp" in payload


def test_login_with_wrong_password_is_401(client):
    client.post("/register", json={"email": "pw@example.com", "password": PASSWORD})
    response = client.post("/login", json={"email": "pw@example.com", "password": "nope"})

    assert response.status_code == 401


def test_login_for_unknown_email_gives_the_same_error(client):
    client.post("/register", json={"email": "known@example.com", "password": PASSWORD})

    wrong_pw = client.post("/login", json={"email": "known@example.com", "password": "nope"})
    unknown = client.post("/login", json={"email": "ghost@example.com", "password": "nope"})

    # Identical responses, so login cannot be used to enumerate accounts.
    assert wrong_pw.status_code == unknown.status_code == 401
    assert wrong_pw.json() == unknown.json()


# --------------------------------------------------------------------------
# Protected endpoints
# --------------------------------------------------------------------------


def test_me_returns_the_authenticated_user(client):
    token = register_and_login(client, "me@example.com")
    response = client.get("/me", headers=auth(token))

    assert response.status_code == 200
    assert response.json()["email"] == "me@example.com"
    assert "password_hash" not in response.json()


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer not-a-jwt"},
        {"Authorization": "Basic abc123"},
    ],
    ids=["no-header", "garbage-token", "wrong-scheme"],
)
def test_me_rejects_bad_credentials(client, headers):
    assert client.get("/me", headers=headers).status_code == 401


def test_token_signed_with_another_secret_is_rejected(client):
    register_and_login(client, "victim@example.com")
    forged = jwt.encode({"sub": "1"}, "attacker-secret", algorithm="HS256")

    assert client.get("/me", headers=auth(forged)).status_code == 401


def test_expired_token_is_rejected(client, monkeypatch):
    monkeypatch.setattr(settings, "jwt_expire_minutes", -1)
    token = create_access_token(user_id=1)

    assert client.get("/me", headers=auth(token)).status_code == 401


def test_token_for_deleted_user_is_rejected(client):
    token = create_access_token(user_id=99999)  # valid signature, no such user

    assert client.get("/me", headers=auth(token)).status_code == 401
