"""Shared test fixtures.

Environment variables are set before importing anything from `src`, because
`src.config.settings` is instantiated at import time.
"""

import os

os.environ["JWT_SECRET"] = "test-secret-key-that-is-long-enough-for-hs256"
os.environ["GEMINI_API_KEY"] = "test-key-not-used-because-gemini-is-mocked"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from src import models  # noqa: E402,F401  - registers mappers
from src.actions import Action  # noqa: E402
from src.api import app  # noqa: E402
from src.config import settings  # noqa: E402
from src.database import Base, get_db  # noqa: E402
from src.schemas import LLMDecision, PhotoAnalysis  # noqa: E402

STUB_DECISION = LLMDecision(
    action=Action.REQUEST_PHOTOS,
    confidence=0.91,
    reason="Stubbed decision used by the automated tests.",
    sources=["damaged_goods.md"],
)

STUB_ANALYSIS = PhotoAnalysis(
    description="A white mug with a large crack along one side, next to a torn box.",
    is_clear=True,
    is_relevant=True,
    shows_issue=True,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if a test reaches for the real Gemini client.

    Without this a test that forgets to stub something makes a live API call:
    slow, flaky, and it fails in CI where there is no key. Anything that needs
    the client must stub it explicitly.
    """
    import google.genai

    from src import retrieval

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "A test tried to construct a real Gemini client. Stub the call instead."
        )

    monkeypatch.setattr(google.genai, "Client", forbidden)
    retrieval._load_index.cache_clear()
    retrieval._make_client.cache_clear()
    yield
    retrieval._load_index.cache_clear()
    retrieval._make_client.cache_clear()


@pytest.fixture
def db_session():
    """A fresh in-memory database per test.

    StaticPool keeps every connection pointed at the same in-memory database;
    without it each connection would get its own empty one.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def client(db_session, monkeypatch, tmp_path):
    """TestClient wired to the in-memory DB, with both LLM calls stubbed out.

    Uploads go to a per-test temporary folder, never the real uploads/ dir.
    """
    monkeypatch.setattr("src.api.generate_decision", lambda ticket, history=None: STUB_DECISION)
    monkeypatch.setattr(
        "src.api.analyze_photos",
        lambda photos, complaint: [STUB_ANALYSIS for _ in photos],
    )
    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "uploads")

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def register_and_login(client, email: str, password: str = "correct-horse-battery") -> str:
    """Create an account and return its bearer token."""
    response = client.post("/register", json={"email": email, "password": password})
    assert response.status_code == 201, response.text
    response = client.post("/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
