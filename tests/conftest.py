from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.auth import models as _auth_models  # noqa: F401 — register tables
from backend.auth.models import User
from backend.clients import models as _client_models  # noqa: F401
from backend.db import Base, get_db
from backend.main import app
from backend.projects import models as _project_models  # noqa: F401


@pytest.fixture()
def db_session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture()
def client(db_session: Session) -> Iterator[TestClient]:
    def override_get_db() -> Iterator[Session]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def auth_headers(
    client: TestClient,
    db: Session,
    email: str = "dev@example.com",
    *,
    admin: bool = False,
) -> dict[str, str]:
    client.post("/auth/register", json={"email": email, "password": "supersecret"})
    if admin:
        user = db.query(User).filter(User.email == email).one()
        user.is_admin = True
        db.commit()
    token = client.post(
        "/auth/login",
        json={"email": email, "password": "supersecret"},
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def create_project(
    client: TestClient,
    headers: dict[str, str],
    name: str = "CIFAR-100 non-IID",
) -> dict:
    resp = client.post(
        "/projects",
        headers=headers,
        json={
            "name": name,
            "summary": "Short debug experiment blurb",
            "description": "Debug experiment",
            "requirements": "1 vCPU, 2 GB RAM",
        },
    )
    assert resp.status_code == 201
    return resp.json()
