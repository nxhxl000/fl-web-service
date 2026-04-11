from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.clients.models import ClientToken
from tests.conftest import auth_headers, create_project


def _issue_token(client: TestClient, db_session: Session) -> str:
    admin = auth_headers(client, db_session, "admin@example.com", admin=True)
    project = create_project(client, admin)
    user = auth_headers(client, db_session, "user@example.com")
    created = client.post(
        f"/projects/{project['id']}/tokens",
        headers=user,
        json={"name": "laptop"},
    ).json()
    return created["token"]


def test_heartbeat_updates_last_seen(client: TestClient, db_session: Session) -> None:
    token = _issue_token(client, db_session)

    assert db_session.query(ClientToken).one().last_seen_at is None

    resp = client.post(
        "/client/heartbeat",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 204

    db_session.expire_all()
    assert db_session.query(ClientToken).one().last_seen_at is not None


def test_heartbeat_advances_timestamp(client: TestClient, db_session: Session) -> None:
    token = _issue_token(client, db_session)
    headers = {"Authorization": f"Bearer {token}"}

    client.post("/client/heartbeat", headers=headers)
    db_session.expire_all()
    first = db_session.query(ClientToken).one().last_seen_at

    client.post("/client/heartbeat", headers=headers)
    db_session.expire_all()
    second = db_session.query(ClientToken).one().last_seen_at

    assert first is not None and second is not None
    assert second >= first


def test_heartbeat_requires_bearer(client: TestClient) -> None:
    assert client.post("/client/heartbeat").status_code == 401


def test_heartbeat_rejects_unknown_token(client: TestClient) -> None:
    resp = client.post(
        "/client/heartbeat",
        headers={"Authorization": "Bearer flwc_not-a-real-token"},
    )
    assert resp.status_code == 401


def test_heartbeat_rejects_deleted_token(client: TestClient, db_session: Session) -> None:
    admin = auth_headers(client, db_session, "admin@example.com", admin=True)
    project = create_project(client, admin)
    user = auth_headers(client, db_session, "user@example.com")
    created = client.post(
        f"/projects/{project['id']}/tokens",
        headers=user,
        json={"name": "laptop"},
    ).json()
    token = created["token"]

    client.delete(f"/projects/{project['id']}/tokens/{created['id']}", headers=user)

    resp = client.post(
        "/client/heartbeat",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401
