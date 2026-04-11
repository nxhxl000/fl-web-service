from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from tests.conftest import auth_headers, create_project


def test_create_list_delete_token(client: TestClient, db_session: Session) -> None:
    admin = auth_headers(client, db_session, "admin@example.com", admin=True)
    project = create_project(client, admin)
    pid = project["id"]

    user = auth_headers(client, db_session, "user@example.com")

    create_resp = client.post(
        f"/projects/{pid}/tokens",
        json={"name": "laptop"},
        headers=user,
    )
    assert create_resp.status_code == 201
    created = create_resp.json()
    assert created["name"] == "laptop"
    assert created["token"].startswith("flwc_")
    token_id = created["id"]

    list_resp = client.get(f"/projects/{pid}/tokens", headers=user)
    assert list_resp.status_code == 200
    tokens = list_resp.json()
    assert len(tokens) == 1
    assert tokens[0]["id"] == token_id
    assert "token" not in tokens[0]

    delete_resp = client.delete(f"/projects/{pid}/tokens/{token_id}", headers=user)
    assert delete_resp.status_code == 204
    assert client.get(f"/projects/{pid}/tokens", headers=user).json() == []


def test_tokens_are_scoped_per_user(client: TestClient, db_session: Session) -> None:
    admin = auth_headers(client, db_session, "admin@example.com", admin=True)
    project = create_project(client, admin)
    pid = project["id"]

    alice = auth_headers(client, db_session, "alice@example.com")
    bob = auth_headers(client, db_session, "bob@example.com")

    client.post(
        f"/projects/{pid}/tokens",
        headers=alice,
        json={"name": "alice-laptop"},
    )

    bob_tokens = client.get(f"/projects/{pid}/tokens", headers=bob).json()
    assert bob_tokens == []


def test_cannot_delete_another_users_token(client: TestClient, db_session: Session) -> None:
    admin = auth_headers(client, db_session, "admin@example.com", admin=True)
    project = create_project(client, admin)
    pid = project["id"]

    alice = auth_headers(client, db_session, "alice@example.com")
    bob = auth_headers(client, db_session, "bob@example.com")

    created = client.post(
        f"/projects/{pid}/tokens",
        headers=alice,
        json={"name": "alice-laptop"},
    ).json()

    resp = client.delete(f"/projects/{pid}/tokens/{created['id']}", headers=bob)
    assert resp.status_code == 404


def test_tokens_require_auth(client: TestClient) -> None:
    assert client.get("/projects/1/tokens").status_code == 401
    assert client.post("/projects/1/tokens", json={"name": "x"}).status_code == 401


def test_tokens_require_existing_project(client: TestClient, db_session: Session) -> None:
    headers = auth_headers(client, db_session)
    resp = client.post(
        "/projects/999/tokens",
        headers=headers,
        json={"name": "laptop"},
    )
    assert resp.status_code == 404


def test_deleting_project_cascades_tokens(client: TestClient, db_session: Session) -> None:
    admin = auth_headers(client, db_session, "admin@example.com", admin=True)
    project = create_project(client, admin)
    pid = project["id"]
    user = auth_headers(client, db_session, "user@example.com")
    client.post(f"/projects/{pid}/tokens", headers=user, json={"name": "laptop"})

    client.delete(f"/projects/{pid}", headers=admin)
    assert client.get(f"/projects/{pid}/tokens", headers=user).status_code == 404


def test_admin_lists_all_clients_in_project(client: TestClient, db_session: Session) -> None:
    admin = auth_headers(client, db_session, "admin@example.com", admin=True)
    project = create_project(client, admin)
    pid = project["id"]

    alice = auth_headers(client, db_session, "alice@example.com")
    bob = auth_headers(client, db_session, "bob@example.com")

    client.post(f"/projects/{pid}/tokens", headers=alice, json={"name": "alice-laptop"})
    client.post(f"/projects/{pid}/tokens", headers=bob, json={"name": "bob-desktop"})

    resp = client.get(f"/projects/{pid}/clients", headers=admin)
    assert resp.status_code == 200
    clients = resp.json()
    assert len(clients) == 2
    emails = {c["user_email"] for c in clients}
    assert emails == {"alice@example.com", "bob@example.com"}
    for c in clients:
        assert "user_email" in c
        assert "name" in c
        assert "last_seen_at" in c


def test_non_admin_cannot_list_all_clients(client: TestClient, db_session: Session) -> None:
    admin = auth_headers(client, db_session, "admin@example.com", admin=True)
    project = create_project(client, admin)
    pid = project["id"]

    user = auth_headers(client, db_session, "user@example.com")
    assert client.get(f"/projects/{pid}/clients", headers=user).status_code == 403
