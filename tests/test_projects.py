from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from tests.conftest import auth_headers, create_project


def test_admin_creates_project_and_anyone_lists(client: TestClient, db_session: Session) -> None:
    admin = auth_headers(client, db_session, "admin@example.com", admin=True)
    user = auth_headers(client, db_session, "user@example.com")

    project = create_project(client, admin, name="PlantVillage demo")
    assert project["id"] > 0
    assert "dataset" not in project

    admin_list = client.get("/projects", headers=admin).json()
    user_list = client.get("/projects", headers=user).json()
    assert len(admin_list) == 1
    assert len(user_list) == 1
    assert user_list[0]["name"] == "PlantVillage demo"


def test_non_admin_cannot_create_project(client: TestClient, db_session: Session) -> None:
    user = auth_headers(client, db_session, "user@example.com")
    resp = client.post(
        "/projects",
        headers=user,
        json={"name": "X", "summary": "short", "description": "Y", "requirements": "Z"},
    )
    assert resp.status_code == 403


def test_non_admin_cannot_update_or_delete(client: TestClient, db_session: Session) -> None:
    admin = auth_headers(client, db_session, "admin@example.com", admin=True)
    user = auth_headers(client, db_session, "user@example.com")
    project = create_project(client, admin)
    pid = project["id"]

    assert client.patch(f"/projects/{pid}", headers=user, json={"name": "New"}).status_code == 403
    assert client.delete(f"/projects/{pid}", headers=user).status_code == 403


def test_admin_updates_project(client: TestClient, db_session: Session) -> None:
    admin = auth_headers(client, db_session, "admin@example.com", admin=True)
    project = create_project(client, admin)
    resp = client.patch(
        f"/projects/{project['id']}",
        headers=admin,
        json={"name": "Renamed", "requirements": "2 vCPU, 4 GB RAM"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Renamed"
    assert body["requirements"] == "2 vCPU, 4 GB RAM"


def test_admin_deletes_project(client: TestClient, db_session: Session) -> None:
    admin = auth_headers(client, db_session, "admin@example.com", admin=True)
    project = create_project(client, admin)
    resp = client.delete(f"/projects/{project['id']}", headers=admin)
    assert resp.status_code == 204
    assert client.get(f"/projects/{project['id']}", headers=admin).status_code == 404


def test_get_missing_project_404(client: TestClient, db_session: Session) -> None:
    headers = auth_headers(client, db_session)
    assert client.get("/projects/999", headers=headers).status_code == 404


def test_projects_require_auth(client: TestClient) -> None:
    assert client.get("/projects").status_code == 401


def test_me_exposes_is_admin(client: TestClient, db_session: Session) -> None:
    regular = auth_headers(client, db_session, "regular@example.com")
    admin = auth_headers(client, db_session, "admin@example.com", admin=True)

    assert client.get("/auth/me", headers=regular).json()["is_admin"] is False
    assert client.get("/auth/me", headers=admin).json()["is_admin"] is True
