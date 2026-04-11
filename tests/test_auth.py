from fastapi.testclient import TestClient


def test_register_login_me_flow(client: TestClient) -> None:
    register_resp = client.post(
        "/auth/register",
        json={"email": "alice@example.com", "password": "supersecret"},
    )
    assert register_resp.status_code == 201
    user = register_resp.json()
    assert user["email"] == "alice@example.com"
    assert "id" in user
    assert "password_hash" not in user

    login_resp = client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "supersecret"},
    )
    assert login_resp.status_code == 200
    body = login_resp.json()
    assert body["token_type"] == "bearer"
    token = body["access_token"]
    assert token

    me_resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == "alice@example.com"


def test_register_duplicate_email_conflict(client: TestClient) -> None:
    payload = {"email": "bob@example.com", "password": "supersecret"}
    assert client.post("/auth/register", json=payload).status_code == 201
    assert client.post("/auth/register", json=payload).status_code == 409


def test_login_wrong_password(client: TestClient) -> None:
    client.post(
        "/auth/register",
        json={"email": "carol@example.com", "password": "supersecret"},
    )
    resp = client.post(
        "/auth/login",
        json={"email": "carol@example.com", "password": "wrongpass"},
    )
    assert resp.status_code == 401


def test_me_requires_auth(client: TestClient) -> None:
    assert client.get("/auth/me").status_code == 401
