from fastapi.testclient import TestClient


def _auth_headers(client: TestClient, email: str = "dev@example.com") -> dict[str, str]:
    client.post("/auth/register", json={"email": email, "password": "supersecret"})
    token = client.post(
        "/auth/login",
        json={"email": email, "password": "supersecret"},
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_create_list_delete_token(client: TestClient) -> None:
    headers = _auth_headers(client)

    create_resp = client.post(
        "/clients/tokens",
        json={"name": "laptop"},
        headers=headers,
    )
    assert create_resp.status_code == 201
    created = create_resp.json()
    assert created["name"] == "laptop"
    assert created["token"].startswith("flwc_")
    token_id = created["id"]

    list_resp = client.get("/clients/tokens", headers=headers)
    assert list_resp.status_code == 200
    tokens = list_resp.json()
    assert len(tokens) == 1
    assert tokens[0]["id"] == token_id
    assert "token" not in tokens[0]

    delete_resp = client.delete(f"/clients/tokens/{token_id}", headers=headers)
    assert delete_resp.status_code == 204

    assert client.get("/clients/tokens", headers=headers).json() == []


def test_cannot_delete_another_users_token(client: TestClient) -> None:
    alice_headers = _auth_headers(client, "alice@example.com")
    bob_headers = _auth_headers(client, "bob@example.com")

    created = client.post(
        "/clients/tokens",
        json={"name": "alice-laptop"},
        headers=alice_headers,
    ).json()

    resp = client.delete(f"/clients/tokens/{created['id']}", headers=bob_headers)
    assert resp.status_code == 404


def test_tokens_require_auth(client: TestClient) -> None:
    assert client.get("/clients/tokens").status_code == 401
    assert client.post("/clients/tokens", json={"name": "x"}).status_code == 401
