from fastapi import FastAPI

from backend.auth.router import router as auth_router
from backend.clients.agent_router import router as client_agent_router
from backend.clients.router import router as clients_router
from backend.projects.router import router as projects_router

app = FastAPI(title="fl-web-service", version="0.0.0")

app.include_router(auth_router)
app.include_router(projects_router)
app.include_router(clients_router)
app.include_router(client_agent_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
