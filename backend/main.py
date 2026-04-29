import logging
from datetime import datetime, timezone

from fastapi import FastAPI

from backend.auth.router import router as auth_router
from backend.clients.agent_router import router as client_agent_router
from backend.clients.router import router as clients_router
from backend.db import SessionLocal
from backend.inference.router import router as inference_router
from backend.projects.router import router as projects_router
from backend.runs.models import Run
from backend.runs.router import router as runs_router
from backend.trained_models.router import router as trained_models_router

log = logging.getLogger(__name__)

app = FastAPI(title="fl-web-service", version="0.0.0")

app.include_router(auth_router)
app.include_router(projects_router)
app.include_router(clients_router)
app.include_router(client_agent_router)
app.include_router(runs_router)
app.include_router(trained_models_router)
app.include_router(inference_router)


@app.on_event("startup")
def reap_orphan_running_runs() -> None:
    """Mark any 'running' runs as failed at backend startup.

    The orchestrator's process map is in-memory — when uvicorn restarts, it
    forgets all running subprocesses. Without this sweep, the UI would show
    those runs as 'running' forever and block new starts (concurrency check).
    The actual flwr-run subprocess may still be alive on the host, but the
    backend can no longer track or finalize it; mark them failed and let the
    user clean up the stragglers manually.
    """
    with SessionLocal() as db:
        orphans = db.query(Run).filter(Run.status == "running").all()
        if not orphans:
            return
        now = datetime.now(timezone.utc)
        for run in orphans:
            run.status = "failed"
            run.finished_at = now
            run.error_message = (
                "Backend restarted while this run was active — process is no longer "
                "tracked. Check the host for orphan flwr-run processes."
            )
        db.commit()
        log.warning("startup: marked %d orphaned running run(s) as failed", len(orphans))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
