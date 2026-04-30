import logging
import logging.handlers
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI

from backend.auth.router import router as auth_router
from backend.clients.agent_router import router as client_agent_router
from backend.clients.router import router as clients_router
from backend.db import SessionLocal
from backend.inference.router import router as inference_router
from backend.projects.router import router as projects_router
from backend.runs.models import Run
from backend.runs.router import router as runs_router
from backend.superlink.manager import ensure_running as ensure_superlink_running
from backend.trained_models.router import router as trained_models_router


def _configure_ops_logger() -> None:
    """Daily-rotating file logger for system-level events (`ops` namespace).

    Backend startup, SuperLink spawn, run lifecycle, etc. land here so anyone
    can `tail -f ops.log` to see what the platform is doing without scraping
    uvicorn output. Old files are auto-purged after `OPS_LOG_BACKUP` days.
    """
    repo_root = Path(__file__).resolve().parents[1]
    log_dir = Path(os.environ.get("OPS_LOG_DIR", str(repo_root)))
    log_dir.mkdir(parents=True, exist_ok=True)

    ops_logger = logging.getLogger("ops")
    ops_logger.setLevel(logging.INFO)
    # Idempotent: don't double-add handler on uvicorn reload
    if any(
        isinstance(h, logging.handlers.TimedRotatingFileHandler)
        for h in ops_logger.handlers
    ):
        return

    handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_dir / "ops.log",
        when="midnight",
        backupCount=int(os.environ.get("OPS_LOG_BACKUP", "7")),
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    ops_logger.addHandler(handler)


_configure_ops_logger()

log = logging.getLogger(__name__)
ops = logging.getLogger("ops")

app = FastAPI(title="fl-web-service", version="0.0.0")

app.include_router(auth_router)
app.include_router(projects_router)
app.include_router(clients_router)
app.include_router(client_agent_router)
app.include_router(runs_router)
app.include_router(trained_models_router)
app.include_router(inference_router)


@app.on_event("startup")
def announce_startup() -> None:
    ops.info("backend startup pid=%d", os.getpid())


@app.on_event("startup")
def boot_superlink() -> None:
    """Spawn SuperLink if not already running. Best-effort — backend stays up
    even if SuperLink fails to come up, so the admin can still hit the API
    and investigate."""
    try:
        ensure_superlink_running()
    except Exception as exc:
        ops.error("superlink autostart failed: %s", exc)
        log.exception("SuperLink autostart failed — local-sim runs will still work")


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
        ops.warning("startup: marked %d orphaned running run(s) as failed", len(orphans))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
