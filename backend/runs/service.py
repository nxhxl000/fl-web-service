from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.auth.models import User
from backend.runs.models import Run


VALID_STATUSES = {"draft", "running", "completed", "failed", "cancelled"}


def create_run(
    db: Session,
    user: User,
    project_id: int,
    federation: str,
    run_config: dict[str, Any],
) -> Run:
    run = Run(
        project_id=project_id,
        created_by=user.id,
        federation=federation,
        run_config=run_config,
        status="draft",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_run(db: Session, run_id: int) -> Run | None:
    return db.get(Run, run_id)


def list_runs(db: Session, project_id: int) -> list[Run]:
    stmt = select(Run).where(Run.project_id == project_id).order_by(Run.created_at.desc())
    return list(db.scalars(stmt))


def mark_run_started(db: Session, run: Run, pid: int, log_path: str) -> Run:
    run.status = "running"
    run.pid = pid
    run.log_path = log_path
    run.started_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(run)
    return run


def mark_run_cancelled(db: Session, run: Run) -> Run:
    run.status = "cancelled"
    run.finished_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(run)
    return run


def finalize_run(db: Session, run_id: int, exit_code: int) -> None:
    """Called by the orchestrator's background thread when subprocess exits."""
    run = db.get(Run, run_id)
    if run is None:
        return
    run.exit_code = exit_code
    run.finished_at = datetime.now(timezone.utc)
    if run.status == "running":
        if exit_code == 0:
            run.status = "completed"
        else:
            run.status = "failed"
            run.error_message = f"Process exited with code {exit_code}"
    db.commit()
