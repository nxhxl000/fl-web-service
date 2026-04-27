from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from backend.auth.deps import get_current_admin
from backend.auth.models import User
from backend.db import get_db
from backend.projects.deps import get_project_or_404
from backend.projects.models import Project
from backend.runs.models import Run
from backend.runs.orchestrator import get_orchestrator
from backend.runs.schemas import RunCreate, RunOut
from backend.runs.service import (
    create_run,
    get_run,
    list_runs,
    mark_run_cancelled,
    mark_run_started,
)


router = APIRouter(prefix="/projects/{project_id}/runs", tags=["runs"])


def get_run_or_404(
    run_id: int,
    project: Project = Depends(get_project_or_404),
    db: Session = Depends(get_db),
) -> Run:
    run = get_run(db, run_id)
    if run is None or run.project_id != project.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return run


@router.post("", response_model=RunOut, status_code=status.HTTP_201_CREATED)
def create_new_run(
    payload: RunCreate,
    project: Project = Depends(get_project_or_404),
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> Run:
    return create_run(db, admin, project.id, payload.federation, payload.run_config)


@router.get("", response_model=list[RunOut])
def list_project_runs(
    project: Project = Depends(get_project_or_404),
    _admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list[Run]:
    return list_runs(db, project.id)


@router.get("/{run_id}", response_model=RunOut)
def get_one_run(
    run: Run = Depends(get_run_or_404),
    _admin: User = Depends(get_current_admin),
) -> Run:
    return run


@router.post("/{run_id}/start", response_model=RunOut)
def start_run(
    run: Run = Depends(get_run_or_404),
    _admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> Run:
    if run.status != "draft":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot start a run in status {run.status!r}",
        )
    pid, log_path = get_orchestrator().start(run.id)
    return mark_run_started(db, run, pid, str(log_path))


@router.post("/{run_id}/cancel", response_model=RunOut)
def cancel_one_run(
    run: Run = Depends(get_run_or_404),
    _admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> Run:
    if run.status != "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel a run in status {run.status!r}",
        )
    if not get_orchestrator().cancel(run.id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Process is not tracked by this server (likely lost on restart)",
        )
    return mark_run_cancelled(db, run)


@router.get("/{run_id}/log", response_class=PlainTextResponse)
def get_run_log(
    run: Run = Depends(get_run_or_404),
    _admin: User = Depends(get_current_admin),
) -> str:
    if run.log_path is None:
        return ""
    log_path = Path(run.log_path)
    if not log_path.exists():
        return ""
    return log_path.read_text(errors="replace")
