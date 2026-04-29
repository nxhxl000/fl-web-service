import json
from pathlib import Path
from typing import Any

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
    delete_run,
    get_run,
    list_runs,
    mark_run_cancelled,
    mark_run_started,
)
from backend.trained_models.models import TrainedModel


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
    project: Project = Depends(get_project_or_404),
    _admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> Run:
    if run.status != "draft":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot start a run in status {run.status!r}",
        )
    # Inject partition-name from the project's analyzed test dataset.
    # The UI form doesn't expose `partition-name` (per project deployment policy);
    # without injection flwr falls back to the pyproject default, which can mismatch
    # the model's class count and trigger CUDA assert at eval time.
    effective_rc = dict(run.run_config or {})
    info = project.test_dataset_info or {}
    partition_name = info.get("name")
    if partition_name and "partition-name" not in effective_rc:
        effective_rc["partition-name"] = partition_name

    # Project contract for the training subprocess (server_app + sim clients).
    # Real Docker clients fetch the same shape from /client/dataset-manifest.
    if not info.get("class_names"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project has no analyzed dataset spec — run Analyze first.",
        )
    if not project.test_dataset_path:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project has no test dataset path — run Analyze first.",
        )
    from backend.projects.dataset_analyzer import REPO_ROOT

    test_path = Path(project.test_dataset_path)
    if not test_path.is_absolute():
        test_path = REPO_ROOT / test_path
    contract = {
        "class_names": info["class_names"],
        "image_size": info.get("image_size"),
        "image_mode": info.get("image_mode"),
        "mean": info.get("mean"),
        "std": info.get("std"),
        "test_dataset_path": str(test_path),
    }
    pid, log_path, exp_dir = get_orchestrator().start(
        run.id, run.federation, effective_rc, contract
    )
    return mark_run_started(db, run, pid, str(log_path), str(exp_dir))


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


@router.get("/{run_id}/events", response_model=list[dict])
def get_run_events(
    run: Run = Depends(get_run_or_404),
    _admin: User = Depends(get_current_admin),
) -> list[dict[str, Any]]:
    """Return parsed events.jsonl. Tail-safe: a partial last line is skipped."""
    if not run.exp_dir:
        return []
    events_path = Path(run.exp_dir) / "events.jsonl"
    if not events_path.exists():
        return []
    events: list[dict[str, Any]] = []
    with events_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # writer may be mid-append; skip the partial tail line
                break
    return events


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_one_run(
    run: Run = Depends(get_run_or_404),
    _admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> None:
    if run.status == "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cancel the run first; live subprocess is still tracked.",
        )
    linked = db.query(TrainedModel).filter(TrainedModel.run_id == run.id).first()
    if linked is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Run is referenced by trained model #{linked.id} ({linked.display_name!r}). "
                "Delete the model from the registry instead — it will cascade-delete the run."
            ),
        )
    delete_run(db, run)
