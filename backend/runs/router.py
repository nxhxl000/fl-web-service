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


_HEARTBEAT_FRESH_S = 90      # client is "online" if heartbeat ≤ 90 seconds old
_SUPERLINK_CONTROL_PORT = 9093
_SUPERLINK_FLEET_PORT = 9092


# Mirror of `frontend/src/api/flStrategies.ts:STRATEGY_SPECIFIC_KEYS`. Defense
# in depth: drop strategy-gated keys that don't belong to the chosen aggregation
# (e.g. proximal-mu when aggregation=fedavgm). Catches stale values left over
# in saved drafts from older form versions.
_STRATEGY_GATED_KEYS = {"server-momentum", "server-lr", "proximal-mu"}
_STRATEGY_SPECIFIC: dict[str, set[str]] = {
    "fedavg":   set(),
    "fedavgm":  {"server-momentum", "server-lr"},
    "fedprox":  {"proximal-mu"},
    "fednovam": {"server-momentum", "server-lr"},
}


def _sanitize_run_config(rc: dict[str, Any], aggregation: str) -> dict[str, Any]:
    """Drop strategy-gated keys that don't apply to `aggregation`."""
    agg = (aggregation or "").lower()
    allowed = _STRATEGY_SPECIFIC.get(agg, set())
    return {
        k: v for k, v in rc.items()
        if k not in _STRATEGY_GATED_KEYS or k in allowed
    }


def _tcp_alive(host: str, port: int, timeout: float = 2.0) -> bool:
    """Quick TCP socket probe — returns True if port is accepting connections."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _online_client_tokens(db: Session, project_id: int) -> int:
    """Count tokens that heartbeated within the freshness window."""
    from datetime import datetime, timedelta, timezone
    from backend.clients.models import ClientToken

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=_HEARTBEAT_FRESH_S)
    return db.query(ClientToken).filter(
        ClientToken.project_id == project_id,
        ClientToken.last_seen_at >= cutoff,
    ).count()


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

    # Pre-flight checks. Failing fast here is much friendlier than letting the
    # subprocess hang or OOM the box.

    # 1. Single concurrent run across the whole DB (orchestrator handles many in
    # principle, but SuperLink + supernode pool are shared resources).
    other = db.query(Run).filter(
        Run.id != run.id, Run.status == "running"
    ).first()
    if other is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Another run is already running (#{other.id} in project "
                f"#{other.project_id}). Cancel it before starting a new one."
            ),
        )

    # 2. Dataset must have been analyzed.
    info = project.test_dataset_info or {}
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

    rc = dict(run.run_config or {})
    min_train = int(rc.get("min-train-nodes", 1))
    min_avail = int(rc.get("min-available-nodes", 1))
    if min_train < 1 or min_avail < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="min-train-nodes and min-available-nodes must be ≥ 1.",
        )

    # 3. For remote federation: SuperLink must be reachable + enough clients
    # heartbeating. (local-sim auto-spawns its own SuperLink — skip the probe.)
    if run.federation == "remote":
        if not _tcp_alive("127.0.0.1", _SUPERLINK_CONTROL_PORT):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"SuperLink Control API is not responding on "
                    f"127.0.0.1:{_SUPERLINK_CONTROL_PORT}. Make sure the SuperLink is running."
                ),
            )
        if not _tcp_alive("127.0.0.1", _SUPERLINK_FLEET_PORT):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"SuperLink Fleet API is not responding on "
                    f"127.0.0.1:{_SUPERLINK_FLEET_PORT}. Clients cannot connect."
                ),
            )
        online = _online_client_tokens(db, project.id)
        if online < min_train:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Only {online} client(s) heartbeating within "
                    f"{_HEARTBEAT_FRESH_S}s; min-train-nodes={min_train} required. "
                    "Start the missing clients (or lower the threshold) before launching."
                ),
            )

    # Inject partition-name from the project's analyzed test dataset.
    # The UI form doesn't expose `partition-name` (per project deployment policy);
    # without injection flwr falls back to the pyproject default, which can mismatch
    # the model's class count and trigger CUDA assert at eval time.
    effective_rc = dict(run.run_config or {})
    partition_name = info.get("name")
    if partition_name and "partition-name" not in effective_rc:
        effective_rc["partition-name"] = partition_name
    effective_rc = _sanitize_run_config(effective_rc, str(effective_rc.get("aggregation", "")))
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

    # Best-effort: ask the orchestrator to send `flwr stop` + SIGTERM. If the
    # backend was restarted since the run started, the orchestrator no longer
    # tracks the process — but we still want to mark the row cancelled so the
    # user isn't stuck on a phantom "running" status. As a fallback, try to
    # parse the flwr_run_id from the log file and call `flwr stop` directly.
    if not get_orchestrator().cancel(run.id) and run.exp_dir:
        import subprocess as _sp
        from backend.runs.orchestrator import (
            _parse_flwr_run_id, _flwr_bin, REPO_ROOT,
        )
        log_path = Path(run.exp_dir) / "stdout.log"
        flwr_run_id = _parse_flwr_run_id(log_path)
        if flwr_run_id is not None:
            try:
                _sp.run(
                    [_flwr_bin(), "stop", str(flwr_run_id), run.federation],
                    cwd=REPO_ROOT, timeout=10, capture_output=True,
                )
            except (_sp.TimeoutExpired, OSError):
                pass  # best-effort; mark cancelled regardless

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


@router.get("/{run_id}/config", response_model=dict)
def get_run_effective_config(
    run: Run = Depends(get_run_or_404),
    _admin: User = Depends(get_current_admin),
) -> dict[str, Any]:
    """Return the run-config that was actually passed to `flwr run --run-config`.

    Comes from `runs_data/run_<id>/_run_config.json` written by the orchestrator
    at run start time. Includes backend injections (partition-name, output-dir).
    Falls back to `run.run_config` (the saved draft) if the run never started.
    """
    if run.exp_dir:
        config_path = Path(run.exp_dir) / "_run_config.json"
        if config_path.exists():
            return json.loads(config_path.read_text())
    return dict(run.run_config or {})


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
