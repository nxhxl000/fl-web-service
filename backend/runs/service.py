import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.auth.models import User
from backend.runs.models import Run
from backend.trained_models.models import TrainedModel

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_MODELS_DIR = REPO_ROOT / "inference_models"
RUNS_DATA_DIR = REPO_ROOT / "runs_data"


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def delete_run(db: Session, run: Run) -> None:
    """Hard-delete a run + its exp_dir + standalone log_path (legacy layout).
    Caller must check status != 'running' and that no TrainedModel references this run.
    """
    exp_dir = Path(run.exp_dir) if run.exp_dir else None
    log_path = Path(run.log_path) if run.log_path else None
    db.delete(run)
    db.commit()

    if exp_dir is not None and exp_dir.exists():
        if _is_under(exp_dir, RUNS_DATA_DIR):
            try:
                shutil.rmtree(exp_dir)
            except OSError:
                log.exception("failed to rmtree run dir %s", exp_dir)
        else:
            log.info("skipping run dir rmtree — outside runs_data: %s", exp_dir)

    # Legacy: pre-Day-6 runs wrote stdout to runs_data/logs/run_<id>.log
    # (separate from exp_dir). Also wipe that.
    if log_path is not None and log_path.exists() and (exp_dir is None or log_path.parent != exp_dir):
        if _is_under(log_path, RUNS_DATA_DIR):
            try:
                log_path.unlink()
            except OSError:
                log.exception("failed to unlink log file %s", log_path)


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


def mark_run_started(
    db: Session, run: Run, pid: int, log_path: str, exp_dir: str
) -> Run:
    run.status = "running"
    run.pid = pid
    run.log_path = log_path
    run.exp_dir = exp_dir
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
            try:
                _register_trained_model_from_run(db, run)
            except Exception as e:  # registration is best-effort, never poison the run record
                log.exception("auto-register failed for run %s", run_id)
                run.error_message = f"Run completed but auto-register failed: {e}"
        else:
            run.status = "failed"
            run.error_message = f"Process exited with code {exit_code}"
    db.commit()


def _register_trained_model_from_run(db: Session, run: Run) -> None:
    if not run.exp_dir:
        log.warning("run %s has no exp_dir; cannot auto-register", run.id)
        return
    events_path = Path(run.exp_dir) / "events.jsonl"
    if not events_path.exists():
        log.warning("events.jsonl not found at %s", events_path)
        return

    run_done: dict[str, Any] | None = None
    with events_path.open() as f:
        for line in f:
            event = json.loads(line)
            if event.get("type") == "run_done":
                run_done = event
                break

    if run_done is None:
        log.warning("run_done event not found for run %s", run.id)
        return
    if run_done.get("best_acc") is None:
        log.info("run %s has no best_acc; skipping registration", run.id)
        return

    src_weights = Path(run_done["model_best_path"])
    if not src_weights.exists():
        log.error("model_best_path %s does not exist", src_weights)
        return

    INFERENCE_MODELS_DIR.mkdir(exist_ok=True)
    dst_weights = INFERENCE_MODELS_DIR / f"run_{run.id}.pt"
    shutil.copy2(src_weights, dst_weights)

    rc = run.run_config or {}
    model_name = run_done.get("model") or rc.get("model", "unknown")
    aggregation = run_done.get("aggregation") or rc.get("aggregation", "fedavg")
    dataset = (
        run_done.get("dataset")
        or rc.get("partition-name", "").split("__", 1)[0]
        or "unknown"
    )
    num_rounds = run_done.get("rounds_completed") or run_done.get("num_rounds")
    display_name = f"{model_name}/{aggregation} (run #{run.id})"

    tm = TrainedModel(
        project_id=run.project_id,
        run_id=run.id,
        display_name=display_name,
        model_name=model_name,
        dataset=dataset,
        weights_path=str(dst_weights),
        accuracy=run_done.get("best_acc"),
        f1_score=run_done.get("best_f1"),
        num_rounds=num_rounds,
    )
    db.add(tm)
    db.flush()
    log.info("auto-registered TrainedModel %s for run %s", tm.id, run.id)
