import logging
import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.runs.models import Run
from backend.trained_models.models import TrainedModel
from backend.trained_models.schemas import TrainedModelCreate

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_MODELS_DIR = REPO_ROOT / "inference_models"
RUNS_DATA_DIR = REPO_ROOT / "runs_data"


def create_trained_model(
    db: Session, project_id: int, payload: TrainedModelCreate
) -> TrainedModel:
    tm = TrainedModel(
        project_id=project_id,
        display_name=payload.display_name,
        model_name=payload.model_name,
        dataset=payload.dataset,
        weights_path=payload.weights_path,
        accuracy=payload.accuracy,
        f1_score=payload.f1_score,
        num_rounds=payload.num_rounds,
    )
    db.add(tm)
    db.commit()
    db.refresh(tm)
    return tm


def list_trained_models(db: Session, project_id: int) -> list[TrainedModel]:
    stmt = (
        select(TrainedModel)
        .where(TrainedModel.project_id == project_id)
        .order_by(TrainedModel.created_at.desc())
    )
    return list(db.scalars(stmt))


def get_trained_model(db: Session, model_id: int) -> TrainedModel | None:
    return db.get(TrainedModel, model_id)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def delete_trained_model(db: Session, tm: TrainedModel) -> None:
    """Hard-delete: TrainedModel row + its weights file (if managed) + the Run
    that produced it (row + exp_dir). Filesystem ops are best-effort and
    constrained to inference_models/ and runs_data/ to avoid touching paths
    outside our control (e.g. manually imported `/tmp/...` weights).
    """
    weights_path = Path(tm.weights_path) if tm.weights_path else None
    run_id = tm.run_id

    db.delete(tm)
    db.flush()

    run_dir: Path | None = None
    if run_id is not None:
        run = db.get(Run, run_id)
        if run is not None:
            if run.exp_dir:
                run_dir = Path(run.exp_dir)
            db.delete(run)
            db.flush()

    db.commit()

    if weights_path is not None and weights_path.exists():
        if _is_under(weights_path, INFERENCE_MODELS_DIR):
            try:
                weights_path.unlink()
            except OSError:
                log.exception("failed to unlink weights file %s", weights_path)
        else:
            log.info("skipping weights unlink — outside inference_models: %s", weights_path)

    if run_dir is not None and run_dir.exists():
        if _is_under(run_dir, RUNS_DATA_DIR):
            try:
                shutil.rmtree(run_dir)
            except OSError:
                log.exception("failed to rmtree run dir %s", run_dir)
        else:
            log.info("skipping run dir rmtree — outside runs_data: %s", run_dir)
