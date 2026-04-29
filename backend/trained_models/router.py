from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.auth.deps import get_current_admin
from backend.auth.models import User
from backend.db import get_db
from backend.inference.datasets import DATASET_INFO
from backend.inference.service import resolve_weights_path
from backend.projects.deps import get_project_or_404, get_project_or_404_public
from backend.projects.models import Project
from backend.projects.schemas import ProjectOut
from backend.trained_models.models import TrainedModel
from backend.trained_models.schemas import TrainedModelCreate, TrainedModelOut
from backend.trained_models.service import (
    create_trained_model,
    delete_trained_model,
    get_trained_model,
    list_trained_models,
)


router = APIRouter(prefix="/projects/{project_id}/models", tags=["trained_models"])


def get_trained_model_or_404(
    model_id: int,
    project: Project = Depends(get_project_or_404_public),
    db: Session = Depends(get_db),
) -> TrainedModel:
    tm = get_trained_model(db, model_id)
    if tm is None or tm.project_id != project.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Trained model not found"
        )
    return tm


@router.get("", response_model=list[TrainedModelOut])
def list_project_models(
    project: Project = Depends(get_project_or_404_public),
    db: Session = Depends(get_db),
) -> list[TrainedModel]:
    """Public — registered models for a project (without weights paths)."""
    return list_trained_models(db, project.id)


@router.post("", response_model=TrainedModelOut, status_code=status.HTTP_201_CREATED)
def create_model(
    payload: TrainedModelCreate,
    project: Project = Depends(get_project_or_404),
    _admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> TrainedModel:
    if payload.dataset not in DATASET_INFO:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown dataset {payload.dataset!r}; supported: {sorted(DATASET_INFO)}",
        )
    resolved = resolve_weights_path(payload.weights_path)
    if not resolved.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Weights file not found: {resolved}",
        )
    return create_trained_model(db, project.id, payload)


@router.get("/{model_id}", response_model=TrainedModelOut)
def get_one_model(
    tm: TrainedModel = Depends(get_trained_model_or_404),
) -> TrainedModel:
    return tm


@router.delete("/{model_id}", response_model=ProjectOut)
def delete_model(
    tm: TrainedModel = Depends(get_trained_model_or_404),
    project: Project = Depends(get_project_or_404),
    _admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> Project:
    """Delete the registry entry. Cascades from FK SET NULL on projects.inference_target_id."""
    delete_trained_model(db, tm)
    db.refresh(project)
    return project


@router.post("/{model_id}/promote", response_model=ProjectOut)
def promote_model(
    tm: TrainedModel = Depends(get_trained_model_or_404),
    project: Project = Depends(get_project_or_404),
    _admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> Project:
    """Set this model as the project's public inference target."""
    project.inference_target_id = tm.id
    db.commit()
    db.refresh(project)
    return project
