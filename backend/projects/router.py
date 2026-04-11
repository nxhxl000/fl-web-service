from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.auth.deps import get_current_admin, get_current_user
from backend.auth.models import User
from backend.clients.schemas import ClientTokenWithOwnerOut
from backend.clients.service import list_all_tokens_in_project
from backend.db import get_db
from backend.projects.deps import get_project_or_404
from backend.projects.models import Project
from backend.projects.schemas import ProjectCreate, ProjectOut, ProjectUpdate
from backend.projects.service import (
    create_project,
    delete_project,
    list_projects,
    update_project,
)

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=list[ProjectOut])
def list_all_projects(
    _current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Project]:
    return list_projects(db)


@router.get("/{project_id}", response_model=ProjectOut)
def get_one_project(
    project: Project = Depends(get_project_or_404),
) -> Project:
    return project


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def create_new_project(
    payload: ProjectCreate,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> Project:
    return create_project(db, admin, payload)


@router.patch("/{project_id}", response_model=ProjectOut)
def update_existing_project(
    payload: ProjectUpdate,
    project: Project = Depends(get_project_or_404),
    _admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> Project:
    return update_project(db, project, payload)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_existing_project(
    project: Project = Depends(get_project_or_404),
    _admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> None:
    delete_project(db, project)


@router.get("/{project_id}/clients", response_model=list[ClientTokenWithOwnerOut])
def list_project_clients(
    project: Project = Depends(get_project_or_404),
    _admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list:
    return list_all_tokens_in_project(db, project.id)
