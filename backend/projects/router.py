from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.auth.deps import get_current_admin, get_current_user
from backend.auth.models import User
from backend.clients.schemas import ClientTokenWithOwnerOut
from backend.clients.service import list_all_tokens_in_project
from backend.db import get_db
from backend.projects.dataset_analyzer import (
    DatasetAnalysisError,
    analyze_dataset,
    browse_directory,
)
from backend.projects.deps import get_project_or_404, get_project_or_404_public
from backend.projects.models import Project
from backend.projects.schemas import (
    DatasetAnalyzeRequest,
    ProjectAdminOut,
    ProjectCreate,
    ProjectOut,
    ProjectUpdate,
)
from backend.projects.service import (
    create_project,
    delete_project,
    list_projects,
    update_project,
)

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=list[ProjectOut])
def list_all_projects(db: Session = Depends(get_db)) -> list[Project]:
    """Public — projects list is part of the public showcase."""
    return list_projects(db)


@router.get("/joined")
def list_joined_project_ids(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[int]:
    """Project IDs where the current user owns at least one client token.

    Frontend uses this to swap "Join To Project" → "Open project" on the
    landing/projects list. Lighter than embedding a flag into ProjectOut.
    """
    from backend.clients.models import ClientToken
    from sqlalchemy import select

    rows = db.execute(
        select(ClientToken.project_id)
        .where(ClientToken.user_id == current_user.id)
        .distinct()
    ).all()
    return [r[0] for r in rows]


@router.get("/{project_id}", response_model=ProjectOut)
def get_one_project(
    project: Project = Depends(get_project_or_404_public),
) -> Project:
    """Public — basic project info is non-sensitive."""
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


@router.get("/dataset/browse")
def browse_dataset_directory(
    path: str = "",
    _admin: User = Depends(get_current_admin),
) -> dict:
    """Admin-only — list subdirectories under REPO_ROOT/path. Sandboxed."""
    try:
        return browse_directory(path)
    except DatasetAnalysisError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.get("/{project_id}/admin", response_model=ProjectAdminOut)
def get_project_admin(
    project: Project = Depends(get_project_or_404),
    _admin: User = Depends(get_current_admin),
) -> Project:
    """Admin-only — same as GET /projects/{id} but exposes the local server
    `test_dataset_path` so admins can edit it."""
    return project


@router.post("/{project_id}/dataset/analyze", response_model=ProjectAdminOut)
def analyze_project_dataset(
    payload: DatasetAnalyzeRequest,
    project: Project = Depends(get_project_or_404),
    _admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> Project:
    try:
        info = analyze_dataset(payload.path)
    except DatasetAnalysisError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    project.test_dataset_path = payload.path
    project.test_dataset_info = info
    db.commit()
    db.refresh(project)
    return project


@router.get("/{project_id}/clients", response_model=list[ClientTokenWithOwnerOut])
def list_project_clients(
    project: Project = Depends(get_project_or_404),
    _admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list:
    return list_all_tokens_in_project(db, project.id)
