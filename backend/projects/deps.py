from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.auth.deps import get_current_user
from backend.auth.models import User
from backend.db import get_db
from backend.projects.models import Project
from backend.projects.service import get_project


def get_project_or_404_public(
    project_id: int,
    db: Session = Depends(get_db),
) -> Project:
    """Lookup project without requiring authentication (used for public endpoints)."""
    project = get_project(db, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    return project


def get_project_or_404(
    project_id: int,
    _current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Project:
    return get_project_or_404_public(project_id=project_id, db=db)
