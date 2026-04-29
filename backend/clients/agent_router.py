from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.clients.models import ClientToken
from backend.clients.service import touch_client_token
from backend.db import get_db
from backend.projects.models import Project

router = APIRouter(prefix="/client", tags=["client agent"])

bearer = HTTPBearer(auto_error=False)


def _authenticate(
    creds: HTTPAuthorizationCredentials | None, db: Session
) -> ClientToken:
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )
    token = touch_client_token(db, creds.credentials)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client token",
        )
    return token


@router.post("/heartbeat", status_code=status.HTTP_204_NO_CONTENT)
def heartbeat(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> None:
    _authenticate(creds, db)


@router.get("/dataset-manifest")
def dataset_manifest(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Project's data contract for the authenticated client.

    Returns the canonical class list (defines name→index mapping), expected
    image size and mode. Client must prepare its local data to match.
    """
    token = _authenticate(creds, db)
    project = db.get(Project, token.project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
        )
    info = project.test_dataset_info or {}
    class_names = info.get("class_names")
    if not class_names:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project has no published dataset spec yet",
        )
    return {
        "project_id": project.id,
        "project_name": project.name,
        "node_name": token.name,
        "class_names": class_names,
        "num_classes": len(class_names),
        "image_size": info.get("image_size"),
        "image_mode": info.get("image_mode"),
        "mean": info.get("mean"),
        "std": info.get("std"),
    }
