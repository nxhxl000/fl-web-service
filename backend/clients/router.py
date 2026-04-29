from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.auth.deps import get_current_user
from backend.auth.models import User
from backend.clients.schemas import ClientTokenCreate, ClientTokenCreated, ClientTokenOut
from backend.clients.service import (
    ClientTokenNotFoundError,
    create_client_token,
    delete_client_token,
    list_user_tokens_in_project,
)
from backend.config import get_settings
from backend.db import get_db
from backend.projects.deps import get_project_or_404
from backend.projects.models import Project

router = APIRouter(prefix="/projects/{project_id}/tokens", tags=["client tokens"])


def _build_docker_command(token: str) -> str:
    s = get_settings()
    # Multi-line for readability when participant pastes into a terminal.
    return (
        f"docker run -d --rm \\\n"
        f"  -e FL_TOKEN={token} \\\n"
        f"  -e FL_SERVER_URL={s.public_server_url} \\\n"
        f"  -e FL_SUPERLINK={s.public_superlink_addr} \\\n"
        f"  -v /PATH/TO/YOUR/DATA:/data \\\n"
        f"  {s.fl_client_image}"
    )


@router.post("", response_model=ClientTokenCreated, status_code=status.HTTP_201_CREATED)
def create_token(
    payload: ClientTokenCreate,
    project: Project = Depends(get_project_or_404),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ClientTokenCreated:
    record, plaintext = create_client_token(db, current_user, project.id, payload.name)
    return ClientTokenCreated(
        id=record.id,
        name=record.name,
        created_at=record.created_at,
        last_seen_at=record.last_seen_at,
        token=plaintext,
        docker_command=_build_docker_command(plaintext),
    )


@router.get("", response_model=list[ClientTokenOut])
def list_tokens(
    project: Project = Depends(get_project_or_404),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list:
    return list_user_tokens_in_project(db, current_user, project.id)


@router.delete("/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_token(
    token_id: int,
    project: Project = Depends(get_project_or_404),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    try:
        delete_client_token(db, current_user, project.id, token_id)
    except ClientTokenNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Token not found",
        ) from exc
