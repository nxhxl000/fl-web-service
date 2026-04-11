from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.auth.deps import get_current_user
from backend.auth.models import User
from backend.clients.schemas import ClientTokenCreate, ClientTokenCreated, ClientTokenOut
from backend.clients.service import (
    ClientTokenNotFoundError,
    create_client_token,
    delete_client_token,
    list_user_tokens,
)
from backend.db import get_db

router = APIRouter(prefix="/clients/tokens", tags=["clients"])


@router.post("", response_model=ClientTokenCreated, status_code=status.HTTP_201_CREATED)
def create_token(
    payload: ClientTokenCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ClientTokenCreated:
    record, plaintext = create_client_token(db, current_user, payload.name)
    return ClientTokenCreated(
        id=record.id,
        name=record.name,
        created_at=record.created_at,
        last_seen_at=record.last_seen_at,
        token=plaintext,
    )


@router.get("", response_model=list[ClientTokenOut])
def list_tokens(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list:
    return list_user_tokens(db, current_user)


@router.delete("/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_token(
    token_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    try:
        delete_client_token(db, current_user, token_id)
    except ClientTokenNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Token not found",
        ) from exc
