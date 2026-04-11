import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.auth.models import User
from backend.clients.models import ClientToken


@dataclass
class ClientTokenWithOwner:
    id: int
    name: str
    user_email: str
    created_at: datetime
    last_seen_at: datetime | None

TOKEN_PREFIX = "flwc_"


class ClientTokenNotFoundError(Exception):
    pass


def generate_client_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def hash_client_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_client_token(
    db: Session,
    user: User,
    project_id: int,
    name: str,
) -> tuple[ClientToken, str]:
    plaintext = generate_client_token()
    record = ClientToken(
        user_id=user.id,
        project_id=project_id,
        name=name,
        token_hash=hash_client_token(plaintext),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record, plaintext


def list_user_tokens_in_project(
    db: Session,
    user: User,
    project_id: int,
) -> list[ClientToken]:
    stmt = (
        select(ClientToken)
        .where(ClientToken.user_id == user.id, ClientToken.project_id == project_id)
        .order_by(ClientToken.created_at.desc())
    )
    return list(db.scalars(stmt))


def list_all_tokens_in_project(
    db: Session,
    project_id: int,
) -> list[ClientTokenWithOwner]:
    stmt = (
        select(ClientToken, User.email)
        .join(User, User.id == ClientToken.user_id)
        .where(ClientToken.project_id == project_id)
        .order_by(ClientToken.created_at.desc())
    )
    return [
        ClientTokenWithOwner(
            id=token.id,
            name=token.name,
            user_email=email,
            created_at=token.created_at,
            last_seen_at=token.last_seen_at,
        )
        for token, email in db.execute(stmt).all()
    ]


def delete_client_token(
    db: Session,
    user: User,
    project_id: int,
    token_id: int,
) -> None:
    record = db.get(ClientToken, token_id)
    if (
        record is None
        or record.user_id != user.id
        or record.project_id != project_id
    ):
        raise ClientTokenNotFoundError()
    db.delete(record)
    db.commit()
