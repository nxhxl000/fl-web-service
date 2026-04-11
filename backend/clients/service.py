import hashlib
import secrets

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.auth.models import User
from backend.clients.models import ClientToken

TOKEN_PREFIX = "flwc_"


class ClientTokenNotFoundError(Exception):
    pass


def generate_client_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def hash_client_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_client_token(db: Session, user: User, name: str) -> tuple[ClientToken, str]:
    plaintext = generate_client_token()
    record = ClientToken(
        user_id=user.id,
        name=name,
        token_hash=hash_client_token(plaintext),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record, plaintext


def list_user_tokens(db: Session, user: User) -> list[ClientToken]:
    stmt = (
        select(ClientToken)
        .where(ClientToken.user_id == user.id)
        .order_by(ClientToken.created_at.desc())
    )
    return list(db.scalars(stmt))


def delete_client_token(db: Session, user: User, token_id: int) -> None:
    record = db.get(ClientToken, token_id)
    if record is None or record.user_id != user.id:
        raise ClientTokenNotFoundError()
    db.delete(record)
    db.commit()
