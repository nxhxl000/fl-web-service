from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ClientTokenCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class ClientTokenOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    created_at: datetime
    last_seen_at: datetime | None


class ClientTokenCreated(ClientTokenOut):
    token: str


class ClientTokenWithOwnerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    user_email: str
    created_at: datetime
    last_seen_at: datetime | None
