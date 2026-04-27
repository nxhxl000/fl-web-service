from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunCreate(BaseModel):
    federation: str = Field(default="dummy", max_length=50)
    run_config: dict[str, Any]


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    federation: str
    run_config: dict[str, Any]
    status: str
    pid: int | None
    log_path: str | None
    exp_dir: str | None
    started_at: datetime | None
    finished_at: datetime | None
    exit_code: int | None
    error_message: str | None
    created_at: datetime
