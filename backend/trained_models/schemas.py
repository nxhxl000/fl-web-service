from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TrainedModelCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=150)
    model_name: str = Field(min_length=1, max_length=50)
    dataset: str = Field(min_length=1, max_length=50)
    weights_path: str = Field(min_length=1, max_length=500)
    accuracy: float | None = Field(default=None, ge=0, le=1)
    f1_score: float | None = Field(default=None, ge=0, le=1)
    num_rounds: int | None = Field(default=None, ge=0)


class TrainedModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    run_id: int | None
    display_name: str
    model_name: str
    dataset: str
    accuracy: float | None
    f1_score: float | None
    num_rounds: int | None
    created_at: datetime
