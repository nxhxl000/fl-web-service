from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1)
    requirements: str = Field(min_length=1)


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    summary: str | None = Field(default=None, min_length=1, max_length=300)
    description: str | None = Field(default=None, min_length=1)
    requirements: str | None = Field(default=None, min_length=1)


class DatasetAnalyzeRequest(BaseModel):
    path: str = Field(min_length=1, max_length=500)


class DatasetInfo(BaseModel):
    name: str
    format: str
    num_samples: int
    num_classes: int
    class_names: list[str]
    label_column: str | None = None
    image_column: str | None = None
    image_size: list[int] | None = None
    image_mode: str | None = None


class TrainedModelInline(BaseModel):
    """Embedded trained-model snapshot inside ProjectOut."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    display_name: str
    model_name: str
    dataset: str
    accuracy: float | None = None
    f1_score: float | None = None


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    summary: str
    description: str
    requirements: str
    created_at: datetime
    inference_target_id: int | None = None
    inference_target: TrainedModelInline | None = None
    test_dataset_info: dict[str, Any] | None = None
    # `test_dataset_path` is server-only and intentionally absent here.


class ProjectAdminOut(ProjectOut):
    """Admin-only view that exposes the local server path (for editing)."""

    test_dataset_path: str | None = None
