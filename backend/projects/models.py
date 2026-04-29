from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db import Base


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    summary: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    requirements: Mapped[str] = mapped_column(Text, nullable=False)
    created_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    inference_target_id: Mapped[int | None] = mapped_column(
        ForeignKey("trained_models.id", ondelete="SET NULL"), nullable=True
    )
    inference_target: Mapped["TrainedModel | None"] = relationship(  # noqa: F821
        foreign_keys=[inference_target_id]
    )

    # Path on the server to the test dataset directory (admin-only).
    # Stays server-side: never exposed in public API responses.
    test_dataset_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # Public dataset specs derived from analyze_dataset(test_dataset_path).
    # Shape: {format, num_samples, num_classes, class_names, image_size, image_mode}
    test_dataset_info: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    tokens: Mapped[list["ClientToken"]] = relationship(  # noqa: F821
        back_populates="project", cascade="all, delete-orphan"
    )
