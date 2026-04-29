"""add project inference target fields

Revision ID: d8e2b9f5c1a4
Revises: c1b8a4e7d3f0
Create Date: 2026-04-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d8e2b9f5c1a4"
down_revision: Union[str, Sequence[str], None] = "c1b8a4e7d3f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("projects", sa.Column("inference_model_name", sa.String(length=50), nullable=True))
    op.add_column("projects", sa.Column("inference_dataset", sa.String(length=50), nullable=True))
    op.add_column("projects", sa.Column("inference_weights_path", sa.String(length=500), nullable=True))
    op.add_column("projects", sa.Column("inference_accuracy", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("projects", "inference_accuracy")
    op.drop_column("projects", "inference_weights_path")
    op.drop_column("projects", "inference_dataset")
    op.drop_column("projects", "inference_model_name")
