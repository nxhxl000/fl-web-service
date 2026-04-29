"""trained models registry; project FK to inference target

Revision ID: e7f3a9d2b015
Revises: d8e2b9f5c1a4
Create Date: 2026-04-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e7f3a9d2b015"
down_revision: Union[str, Sequence[str], None] = "d8e2b9f5c1a4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "trained_models",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("display_name", sa.String(length=150), nullable=False),
        sa.Column("model_name", sa.String(length=50), nullable=False),
        sa.Column("dataset", sa.String(length=50), nullable=False),
        sa.Column("weights_path", sa.String(length=500), nullable=False),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("f1_score", sa.Float(), nullable=True),
        sa.Column("num_rounds", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.id"], name="fk_trained_models_project_id", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], name="fk_trained_models_run_id", ondelete="SET NULL"
        ),
    )
    op.create_index("ix_trained_models_project_id", "trained_models", ["project_id"])

    op.add_column("projects", sa.Column("inference_target_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_projects_inference_target_id",
        "projects",
        "trained_models",
        ["inference_target_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Migrate existing inline targets into the registry, then drop the inline columns.
    conn = op.get_bind()
    rows = conn.execute(
        sa.text(
            "SELECT id, inference_model_name, inference_dataset, inference_weights_path, inference_accuracy "
            "FROM projects WHERE inference_model_name IS NOT NULL"
        )
    ).fetchall()
    for row in rows:
        new_id = conn.execute(
            sa.text(
                "INSERT INTO trained_models "
                "(project_id, display_name, model_name, dataset, weights_path, accuracy, created_at) "
                "VALUES (:pid, :name, :mname, :ds, :path, :acc, NOW()) RETURNING id"
            ),
            {
                "pid": row.id,
                "name": f"{row.inference_model_name} (imported)",
                "mname": row.inference_model_name,
                "ds": row.inference_dataset,
                "path": row.inference_weights_path,
                "acc": row.inference_accuracy,
            },
        ).scalar()
        conn.execute(
            sa.text("UPDATE projects SET inference_target_id = :tid WHERE id = :pid"),
            {"tid": new_id, "pid": row.id},
        )

    op.drop_column("projects", "inference_accuracy")
    op.drop_column("projects", "inference_weights_path")
    op.drop_column("projects", "inference_dataset")
    op.drop_column("projects", "inference_model_name")


def downgrade() -> None:
    op.add_column("projects", sa.Column("inference_model_name", sa.String(length=50), nullable=True))
    op.add_column("projects", sa.Column("inference_dataset", sa.String(length=50), nullable=True))
    op.add_column("projects", sa.Column("inference_weights_path", sa.String(length=500), nullable=True))
    op.add_column("projects", sa.Column("inference_accuracy", sa.Float(), nullable=True))

    conn = op.get_bind()
    rows = conn.execute(
        sa.text(
            "SELECT p.id, tm.model_name, tm.dataset, tm.weights_path, tm.accuracy "
            "FROM projects p JOIN trained_models tm ON tm.id = p.inference_target_id"
        )
    ).fetchall()
    for row in rows:
        conn.execute(
            sa.text(
                "UPDATE projects SET inference_model_name = :mn, inference_dataset = :ds, "
                "inference_weights_path = :wp, inference_accuracy = :acc WHERE id = :pid"
            ),
            {
                "mn": row.model_name,
                "ds": row.dataset,
                "wp": row.weights_path,
                "acc": row.accuracy,
                "pid": row.id,
            },
        )

    op.drop_constraint("fk_projects_inference_target_id", "projects", type_="foreignkey")
    op.drop_column("projects", "inference_target_id")
    op.drop_index("ix_trained_models_project_id", table_name="trained_models")
    op.drop_table("trained_models")
