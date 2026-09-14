"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-09-14

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    role_enum = sa.Enum("viewer", "clinician", "ml_engineer", name="role_enum")
    model_status_enum = sa.Enum("candidate", "staging", "production", "archived", name="model_status_enum")
    framework_enum = sa.Enum("sklearn", "keras", name="framework_enum")
    job_status_enum = sa.Enum("queued", "running", "succeeded", "failed", "partial", name="job_status_enum")

    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("username", sa.String(64), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", role_enum, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("username", name="uq_users_username"),
    )
    op.create_index("ix_users_username", "users", ["username"])

    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("version", sa.String(32), nullable=False),
        sa.Column("framework", framework_enum, nullable=False),
        sa.Column("artifact_path", sa.String(512), nullable=False),
        sa.Column("artifact_sha256", sa.String(64), nullable=False),
        sa.Column("scaler_artifact_path", sa.String(512), nullable=False),
        sa.Column("scaler_sha256", sa.String(64), nullable=False),
        sa.Column("preprocessing_version", sa.String(64), nullable=False),
        sa.Column("status", model_status_enum, nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("name", "version", name="uq_model_name_version"),
    )
    op.create_index(
        "uq_one_production_per_name", "model_versions", ["name", "status"],
        unique=True, postgresql_where=sa.text("status = 'production'"),
        sqlite_where=sa.text("status = 'production'"),
    )

    op.create_table(
        "promotion_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_version_id", sa.String(36), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("from_status", sa.String(32), nullable=True),
        sa.Column("to_status", sa.String(32), nullable=False),
        sa.Column("actor_user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("metrics_snapshot", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "batch_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("status", job_status_enum, nullable=False),
        sa.Column("model_version_id", sa.String(36), sa.ForeignKey("model_versions.id"), nullable=True),
        sa.Column("total_rows", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("processed_rows", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("succeeded_rows", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_rows", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "inference_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("batch_job_id", sa.String(36), sa.ForeignKey("batch_jobs.id"), nullable=True),
        sa.Column("row_index", sa.Integer(), nullable=True),
        sa.Column("model_version_id", sa.String(36), sa.ForeignKey("model_versions.id"), nullable=True),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("input_echo", sa.JSON(), nullable=False),
        sa.Column("prediction", sa.Integer(), nullable=True),
        sa.Column("probability", sa.Float(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="ok"),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_inference_logs_input_hash", "inference_logs", ["input_hash"])
    op.create_index("ix_inference_logs_created_at", "inference_logs", ["created_at"])

    op.create_table(
        "request_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("request_id", sa.String(36), nullable=False),
        sa.Column("endpoint", sa.String(255), nullable=False),
        sa.Column("method", sa.String(8), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_request_logs_created_at", "request_logs", ["created_at"])


def downgrade() -> None:
    op.drop_table("request_logs")
    op.drop_index("ix_inference_logs_created_at", table_name="inference_logs")
    op.drop_index("ix_inference_logs_input_hash", table_name="inference_logs")
    op.drop_table("inference_logs")
    op.drop_table("batch_jobs")
    op.drop_table("promotion_events")
    op.drop_index("uq_one_production_per_name", table_name="model_versions")
    op.drop_table("model_versions")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")

    sa.Enum(name="job_status_enum").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="framework_enum").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="model_status_enum").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="role_enum").drop(op.get_bind(), checkfirst=True)
