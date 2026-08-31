"""Create TrainingRun and ModelVersion tables

Revision ID: 0003_create_training_tables
Revises: 0002_create_dataset_tables
Create Date: 2026-08-30

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '0003_create_training_tables'
down_revision: Union[str, None] = '0002_create_dataset_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'training_run',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('dataset_version_id', sa.String(length=36), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('config', sa.JSON(), nullable=True),
        sa.Column('comparison_results', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['dataset_version_id'], ['dataset_version.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_training_run_dataset_version_id'), 'training_run', ['dataset_version_id'], unique=False)

    op.create_table(
        'model_version',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('training_run_id', sa.String(length=36), nullable=False),
        sa.Column('dataset_version_id', sa.String(length=36), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('algorithm', sa.String(length=100), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('metrics', sa.JSON(), nullable=True),
        sa.Column('hyperparameters', sa.JSON(), nullable=True),
        sa.Column('artifact_path', sa.String(length=512), nullable=True),
        sa.Column('preprocessor_path', sa.String(length=512), nullable=True),
        sa.Column('training_time_seconds', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['training_run_id'], ['training_run.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['dataset_version_id'], ['dataset_version.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_model_version_training_run_id'), 'model_version', ['training_run_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_model_version_training_run_id'), table_name='model_version')
    op.drop_table('model_version')
    op.drop_index(op.f('ix_training_run_dataset_version_id'), table_name='training_run')
    op.drop_table('training_run')
