"""Create Dataset and DatasetVersion tables

Revision ID: 0002_create_dataset_tables
Revises: 0001_initial_schema
Create Date: 2026-08-30

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '0002_create_dataset_tables'
down_revision: Union[str, None] = '0001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'dataset',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('adapter_type', sa.String(length=100), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index(op.f('ix_dataset_name'), 'dataset', ['name'], unique=True)

    op.create_table(
        'dataset_version',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('dataset_id', sa.String(length=36), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('file_path', sa.String(length=512), nullable=False),
        sa.Column('row_count', sa.Integer(), nullable=False),
        sa.Column('column_info', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['dataset_id'], ['dataset.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_dataset_version_dataset_id'), 'dataset_version', ['dataset_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_dataset_version_dataset_id'), table_name='dataset_version')
    op.drop_table('dataset_version')
    op.drop_index(op.f('ix_dataset_name'), table_name='dataset')
    op.drop_table('dataset')
