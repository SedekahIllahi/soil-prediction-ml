"""Create Prediction table

Revision ID: 0004_create_prediction_table
Revises: 0003_create_training_tables
Create Date: 2026-08-31

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '0004_create_prediction_table'
down_revision: Union[str, None] = '0003_create_training_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'prediction',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('model_version_id', sa.String(length=36), nullable=True),
        sa.Column('features', sa.JSON(), nullable=False),
        sa.Column('predicted_class', sa.String(length=50), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('probabilities', sa.JSON(), nullable=False),
        sa.Column('latitude', sa.Float(), nullable=True),
        sa.Column('longitude', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['model_version_id'], ['model_version.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_prediction_model_version_id'), 'prediction', ['model_version_id'], unique=False)
    op.create_index(op.f('ix_prediction_predicted_class'), 'prediction', ['predicted_class'], unique=False)
    op.create_index(op.f('ix_prediction_created_at'), 'prediction', ['created_at'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_prediction_created_at'), table_name='prediction')
    op.drop_index(op.f('ix_prediction_predicted_class'), table_name='prediction')
    op.drop_index(op.f('ix_prediction_model_version_id'), table_name='prediction')
    op.drop_table('prediction')
