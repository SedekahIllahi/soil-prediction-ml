"""
Dataset adapters for mapping raw datasets to the canonical ML schema.
"""

from .base import DatasetAdapter
from .urban_road_collapse import UrbanRoadCollapseAdapter
from .adapter_registry import get_adapter, register_adapter

__all__ = ["DatasetAdapter", "UrbanRoadCollapseAdapter", "get_adapter", "register_adapter"]
