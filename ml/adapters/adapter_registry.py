from .base import DatasetAdapter
from .urban_road_collapse import UrbanRoadCollapseAdapter

_registry: dict[str, type[DatasetAdapter]] = {}

def register_adapter(name: str, adapter_class: type[DatasetAdapter]):
    """Registers a dataset adapter class under a string name."""
    _registry[name] = adapter_class

def get_adapter(name: str) -> DatasetAdapter:
    """Instantiates and returns a registered dataset adapter."""
    if name not in _registry:
        raise ValueError(f"Dataset adapter '{name}' not found. Available adapters: {list(_registry.keys())}")
    return _registry[name]()

# Register the default canonical adapter
register_adapter("urban_road_collapse", UrbanRoadCollapseAdapter)
register_adapter("default", UrbanRoadCollapseAdapter)
