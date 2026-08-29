import pytest
import pandas as pd
from pathlib import Path
import os
import sys

# Ensure project root is in sys.path
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.adapters.adapter_registry import get_adapter

@pytest.fixture
def sample_dataset_path():
    """Returns the path to the small synthetic test dataset."""
    return Path(__file__).parent.parent / "fixtures" / "sample_dataset.csv"

@pytest.fixture
def sample_raw_df(sample_dataset_path):
    """Returns the raw DataFrame of the test dataset."""
    return pd.read_csv(sample_dataset_path)

@pytest.fixture
def adapter():
    """Returns the configured DatasetAdapter."""
    return get_adapter("urban_road_collapse")
    
@pytest.fixture(autouse=True)
def set_random_seed():
    """Ensures a fixed random seed is set in the environment for tests."""
    os.environ["RANDOM_SEED"] = "42"
    yield
