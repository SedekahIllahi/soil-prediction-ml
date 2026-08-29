from abc import ABC, abstractmethod
import pandas as pd

from ml.schema import FeatureSchema
from ml.validation.validation import DatasetValidator, ValidationReport

class DatasetAdapter(ABC):
    """
    Abstract base class for dataset adapters.
    Adapters are responsible for loading raw datasets, validating them,
    and transforming them into a format that conforms to the canonical FeatureSchema.
    """
    
    def __init__(self):
        self.validator = DatasetValidator()
        self._schema = FeatureSchema()
        
    def get_schema(self) -> FeatureSchema:
        """Returns the canonical feature schema expected by the pipeline."""
        return self._schema
        
    def load(self, file_path: str) -> pd.DataFrame:
        """Loads the raw dataset from a file into a DataFrame."""
        return pd.read_csv(file_path)
        
    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Validates the dataset against the canonical schema."""
        return self.validator.validate(df)
        
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Transforms the raw DataFrame into the canonical format.
        
        Returns:
            Tuple of (X, y, metadata):
            - X: DataFrame containing only the canonical MODEL_FEATURES
            - y: Series containing the TARGET_COLUMN
            - metadata: DataFrame containing METADATA_COLUMNS (if present)
        """
        pass
