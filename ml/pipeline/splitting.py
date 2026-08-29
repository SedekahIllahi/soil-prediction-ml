import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os

@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    
    @property
    def train_size(self) -> int:
        return len(self.X_train)
        
    @property
    def val_size(self) -> int:
        return len(self.X_val)
        
    @property
    def test_size(self) -> int:
        return len(self.X_test)

class DataSplitter:
    """
    Handles reproducible, stratified dataset splitting.
    Follows canonical spec D08: Random stratified 70/15/15 split.
    """
    
    def __init__(self, test_size: float = 0.15, val_size: float = 0.15):
        """
        Args:
            test_size: Proportion of the dataset to include in the test split.
            val_size: Proportion of the dataset to include in the validation split.
        """
        self.test_size = test_size
        self.val_size = val_size
        
        # Calculate the intermediate test_size for the first split
        # We need to hold out (test_size + val_size) in the first step.
        self.holdout_size = test_size + val_size
        
        # In the second step, we split the holdout into val and test.
        # The proportion of val inside the holdout is val_size / (val_size + test_size)
        self.val_ratio_in_holdout = val_size / self.holdout_size

    def _get_random_seed(self) -> int:
        """Retrieves the canonical random seed from the environment, defaults to 42."""
        try:
            return int(os.environ.get("RANDOM_SEED", 42))
        except ValueError:
            return 42
            
    def split(self, X: pd.DataFrame, y: pd.Series) -> SplitResult:
        """
        Performs the 70/15/15 stratified split.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            SplitResult containing all 6 resulting datasets.
        """
        seed = self._get_random_seed()
        
        # Step 1: Split off the training set (70%) and the holdout set (30%)
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y, 
            test_size=self.holdout_size, 
            stratify=y, 
            random_state=seed
        )
        
        # Step 2: Split the holdout set (30%) into validation (15%) and test (15%)
        # For equal val/test sizes, test_size=0.5
        X_val, X_test, y_val, y_test = train_test_split(
            X_holdout, y_holdout,
            test_size=1.0 - self.val_ratio_in_holdout,
            stratify=y_holdout,
            random_state=seed
        )
        
        return SplitResult(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test
        )
