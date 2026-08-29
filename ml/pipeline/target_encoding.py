import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from ml.schema import ORDINAL_ENCODING

class TargetEncoderWrapper:
    """
    A unified wrapper for target encoding that handles both tree (nominal)
    and linear (ordinal) encoding schemes, and provides a decoding method.
    """
    def __init__(self, is_linear: bool = False):
        self.is_linear = is_linear
        self.label_encoder = None
        
    def fit_transform(self, y: pd.Series) -> np.ndarray:
        if self.is_linear:
            # Ordinal Encoding based on canonical schema ordering
            y_encoded = y.map(ORDINAL_ENCODING)
            if y_encoded.isna().any():
                raise ValueError("Found unknown target classes during ordinal encoding")
            return y_encoded.values
        else:
            # Nominal Encoding (LabelEncoder) for tree models
            self.label_encoder = LabelEncoder()
            return self.label_encoder.fit_transform(y)
            
    def decode(self, y_encoded: np.ndarray) -> np.ndarray:
        if self.is_linear:
            # Reverse the ordinal encoding dict
            inverse_mapping = {v: k for k, v in ORDINAL_ENCODING.items()}
            return np.vectorize(inverse_mapping.get)(y_encoded)
        else:
            if self.label_encoder is None:
                raise RuntimeError("LabelEncoder not fitted. Cannot decode.")
            return self.label_encoder.inverse_transform(y_encoded)
