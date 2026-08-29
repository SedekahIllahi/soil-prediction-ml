import pandas as pd
import numpy as np
import pytest

from ml.schema import MODEL_FEATURES
from ml.pipeline.preprocessing import build_linear_preprocessor, build_tree_preprocessor
from ml.pipeline.target_encoding import TargetEncoderWrapper

def test_linear_preprocessor_imputes_and_scales(sample_raw_df, adapter):
    X, _, _ = adapter.transform(sample_raw_df)
    
    # Introduce some missing values
    X.loc[0, MODEL_FEATURES[0]] = np.nan
    
    prep = build_linear_preprocessor(MODEL_FEATURES)
    X_proc = prep.fit_transform(X)
    
    # Should be a numpy array with no NaNs
    assert isinstance(X_proc, np.ndarray)
    assert not np.isnan(X_proc).any()
    
    # Should be scaled (mean approx 0, std approx 1)
    # Check the first column
    assert abs(X_proc[:, 0].mean()) < 1e-10
    assert abs(X_proc[:, 0].std() - 1.0) < 1e-10
    
def test_tree_preprocessor_imputes_only(sample_raw_df, adapter):
    X, _, _ = adapter.transform(sample_raw_df)
    
    # Introduce some missing values
    X.loc[0, MODEL_FEATURES[0]] = np.nan
    
    prep = build_tree_preprocessor(MODEL_FEATURES)
    X_proc = prep.fit_transform(X)
    
    # Should be a numpy array with no NaNs
    assert isinstance(X_proc, np.ndarray)
    assert not np.isnan(X_proc).any()
    
    # Should NOT be scaled (mean/std will depend on data)
    assert abs(X_proc[:, 0].std() - 1.0) > 1e-2 # Almost certainly not 1.0 naturally
    
def test_preprocessor_prevents_leakage(sample_raw_df, adapter):
    X, _, _ = adapter.transform(sample_raw_df)
    
    # Split manually
    X_train = X.iloc[:50].copy()
    X_test = X.iloc[50:].copy()
    
    # Plant a massive outlier in test
    X_test.loc[50, MODEL_FEATURES[0]] = 999999
    
    prep = build_linear_preprocessor(MODEL_FEATURES)
    prep.fit(X_train)
    
    # Retrieve the scaler from the pipeline
    scaler = prep.named_steps['preprocessor'].transformers_[0][1].named_steps['scaler']
    
    mean_before = scaler.mean_[0]
    
    # Transform test
    prep.transform(X_test)
    
    # Check that the scaler's internal state did not change
    mean_after = scaler.mean_[0]
    assert mean_before == mean_after
    
def test_target_encoder_linear():
    y = pd.Series(["Low", "Moderate", "High", "Critical"])
    enc = TargetEncoderWrapper(is_linear=True)
    
    y_enc = enc.fit_transform(y)
    assert (y_enc == [0, 1, 2, 3]).all()
    
    y_dec = enc.decode(y_enc)
    assert (y_dec == ["Low", "Moderate", "High", "Critical"]).all()
    
def test_target_encoder_tree():
    y = pd.Series(["Low", "Moderate", "High", "Critical"])
    enc = TargetEncoderWrapper(is_linear=False)
    
    y_enc = enc.fit_transform(y)
    
    # Note: LabelEncoder sorts alphabetically, so classes will be:
    # Critical (0), High (1), Low (2), Moderate (3)
    y_dec = enc.decode(y_enc)
    assert (y_dec == ["Low", "Moderate", "High", "Critical"]).all()
