import pandas as pd
import pytest

from ml.validation.validation import DatasetValidator

def test_validation_success(sample_raw_df):
    validator = DatasetValidator()
    report = validator.validate(sample_raw_df)
    
    assert report.is_valid
    assert len(report.errors) == 0
    # There might be some IQR warnings depending on the random generation,
    # but there should be no errors.
    
def test_validation_missing_target(sample_raw_df):
    df = sample_raw_df.drop(columns=["collapse_risk_level"])
    validator = DatasetValidator()
    report = validator.validate(df)
    
    assert not report.is_valid
    assert any(e.code == "MISSING_TARGET" for e in report.errors)
    
def test_validation_unknown_target_class(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[0, "collapse_risk_level"] = "Extreme Risk"
    validator = DatasetValidator()
    report = validator.validate(df)
    
    assert not report.is_valid
    assert any(e.code == "UNKNOWN_TARGET_CLASSES" for e in report.errors)
    
def test_validation_missing_feature(sample_raw_df):
    df = sample_raw_df.drop(columns=["road_age_years"])
    validator = DatasetValidator()
    report = validator.validate(df)
    
    assert not report.is_valid
    assert any(e.code == "MISSING_FEATURE" for e in report.errors)
    
def test_validation_leakage_feature(sample_raw_df):
    df = sample_raw_df.copy()
    df["historical_collapse_count"] = 5
    validator = DatasetValidator()
    report = validator.validate(df)
    
    assert not report.is_valid
    assert any(e.code == "LEAKAGE_COLUMN_PRESENT" for e in report.errors)

def test_validation_missing_value_warning(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[0, "road_age_years"] = pd.NA
    validator = DatasetValidator()
    report = validator.validate(df)
    
    # Missing values should NOT be an error (handled by preprocessing)
    assert report.is_valid
    assert any(w.code == "MISSING_VALUES" for w in report.warnings)
    
def test_validation_out_of_range_warning(sample_raw_df):
    df = sample_raw_df.copy()
    # road_age_years range is [1, 60]
    df.loc[0, "road_age_years"] = 100
    validator = DatasetValidator()
    report = validator.validate(df)
    
    assert report.is_valid
    assert any(w.code == "OUT_OF_RANGE" for w in report.warnings)
    
def test_validation_never_deletes_rows(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[0, "road_age_years"] = pd.NA
    df.loc[1, "road_age_years"] = 100
    
    validator = DatasetValidator()
    report = validator.validate(df)
    
    # Ensure validation report has no output dataframe, it just validates.
    # The dataframe passing through the adapter should retain all rows.
    from ml.adapters.adapter_registry import get_adapter
    adapter = get_adapter("urban_road_collapse")
    
    X, y, meta = adapter.transform(df)
    assert len(X) == len(df)
