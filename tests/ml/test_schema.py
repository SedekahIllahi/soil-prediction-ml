from ml.schema import (
    FeatureSchema,
    MODEL_FEATURES,
    METADATA_COLUMNS,
    REMOVED_COLUMNS,
    TARGET_COLUMN,
    TARGET_CLASSES,
    CAUTION_FEATURES
)

def test_schema_feature_count():
    schema = FeatureSchema()
    
    # 34 exact features
    assert len(schema.model_features) == 34
    
    # 5 caution features, all must be in model features
    assert len(CAUTION_FEATURES) == 5
    assert CAUTION_FEATURES.issubset(set(schema.model_features))

def test_schema_no_duplicates():
    # model_features is a tuple, casting to set removes duplicates
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES))
    
def test_schema_no_overlap():
    # REMOVED columns and METADATA columns must not be in MODEL features
    feature_set = set(MODEL_FEATURES)
    assert not feature_set.intersection(REMOVED_COLUMNS)
    assert not feature_set.intersection(METADATA_COLUMNS)
    
def test_target_schema():
    assert TARGET_COLUMN == "collapse_risk_level"
    assert len(TARGET_CLASSES) == 4
    assert set(TARGET_CLASSES) == {"Low", "Moderate", "High", "Critical"}
