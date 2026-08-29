import pandas as pd
import numpy as np

from ml.pipeline.splitting import DataSplitter

def test_splitting_sizes(sample_raw_df, adapter):
    X, y, _ = adapter.transform(sample_raw_df)
    
    splitter = DataSplitter(test_size=0.15, val_size=0.15)
    res = splitter.split(X, y)
    
    total = len(sample_raw_df)
    # Expected proportions
    assert abs(res.train_size / total - 0.70) < 0.05
    assert abs(res.val_size / total - 0.15) < 0.05
    assert abs(res.test_size / total - 0.15) < 0.05
    
    assert res.train_size + res.val_size + res.test_size == total
    
def test_splitting_stratification(sample_raw_df, adapter):
    X, y, _ = adapter.transform(sample_raw_df)
    
    splitter = DataSplitter()
    res = splitter.split(X, y)
    
    # Calculate original proportions
    orig_props = y.value_counts(normalize=True)
    
    # Calculate split proportions
    train_props = res.y_train.value_counts(normalize=True)
    val_props = res.y_val.value_counts(normalize=True)
    test_props = res.y_test.value_counts(normalize=True)
    
    # Check that they match within 5% tolerance
    for class_label in orig_props.index:
        assert abs(orig_props[class_label] - train_props.get(class_label, 0)) < 0.05
        assert abs(orig_props[class_label] - val_props.get(class_label, 0)) < 0.05
        assert abs(orig_props[class_label] - test_props.get(class_label, 0)) < 0.05

def test_splitting_reproducibility(sample_raw_df, adapter):
    X, y, _ = adapter.transform(sample_raw_df)
    
    splitter = DataSplitter()
    res1 = splitter.split(X, y)
    res2 = splitter.split(X, y)
    
    assert (res1.X_train.index == res2.X_train.index).all()
    assert (res1.X_val.index == res2.X_val.index).all()
    assert (res1.X_test.index == res2.X_test.index).all()
