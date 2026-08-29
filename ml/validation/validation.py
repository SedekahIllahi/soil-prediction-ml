import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal

from ml.schema import (
    TARGET_COLUMN, 
    TARGET_CLASSES,
    MODEL_FEATURES,
    REMOVED_COLUMNS,
    METADATA_COLUMNS,
    FEATURE_RANGES
)

@dataclass
class ValidationMessage:
    level: Literal["ERROR", "WARNING", "INFO"]
    code: str
    message: str
    column: str | None = None
    count: int | None = None
    
    def __str__(self):
        col_part = f"[{self.column}] " if self.column else ""
        count_part = f" (n={self.count})" if self.count is not None else ""
        return f"{self.level}: {self.code} - {col_part}{self.message}{count_part}"

@dataclass
class ValidationReport:
    errors: list[ValidationMessage]
    warnings: list[ValidationMessage]
    infos: list[ValidationMessage]
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0
        
    def summary(self) -> str:
        out = []
        out.append(f"Validation {'PASSED' if self.is_valid else 'FAILED'}")
        out.append(f"Errors: {len(self.errors)}")
        out.append(f"Warnings: {len(self.warnings)}")
        out.append(f"Infos: {len(self.infos)}")
        out.append("\nDetails:")
        for e in self.errors:
            out.append(str(e))
        for w in self.warnings:
            out.append(str(w))
        for i in self.infos:
            out.append(str(i))
        return "\n".join(out)

class DatasetValidator:
    """
    Validates a loaded dataset against the canonical schema.
    Performs structural checks (columns, target), missing value detection,
    range validation, and IQR outlier detection (for warnings).
    Does NOT remove rows.
    """
    
    def __init__(self):
        pass
        
    def validate(self, df: pd.DataFrame) -> ValidationReport:
        errors = []
        warnings = []
        infos = []
        
        # 1. Structural Validation
        # Check target column
        if TARGET_COLUMN not in df.columns:
            errors.append(ValidationMessage("ERROR", "MISSING_TARGET", "Target column is missing", TARGET_COLUMN))
        else:
            # Check target classes
            unique_targets = df[TARGET_COLUMN].dropna().unique()
            unknown_targets = set(unique_targets) - set(TARGET_CLASSES)
            if unknown_targets:
                errors.append(ValidationMessage(
                    "ERROR", "UNKNOWN_TARGET_CLASSES", 
                    f"Found unknown target classes: {unknown_targets}", TARGET_COLUMN
                ))
            
            # Check for missing targets
            missing_target_count = df[TARGET_COLUMN].isna().sum()
            if missing_target_count > 0:
                errors.append(ValidationMessage(
                    "ERROR", "MISSING_TARGET_VALUES", 
                    "Target column contains missing values", TARGET_COLUMN, missing_target_count
                ))
                
        # Check model features
        missing_features = set(MODEL_FEATURES) - set(df.columns)
        if missing_features:
            for feature in missing_features:
                errors.append(ValidationMessage("ERROR", "MISSING_FEATURE", "Required feature column is missing", feature))
                
        # Check removed columns
        found_removed = set(REMOVED_COLUMNS).intersection(df.columns)
        if found_removed:
            for col in found_removed:
                errors.append(ValidationMessage("ERROR", "LEAKAGE_COLUMN_PRESENT", "Removed/Leakage column is present and must be excluded", col))
                
        # Check metadata columns
        missing_metadata = set(METADATA_COLUMNS) - set(df.columns)
        if missing_metadata:
            for col in missing_metadata:
                warnings.append(ValidationMessage("WARNING", "MISSING_METADATA", "Metadata column is missing (not required for ML)", col))
                
        # Check for extra columns
        expected_cols = set(MODEL_FEATURES) | set([TARGET_COLUMN]) | set(METADATA_COLUMNS)
        extra_cols = set(df.columns) - expected_cols - set(REMOVED_COLUMNS)
        if extra_cols:
            for col in extra_cols:
                warnings.append(ValidationMessage("WARNING", "EXTRA_COLUMN", "Unexpected column found", col))
                
        # If structural validation fails, we might not be able to do deeper checks cleanly,
        # but we'll proceed on the features we do have.
        
        present_features = [f for f in MODEL_FEATURES if f in df.columns]
        
        # 2. Missing Values Validation
        total_rows = len(df)
        has_missing = False
        for feature in present_features:
            missing_count = df[feature].isna().sum()
            if missing_count > 0:
                has_missing = True
                warnings.append(ValidationMessage(
                    "WARNING", "MISSING_VALUES", 
                    f"Feature contains {missing_count} missing values ({missing_count/total_rows*100:.1f}%)", 
                    feature, missing_count
                ))
        if not has_missing:
            infos.append(ValidationMessage("INFO", "NO_MISSING_VALUES", "No missing values detected in model features"))
            
        # 3. Range Validation & 4. IQR Outlier Detection
        # Only check numeric features
        for feature in present_features:
            if not pd.api.types.is_numeric_dtype(df[feature]):
                errors.append(ValidationMessage("ERROR", "NON_NUMERIC_FEATURE", "Feature must be numeric", feature))
                continue
                
            # Range check
            if feature in FEATURE_RANGES:
                min_val, max_val = FEATURE_RANGES[feature]
                out_of_range = ((df[feature] < min_val) | (df[feature] > max_val)).sum()
                if out_of_range > 0:
                    warnings.append(ValidationMessage(
                        "WARNING", "OUT_OF_RANGE", 
                        f"Values outside canonical range [{min_val}, {max_val}]", 
                        feature, out_of_range
                    ))
                    
            # IQR Outlier check
            q1 = df[feature].quantile(0.25)
            q3 = df[feature].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                outliers = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).sum()
                if outliers > 0:
                    warnings.append(ValidationMessage(
                        "WARNING", "EXTREME_OUTLIERS", 
                        f"Values outside 3x IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]", 
                        feature, outliers
                    ))
                    
        infos.append(ValidationMessage("INFO", "DATASET_SIZE", f"Dataset validated with {total_rows} rows", None, total_rows))
        
        return ValidationReport(errors=errors, warnings=warnings, infos=infos)
