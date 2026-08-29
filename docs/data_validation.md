# Data Validation Strategy

The ML pipeline implements a strict, multi-tiered validation approach. All logic resides in `ml/validation/validation.py`.

## Validation Tiers

1. **ERROR (Blocks execution):**
   - Missing target column.
   - Unknown target classes (anything other than Low, Moderate, High, Critical).
   - Missing expected model features.
   - Presence of explicit leakage features (e.g. `historical_collapse_count`).

2. **WARNING (Logged, execution continues):**
   - Missing metadata columns (e.g. `segment_id`).
   - Missing values (handled downstream by the Imputer).
   - Values outside the theoretical bounds defined in `FEATURE_RANGES`.
   - Extreme outliers (Values outside $Q1 - 3 \times IQR$ and $Q3 + 3 \times IQR$).

3. **INFO:**
   - General summary statistics.

## Row Preservation Policy

**The validator NEVER drops rows.** 
Row dropping must be an explicit action taken by the preprocessor or adapter if specifically configured to do so. In Phase 1, we rely on the `SimpleImputer` to handle `NaN` values to ensure the pipeline is robust to messy inputs without silently destroying data volume.
