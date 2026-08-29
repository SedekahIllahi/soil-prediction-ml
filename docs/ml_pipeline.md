# ML Pipeline Architecture & Implementation

This document describes the design and flow of the ML pipeline implemented in `ml/`.

## Core Philosophy

1. **Defensibility**: Every transformation step is deliberate.
2. **Reproducibility**: Random seeds are centrally managed via `RANDOM_SEED`. Split stratifications are deterministic.
3. **No Leakage**: Validation and Test sets NEVER influence the fitting of the preprocessor.

## Pipeline Flow

1. **Adapter**: `ml.adapters.urban_road_collapse.UrbanRoadCollapseAdapter` maps the raw CSV to the unified schema.
2. **Validator**: `ml.validation.validation.DatasetValidator` checks structural invariants.
3. **Splitter**: `ml.pipeline.splitting.DataSplitter` creates the 70/15/15 stratified holdouts.
4. **Preprocessors**: 
   - `build_linear_preprocessor`: Imputation + Scaling (StandardScaler)
   - `build_tree_preprocessor`: Imputation only.
5. **Target Encoder**: `TargetEncoderWrapper` maps `Low, Moderate, High, Critical` to numerical vectors. Ordinal for linear models (0,1,2,3), Nominal for tree models.
6. **Training**: `ml.pipeline.training.run_baseline_training` iterates over defined `ModelConfig`s.
7. **Evaluation**: `ml.pipeline.evaluation.evaluate_model` produces comprehensive metrics including per-class recall to guard against missing Critical risks.

## Extending the Pipeline

To add a new model family (e.g. neural networks), define a new preprocessor in `ml/pipeline/preprocessing.py`, configure its `ModelConfig` in `registry.py` under the new model family name, and update the dispatching logic in `run_baseline_training`.
