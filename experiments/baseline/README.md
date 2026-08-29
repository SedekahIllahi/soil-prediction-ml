# Baseline Experiment Results

This directory contains the output of the Phase 1 baseline model training.

## Contents
- `metrics.json`: Contains the full evaluation metrics (accuracy, precision, recall, f1) overall and per-class for all baseline models.
- `confusion_matrices/`: Contains the JSON representation of the confusion matrix for each model.

## Reproducibility
To reproduce these results, ensure your `.env` has `RANDOM_SEED=42` and run:

```bash
# Locally
python -m ml.experiments.run_baseline

# Or via Docker
docker compose run ml python -m ml.experiments.run_baseline
```
