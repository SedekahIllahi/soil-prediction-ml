# Development Guide

## Environment Setup

1. Copy `.env.example` to `.env`.
2. Ensure you have Docker and Docker Compose installed.

## Running Tests

Tests are written using `pytest`.

```bash
# Run locally (requires pip install -r backend/requirements.txt)
PYTHONPATH=. python -m pytest tests/ml/ -v

# Run via Docker
docker compose run ml python -m pytest tests/ml/ -v
```

## Running Baseline Training

```bash
docker compose run ml python -m ml.experiments.run_baseline
```

Output is written to `experiments/baseline/`.
