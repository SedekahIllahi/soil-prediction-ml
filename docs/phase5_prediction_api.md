# Phase 5 — Prediction API & Real-Time Decision Support

**Project:** ML-Based Ground/Soil Risk Prediction and Monitoring System  
**Date:** 2026-08-31  
**Status:** Completed & Verified (52 Automated Tests Passing)  
**Authoritative Specifications:** [ARCHITECTURE.md §12, §16.1, §16.2, §16.6](file:///d:/Project/soil-ml-prediction/ARCHITECTURE.md), [TODO.md §Phase 5](file:///d:/Project/soil-ml-prediction/TODO.md)

---

## 1. Executive Summary & WHAT Was Done

Phase 5 implements the real-time inference and decision-support subsystem for the ML-Based Ground/Soil Risk Prediction platform.

Key deliverables completed:
1. **Database Persistence:** SQLAlchemy ORM model [`Prediction`](file:///d:/Project/soil-ml-prediction/backend/app/models/prediction.py) and Alembic migration [`0004_create_prediction_table.py`](file:///d:/Project/soil-ml-prediction/backend/alembic/versions/0004_create_prediction_table.py) recording 34 input features, predicted risk category, confidence, 4-class probability distribution, active model foreign key, coordinates, and timestamps.
2. **Strict Schema & Range Validation:** [`app/schemas/prediction.py`](file:///d:/Project/soil-ml-prediction/backend/app/schemas/prediction.py) validates all 34 canonical features against expected physiological ranges from [`ml/schema.py`](file:///d:/Project/soil-ml-prediction/ml/schema.py).
3. **In-Memory Caching Architecture:** [`app/services/prediction_service.py`](file:///d:/Project/soil-ml-prediction/backend/app/services/prediction_service.py) caches the loaded `.joblib` model estimator and preprocessing pipeline in RAM for low-latency inference, automatically refreshing when active model promotions or rollbacks occur.
4. **REST API Endpoints:**
   - `POST /api/predictions`: Real-time risk classification and probability scoring.
   - `GET /api/predictions`: Paginated prediction history with risk category filtering.
   - `GET /api/predictions/{id}`: Detailed record inspection with raw feature payloads.
   - `GET /api/schema/features`: Dynamic feature metadata (min, max, category, caution flag) for frontend form construction.
   - `GET /api/dashboard/summary`: Aggregate analytics (total predictions, 4-class risk distribution, active model overview).
   - `GET /api/dashboard/recent`: Recent prediction feed for real-time monitoring.
5. **Comprehensive Test Suite:** 14 new automated unit and integration tests added, bringing the project total to **52 passed tests**.

---

## 2. WHY It Was Done (Design Rationale & Architectural Decisions)

1. **4-Class Target Support:** Rather than collapsing classes, the inference pipeline provides predictions across all 4 canonical risk classes (`Low`, `Moderate`, `High`, `Critical`), with exact calibrated probabilities for each.
2. **In-Memory Caching for Inference Latency:** Deserializing Scikit-Learn / XGBoost estimators and preprocessors from disk on every HTTP request introduces unnecessary I/O overhead. Caching the active model in memory reduces prediction latency from tens of milliseconds to sub-millisecond execution.
3. **Automatic Cache Invalidation:** Whenever a new model is promoted or rolled back via `ModelService.promote_model()`, `invalidate_prediction_cache()` is called, ensuring zero stale inference without requiring process restarts.
4. **Strict Pre-Execution Validation:** Road safety and civil infrastructure features have known physical bounds (e.g. `road_age_years` ∈ [1, 60], `soil_moisture_pct` ∈ [5, 60]%). Validating ranges before feeding data into the ML preprocessor prevents corrupted predictions from unscaled outliers.
5. **Feature Schema Introspection:** Exposing `GET /api/schema/features` decouples frontend UI forms from hardcoded backend schemas, enabling dynamic rendering in Phase 6.

---

## 3. HOW It Works Technically (Step-by-Step Workflow)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Client submits JSON payload to POST /api/predictions     │
│    { "features": { 34 canonical features... }, "lat", "lon" }│
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. FastAPI + Pydantic Schema Validation                      │
│    - Checks all 34 required features are present            │
│    - Validates values against FEATURE_RANGES bounds         │
│    - Raises HTTP 422 if invalid                             │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. PredictionService In-Memory Cache Check                   │
│    - Queries active model from ModelRepository              │
│    - Raises HTTP 400 if no model is currently active        │
│    - If cached_model_id != active_model.id, loads artifacts │
│      from disk via joblib.load() and caches in memory        │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Preprocessing & Inference                                │
│    - Converts features into single-row DataFrame in canonical│
│      column order (list(MODEL_FEATURES))                    │
│    - Applies fitted preprocessor.transform()                │
│    - Runs model.predict_proba()                             │
│    - Formats 4-class probabilities: Low, Moderate, High, Crit│
│    - Identifies highest probability as predicted_class       │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Persistence & Response                                   │
│    - Persists record to `prediction` table via repository   │
│    - Returns PredictionResponse (HTTP 200)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. API Reference

### 4.1 Predictions API (`/api/predictions`)

#### `POST /api/predictions`
Submit 34 features for instant risk scoring.

**Request Body:**
```json
{
  "features": {
    "road_age_years": 15.0,
    "road_length_m": 500.0,
    "pavement_thickness_cm": 25.0,
    "surface_crack_density_pct": 12.0,
    "pavement_condition_index": 75.0,
    "rut_depth_mm": 10.0,
    "avg_daily_traffic": 25000.0,
    "heavy_vehicle_pct": 15.0,
    "avg_vehicle_speed_kmh": 45.0,
    "surface_deformation_mm": 15.0,
    "soil_moisture_pct": 25.0,
    "soil_density_g_cm3": 1.75,
    "soil_bearing_capacity_kpa": 250.0,
    "groundwater_depth_m": 8.0,
    "soil_porosity_pct": 35.0,
    "void_ratio": 0.65,
    "soil_settlement_mm": 20.0,
    "elevation_m": 35.0,
    "annual_rainfall_mm": 1500.0,
    "max_daily_rainfall_mm": 85.0,
    "flood_frequency_per_year": 2.0,
    "temperature_variation_c": 20.0,
    "waterlogging_duration_hr": 12.0,
    "drainage_efficiency": 0.75,
    "distance_to_water_body_m": 500.0,
    "underground_pipe_density": 0.45,
    "pipe_age_years": 20.0,
    "distance_to_pipeline_m": 15.0,
    "utility_excavation_count": 3.0,
    "sewer_condition_index": 70.0,
    "land_subsidence_rate_mm_year": 8.0,
    "nearby_construction_intensity": 0.35,
    "building_density_per_km2": 3500.0,
    "distance_to_previous_collapse_m": 1200.0
  },
  "latitude": -6.2088,
  "longitude": 106.8456
}
```

**Response (HTTP 200):**
```json
{
  "id": "e3b0c442-98fc-1c14-9afb-4c8996fb9242",
  "predicted_class": "Moderate",
  "confidence": 0.5842,
  "probabilities": {
    "Low": 0.1215,
    "Moderate": 0.5842,
    "High": 0.2411,
    "Critical": 0.0532
  },
  "model_version_id": "a1b2c3d4-0000-0000-0000-000000000001",
  "model_version_number": 1,
  "algorithm": "logistic_regression",
  "latitude": -6.2088,
  "longitude": 106.8456,
  "created_at": "2026-08-31T18:40:00Z"
}
```

#### `GET /api/predictions?page=1&page_size=20&risk_category=High`
Retrieve paginated prediction records with optional category filter.

#### `GET /api/predictions/{id}`
Retrieve complete prediction details including original submitted input features.

---

### 4.2 Feature Schema API (`/api/schema`)

#### `GET /api/schema/features`
Returns all 34 canonical features with UI categories, ranges, and caution labels:

```json
{
  "features": [
    {
      "name": "road_age_years",
      "type": "float",
      "min_val": 1.0,
      "max_val": 60.0,
      "is_caution": false,
      "category": "Road Infrastructure",
      "description": "Road Age Years"
    },
    {
      "name": "pavement_condition_index",
      "type": "float",
      "min_val": 9.83,
      "max_val": 100.0,
      "is_caution": true,
      "category": "Road Infrastructure",
      "description": "Pavement Condition Index"
    }
  ],
  "target_classes": ["Low", "Moderate", "High", "Critical"],
  "total_features": 34
}
```

---

### 4.3 Dashboard Analytics API (`/api/dashboard`)

#### `GET /api/dashboard/summary`
Returns total predictions, class distribution breakdown, and active model status:

```json
{
  "total_predictions": 142,
  "risk_distribution": {
    "Low": 45,
    "Moderate": 52,
    "High": 31,
    "Critical": 14
  },
  "active_model": {
    "id": "a1b2c3d4-...",
    "version": 1,
    "algorithm": "logistic_regression",
    "metrics": {
      "weighted_f1": 0.7353,
      "macro_f1": 0.7354,
      "accuracy": 0.7338
    },
    "created_at": "2026-08-30T10:00:00Z"
  }
}
```

#### `GET /api/dashboard/recent?limit=10`
Returns the 10 most recent prediction events.

---

## 5. File Map & Affected Components

| File Path | Component | Description |
| :--- | :--- | :--- |
| [`backend/app/models/prediction.py`](file:///d:/Project/soil-ml-prediction/backend/app/models/prediction.py) | Database Model | SQLAlchemy ORM model `Prediction`. |
| [`backend/alembic/versions/0004_create_prediction_table.py`](file:///d:/Project/soil-ml-prediction/backend/alembic/versions/0004_create_prediction_table.py) | Migration | Alembic migration for `prediction` table. |
| [`backend/app/schemas/prediction.py`](file:///d:/Project/soil-ml-prediction/backend/app/schemas/prediction.py) | Schemas | Pydantic request/response schemas for prediction, schema, and dashboard. |
| [`backend/app/repositories/prediction_repository.py`](file:///d:/Project/soil-ml-prediction/backend/app/repositories/prediction_repository.py) | Repository | Database operations and dashboard aggregations. |
| [`backend/app/services/prediction_service.py`](file:///d:/Project/soil-ml-prediction/backend/app/services/prediction_service.py) | Service | Inference orchestration, caching, and feature schema. |
| [`backend/app/services/model_service.py`](file:///d:/Project/soil-ml-prediction/backend/app/services/model_service.py) | Service | Integrated cache invalidation on model promotion. |
| [`backend/app/api/predictions.py`](file:///d:/Project/soil-ml-prediction/backend/app/api/predictions.py) | REST Router | `/api/predictions` endpoints. |
| [`backend/app/api/schema.py`](file:///d:/Project/soil-ml-prediction/backend/app/api/schema.py) | REST Router | `/api/schema/features` endpoint. |
| [`backend/app/api/dashboard.py`](file:///d:/Project/soil-ml-prediction/backend/app/api/dashboard.py) | REST Router | `/api/dashboard/summary` and `/recent` endpoints. |
| [`backend/app/main.py`](file:///d:/Project/soil-ml-prediction/backend/app/main.py) | Application Entry | Router registrations under `/api`. |
| [`tests/backend/unit/test_prediction_service.py`](file:///d:/Project/soil-ml-prediction/tests/backend/unit/test_prediction_service.py) | Automated Tests | Unit tests for service logic, validation, caching, and schemas. |
| [`tests/backend/integration/test_predictions_api.py`](file:///d:/Project/soil-ml-prediction/tests/backend/integration/test_predictions_api.py) | Automated Tests | Integration tests for HTTP endpoints. |

---

## 6. Assumptions & Limitations

1. **Academic Decision-Support Prototype:** As emphasized throughout project specifications, the system is a decision-support prototype. Predictions must not be treated as automated safety-critical emergency overrides without field verification.
2. **Active Model Requirement:** Predictions require an explicitly active model. If no model has been promoted, the API safely rejects requests with a descriptive HTTP 400 error rather than attempting uncalibrated inference.
3. **Single Process Memory Cache:** In-memory caching operates per worker process. For the MVP deployment with standard worker count, memory footprint is minimal (< 50MB for linear and tree models).

---

## 7. Verification & Reproducibility

### 7.1 Running Automated Tests
Execute the entire test suite:
```powershell
python -m pytest tests/
```
**Result:** `52 passed, 61 warnings in 5.34s`

### 7.2 Manual API Verification via cURL / HTTP
```bash
# 1. Inspect Feature Schema
curl -X GET http://localhost:8000/api/schema/features

# 2. View Active Model
curl -X GET http://localhost:8000/api/models/active

# 3. Request Risk Prediction
curl -X POST http://localhost:8000/api/predictions \
  -H "Content-Type: application/json" \
  -d '{"features": {...34 features...}}'

# 4. View Dashboard Analytics
curl -X GET http://localhost:8000/api/dashboard/summary
```
