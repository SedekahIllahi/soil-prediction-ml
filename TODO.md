# Project TODO

## ML-Based Ground/Soil Risk Prediction and Monitoring System

**Source:** [ARCHITECTURE.md §23](file:///d:/Project/soil-ml-prediction/ARCHITECTURE.md) — Implementation Phases

**Last Updated:** 2026-08-31

---

## How to Use This File

- Mark tasks `[x]` only when **acceptance criteria are satisfied** — not merely when code exists.
- Add discovered subtasks as needed. Do not silently delete unfinished work.
- If scope changes, document the reason inline or in the [Decisions](#decisions) section.
- Keep this file synchronized with actual project state.
- Completed tasks remain visible unless the list becomes unreasonably large.

---

## Phase 0 — Dataset Research & Selection

**Goal:** Select and validate the dataset for the project.
**Dependencies:** None
**Status:** ✅ Complete

### Tasks

- [x] Identify and document candidate datasets with source URLs
  - [x] Urban Road Collapse Risk Assessment Dataset
  - [x] Geotechnical Slope Stability Analysis Dataset
  - [x] LUCAS 2018 Topsoil / Bulk Density Dataset
  - [x] Hyrcanian Forest Road Soil Properties Dataset
- [x] Evaluate each candidate dataset
  - [x] Verify data provenance and licensing
  - [x] Check sample count (sufficient for 70/15/15 split)
  - [x] Inspect available features (numerical vs. categorical)
  - [x] Inspect target variable / label reliability
  - [x] Check missing values and data quality
  - [x] Check class distribution (balance across categories)
  - [x] Assess whether the dataset represents the project's intended risk concept
  - [x] Check for potential synthetic data or label leakage
- [x] Document evaluation findings for at least 2 datasets
- [x] Select final dataset with documented justification
- [x] Define target variable mapping to Low / Moderate / High / Critical Risk
- [x] Document canonical feature schema (column names, types, target)
- [x] Document sample statistics (row count, class distribution, missing values)
- [x] Implement initial dataset adapter for the selected dataset

### Acceptance Criteria

- [x] At least 2 candidate datasets evaluated with documented findings
- [x] Final dataset selected with documented justification
- [x] Target variable mapping to Low/Moderate/High/Critical Risk is defined and defensible
- [x] Feature list (numerical vs. categorical) is documented
- [x] Dataset adapter loads and validates the selected dataset without errors
- [x] Sample statistics are documented

### Phase 0 Outputs

| Document | Location | Description |
| --- | --- | --- |
| Dataset Audit Report | `docs/dataset2_audit_report.md` | Target provenance, leakage, and validity audit |
| Canonical Dataset Specification | `docs/dataset_specification.md` | Authoritative schema for the ML pipeline |

---

## Phase 1 — Project Foundation

**Goal:** Establish the project skeleton so that subsequent phases have infrastructure.
**Dependencies:** None (can run parallel with Phase 0)
**Status:** ✅ Complete

### Tasks

- [x] Initialize repository structure per ARCHITECTURE.md §3
- [x] Create `docker-compose.yml` with frontend, backend, and PostgreSQL services
- [x] Create `docker-compose.dev.yml` with development overrides
- [x] Create `docker/backend.Dockerfile`
- [x] Create `docker/frontend.Dockerfile`
- [x] Create `docker/nginx.conf` for frontend reverse proxy
- [x] Create FastAPI application skeleton (`backend/app/main.py`)
  - [x] Health check endpoint (`GET /api/health`)
  - [x] CORS middleware
  - [x] Configuration loading from environment
- [x] Create Vue 3 + Vite + TypeScript skeleton (`frontend/`)
  - [x] Vue Router with placeholder pages
  - [x] Basic application layout
- [x] Set up PostgreSQL with initial Alembic migration
- [x] Create `.env.example` with all documented variables
- [x] Create `.gitignore` (secrets, artifacts, node_modules, __pycache__)
- [x] Create initial `README.md` with basic project description
- [x] Verify backend API and frontend component build

### Acceptance Criteria

- [x] Docker setup ready for backend, frontend, and database services
- [x] `GET /api/health` returns 200 from the backend
- [x] Frontend layout and pages configured for `localhost:3000`
- [x] PostgreSQL database engine & Alembic migrations initialized
- [x] `.env.example` exists with documented variables
- [x] `.gitignore` excludes secrets, generated artifacts, `node_modules`, `__pycache__`

---

## Phase 2 — Dataset Pipeline

**Goal:** Build data ingestion, validation, splitting, and preprocessing pipeline.
**Dependencies:** Phase 0 (dataset selected), Phase 1 (infrastructure)
**Status:** ✅ Complete

### Tasks

- [x] Implement dataset adapter base class (`ml/adapters/base.py`)
- [x] Implement concrete adapter for the selected dataset
- [x] Implement adapter registry (`ml/adapters/adapter_registry.py`)
- [x] Create database tables: `Dataset`, `DatasetVersion` (Alembic migration)
- [x] Implement dataset upload API endpoint (`POST /api/datasets/upload`)
  - [x] File type validation (.csv only)
  - [x] File size validation (MAX_UPLOAD_SIZE_MB)
  - [x] Filename sanitization
- [x] Implement dataset schema validation (required columns, types, labels)
- [x] Implement dataset preview endpoint (`GET /api/datasets/preview/{file_id}`)
- [x] Implement dataset integration endpoint (`POST /api/datasets/integrate`)
- [x] Implement dataset versioning (new version on integration)
- [x] Implement dataset listing endpoints (`GET /api/datasets`, `GET /api/datasets/{id}`)
- [x] Implement stratified 70/15/15 split (`ml/pipeline/splitting.py`)
  - [x] Fixed random seed for reproducibility
  - [x] Stratification by target class
- [x] Implement preprocessing pipeline (`ml/pipeline/preprocessing.py`)
  - [x] ColumnTransformer with numerical and categorical pipelines
  - [x] fit_transform() on training data only
  - [x] transform() only on validation/test data
- [x] Implement preprocessor serialization (save/load via Joblib)
- [x] Write tests for dataset validation
- [x] Write tests for splitting (proportions, stratification, reproducibility)
- [x] Write tests for preprocessing (fit/transform, no leakage, serialization round-trip)

### Acceptance Criteria

- [x] CSV upload succeeds for valid files
- [x] Invalid file types are rejected with clear error messages
- [x] Missing required columns are detected and reported
- [x] Invalid data types or label values are detected
- [x] Data preview returns correct sample rows
- [x] Dataset version is created on integration
- [x] 70/15/15 split produces correct proportions
- [x] Split is stratified (class ratios preserved across partitions)
- [x] Split is reproducible with same seed
- [x] Preprocessing fits on training data only
- [x] Preprocessor can be serialized and deserialized
- [x] Deserialized preprocessor produces identical output
- [x] Automated tests cover validation, splitting, and preprocessing


---

## Phase 3 — ML Training & Evaluation

**Goal:** Train multiple models, evaluate them, and compare performance.
**Dependencies:** Phase 2 (dataset pipeline)
**Status:** ✅ Complete

### Tasks

- [x] Implement candidate model registry (`ml/models/registry.py`)
  - [x] Logistic Regression
  - [x] Decision Tree
  - [x] Random Forest
  - [x] XGBoost
  - [x] SVM
- [x] Implement model training logic (`ml/pipeline/training.py`)
- [x] Implement evaluation metrics computation (`ml/pipeline/evaluation.py`)
  - [x] Accuracy
  - [x] Precision (per class)
  - [x] Recall (per class)
  - [x] F1-Score (per class, weighted, macro)
  - [x] Confusion matrix
- [x] Implement multi-model comparison and ranking (`ml/pipeline/comparison.py`)
  - [x] Rank by weighted F1-Score (primary criterion)
  - [x] Track High Risk recall as secondary criterion
- [x] Implement model artifact serialization (Joblib)
- [x] Create database tables: `TrainingRun`, `ModelVersion` (Alembic migration)
- [x] Implement training API endpoint (`POST /api/training`)
  - [x] Background execution (asyncio / BackgroundTasks)
  - [x] Return training run ID immediately
- [x] Implement training status endpoint (`GET /api/training/{id}`)
- [x] Implement training list endpoint (`GET /api/training`)
- [x] Write tests for model training (correct output shape/type)
- [x] Write tests for evaluation (metric calculations)
- [x] Write tests for comparison (ranking logic)
- [x] Verify no fabricated or hardcoded metrics in codebase

### Acceptance Criteria

- [x] All candidate algorithms train without errors on the selected dataset
- [x] Evaluation metrics are computed for each model (accuracy, precision, recall, F1, confusion matrix)
- [x] Per-class metrics are available, especially for High Risk
- [x] Comparison ranks models by the defined primary criterion
- [x] Model artifacts are serialized to `storage/models/`
- [x] Preprocessor artifacts are serialized to `storage/preprocessors/`
- [x] Training run record is created with correct status transitions (pending → running → completed/failed)
- [x] Background training does not block API requests
- [x] Training status can be polled via API
- [x] No fabricated or hardcoded metrics exist in the codebase
- [x] Automated tests cover training, evaluation, and comparison logic

### Phase 3 & 3.1 Outputs

| Document | Location | Description |
| --- | --- | --- |
| ML Training & Evaluation Report | `docs/phase3_training_evaluation_report.md` | Comprehensive empirical training & validation report for 5 candidate algorithms |
| Phase 3.1 Audit & Quality Gate | `docs/phase3_1_audit_report.md` | Pre-Phase 4 audit verifying data integrity, zero leakage, and metrics |
| Client-Facing ML Explanation | `docs/client_ml_model_explanation.md` | Non-technical technical explanation document for clients, lecturers, and stakeholders |

---

## Phase 4 — Model Management

**Goal:** Implement model versioning, promotion, and rollback.
**Dependencies:** Phase 3 (trained models exist)
**Status:** ✅ Complete

### Tasks

- [x] Implement model listing endpoint (`GET /api/models`)
- [x] Implement model detail endpoint (`GET /api/models/{id}`)
- [x] Implement active model endpoint (`GET /api/models/active`)
- [x] Implement model promotion logic (`POST /api/models/{id}/promote`)
  - [x] Set promoted model to `active`
  - [x] Set previously active model to `retired`
  - [x] Enforce: only one model active at a time
  - [x] Enforce: model must have evaluation metrics to be promoted
- [x] Implement rollback logic (re-promote a `retired` model via `/api/models/{id}/rollback` / promote)
- [x] Implement model comparison endpoint (`GET /api/models/compare`)
- [x] Implement model service with lifecycle management (`app/services/model_service.py`)
- [x] Write API tests for promotion, rollback, and edge cases

### Acceptance Criteria

- [x] All model versions are listed with metadata and metrics
- [x] Promoting a model sets it to `active` and retires the previous active model
- [x] Only one model is `active` at any time
- [x] Rollback re-promotes a `retired` model
- [x] A model without evaluation metrics cannot be promoted
- [x] Model comparison endpoint returns side-by-side metrics
- [x] API tests cover promotion, rollback, and edge cases

### Phase 4 Outputs

| Document | Location | Description |
| --- | --- | --- |
| Model Comparison Analysis (EN) | `secret_docs/model_comparison_analysis.md` | Client-facing baseline model comparison, analogies, and selection justification |
| Model Comparison Analysis (ID) | `secret_docs/model_comparison_analysis_id.md` | Indonesian version of baseline model comparison and selection analysis |

---

## Phase 5 — Prediction API

**Goal:** Implement the prediction endpoint and history.
**Dependencies:** Phase 4 (active model exists)
**Status:** ✅ Complete

### Tasks

- [x] Create database table: `Prediction` (Alembic migration)
- [x] Implement feature schema endpoint (`GET /api/schema/features`)
- [x] Implement prediction endpoint (`POST /api/predictions`)
  - [x] Load active model artifact
  - [x] Load associated preprocessor
  - [x] Apply preprocessing to input features
  - [x] Run model.predict() and model.predict_proba()
  - [x] Map numeric label to "Low" / "Moderate" / "High" / "Critical" Risk
  - [x] Save prediction record to database
- [x] Implement model + preprocessor caching
  - [x] Cache invalidation on model promotion
- [x] Implement prediction history endpoint (`GET /api/predictions`)
  - [x] Pagination support & risk category filtering
- [x] Implement single prediction detail (`GET /api/predictions/{id}`)
- [x] Implement dashboard summary endpoint (`GET /api/dashboard/summary`)
- [x] Implement dashboard recent predictions (`GET /api/dashboard/recent`)
- [x] Write tests for valid predictions
- [x] Write tests for invalid input (validation errors)
- [x] Write tests for no-active-model scenario

### Acceptance Criteria

- [x] Valid feature input returns a prediction (Low/Moderate/High/Critical Risk)
- [x] Prediction includes probabilities for all four classes
- [x] Invalid input returns a clear 422 validation error
- [x] Prediction uses the currently active model
- [x] Prediction uses the correct preprocessor paired with the active model
- [x] Each prediction is saved to the database
- [x] Prediction history is retrievable with pagination
- [x] Feature schema endpoint returns the expected input fields for the current dataset
- [x] Model/preprocessor are cached in memory; cache invalidates on promotion
- [x] API tests cover valid predictions, invalid input, and no active model scenarios

### Phase 5 Outputs

| Document | Location | Description |
| --- | --- | --- |
| Prediction API Specification & Guide | `docs/phase5_prediction_api.md` | Comprehensive documentation of inference engine, caching, schema validation, and REST API |

---

## Phase 6 — Frontend

**Goal:** Build the complete user interface.
**Dependencies:** Phase 5 (backend APIs complete)
**Status:** Not started

### Tasks

- [ ] Implement application layout and navigation
- [ ] Implement API client services (`frontend/src/services/`)
  - [ ] Base API client with error handling
  - [ ] Prediction API service
  - [ ] Dataset API service
  - [ ] Training API service
  - [ ] Model API service
  - [ ] Dashboard API service
- [ ] Implement Dashboard page
  - [ ] Prediction count
  - [ ] Risk category distribution (chart)
  - [ ] Recent predictions list
  - [ ] Active model info
  - [ ] Geographic map visualization (if dataset has location data)
- [ ] Implement Prediction page
  - [ ] Dynamic form from feature schema endpoint
  - [ ] Input validation
  - [ ] Prediction result display (risk category + confidence + probabilities)
- [ ] Implement Prediction History page
  - [ ] Paginated table
  - [ ] Filtering/sorting
- [ ] Implement Dataset Management page
  - [ ] Upload form with file validation
  - [ ] Validation result display
  - [ ] Data preview table
  - [ ] Integration confirmation
  - [ ] Dataset version list
- [ ] Implement Model Management page
  - [ ] Model version list with metrics
  - [ ] Side-by-side model comparison
  - [ ] Promote action
  - [ ] Rollback action
- [ ] Implement Training page
  - [ ] Initiate training form
  - [ ] Training status display with auto-refresh
  - [ ] Training results / comparison display
- [ ] Implement error states and user-friendly messages
- [ ] Write frontend tests for form validation and key components

### Acceptance Criteria

- [ ] Dashboard displays prediction count, risk distribution, and recent predictions
- [ ] Prediction form dynamically renders fields based on feature schema
- [ ] Prediction form validates input before submission
- [ ] Prediction result clearly displays the risk category and confidence
- [ ] Dataset upload shows validation results and preview
- [ ] Model management shows model list with metrics
- [ ] Model comparison view displays side-by-side metrics
- [ ] Promote and rollback actions work from the UI
- [ ] Training page shows status and results
- [ ] Error states display user-friendly messages
- [ ] Pages are functional on desktop screen sizes
- [ ] Frontend tests cover form validation and key component logic

---

## Phase 7 — Retraining Workflow

**Goal:** Enable the full retrain-on-new-data workflow end-to-end.
**Dependencies:** Phase 6 (frontend), Phases 3–5 (backend pipeline)
**Status:** Not started

### Tasks

- [ ] Connect dataset upload → validation → integration → training flow in UI
- [ ] Ensure new dataset version is created before training
- [ ] Ensure training run uses the correct (new) dataset version
- [ ] Display comparison of new model candidates vs. current active model
- [ ] Enable promotion of new model or retention of current model
- [ ] Handle case where new models perform worse than current model
- [ ] Test full end-to-end workflow from UI

### Acceptance Criteria

- [ ] New dataset upload creates a new dataset version
- [ ] Training on the new version produces new model candidates
- [ ] Comparison includes reference to current active model metrics
- [ ] User can promote a new model or keep the current one
- [ ] Full workflow functions end-to-end from UI
- [ ] If new models perform worse, current model remains active

---

## Phase 8 — Testing & Security

**Goal:** Harden the system with comprehensive tests and security measures.
**Dependencies:** Phase 7 (all features complete)
**Status:** Not started

### Tasks

#### Testing
- [ ] Write remaining backend unit tests for uncovered services and utilities
- [ ] Write integration tests for critical API flows
  - [ ] Dataset upload → validation → integration
  - [ ] Training → evaluation → model creation
  - [ ] Prediction with active model
  - [ ] Model promotion → rollback
- [ ] Write ML pipeline tests
  - [ ] Split correctness and reproducibility
  - [ ] Preprocessing fit/transform and leakage prevention
  - [ ] Training output shape and type
  - [ ] Serialization round-trip consistency
- [ ] Write frontend tests for remaining components
- [ ] Verify all tests use deterministic data and fixed seeds

#### Security
- [ ] Review file upload security (type, size, filename sanitization, storage path)
- [ ] Review API input validation coverage
- [ ] Implement/verify CORS configuration
- [ ] Review error responses (no stack traces or internal paths exposed)
- [ ] Verify no SQL injection vectors (ORM parameterized queries)
- [ ] Verify file paths are never constructed from raw user input
- [ ] Verify `.env` is gitignored and `.env.example` has no real secrets
- [ ] Review data leakage prevention in ML pipeline

### Acceptance Criteria

- [ ] Backend test suite passes with reasonable coverage of critical paths
- [ ] ML test suite passes with deterministic, seeded test cases
- [ ] Frontend test suite passes
- [ ] Invalid file uploads are rejected (wrong type, too large, malformed)
- [ ] API error responses do not expose internal paths or stack traces
- [ ] CORS is configured to allow only the frontend origin
- [ ] No SQL injection vectors exist (ORM parameterized queries verified)
- [ ] File paths are never constructed from raw user input

---

## Phase 9 — Documentation & Deployment Verification

**Goal:** Finalize documentation and verify the complete deployment experience.
**Dependencies:** Phase 8 (testing complete)
**Status:** Not started

### Tasks

#### Documentation
- [ ] Write/update `README.md` with prerequisites, setup, and usage
- [ ] Document system architecture overview in `docs/`
- [ ] Document dataset structure and adapter pattern
- [ ] Document ML pipeline (preprocessing, training, evaluation, metrics)
- [ ] Document model versioning and promotion workflow
- [ ] Document retraining workflow
- [ ] Document API reference (all endpoints)
- [ ] Document environment variables and configuration
- [ ] Document troubleshooting / known issues

#### Deployment Verification
- [ ] Verify `docker compose up` on a clean environment (no pre-existing data)
- [ ] Verify full user workflow: prediction, dataset upload, training, model promotion
- [ ] Verify `.env.example` is complete and accurate
- [ ] Verify persistent data survives container recreation

### Acceptance Criteria

- [ ] A developer can clone the repo, copy `.env.example` to `.env`, run `docker compose up`, and access the application
- [ ] README covers prerequisites, setup, and basic usage
- [ ] Architecture documentation exists in `docs/`
- [ ] Dataset documentation explains the schema and adapter pattern
- [ ] ML documentation explains the pipeline, metrics, and model selection
- [ ] API reference documents all endpoints
- [ ] A new developer can understand the system from the documentation without reverse-engineering

---

## Blocked

| Item | Blocked By | Notes |
| ---- | ---------- | ----- |
| Phase 2 tasks requiring specific column names | Phase 0 — Dataset not yet selected | Adapter implementation requires knowing the actual dataset schema |

---

## Open Questions

These are unresolved questions from [PRD.md §12](file:///d:/Project/soil-ml-prediction/PRD.md):

| # | Question | Status |
|---|----------|--------|
| Q1 | Which dataset will be selected for the initial implementation? | **Resolved** — Dataset 2 (Urban Road Collapse Risk, Kaggle CC0). See `docs/dataset_specification.md`. |
| Q2 | What are the exact input features for prediction? | **Resolved** — 34 model features. See `docs/dataset_specification.md` §Part 9. |
| Q3 | How should risk categories be derived if the dataset does not provide them directly? | **Resolved** — Dataset provides labels directly: Low / Moderate / High / Critical (4 classes). |
| Q4 | What minimum number of samples is acceptable? | **Resolved** — 15,000 rows; 70/15/15 = 10,500 / 2,250 / 2,250. Sufficient. |
| Q5 | Should the system support multiple concurrent datasets or only one active at a time? | Open — current architecture assumes one active dataset version. |
| Q6 | What level of authentication/authorization is required for the MVP? | Open — currently planned without auth (ARCHITECTURE.md AD-7). |
| Q7 | Are there specific regulatory or academic standards for the documentation? | Open |

---

## Decisions

Record important decisions made during implementation here.

| Date | Decision | Rationale | Reference |
| ---- | -------- | --------- | --------- |
| 2026-08-27 | Modular monolith architecture | Simplicity, appropriate for academic project | ARCHITECTURE.md AD-1 |
| 2026-08-27 | Weighted F1-Score as primary model selection criterion | Balances per-class performance; configurable | ARCHITECTURE.md AD-8 |
| 2026-08-27 | No authentication for initial MVP | PRD Q6 unresolved; can be added later | ARCHITECTURE.md AD-7 |
| 2026-08-27 | asyncio background task for training (no job queue) | Small dataset size; sufficient for MVP | ARCHITECTURE.md AD-4 |
| 2026-08-28 | Dataset 2 selected (Urban Road Collapse Risk) with major caveats | Best available candidate; 15,000 rows; CC0 license; synthetically generated but domain-coherent | `docs/dataset_specification.md` |
| 2026-08-28 | 4-class target retained (Low/Moderate/High/Critical) | All 4 classes are statistically distinct; merging Critical→High loses information | `docs/dataset_specification.md` §Part 1 |
| 2026-08-28 | Project scope narrowed to "Urban Road Collapse Risk Prediction" | Directly supported by dataset; broader scopes would be unsupported overclaims | `docs/dataset_specification.md` §Part 7 |
| 2026-08-28 | 4 features removed from ML pipeline | Leakage (SVI, historical_collapse_count) and derivation (TLI, PLI) | `docs/dataset_specification.md` §Part 2 |
| 2026-08-28 | Latitude/longitude excluded from ML, retained for visualization | Zero predictive value (r<0.005 with target); synthetic bounding box only | `docs/dataset_specification.md` §Part 4 |
| 2026-08-28 | Random stratified 70/15/15 split (no spatial grouping) | No spatial autocorrelation detected (NN same-class rate 0.222 < random 0.250) | `docs/dataset_specification.md` §Part 8 |
| 2026-08-28 | Fuzzy logic deferred/rejected for MVP | Labels are crisp quantile bins; ML calibrated probabilities are more defensible | `docs/dataset_specification.md` — Audit Investigation 6 |
