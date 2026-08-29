# AGENTS.md — Development Rules and Conventions

## ML-Based Ground/Soil Risk Prediction and Monitoring System

This document defines the rules and conventions that all developers (human and AI agents) must follow when working on this repository.

**Guiding principle:**

> Could another developer understand, maintain, test, and extend what I just wrote without needing to ask the original developer?

**Source documents:**

- [PRD.md](file:///d:/Project/soil-ml-prediction/PRD.md) — Product requirements and scope
- [ARCHITECTURE.md](file:///d:/Project/soil-ml-prediction/ARCHITECTURE.md) — Technical architecture and implementation plan

---

## 1. General Development Principles

1. Follow [PRD.md](file:///d:/Project/soil-ml-prediction/PRD.md) for product scope. Do not invent requirements.
2. Follow [ARCHITECTURE.md](file:///d:/Project/soil-ml-prediction/ARCHITECTURE.md) for technical decisions. Do not silently change architectural decisions.
3. Do not add features merely because they seem useful. Every feature must trace to a PRD requirement or be explicitly flagged as a deviation.
4. Prefer simple solutions. Choose the least complex approach that satisfies the requirement.
5. Avoid premature abstraction. Build what is needed now; generalize only when a genuine second use case appears.
6. Avoid unnecessary dependencies. Check whether the existing stack solves the problem before adding a library.
7. Keep modules focused. Each module/file should have a clear, singular responsibility.
8. Favor readable code over clever code. Optimize for the next developer who reads it.
9. Preserve existing behavior unless a change is intentional and documented.
10. Inspect existing code before making changes. Understand what exists before modifying it.

---

## 2. Working with AI Agents

### Agents MUST

1. Read relevant documentation (PRD, ARCHITECTURE, this file) before modifying code.
2. Inspect existing implementation before creating new files or functions.
3. Understand dependencies before changing shared code.
4. Make the smallest reasonable change that satisfies the task.
5. Run relevant tests after changes.
6. Report what was changed (files, functions, logic).
7. Report which tests were run and their results.
8. Report known limitations or unresolved issues.
9. Update [TODO.md](file:///d:/Project/soil-ml-prediction/TODO.md) when work is completed.
10. Verify that claims are backed by actual test execution.

### Agents MUST NOT

- Fabricate test results or claim tests passed without running them.
- Fabricate model metrics or evaluation results.
- Pretend an implementation was tested when it was not.
- Delete working functionality without documented justification.
- Rewrite large sections unnecessarily. Prefer targeted edits.
- Introduce dependencies without a concrete reason.
- Ignore failing tests. If a test fails, diagnose and fix or report it.
- Silently change API contracts, database schemas, or architectural patterns.

---

## 3. Code Style

### 3.1 Python

**Use:**

- PEP 8 formatting.
- Type hints on all function signatures (parameters and return types).
- Clear, descriptive function and variable names.
- Small, focused functions (prefer < 30 lines for business logic).
- Explicit error handling with specific exception types.
- Docstrings for public functions, classes, and non-obvious logic.

**Avoid:**

- Functions exceeding ~50 lines. Break them down.
- Deeply nested logic (> 3 levels). Refactor with early returns or helper functions.
- Unnecessary global state.
- Magic numbers. Use named constants.
- Bare `except:` or `except Exception: pass`. Always handle or log explicitly.

### 3.2 TypeScript / Vue

**Use:**

- TypeScript types and interfaces for all data structures and API contracts.
- Vue 3 Composition API (`<script setup lang="ts">`).
- Clear component responsibilities — one concern per component where practical.
- Reusable components where there is a genuine reuse case.
- Consistent naming conventions (see §4).
- Explicit types for API request/response objects.

**Avoid:**

- `any` type unless genuinely unavoidable. Prefer `unknown` + type narrowing.
- Monolithic components (> 200 lines template + script). Split into subcomponents.
- Business logic buried inside `<template>`. Move logic to composables or utility functions.
- Duplicated API logic. Centralize in `services/`.

---

## 4. Naming Conventions

| Element                | Convention            | Example                              |
| ---------------------- | --------------------- | ------------------------------------ |
| Python files           | `snake_case.py`       | `prediction_service.py`             |
| Python functions       | `snake_case`          | `get_active_model()`                |
| Python classes         | `PascalCase`          | `DatasetService`                    |
| Python constants       | `UPPER_SNAKE_CASE`    | `MAX_UPLOAD_SIZE_MB`                |
| TypeScript files       | `camelCase.ts`        | `predictionApi.ts`                  |
| Vue components         | `PascalCase.vue`      | `PredictionForm.vue`                |
| Vue page files         | `PascalCase.vue`      | `DashboardPage.vue`                 |
| TypeScript interfaces  | `PascalCase`          | `PredictionResponse`                |
| TypeScript variables   | `camelCase`           | `activeModel`                       |
| API endpoints          | `lowercase-kebab`     | `/api/predictions`, `/api/datasets` |
| Database tables        | `snake_case`          | `model_version`                     |
| Database columns       | `snake_case`          | `dataset_version_id`                |
| Environment variables  | `UPPER_SNAKE_CASE`    | `DATABASE_URL`                      |
| ML artifact files      | `{uuid}.joblib`       | `a1b2c3d4.joblib`                   |
| Dataset versions       | `v{integer}`          | `v1`, `v2`                          |
| Model versions         | Sequential integer    | `1`, `2`, `3`                       |

---

## 5. Comments and Documentation

> **Comments should explain *why*, not merely *what*.**

**Good:**

```python
# Fit only on training data to prevent data leakage into validation/test sets
preprocessor.fit(X_train)
```

**Bad:**

```python
# Fit the preprocessor
preprocessor.fit(X_train)
```

**Use comments for:**

- Non-obvious decisions or trade-offs.
- Domain assumptions (e.g., why a particular risk threshold was chosen).
- Workarounds with references to the underlying issue.
- Important ML considerations (leakage prevention, stratification rationale).
- Security-sensitive behavior.
- Architectural reasoning when the "obvious" approach was intentionally avoided.

**Do not** fill code with comments restating obvious syntax or standard library calls.

---

## 6. ML-Specific Rules

### Data Integrity

- **Prevent data leakage.** Fit all transformations (scaling, encoding, imputation) on training data only. Apply learned transformations to validation and test sets via `transform()`.
- **Keep the test set isolated.** The test set must never be used during training, model selection, or hyperparameter tuning. It is used only for final evaluation of the selected model.
- **Use the 70/15/15 split** as defined in ARCHITECTURE.md §7. Use stratified splitting.
- **Use reproducible random seeds** (configured via `RANDOM_SEED` environment variable).

### Model Management

- Record the full training configuration: dataset version, algorithm, hyperparameters, evaluation metrics, timestamp.
- Never fabricate evaluation metrics. All metrics must come from actual model predictions.
- Preserve the preprocessing pipeline alongside each model artifact.
- Keep training and inference preprocessing consistent — the same fitted preprocessor must be used.

### Evaluation

When evaluating or changing ML code, consider:

- Per-class metrics (not just overall accuracy).
- High Risk class recall — the system should not routinely miss high-risk cases.
- Confusion matrix — understand where errors concentrate.
- Class imbalance — document and handle appropriately.
- Reproducibility — same seed + same data = same results.

### Disclaimer

The model must never be presented as a guaranteed ground-safety assessment. It is an academic decision-support prototype (see PRD §1.3).

---

## 7. Dataset Rules

Uploaded datasets are **untrusted input**. Always validate before processing.

### Required Validations

- **File type**: Accept only `.csv` for the MVP.
- **File size**: Enforce `MAX_UPLOAD_SIZE_MB` limit.
- **Required columns**: Verify all expected columns exist.
- **Data types**: Verify column types match the schema.
- **Missing values**: Detect and report.
- **Invalid values**: Detect out-of-range or unexpected values.
- **Target labels**: Verify label values match expected classes.
- **Duplicates**: Detect and report duplicate rows where relevant.

### Safety

- Never execute code from an uploaded dataset.
- Never use uploaded filenames directly in file paths. Generate safe names.
- Do not hardcode assumptions about a specific dataset outside the dataset adapter layer. Dataset-specific logic (column names, label mappings) belongs in `ml/adapters/`.

---

## 8. API Rules

### Route Handlers

- Validate all input using Pydantic schemas.
- Return predictable, consistent response structures.
- Use appropriate HTTP status codes (200, 201, 400, 404, 422, 500).
- Handle errors with clear messages. Do not expose internal paths, stack traces, or implementation details.
- Keep route handlers thin — delegate business logic to service classes.
- Do not place database queries, ML logic, or complex business rules inside route functions.

### Response Format

- Success responses include the requested data.
- Error responses follow the format: `{ "detail": "message", "errors": [...] }`.
- List endpoints support pagination: `?page=1&page_size=20`.

---

## 9. Database Rules

### Access Pattern

- All database access goes through the repository layer (`app/repositories/`).
- Services call repositories. Routes never call repositories directly.
- Use SQLAlchemy ORM with parameterized queries. Never construct SQL from raw user input.

### Schema Management

- Use Alembic for all schema changes. Every schema modification produces a migration file.
- Define explicit ORM models in `app/models/`.
- Add indexes where justified by query patterns — do not add speculative indexes.

### Safety

- Never store secrets (passwords, API keys) in the database unless encrypted.
- Never delete data destructively without an explicit requirement. Prefer soft-delete or status transitions.

---

## 10. Security Rules

### Non-Negotiable

| Rule                                  | Detail                                              |
| ------------------------------------- | --------------------------------------------------- |
| No committed secrets                  | Use `.env` + `.env.example`. `.env` is gitignored   |
| Environment-based configuration       | All secrets and deployment config via env vars      |
| File path sanitization                | Never construct file paths from raw user input       |
| Upload validation                     | Validate type, size, content before any processing  |
| Upload size limits                    | Enforce `MAX_UPLOAD_SIZE_MB`                        |
| API input validation                  | Pydantic schemas on all endpoints                   |
| No arbitrary command execution        | Never pass user input to shell commands             |
| No unsafe deserialization             | Only load Joblib files written by the application   |
| Treat uploaded datasets as untrusted  | Validate fully before ML processing                 |
| Safe error responses                  | No stack traces or internal paths in API responses  |
| CORS restriction                      | Allow only the configured frontend origin           |

Any security-sensitive implementation should be flagged for additional review.

---

## 11. Testing Rules

### Coverage Expectations

Every meaningful feature should have appropriate tests:

| Layer         | Tool     | Location                       | Focus                                 |
| ------------- | -------- | ------------------------------ | ------------------------------------- |
| Backend unit  | Pytest   | `tests/backend/unit/`          | Services, utilities, schemas          |
| Backend integ | Pytest   | `tests/backend/integration/`   | API endpoints with test DB            |
| ML pipeline   | Pytest   | `tests/ml/`                    | Split, preprocess, train, evaluate    |
| Frontend      | Vitest   | `frontend/src/**/*.test.ts`    | Components, composables, validation   |

### Testing Behavior

- Run the narrowest relevant tests first, then broader tests.
- Use deterministic test data. Commit small test datasets to `tests/fixtures/`.
- Use fixed random seeds for ML reproducibility in tests.
- Never remove a test merely to make the test suite pass. If a test is changed because requirements changed, document why.
- Never fabricate test results.

### Test Commands

```bash
# Backend + ML
cd backend && pytest

# Frontend
cd frontend && npx vitest run

# Via Docker
docker compose run backend pytest
docker compose run frontend npx vitest run
```

---

## 12. Git / Change Rules

- Keep commits focused on a single logical change.
- Write meaningful commit messages describing *what* and *why*.
- Do not commit: `.env`, `storage/` artifacts, `node_modules/`, `__pycache__/`, IDE configs.
- Do not commit large generated files (model binaries, full datasets).
- Review changes before committing. Verify no unintended files are staged.
- Do not rewrite history or perform destructive Git operations unless explicitly instructed.
- Avoid giant commits mixing unrelated changes.

---

## 13. Dependency Rules

Before adding a new dependency:

1. **Check the existing stack.** Does scikit-learn, FastAPI, or Vue already provide this capability?
2. **Verify necessity.** Can the problem be solved with standard library code?
3. **Prefer mature, maintained libraries.** Check maintenance status and community support.
4. **Keep dependency count reasonable.** Every dependency is a maintenance burden.
5. **Document the decision.** If a dependency is non-obvious, add a comment explaining why it was chosen.

Do not introduce infrastructure (Redis, Kafka, Celery, MLflow, Kubernetes) unless a concrete requirement justifies it. The MVP is a university project with a modular-monolith architecture.

---

## 14. Docker Rules

The application must remain runnable through `docker compose up`.

- Changes to Dockerfiles, Docker Compose, volumes, networking, or environment variables must be tested where practical.
- Persistent data (PostgreSQL, model artifacts, datasets) lives in Docker volumes. It must not disappear when containers are recreated.
- The `storage/` directory is mounted as a volume. Do not rely on data inside the container filesystem.
- Keep Docker configuration minimal. Do not add services (Redis, workers, etc.) unless required.
- Verify that `.env.example` stays in sync with any new environment variables.

---

## 15. Definition of Done

A task is considered complete only when:

| # | Criterion                                                            |
|---|----------------------------------------------------------------------|
| 1 | Implementation exists and is functional                              |
| 2 | Relevant tests exist, or the absence of tests is documented with reason |
| 3 | Relevant tests pass                                                  |
| 4 | Error handling covers expected failure modes                         |
| 5 | Documentation is updated if the change affects system understanding  |
| 6 | [TODO.md](file:///d:/Project/soil-ml-prediction/TODO.md) is updated |
| 7 | No known regression is introduced                                    |
| 8 | Acceptance criteria from ARCHITECTURE.md are satisfied               |

A component is **not** complete merely because code exists.
