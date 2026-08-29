# SKILL.md — Reusable Development Workflows

## ML-Based Ground/Soil Risk Prediction and Monitoring System

This document defines standardized workflows for common engineering tasks. Agents and developers can follow these procedures instead of improvising each time.

**Source documents:**

- [PRD.md](file:///d:/Project/soil-ml-prediction/PRD.md)
- [ARCHITECTURE.md](file:///d:/Project/soil-ml-prediction/ARCHITECTURE.md)
- [AGENTS.md](file:///d:/Project/soil-ml-prediction/AGENTS.md)

---

## Skill 1 — Feature Implementation

### Purpose

Implement a new feature safely, following the project's architecture and conventions.

### When to Use

When implementing a new functional requirement from the PRD or a task from TODO.md.

### Procedure

1. Read the relevant PRD requirement(s).
2. Read the relevant ARCHITECTURE.md section(s).
3. Read AGENTS.md rules that apply to the feature area.
4. Inspect existing code in affected modules. Do not duplicate existing functionality.
5. Identify affected files and modules. Map the change to the layer (API → Service → Repository/ML).
6. Plan the implementation. Identify what changes, what is new, and what tests are needed.
7. Implement the smallest reasonable change.
8. Add or update tests.
9. Run relevant tests.
10. Update TODO.md to reflect progress.
11. Report changes and test results.

### Checklist

- [ ] Change is traced to a PRD requirement or architecture section
- [ ] Existing code was inspected before writing new code
- [ ] Implementation follows the project's layering (routes → services → repositories/ML)
- [ ] Naming conventions from AGENTS.md §4 are followed
- [ ] Tests are added or updated
- [ ] Tests pass
- [ ] No existing tests are broken
- [ ] TODO.md is updated

### Expected Output

Report:
- Files created or modified
- Functions/classes added or changed
- Tests written and their results
- Any unresolved issues or limitations

---

## Skill 2 — Testing

### Purpose

Test a feature or module thoroughly.

### When to Use

When writing tests for new or existing functionality, or when verifying test coverage.

### Procedure

1. Identify the expected behavior from PRD/ARCHITECTURE requirements.
2. Identify edge cases: invalid input, missing data, boundary values, error states.
3. Write or update tests.
   - Use deterministic test data from `tests/fixtures/`.
   - Use fixed random seeds for ML tests.
   - Mock external dependencies (database, filesystem) in unit tests.
4. Run focused tests first (single file or module).
5. Run broader tests (full test suite) to check for regressions.
6. Diagnose any failures. Identify whether the failure is in the test or the implementation.
7. Fix root causes — do not patch symptoms or delete failing tests.
8. Report results.

### Checklist

- [ ] Happy-path test exists
- [ ] Invalid input test exists
- [ ] Edge case tests exist where applicable
- [ ] Tests are deterministic (no randomness without seed)
- [ ] Tests do not depend on external services or production data
- [ ] All tests pass
- [ ] No existing tests were removed without documented justification

### Expected Output

Report:
- Tests written (file, test name, what it covers)
- Test execution command and results
- Any failing tests with diagnosis
- Coverage gaps noted

---

## Skill 3 — Security Review

### Purpose

Review implementation for common security problems.

### When to Use

When implementing file uploads, API endpoints, database queries, user input handling, or Docker configuration changes. Also during Phase 8 security hardening.

### Procedure

1. Review the target code systematically against each check below.
2. Document each finding.
3. Implement fixes for critical and high-severity issues.
4. Flag medium/low issues for future attention.

### Checklist

- [ ] **Input validation**: All API inputs validated via Pydantic schemas
- [ ] **File uploads**: Type checked (.csv only), size limited, filename sanitized
- [ ] **Path traversal**: No file paths constructed from raw user input
- [ ] **SQL injection**: All queries use ORM parameterized queries (no raw SQL with string formatting)
- [ ] **Command injection**: No user input passed to shell commands
- [ ] **Unsafe deserialization**: Only Joblib files written by the application are loaded
- [ ] **Authentication**: Auth controls applied where required (or documented as intentionally absent per AD-7)
- [ ] **Secrets**: No secrets committed to repository; `.env` is gitignored
- [ ] **Environment variables**: All sensitive config loaded from env vars
- [ ] **Error leakage**: API error responses contain no stack traces, internal paths, or system info
- [ ] **CORS**: Restricted to configured frontend origin only
- [ ] **Dependencies**: No known critical vulnerabilities in dependencies
- [ ] **Docker**: No unnecessary ports exposed; volumes configured correctly

### Expected Output

For each finding:

| Field | Content |
|-------|---------|
| **Finding** | Brief description of the issue |
| **Severity** | Critical / High / Medium / Low |
| **Location** | File and line/function |
| **Explanation** | Why this is a problem |
| **Recommended Fix** | How to fix it |
| **Fixed?** | Yes / No / Deferred |

---

## Skill 4 — ML Pipeline Review

### Purpose

Review the ML implementation for correctness, reproducibility, and leakage prevention.

### When to Use

When implementing or modifying ML pipeline code (preprocessing, splitting, training, evaluation). Also as a periodic review during Phase 3, 7, and 8.

### Procedure

1. Trace the data flow from raw dataset to prediction output.
2. Check each stage against the checklist below.
3. Pay particular attention to the preprocessing boundary (fit on train only).
4. Verify test set isolation.
5. Document any findings or concerns.

### Checklist

- [ ] **Dataset loading**: Adapter correctly loads and validates the dataset
- [ ] **Feature/target separation**: Features and target are separated before any processing
- [ ] **70/15/15 split**: Split proportions are correct
- [ ] **Stratification**: Split preserves class distribution across all partitions
- [ ] **Random seed**: Fixed seed used; split is reproducible
- [ ] **Missing-value handling**: Imputation strategy is appropriate for the data
- [ ] **Categorical encoding**: Encoder is fit on training data only
- [ ] **Scaling**: Scaler is fit on training data only
- [ ] **Data leakage**: No information from validation/test flows into training
  - [ ] Preprocessing fit() uses only training data
  - [ ] No target leakage through features
  - [ ] No future-data leakage
- [ ] **Test set isolation**: Test set is used only for final evaluation of the selected model
- [ ] **Class imbalance**: Handling strategy (if any) is applied only to training set
- [ ] **Evaluation metrics**: Accuracy, precision, recall, F1, confusion matrix are computed correctly
- [ ] **Per-class metrics**: Available for all three classes
- [ ] **High Risk recall**: Specifically tracked and reported
- [ ] **Model serialization**: Joblib save/load round-trip produces identical predictions
- [ ] **Inference consistency**: Prediction uses the same fitted preprocessor as training

### Expected Output

Report:
- Each checklist item: Pass / Fail / N/A
- Any leakage vectors found
- Any reproducibility issues
- Recommendations for fixes

---

## Skill 5 — Model Comparison Review

### Purpose

Verify that model comparison is technically meaningful and fair.

### When to Use

When reviewing training results, comparison logic, or model selection decisions.

### Procedure

1. Verify all models were trained and evaluated under identical conditions.
2. Review metrics for each model.
3. Check whether the selection criterion is applied correctly.
4. Verify no fabricated or hardcoded results exist.

### Checklist

- [ ] **Same data**: All models trained on the same training set
- [ ] **Same split**: All models evaluated on the same validation set
- [ ] **Same preprocessing**: All models use the same fitted preprocessor
- [ ] **Same target**: All models predict the same target variable
- [ ] **Consistent metrics**: Same metrics computed for all models
- [ ] **Accuracy**: Reported and reasonable
- [ ] **Precision**: Per-class values reported
- [ ] **Recall**: Per-class values reported; High Risk recall specifically noted
- [ ] **F1-Score**: Weighted and macro reported
- [ ] **Confusion matrix**: Available for each model
- [ ] **Selection criterion**: Primary criterion (weighted F1) applied correctly
- [ ] **No fabrication**: All metrics come from actual model predictions, not hardcoded values
- [ ] **Misleading accuracy**: If classes are imbalanced, accuracy alone is not used for selection

### Expected Output

Report:
- Whether comparison is fair and valid
- Any methodological issues found
- Whether the selected model's performance justifies its selection
- Note if any model is suspiciously perfect (may indicate data leakage)

---

## Skill 6 — Dataset Review

### Purpose

Evaluate a candidate dataset before committing to using it in the project.

### When to Use

During Phase 0 when evaluating candidate datasets. Also when new datasets are proposed for retraining.

### Procedure

1. Obtain the dataset and document its source.
2. Inspect it systematically against the checklist below.
3. Produce a structured evaluation report.

### Checklist

- [ ] **Source / provenance**: Where does the dataset come from? Is it a reputable source?
- [ ] **License**: Is the dataset legally usable for this project?
- [ ] **Accessibility**: Can the dataset be downloaded and used programmatically?
- [ ] **Sample count**: How many rows? Is it sufficient for 70/15/15 split with meaningful partition sizes?
- [ ] **Features**: What features are available? Are they relevant to ground/soil risk?
- [ ] **Feature types**: Which are numerical? Which are categorical?
- [ ] **Target variable**: Does a clear target/label exist? What are its values?
- [ ] **Target reliability**: Is the target based on measurement, expert judgment, or synthetic derivation?
- [ ] **Missing data**: What percentage of values is missing? Which columns are affected?
- [ ] **Duplicate data**: Are there duplicate rows?
- [ ] **Class distribution**: How balanced are the classes? Severe imbalance?
- [ ] **Data types**: Are values in expected ranges and formats?
- [ ] **Geographic relevance**: Does location data exist? Is it relevant?
- [ ] **Domain relevance**: Does the dataset actually represent ground/soil risk as intended by the PRD?
- [ ] **Potential synthetic data**: Does the data look artificially generated?
- [ ] **Potential label leakage**: Does any feature directly encode or proxy the target?
- [ ] **Defensible risk mapping**: Can the target variable be defensibly mapped to Low/Moderate/High Risk?

### Expected Output

| Section | Content |
|---------|---------|
| **Dataset** | Name, source, URL |
| **Summary** | Row count, feature count, target description |
| **Strengths** | What makes this dataset suitable |
| **Weaknesses** | Quality issues, limitations, gaps |
| **Risks** | Leakage, synthetic data, domain mismatch |
| **Recommendation** | Use / Do not use / Use with caveats |

> Do not automatically accept a dataset merely because it has the expected column names.

---

## Skill 7 — Refactoring

### Purpose

Improve code quality without changing intended behavior.

### When to Use

When code is difficult to read, maintain, or extend, but the behavior is correct. Not for adding new features.

### Procedure

1. Read relevant architecture documentation for the module being refactored.
2. Identify the specific problem (duplication, complexity, unclear naming, etc.).
3. Ensure tests exist for the code being refactored. If not, add them first.
4. Make small, incremental changes. Run tests after each meaningful change.
5. Remove duplication only when it genuinely simplifies the code.
6. Do not introduce speculative abstractions ("we might need this later").
7. Verify behavior remains unchanged by running all relevant tests.
8. Update documentation if the refactoring changes module structure or public APIs.

### Checklist

- [ ] Tests exist before refactoring begins
- [ ] Changes are incremental
- [ ] Tests pass after each change
- [ ] No behavior change (unless intentional and documented)
- [ ] No speculative abstractions added
- [ ] Documentation updated if structure changed
- [ ] Final test suite passes

### Expected Output

Report:
- What was refactored and why
- Tests run before and after
- Confirmation that behavior is unchanged
- Any remaining code-quality concerns

---

## Skill 8 — Bug Investigation

### Purpose

Diagnose and fix a bug methodically.

### When to Use

When unexpected behavior is reported or observed.

### Procedure

1. **Reproduce** the problem. Define the exact steps, input, and expected vs. actual output.
2. **Inspect** relevant logs, error messages, and stack traces.
3. **Trace** the execution path through the code.
4. **Identify** the root cause (not just the symptom).
5. **Write a regression test** that fails before the fix and passes after.
6. **Implement the smallest fix** that addresses the root cause.
7. **Run relevant tests** including the new regression test.
8. **Verify no regression** in related functionality.
9. **Document** the bug and fix if it reveals a non-obvious system behavior.

### Checklist

- [ ] Bug is reproducible
- [ ] Root cause identified (not just symptom patched)
- [ ] Regression test written
- [ ] Fix implemented
- [ ] Regression test passes
- [ ] Existing tests pass
- [ ] No unintended side effects

### Expected Output

Report:
- Bug description (input, expected, actual)
- Root cause
- Fix applied (file, line, change)
- Regression test added
- Test results

---

## Skill 9 — Code Review

### Purpose

Review a proposed change for correctness, quality, and alignment with project standards.

### When to Use

Before merging a change or after an agent completes a task.

### Procedure

1. Read the change description. Understand the intent.
2. Review the diff against each category below.
3. Classify findings as blocking, non-blocking, or positive.

### Checklist

- [ ] **Correctness**: Does the code do what it claims?
- [ ] **Architecture alignment**: Does the change follow ARCHITECTURE.md patterns?
- [ ] **Readability**: Can another developer understand this without asking questions?
- [ ] **Maintainability**: Will this be easy to modify in the future?
- [ ] **Security**: Any security concerns per AGENTS.md §10?
- [ ] **Error handling**: Are failure cases handled appropriately?
- [ ] **Tests**: Are new/modified behaviors covered by tests?
- [ ] **Performance**: Any obvious inefficiencies? (Do not over-optimize)
- [ ] **ML correctness**: If ML code, does it follow Skill 4 checklist?
- [ ] **Scope creep**: Does the change introduce unrelated features?
- [ ] **Naming**: Follows conventions from AGENTS.md §4?
- [ ] **Comments**: Non-obvious logic is explained?

### Expected Output

#### Blocking Issues
Issues that must be fixed before the change is accepted.

#### Non-Blocking Suggestions
Improvements that can be addressed later.

#### Positive Findings
Things implemented correctly or well.

> Do not request changes for purely stylistic preference.

---

## Skill 10 — Docker / Deployment Verification

### Purpose

Verify that the application can be run by a new developer or user from scratch.

### When to Use

After changes to Dockerfiles, Docker Compose, environment configuration, or during Phase 9 verification.

### Procedure

1. Start from a clean state (remove existing containers and volumes if testing fresh install).
2. Verify prerequisites are documented.
3. Follow the documented setup steps exactly.
4. Test each service.
5. Test a basic user workflow.

### Checklist

- [ ] **`.env.example`**: Complete, accurate, no real secrets
- [ ] **`docker compose up --build`**: All containers start without errors
- [ ] **PostgreSQL**: Database initializes, migrations run
- [ ] **Backend**: `GET /api/health` returns 200
- [ ] **Frontend**: Loads in browser at the documented URL
- [ ] **Networking**: Frontend can reach backend API (proxy works)
- [ ] **Volumes**: Persistent data survives `docker compose down && docker compose up`
- [ ] **Health checks**: All services report healthy
- [ ] **Basic workflow**: Can make a prediction (if model exists) or upload a dataset
- [ ] **Logs**: No unexpected errors in container logs

### Expected Output

Report:
- Setup steps followed
- Each checklist item: Pass / Fail
- Any issues encountered and their resolution
- Time from `git clone` to working application

---

## Skill 11 — Documentation / Handover Review

### Purpose

Verify that another developer can understand and continue the project using the documentation alone.

### When to Use

During Phase 9 or when documentation is added/updated.

### Procedure

1. Approach the documentation as if you are a new developer joining the project.
2. Verify each area is covered and understandable.
3. Note gaps, unclear sections, or missing information.

### Checklist

- [ ] **What the project does**: Clear project description and purpose
- [ ] **Architecture**: System components and their relationships explained
- [ ] **Setup**: Prerequisites and installation steps documented
- [ ] **Docker**: `docker compose` usage explained
- [ ] **Environment variables**: All variables documented with purpose and examples
- [ ] **Dataset structure**: Feature schema and adapter pattern explained
- [ ] **ML pipeline**: Preprocessing, training, evaluation documented
- [ ] **Model training**: How to train and compare models
- [ ] **Model versioning**: Lifecycle and promotion workflow documented
- [ ] **Retraining**: How to add data and retrain documented
- [ ] **API reference**: All endpoints documented with request/response examples
- [ ] **Troubleshooting**: Common issues and solutions documented
- [ ] **Known limitations**: Documented honestly (including the academic prototype disclaimer)

### Expected Output

| Area | Status | Notes |
|------|--------|-------|
| Project overview | ✅ / ❌ / ⚠️ | Details |
| Architecture | ✅ / ❌ / ⚠️ | Details |
| Setup | ✅ / ❌ / ⚠️ | Details |
| ... | ... | ... |

**Final assessment:** Could a competent developer who did not build this project get it running and understand the major implementation decisions?
