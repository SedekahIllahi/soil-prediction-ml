# Product Requirements Document (PRD)

## ML-Based Ground/Soil Risk Prediction and Monitoring System

| Field              | Value                                    |
| ------------------ | ---------------------------------------- |
| **Status**         | Active — Phase 0 Complete                |
| **Last Updated**   | 2026-08-28                               |
| **Project Type**   | Academic / Technical Decision-Support Prototype |

---

## 1. Overview

### 1.1 What Are We Building?

A machine-learning system that accepts urban road segment measurements (pavement condition, soil properties, hydrological data, and infrastructure state) and classifies the segment's collapse susceptibility into one of four risk categories:

- **Low Risk**
- **Moderate Risk**
- **High Risk**
- **Critical Risk**

The system includes a web-based interface for entering data, viewing predictions, monitoring historical results, managing datasets, and managing ML model versions.

**Project scope:** Urban Road Collapse Risk Prediction. Input features are road segment condition indicators grounded in geotechnical and infrastructure measurements. The system does not claim to be a general ground-stability or soil-quality assessment tool.

### 1.2 Why Are We Building It?

To demonstrate how supervised machine learning can be applied to urban infrastructure condition data to support road collapse risk decisions. The project serves as an academic prototype that integrates data management, ML training, model versioning, and a usable front-end into a single deployable application.

### 1.3 What This System Is NOT

> [!CAUTION]
> This system is an **academic/technical prototype** built on a **synthetic dataset**. It must **not** be presented as:
>
> - A replacement for professional geotechnical or structural engineering assessment.
> - A guaranteed safety system.
> - A tool for making automatic decisions that replace professional engineering judgment.
> - A system validated against real-world road collapse events.
>
> The dataset used is publicly available synthetic data (Kaggle CC0). Risk predictions produced by this system have not been validated against real infrastructure observations.

---

## 2. Target Users

### 2.1 Engineering / Technical Users

Users who possess soil/ground measurements and want to obtain a risk assessment.

**Capabilities:**

- Enter soil/ground parameters.
- Request a risk prediction.
- View the predicted risk category (and prediction confidence, if available).
- View historical prediction and monitoring information.

### 2.2 Project Administrator / Operator

Users responsible for maintaining the dataset and ML models.

**Capabilities:**

- Upload new labeled datasets.
- Validate and preview datasets before integration.
- Retrain ML models on updated data.
- Compare model performance across algorithms and versions.
- View model version history.
- Promote a newly trained model to production.
- Roll back to a previous model version.

### 2.3 Usability Constraint

The system should remain usable by people who are **not** software or IT specialists. Avoid unnecessary technical complexity in the UI.

---

## 3. MVP Functional Requirements

### 3.1 Risk Prediction

| ID   | Requirement                                                                                                | Priority |
| ---- | ---------------------------------------------------------------------------------------------------------- | -------- |
| F-01 | User can enter relevant road segment parameters via a form.                                                 | Must     |
| F-02 | System returns a prediction of **Low Risk**, **Moderate Risk**, **High Risk**, or **Critical Risk**.        | Must     |
| F-03 | System displays prediction probabilities/confidence for all four classes where appropriate.                 | Should   |

> [!NOTE]
> The exact input parameters are defined in `docs/dataset_specification.md` (Phase 0 output). The canonical feature set contains 34 model features. Input form fields must match this schema.

---

### 3.2 Monitoring Dashboard

| ID   | Requirement                                                                             | Priority |
| ---- | --------------------------------------------------------------------------------------- | -------- |
| F-04 | Display the total number of observations/predictions made.                              | Must     |
| F-05 | Display the distribution of predicted risk categories.                                  | Must     |
| F-06 | Display a list of recent predictions.                                                   | Must     |
| F-07 | Provide access to historical prediction data.                                           | Must     |
| F-08 | Display geographic/map visualization if the dataset contains location data.             | Should   |

The dashboard should prioritize **useful information** over visual complexity.

---

### 3.3 Dataset Management

| ID   | Requirement                                                                       | Priority |
| ---- | --------------------------------------------------------------------------------- | -------- |
| F-09 | Authorized users can upload a new labeled dataset.                                | Must     |
| F-10 | System validates the uploaded dataset's structure against expected schema.         | Must     |
| F-11 | User can preview uploaded data before integration.                                | Must     |
| F-12 | Valid data can be integrated into the training dataset.                            | Must     |
| F-13 | The ML pipeline is not permanently tied to one specific dataset.                  | Must     |

---

### 3.4 Model Comparison

| ID   | Requirement                                                                          | Priority |
| ---- | ------------------------------------------------------------------------------------ | -------- |
| F-14 | System trains multiple ML models on the same dataset.                                | Must     |
| F-15 | System evaluates and compares model performance using defined metrics.                | Must     |
| F-16 | Final model selection is based on actual evaluation results, not assumptions.         | Must     |
| F-17 | System must not contain fabricated or hard-coded performance results.                 | Must     |

**Candidate algorithms:**

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- SVM

The final set of algorithms may be adjusted based on dataset characteristics and evaluation outcomes.

---

### 3.5 Model Retraining

| ID   | Requirement                                                                     | Priority |
| ---- | ------------------------------------------------------------------------------- | -------- |
| F-18 | System supports retraining when new labeled data is integrated.                 | Must     |
| F-19 | Retraining follows a defined pipeline (validate → integrate → preprocess → train → evaluate → compare → version). | Must |
| F-20 | A newly trained model does not blindly overwrite the production model.          | Must     |

**Retraining workflow:**

```
Existing Dataset + New Labeled Dataset
        ↓
     Validate
        ↓
     Integrate
        ↓
    Preprocess
        ↓
   Train Models
        ↓
     Evaluate
        ↓
     Compare
        ↓
 New Model Version
        ↓
  Promote / Reject
```

> [!NOTE]
> For the MVP, conventional batch retraining is sufficient. Online/incremental learning is out of scope.

---

### 3.6 Model Versioning

| ID   | Requirement                                                                       | Priority |
| ---- | --------------------------------------------------------------------------------- | -------- |
| F-21 | Each trained model is stored as a versioned artifact.                             | Must     |
| F-22 | Model metadata includes: version, training dataset/version, training timestamp, algorithm, and evaluation metrics. | Must |
| F-23 | Users can view the list of model versions and their metadata.                     | Must     |
| F-24 | Users can promote a model version to production.                                  | Must     |
| F-25 | Users can roll back to a previous model version.                                  | Must     |

---

## 4. Dataset

### 4.1 Selected Dataset

> [!IMPORTANT]
> **Phase 0 complete.** Dataset 2 has been provisionally selected with major caveats.
> Full evaluation is documented in `docs/dataset_specification.md`.

**Selected:** Urban Road Collapse Risk Assessment Dataset

| Property | Value |
| --- | --- |
| Source | Kaggle — user `colabsss` |
| License | CC0: Public Domain |
| File | `storage/datasets/urban_road_collapse_risk_dataset.csv` |
| Rows | 15,000 |
| Features (raw) | 41 |
| Model features (after audit) | 34 |
| Target | `collapse_risk_level` — 4 classes |
| Nature | **Synthetic** — no real-world collection protocol documented |
| Suitability score | 6.5 / 10 |

> [!WARNING]
> This dataset is almost certainly fully synthetic with no documented target-generation methodology.
> All project documentation, the application UI, and the academic report must declare this explicitly.
> Risk predictions are not validated against real collapse events.

### 4.2 Removed Features

Four columns from the raw dataset are **excluded** from the ML pipeline:

| Feature | Reason |
| --- | --- |
| `spatial_vulnerability_index` | Target leakage — likely used in label generation |
| `historical_collapse_count` | Temporal leakage — unavailable for uncollapsed segments |
| `traffic_load_index` | Derived from `avg_daily_traffic × heavy_vehicle_pct` (r=0.896) |
| `pipe_leakage_index` | Derived from `pipe_age_years` (r=0.897, R²=0.804) |

### 4.3 Features Retained with Caution

| Feature | Caution |
| --- | --- |
| `pavement_condition_index` | Composite of distress indicators; document as derived KPI |
| `soil_settlement_mm` | Condition indicator; may be consequence not only precursor |
| `land_subsidence_rate_mm_year` | Requires satellite/leveling data; may overlap with collapse onset |
| `sewer_condition_index` | Strongly correlated with `pipe_age_years` (r=−0.939) |
| `distance_to_previous_collapse_m` | Requires collapse database; weak signal (r=−0.078 with target) |

### 4.4 Dataset Assumptions

- The dataset is treated as **synthetic** throughout the project.
- Risk labels were almost certainly generated via a weighted composite score binned into 4 equal-size classes.
- The geographic coordinates (Chennai Metro area) are cosmetic and are excluded from the ML model.
- Class distribution is perfectly balanced (25% per class) — artificially so.
- The 70/15/15 split will yield 10,500 / 2,250 / 2,250 rows per partition.

---

## 5. Machine Learning

### 5.1 Task

Four-class classification:

```
Input road segment features → ML Classifier → Low / Moderate / High / Critical Risk
```

Target label: `collapse_risk_level` with classes `Low`, `Moderate`, `High`, `Critical` (string, sentence-case).

### 5.2 Data Split

| Set        | Proportion | Purpose                         |
| ---------- | ---------- | ------------------------------- |
| Training   | 70%        | Model fitting                   |
| Validation | 15%        | Hyperparameter tuning / selection |
| Test       | 15%        | Final, isolated evaluation      |

The split must be **reproducible** (fixed random seed). The test set must remain **isolated** until final model evaluation.

### 5.3 Preprocessing

- Preprocessing should be **modular**, so that different compatible datasets can be introduced without rewriting the ML pipeline.
- **Data leakage** must be avoided, particularly during feature scaling, encoding, and transformation. Fit transformations on training data only.

### 5.4 Evaluation Metrics

| Metric           | Purpose                                    |
| ---------------- | ------------------------------------------ |
| Accuracy         | Overall correctness                        |
| Precision        | Per-class positive predictive value        |
| Recall           | Per-class sensitivity                      |
| F1-Score         | Harmonic mean of precision and recall      |
| Confusion Matrix | Error distribution across classes          |
| Per-class detail | Individual class performance breakdown     |

> [!IMPORTANT]
> Because this is a risk-related system, performance on the **High Risk** and **Critical Risk** classes should receive particular attention during evaluation. A model that systematically misses Critical segments is dangerous regardless of overall accuracy. Per-class recall for High and Critical are secondary evaluation criteria alongside weighted F1-score.

### 5.5 Integrity Rules

- Do not claim scientific reliability without appropriate data and validation.
- Do not fabricate or hard-code model performance results.
- Do not invent risk thresholds or engineering definitions.

---

## 6. User Experience

### 6.1 Prediction Flow

```
Open application → Enter/view data → Get prediction → View result
```

### 6.2 Maintenance / Retraining Flow

```
Open application → Upload dataset → Validate → Train models → Compare results → Select model
```

### 6.3 Design Principles

- Prioritize clarity and usability over visual complexity.
- Design for users who may not be software developers.
- Avoid exposing unnecessary technical details in the UI.

---

## 7. Technology Stack

| Layer            | Technology             |
| ---------------- | ---------------------- |
| Frontend         | Vue 3, TypeScript, Vite |
| Backend / API    | Python, FastAPI        |
| Machine Learning | Python, scikit-learn, XGBoost |
| Database         | PostgreSQL             |
| Model Storage    | Joblib (serialized model files) |
| Testing          | Pytest (backend), Vitest (frontend) |

> [!NOTE]
> These are the current technology direction. Specifics may be refined during architecture planning.

---

## 8. Deployment

### 8.1 MVP Deployment

The application must be easy to run locally using Docker and Docker Compose.

**Target experience:**

```
Install Docker Desktop → docker compose up → Open application locally
```

Users should **not** need to manually install Python, Node.js, PostgreSQL, or ML dependencies.

### 8.2 Future Deployment

The application should be structured so that it **could** be deployed to a server in the future. Internet hosting is not required for the MVP.

---

## 9. Documentation & Handover

The project must include technical documentation covering:

- System architecture
- Setup and deployment instructions
- Dataset structure and requirements
- Data preprocessing pipeline
- ML training process
- Model evaluation methodology
- Retraining workflow
- Model versioning scheme
- Important implementation decisions and trade-offs

**Purpose:** Enable a future developer to understand and continue the project without reverse-engineering the codebase.

---

## 10. Out of Scope (MVP)

The following are explicitly **out of scope** unless requirements change:

| Item                                         | Reason                          |
| -------------------------------------------- | ------------------------------- |
| IoT sensor hardware integration              | Hardware dependency             |
| Real-time physical sensor networks           | Hardware dependency             |
| Online/incremental machine learning          | Complexity beyond MVP           |
| Deep-learning models (unless dataset-justified) | Complexity beyond MVP        |
| Complex distributed infrastructure           | Complexity beyond MVP           |
| Kubernetes                                   | Complexity beyond MVP           |
| Microservice architecture                    | Complexity beyond MVP           |
| Automatic cloud deployment                   | Not required for academic scope |
| Professional engineering certification       | Out of project scope            |
| Guaranteed real-world safety predictions     | Cannot be claimed               |
| Automatic decisions replacing engineering judgment | Must not be implemented    |

The MVP should remain a **manageable academic project**.

---

## 11. Success Criteria

The MVP is considered successful if it can:

| #  | Criterion                                                                                      |
| -- | ---------------------------------------------------------------------------------------------- |
| 1  | Load and validate the selected dataset against the canonical schema in `docs/dataset_specification.md`. |
| 2  | Preprocess the data correctly (without data leakage; fit on training set only).                |
| 3  | Train multiple classification models.                                                          |
| 4  | Evaluate and compare those models using defined metrics including per-class recall for High and Critical. |
| 5  | Select a model according to weighted F1-score (primary) and High/Critical recall (secondary).  |
| 6  | Use the selected model to predict Low / Moderate / High / Critical Risk.                       |
| 7  | Display predictions through a usable UI including risk level and per-class probabilities.      |
| 8  | Provide basic monitoring and prediction history.                                               |
| 9  | Accept new labeled data for retraining.                                                        |
| 10 | Maintain model versions with metadata.                                                         |
| 11 | Run locally through Docker with minimal setup.                                                 |
| 12 | Provide sufficient documentation for another developer to continue work.                       |
| 13 | Display a synthetic-data disclaimer prominently in the application UI.                         |

---

## 12. Open Questions

| #  | Question                                                                                            | Status |
| -- | --------------------------------------------------------------------------------------------------- | ------ |
| Q1 | Which dataset will be selected for the initial implementation?                                       | **Resolved** — Dataset 2 (Urban Road Collapse Risk, Kaggle CC0). See `docs/dataset_specification.md`. |
| Q2 | What are the exact input features for prediction?                                                   | **Resolved** — 34 model features defined in `docs/dataset_specification.md` §Part 9. |
| Q3 | How should risk categories be derived?                                                              | **Resolved** — Dataset provides `collapse_risk_level` directly: Low / Moderate / High / Critical (4 classes). |
| Q4 | What minimum number of samples is acceptable for the selected dataset?                               | **Resolved** — 15,000 rows; 70/15/15 split yields 10,500 / 2,250 / 2,250. Sufficient. |
| Q5 | Should the system support multiple concurrent datasets or only one active dataset at a time?         | Open — current architecture assumes one active dataset version. |
| Q6 | What level of user authentication/authorization is required for the MVP?                             | Open — currently planned without auth (ARCHITECTURE.md AD-7). |
| Q7 | Are there specific regulatory or academic standards the documentation must satisfy?                  | Open |

---

## 13. Assumptions

These are current assumptions that may need validation:

1. Dataset 2 (synthetic, Kaggle) is sufficiently representative for demonstrating the ML pipeline in an academic context, provided all synthetic caveats are disclosed.
2. The 4-class label (`Low`, `Moderate`, `High`, `Critical`) in the dataset, though synthetically generated, provides meaningful class separation for training and evaluation.
3. Conventional ML algorithms (not deep learning) will be sufficient for this tabular dataset.
4. A single PostgreSQL database is adequate for the MVP's data storage needs.
5. Joblib serialization is sufficient for model artifact storage at MVP scale.
6. The 70/15/15 random stratified split yields 10,500 / 2,250 / 2,250 rows — adequate for training and evaluation of all candidate algorithms.
7. The 34-feature canonical schema defined in `docs/dataset_specification.md` is stable for Phase 1+ implementation.

---

## 14. Future Possibilities

These items are **not** part of the MVP but may be considered in later phases:

- Real-time data ingestion from sensors or external APIs.
- Advanced visualization (3D terrain, time-series trends).
- User role management and access control beyond basic authorization.
- Automated retraining triggers (e.g., on data drift detection).
- Model explainability features (SHAP, LIME).
- Export/reporting functionality.
- Multi-language UI support.
- Cloud deployment pipeline.

---

*This document is the product "north star" for the project. All architecture, design, and implementation decisions should trace back to the requirements defined here.*
