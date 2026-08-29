# Dataset 2 — Target Provenance, Synthetic Data, Leakage & Scientific Validity Audit

**Project:** ML-Based Ground/Soil Risk Prediction and Monitoring System  
**Dataset:** Urban Road Collapse Risk Assessment Dataset  
**Audit Date:** 2026-08-28  
**Status:** Phase 0 — Dataset Research & Selection  
**Auditor:** Antigravity (AI Agent)

---

## 1. Executive Conclusion

Dataset 2 is a **Kaggle-hosted, almost certainly fully synthetic dataset** with no documented origin, no publication record, no disclosed target-generation formula, and no peer review. The target variable `collapse_risk_level` was almost certainly generated from a **weighted multi-variable scoring rule** applied to synthetically generated features, then binned into four perfectly balanced classes. Despite this, the dataset is **defensible for a university project** under clearly declared caveats, primarily because:

- The feature set is geotechnically plausible and internally consistent.
- No hard leakage (exact deterministic reconstruction) was detected.
- Soft leakage exists in several features but can be managed.
- The 15,000 rows support rigorous ML experiments.
- The domain framing (urban road collapse) is scientifically coherent.

**The project must explicitly declare its research context, synthetic nature, and limitations.**

---

## 2. Dataset Provenance

### 2.1 Verified Facts

| Property | Value |
|---|---|
| Dataset name | Urban Road Collapse Risk Assessment Dataset |
| Kaggle URL | https://www.kaggle.com/datasets/colabsss/urban-road-collapse-risk-assessment-dataset |
| Author | `colabsss` (Kaggle username) |
| Author dataset count | 499 public datasets |
| Last updated | ~July/August 2026 |
| License | CC0: Public Domain |
| File | `urban_road_collapse_risk_dataset.csv` |
| Size | 4.46 MB |
| Rows | 15,000 |
| Columns | 42 |
| Usability score | 5.88/10 |
| Notebooks | 0 |
| Discussions | 0 |
| Tags | None |

### 2.2 Strong Evidence / Inference

- **Almost certainly synthetic.** An account with 499 datasets and 0 discussions/notebooks on this dataset is a strong signal of programmatic generation. No real-world infrastructure dataset of this type would be produced without a source institution, publication, or collection protocol.
- **No linked publication, research institution, or government agency.** There is no link to a paper, engineering report, or data collection authority.
- **The segment_id format (`URC_00001` … `URC_15000`)** is sequentially generated — consistent with synthetic data.

### 2.3 Unknown / Unverifiable

- The actual Python/R script used to generate the dataset.
- Whether any real-world data was used as a statistical seed or calibration source.
- What specific formula generated `collapse_risk_level`.
- Whether the geographic coordinates (latitude 12.80–13.25°N, longitude 80.05–80.35°E) represent actual Chennai/Tamil Nadu road segments or are merely plausible coordinates used for cosmetic realism.

### 2.4 Coordinate Analysis

The coordinates fall **entirely within the Chennai Metropolitan Area (Tamil Nadu, India)**. This is a geographically realistic bounding box for an urban road study. However, whether these are actual road segment centroids or randomly sampled coordinates within this bounding box **cannot be verified** from the data alone.

---

## 3. Target Generation Analysis

### 3.1 What Was Found

The Kaggle page description states:

> "The categorical target column, collapse_risk_level, represents the estimated collapse susceptibility of each urban road segment using four categories: Low, Moderate, High, and Critical."

**No formula, no methodology, no code, and no explanation of how the labels were assigned is provided.**

### 3.2 Statistical Evidence Pointing to the Generation Mechanism

Based on data analysis:

**Class distribution is exactly uniform:** 3,750 records per class (exactly 25.00% each). The probability of this occurring in naturally observed data approaches zero. This is the single most decisive indicator of **intentional synthetic construction**.

**Feature means show monotonic ordering across classes** for multiple variables simultaneously:

| Feature | Low mean | Moderate mean | High mean | Critical mean |
|---|---|---|---|---|
| `road_age_years` | ~14 yrs | ~25 yrs | ~36 yrs | ~46 yrs |
| `surface_crack_density_pct` | 13.2% | 19.2% | 24.2% | 30.8% |
| `pavement_condition_index` | 81.1 | 71.2 | 63.3 | 53.4 |
| `soil_settlement_mm` | 31.0 mm | 45.2 mm | 54.6 mm | 69.2 mm |
| `land_subsidence_rate_mm_year` | 10.7 | 14.2 | 16.6 | 20.4 |
| `soil_bearing_capacity_kpa` | 432 | 394 | 369 | 329 |
| `groundwater_depth_m` | 18.4 | 16.0 | 14.5 | 12.2 |

Every single feature listed moves in the expected direction across all four risk classes. The consistency is implausibly clean for real-world data where noise, measurement error, and confounding factors exist.

**Top correlations with numeric target (Low=1, Moderate=2, High=3, Critical=4):**

- `road_age_years`: r=0.566
- `surface_crack_density_pct`: r=0.533
- `pavement_condition_index`: r=−0.539
- `land_subsidence_rate_mm_year`: r=0.492
- `soil_settlement_mm`: r=0.486
- `soil_bearing_capacity_kpa`: r=−0.472

None of these correlations are perfect (>0.95), which argues **against** a single-feature threshold and **toward** a multi-variable composite score.

### 3.3 Most Likely Target Generation Mechanism

**Strong evidence for:** Weighted multi-variable composite score → quantile-based 4-class binning

This is the standard method used by Kaggle dataset generators:

```
composite_score = w1*normalize(road_age) 
               + w2*normalize(surface_crack) 
               + w3*(1 - normalize(pavement_condition)) 
               + w4*normalize(soil_settlement)
               + w5*normalize(land_subsidence)
               + w6*normalize(waterlogging_duration)
               + ... (several more features)

collapse_risk_level = bin(composite_score, 4 equal-size bins) 
                    → Low / Moderate / High / Critical
```

This mechanism produces:
- Exactly balanced classes (confirmed: 3,750 each)
- Monotonic feature-class ordering (confirmed)
- Overlapping class ranges in each individual feature (confirmed — no hard boundaries)
- High F-statistics on ANOVA tests (confirmed — all suspicious features have p < 10⁻⁷⁰)

### 3.4 What Was NOT Found

- **No hard class boundaries** in any single feature. All classes overlap in all individual features, which rules out single-threshold binning.
- **No perfect decision-tree reconstruction.** A depth-5 decision tree achieves only 58.9% accuracy and depth-3 achieves 50.8%, confirming the target is NOT a simple deterministic function of any small feature subset.
- **No exact formula.** The target cannot be exactly reconstructed without knowing the original weights and which features were included in the composite.

---

## 4. Synthetic Data Assessment

### 4.1 Evidence Summary

| Evidence | Finding | Conclusion |
|---|---|---|
| Author profile | 499 datasets, 0 discussions | Strong synthetic indicator |
| Class distribution | Exactly 25.00% each class | Near-certain synthetic |
| `segment_id` | Sequential URC_00001…URC_15000 | Strong synthetic indicator |
| Missing values | 0 missing across all 42 columns | Atypically clean |
| Duplicate rows | 0 duplicates (feature space) | Consistent with generative model |
| Feature rounding | No suspicious rounding detected | Appears normally distributed |
| Decision tree depth 5 | 58.9% accuracy | No simple determinism |
| Feature monotonicity | Perfect ordering across all classes | Synthetic correlation structure |
| Documentation | Zero methodology description | Undocumented generation |

### 4.2 Derived/Engineered Features

Two composite features show strong evidence of being **derived from other features:**

- **`traffic_load_index`** (range 0.0–1.0): Correlation with `avg_daily_traffic × heavy_vehicle_pct / 1M` = **0.896**. Strongly suggests it was computed as a normalized product of traffic volume and heavy vehicle fraction.
- **`pipe_leakage_index`** (range 0.0–1.0): Correlation with `pipe_age_years` = **0.897**. Strongly suggests it was computed as a normalized function of pipe age.
- **`spatial_vulnerability_index`** (range 0.117–0.720): Highest correlation is with `building_density_per_km2` (r=0.681). Appears to be a composite of urban density and infrastructure deterioration indicators.

These engineered composites are internally consistent, but they are derived features masquerading as measured observations.

---

## 5. Leakage Analysis

### Leakage Classification Table

| Feature | Leakage Concern | Reason | Evidence | Recommendation |
|---|---|---|---|---|
| `spatial_vulnerability_index` | **HIGH** | Name explicitly says "vulnerability index" — almost certainly derived from same process that generated the target. Range limited (0.117–0.720), non-zero lower bound. Corr with target: 0.283. | Likely used in composite score generation | **Remove** |
| `historical_collapse_count` | **MEDIUM-HIGH** | Post-event data — only known after collapses occur. If this segment hasn't collapsed yet, this count is zero (not an input). Corr with target: 0.178. F=167, p<10⁻¹⁰⁶. | Would not be available at prediction time for new segments | **Remove** |
| `distance_to_previous_collapse_m` | **MEDIUM** | Similar to above — requires knowledge of prior collapses. Could be used as a spatial proximity feature if collapse history is maintained. Corr with target: negative (closer = higher risk). | Possible if collapse database exists, but requires existing data | **Keep with caution** (document assumption) |
| `soil_settlement_mm` | **MEDIUM** | Settlement is an effect, not just a cause. High settlement may be a consequence of risk rather than purely a predictor. Corr with target: 0.486 (strongest continuous predictor). | Mean monotonically ordered across all 4 classes (30.9→45.2→54.6→69.2). Likely in target formula. | **Keep with caution** (legitimate geotechnical predictor; settlement precedes collapse) |
| `land_subsidence_rate_mm_year` | **MEDIUM** | Similar to soil settlement — can be a consequence. However, land subsidence rate is a real observable measured by InSAR/leveling and precedes collapse. Corr: 0.492. | Likely in target formula | **Keep with caution** |
| `pipe_leakage_index` | **LOW-MEDIUM** | Derived from `pipe_age_years` (r=0.897). Represents estimated (not measured) leakage probability. Corr: 0.272. | Soft leakage — derived feature used in scoring | **Keep with caution** (acknowledge derivation) |
| `sewer_condition_index` | **LOW-MEDIUM** | Condition index — could be assessed or modeled. Inversely correlated with risk (higher condition = lower risk). | Legitimate infrastructure monitoring metric | **Keep** |
| `flood_frequency_per_year` | **LOW** | Historical frequency — a legitimate observable predictor. Corr: 0.270. | Standard environmental predictor | **Keep** |
| `waterlogging_duration_hr` | **LOW** | Measurable environmental condition. Corr: 0.270. | Standard environmental predictor | **Keep** |
| `traffic_load_index` | **LOW** | Derived from `avg_daily_traffic × heavy_vehicle_pct`. Corr: 0.220. | Redundant if raw traffic features included | **Remove or keep but not alongside raw components** |

### 5.1 Critical Note on `spatial_vulnerability_index`

This feature has the word "vulnerability" in its name and a suspicious non-zero lower bound (0.117). Its description says "overall spatial vulnerability of the road segment based on surrounding risk conditions." This is almost certainly a **pre-computed risk proxy** that was used in the target's generation formula. It is not an independent predictor — it is a **leaky summary statistic**.

**Recommendation: Remove from the feature set.**

---

## 6. Class Distribution Analysis

### 6.1 Exact Distribution

| Class | Count | Percentage |
|---|---|---|
| Low | 3,750 | 25.000% |
| Moderate | 3,750 | 25.000% |
| High | 3,750 | 25.000% |
| Critical | 3,750 | 25.000% |

### 6.2 Assessment

**This distribution is not naturally observed.** In any real urban infrastructure study:
- High-risk segments are rare events.
- A perfectly balanced 25/25/25/25 distribution does not occur in nature.
- Real road condition surveys typically show heavily left-skewed or bimodal distributions.

### 6.3 Implications for ML Evaluation

**Positive implications:**
- No class imbalance problem — all algorithms will train on equal class representation.
- Cross-validation folds will be balanced without special treatment.
- All four classes will have adequate representation for evaluation.
- Weighted F1-score and macro F1-score converge — simpler evaluation.

**Negative implications:**
- A random baseline achieves **25% accuracy** without any information.
- Any model achieving >70% accuracy on this dataset may be exploiting the synthetic correlation structure, not learning generalizable real-world patterns.
- The perfect balance **inflates** apparent model performance relative to real-world deployment, where class frequencies are unknown and likely imbalanced.
- Results are not transferable to real-world deployment without significant recalibration.

**For the university project:** The balanced distribution is pedagogically convenient but must be disclosed. Evaluation metrics must be interpreted in the context of a balanced 4-class baseline of 25%.

---

## 7. Feature/Target Relationship Findings

### 7.1 Decision Tree Analysis Results

| Tree Configuration | Train Accuracy |
|---|---|
| Full depth, all features | **100.00%** (trivial overfitting) |
| Depth 1, all features | 41.9% (primary split: `road_age_years ≤ 29.5`) |
| Depth 2, all features | 49.2% |
| Depth 3, all features | 50.8% |
| Depth 5, all features | 58.9% |
| Full depth, without top-3 suspicious | **100.00%** (still overfits) |
| Depth 5, without top-3 suspicious | 58.9% (identical — suspicious features not primary drivers) |

### 7.2 Key Interpretation

- The full tree perfectly memorizes the training data — expected behavior for synthetic data without noise that would prevent this.
- The fact that depth-5 achieves only ~59% accuracy confirms that the target is a **soft multi-variable composite**, not a simple rule-based label.
- Removing the 3 most suspicious features (`spatial_vulnerability_index`, `historical_collapse_count`, `distance_to_previous_collapse_m`) does **not change the depth-5 accuracy**, which means these features are not the primary predictors and their removal does not meaningfully impair model learning.

### 7.3 Feature Importance Rankings (Top-10, Full Tree)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `road_age_years` | 0.123 |
| 2 | `soil_bearing_capacity_kpa` | 0.076 |
| 3 | `soil_settlement_mm` | 0.068 |
| 4 | `land_subsidence_rate_mm_year` | 0.063 |
| 5 | `waterlogging_duration_hr` | 0.059 |
| 6 | `surface_deformation_mm` | 0.047 |
| 7 | `pipe_leakage_index` | 0.046 |
| 8 | `surface_crack_density_pct` | 0.032 |
| 9 | `soil_moisture_pct` | 0.028 |
| 10 | `traffic_load_index` | 0.027 |

**Note:** `spatial_vulnerability_index` ranks only 13th despite being a composite vulnerability indicator. This is consistent with it having a large range overlap across classes (0.117–0.720 for all classes combined).

---

## 8. Fuzzy Logic Assessment

### 8.1 Is There a Fuzzy Logic Case Here?

The question is: would fuzzy logic add scientific value over raw ML probabilities?

**Finding: The dataset's labels almost certainly already came from a scoring/binning process.** The mechanism was:
1. Generate feature values.
2. Compute a composite risk score.
3. Bin into 4 quantile-balanced classes.

This is a **crisp rule-based** (not fuzzy) quantile-based binning, not fuzzy membership.

### 8.2 Could Fuzzy Logic Define Defensible Categories?

In principle, yes — fuzzy logic could define membership functions such as:
- "Low risk" = composite score ∈ [0, 0.4] with full membership; [0.4, 0.6] with decreasing membership
- "Moderate risk" = composite score ∈ [0.3, 0.7] with peak at 0.5
- etc.

However, to build defensible fuzzy membership functions, one would need:
1. **Domain expert validation** of the boundary values.
2. **Real data** to calibrate where transitions actually occur.
3. Access to the underlying composite score distribution.

None of these are available for this dataset.

### 8.3 Would Fuzzy Logic Reproduce Existing Labels?

Almost certainly yes. Since the labels were generated by binning a composite score, any well-calibrated fuzzy system built on the same features would approximate the same label assignment. Adding fuzzy logic would therefore not add independent scientific value — it would reproduce the dataset's existing mechanism.

### 8.4 Are ML Probabilities + Calibration More Appropriate?

**Yes.** ML model output probabilities (from `predict_proba`) with Platt scaling or isotonic regression calibration are:
- More statistically rigorous.
- Directly interpretable as confidence.
- Consistent with the existing label structure.
- Computationally simpler.
- Standard in the ML literature.

### 8.5 Fuzzy Logic Recommendation

**Recommendation: DEFER or COMPLETELY REJECT for this dataset.**

Rationale:
- The labels are crisp bins over an unobservable composite score.
- No real-world calibration data for membership functions exists.
- Adding fuzzy logic without a principled basis would be scientifically indefensible.
- ML calibrated probabilities are a superior and more defensible alternative.
- Fuzzy logic could be noted as a future research direction but should not be part of the MVP.

---

## 9. 3-Class vs 4-Class Recommendation

### 9.1 Current State

- **PRD requirement:** 3 classes (Low, Moderate, High)
- **Dataset:** 4 classes (Low, Moderate, High, Critical)

### 9.2 Analysis

**Against merging Critical → High:**
- Critical and High have statistically distinguishable feature distributions. ANOVA F-statistics are highly significant for all key features.
- Information loss is quantifiable: the merged class would contain 7,500 records (50%) representing two genuinely different risk profiles.
- In engineering practice, "Critical" carries substantially different response urgency than "High." Merging them discards this distinction without domain justification.

**For maintaining 4 classes:**
- All 4 classes are equally represented (3,750 each), making 4-class classification no harder than 3-class from an ML perspective.
- `Critical` is scientifically meaningful: it represents segments where collapse is imminent or highly probable.
- The project's description as a "decision-support prototype" is better served by 4 classes if the risk levels map to concrete response categories (routine maintenance / planned repair / urgent inspection / emergency intervention).
- In road safety literature, 4-tier risk classification (Low/Moderate/High/Critical or equivalent) is standard.

**For updating the PRD:**
- The original "Low/Moderate/High" specification was written before the dataset was selected.
- Now that the dataset contains 4 classes with engineered separation, the PRD should be updated to match the data.

### 9.3 Recommendation

**Redefine the project to 4 classes.** Update the PRD target definition to:
- Low Risk
- Moderate Risk
- High Risk
- Critical Risk

Do NOT merge Critical into High. This would be a scientifically unsupportable simplification that discards information deliberately included in the dataset and reduces the system's practical utility.

---

## 10. Scientifically Defensible Project Scope

### 10.1 Scope Comparison

| Scope Wording | Assessment |
|---|---|
| A. "Ground/Soil Risk Prediction" | **Too broad.** The dataset is about urban road collapse, not general soil/ground risk. It does not cover slope failure, erosion, soil contamination, or agricultural soil quality. |
| B. "Ground Stability Risk Prediction" | **Partially supported.** Soil stability features exist (soil_bearing_capacity, soil_settlement, groundwater_depth), but the target is road collapse risk specifically, not ground stability in general. Misleading scope. |
| C. "Urban Road Collapse Risk Prediction" | **Directly supported.** This is exactly what the dataset was designed to represent. The features, target, and segment structure are all road-collapse focused. |
| D. "Ground-Related Urban Road Collapse Risk Prediction" | **Also supported.** A more precise variant of C that emphasizes the soil/ground factors, distinguishing from purely structural failure. Appropriate if the geotechnical angle is to be emphasized. |

### 10.2 Recommendation

**Use scope C: "Urban Road Collapse Risk Prediction"** as the primary framing.

This is the most accurate, verifiable, and intellectually honest scope. It does not overclaim and is directly supported by the dataset.

If the university brief requires a "soil/ground" framing (as suggested by the project title "ML-Based Ground/Soil Risk Prediction and Monitoring System"), the correct disclosure is:

> "This system predicts urban road collapse risk using ground and soil condition indicators as primary input features. The risk assessment is specific to road segment collapse susceptibility and does not represent general ground stability or soil quality assessment."

---

## 11. Recommended Feature Removals and Cautions

### Remove (leakage / derivation concerns)

| Feature | Reason |
|---|---|
| `spatial_vulnerability_index` | Almost certainly used in target generation; not an independent predictor |
| `historical_collapse_count` | Post-event data; not available for new segments without collapse history |
| `traffic_load_index` | Derived composite of `avg_daily_traffic × heavy_vehicle_pct`; redundant |

### Keep with Documented Caution

| Feature | Caution |
|---|---|
| `soil_settlement_mm` | May be an effect/consequence of risk, not purely a precursor; document assumption |
| `land_subsidence_rate_mm_year` | Measurable by satellite/leveling but may correlate with collapse onset; document assumption |
| `pipe_leakage_index` | Computed from `pipe_age_years`; not an independent measurement; document derivation |
| `distance_to_previous_collapse_m` | Requires existing collapse database; document that this assumes historical records exist |

### Keep (no significant concerns)

All remaining features represent physically plausible, pre-event, independently measurable road segment characteristics consistent with standard pavement management and geotechnical assessment practice.

---

## 12. Overall Dataset Suitability Score

**Score: 6.5 / 10**

| Criterion | Score | Notes |
|---|---|---|
| Sample size | 9/10 | 15,000 rows; adequate for 70/15/15 split |
| Feature richness | 8/10 | 41 non-target features; good domain coverage |
| Feature plausibility | 7/10 | All features are physically meaningful; but several are derived |
| Target reliability | 4/10 | Almost certainly synthetically binned; no documented methodology |
| Provenance | 2/10 | No institution, no publication, no data collection protocol |
| Leakage risk | 5/10 | Several moderate-risk features; manageable with removals |
| Class quality | 5/10 | Perfectly balanced = almost certainly artificial |
| Geographic specificity | 5/10 | Chennai coordinates; plausible but unverifiable |
| Documentation quality | 1/10 | Zero methodology; zero code; zero peer review |
| Academic defensibility | 5/10 | Defensible *only* with honest disclosure |

---

## 13. Confidence Level

**Confidence in the recommendation: MEDIUM-HIGH (7/10)**

High confidence that:
- The dataset is synthetic.
- The target was generated by composite scoring and binning.
- The feature set is appropriate for the stated domain.
- The recommended feature removals reduce but do not eliminate leakage risk.

Lower confidence in:
- The exact target generation formula (unverifiable without the generation code).
- Whether the geographic coordinates correspond to real road segments.
- The physical calibration of feature ranges against real-world data.

---

## 14. Remaining Unknowns That Must Be Acknowledged

The following cannot be determined from the available data and must be disclosed in the project:

1. **Exact target generation formula.** The precise weights and formula used to generate `collapse_risk_level` are unknown. The label assignment mechanism is inferred but not confirmed.
2. **Real-world calibration.** Whether the feature value ranges are physically calibrated against real Chennai road data or are arbitrarily chosen is unknown.
3. **Whether `spatial_vulnerability_index` was used in the target formula.** Strongly inferred but not proven.
4. **Geographic ground truth.** The coordinates cannot be verified against real urban road segments.
5. **Whether any real data seeded the synthetic generation.** The author may have used real statistical parameters even if individual records are synthetic.
6. **The role of `soil_settlement_mm` and `land_subsidence_rate_mm_year`** as predictors vs. consequences — this is a domain question that cannot be resolved from the data alone.

---

## Final Recommendation

> **SELECT Dataset 2 WITH MAJOR CAVEATS**

### Justification

Dataset 2 is the **most suitable available candidate** for this university project, but it carries significant scientific baggage that must be honestly acknowledged. The caveats are:

| Caveat | Required Action |
|---|---|
| Dataset is almost certainly fully synthetic | Must be declared in all project documentation, the README, and the application disclaimer |
| Target generation methodology is unknown | Cannot claim the ML model learns from real-world collapse patterns; frame as "synthetic-data decision-support prototype" |
| Three features should be removed | `spatial_vulnerability_index`, `historical_collapse_count`, `traffic_load_index` |
| Project scope must be narrowed | "Urban Road Collapse Risk Prediction" not "Ground/Soil Risk Prediction" |
| PRD must be updated | 4 classes (Low/Moderate/High/Critical), not 3 |
| Perfect class balance is artificial | Evaluation results must be interpreted against a 25% random baseline |
| ML evaluation must be honest | No claims of real-world accuracy without real-world validation data |

### Minimum Disclosure Requirements

All project artifacts (README, application UI, academic report) must include a statement equivalent to:

> "The dataset used in this system is a publicly available synthetic dataset created for ML research purposes. It does not represent actual engineering assessments of real road segments. The risk predictions produced by this system are not validated against real-world collapse events and must not be used for actual infrastructure safety decisions."

---

*This audit was performed on the raw dataset file and Kaggle metadata. No fabricated results were included. All statistics are derived from actual dataset analysis. The target-generation mechanism is inferred from statistical evidence and is not confirmed by documentation.*
