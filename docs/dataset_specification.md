# Canonical Dataset Specification — Dataset 2
## Urban Road Collapse Risk Assessment Dataset

**Version:** 1.0  
**Status:** AUTHORITATIVE — Phase 0 Output  
**Date:** 2026-08-28  
**Source file:** `storage/datasets/urban_road_collapse_risk_dataset.csv`  
**Produced by:** Phase 0 — Dataset Research & Selection  

> This document is the single source of truth for the ML pipeline feature schema.
> No production preprocessing, model training, or application code may contradict decisions made here without explicit revision and documented justification.

---

## PART 1 — Canonical Target Definition

### Verified Target Values (from raw CSV)

```
collapse_risk_level:
  Low        → 3,750 rows (25.000%)
  Moderate   → 3,750 rows (25.000%)
  High       → 3,750 rows (25.000%)
  Critical   → 3,750 rows (25.000%)
  Total      → 15,000 rows
  Missing    → 0
  Dtype      → string (object)
```

**Exact string values verified:** `"Low"`, `"Moderate"`, `"High"`, `"Critical"` — sentence-case, no leading/trailing spaces, no alternatives observed.

No unexpected or malformed target values exist.

### Target Encoding

The natural ordinal interpretation is:

| Label | Ordinal Value | Interpretation |
|---|---|---|
| Low | 1 | Low collapse risk |
| Moderate | 2 | Moderate collapse risk |
| High | 3 | High collapse risk |
| Critical | 4 | Critical collapse risk (imminent/high probability) |

The target will be treated as **nominal 4-class classification** by default. The ordinal structure is used only for correlation analysis; the model will not impose ordinality.

### Why 4 Classes Are Retained (Do Not Merge Critical → High)

1. **Statistical distinguishability.** All key features have statistically significant ANOVA F-statistics separating all four classes (p < 10⁻⁷⁰ for most). Critical and High are genuinely separated in feature space.
2. **Information preservation.** Merging 3,750 Critical records into 3,750 High records produces a 7,500-record High class (50%) with genuinely heterogeneous feature profiles. This discards engineered structure without domain justification.
3. **Engineering meaning.** In road safety practice, a 4-tier classification (Low/Moderate/High/Critical) maps to concrete response protocols: routine monitoring / planned inspection / urgent repair / emergency intervention. The 4-class target preserves this utility.
4. **No external justification for merging.** The original 3-class PRD requirement was written before dataset selection. It must be revised to match the data.

---

## PART 2 — Complete 42-Column Classification Table

Decision codes:
- **TARGET** — the label to predict
- **ID** — identifier/metadata, not a model feature
- **METADATA** — geographic or administrative, kept for visualization but excluded from ML
- **KEEP** — include as a model feature without reservation
- **KEEP-CAUTION** — include with documented assumption or caveats
- **REMOVE-LEAKAGE** — information that was used to generate the target or is unavailable at prediction time
- **REMOVE-DERIVED** — redundant computed feature derivable from kept features
- **REMOVE-UNSUITABLE** — provides no independent predictive value

| # | Column | Role | Decision | Reason | Available at Prediction Time? | Derived From Other Features? |
|---|---|---|---|---|---|---|
| 1 | `segment_id` | Identifier | **ID** | Sequential integer suffix (URC_00001…15000), zero correlation with target (r=0.007). No predictive value. | N/A | No |
| 2 | `latitude` | Geographic | **METADATA** | Zero correlation with target (r=−0.002). Purely cosmetic — encodes the synthetic generation bounding box, not actual risk signal. Keep for map visualization only. | Yes | No |
| 3 | `longitude` | Geographic | **METADATA** | Zero correlation with target (r=0.004). Same reason as latitude. Keep for map visualization only. | Yes | No |
| 4 | `elevation_m` | Physical | **KEEP** | Topographic feature; drainage and flood risk depend on elevation. Range 1–69.7m. No quality issues. | Yes | No |
| 5 | `road_length_m` | Infrastructure | **KEEP** | Longer segments expose more soil area; relevant to risk scale. Range 50–2000m. No quality issues. | Yes | No |
| 6 | `distance_to_water_body_m` | Environmental | **KEEP** | Proximity to water affects drainage and saturation. Range 25–5000m. No quality issues. | Yes | No |
| 7 | `road_age_years` | Infrastructure | **KEEP** | Strongest single predictor (importance 0.123). Directly reflects pavement degradation timeline. Range 1–60 years (integer). | Yes | No |
| 8 | `pavement_thickness_cm` | Infrastructure | **KEEP** | Structural property. Range 10–60cm. Measurable at survey time. | Yes | No |
| 9 | `surface_crack_density_pct` | Condition | **KEEP** | Key pavement distress indicator. Corr with target 0.533. Range 0–60%. Observable at inspection. | Yes | No |
| 10 | `pavement_condition_index` | Condition | **KEEP-CAUTION** | PCI is a standard pavement management metric. Strong corr with target (−0.539) and with `road_age_years` (r=−0.895). **Caution: computed from crack density, rut depth, and deformation — partially derived.** Keep because PCI is a first-class engineering KPI; document the redundancy. | Yes | Partially — computed from other features |
| 11 | `rut_depth_mm` | Condition | **KEEP** | Independent pavement distress measurement. Range 0–46.6mm. Corr with `surface_deformation_mm` = 0.894 (see Part 6). Both have distinct physical meanings; keep both. | Yes | No |
| 12 | `surface_deformation_mm` | Condition | **KEEP** | Vertical deformation of road surface; distinct from rutting (lateral deformation). Corr with target 0.548 — strongest continuous predictor after removing soil features. Corr with `rut_depth` = 0.894. Keep both. | Yes | No |
| 13 | `avg_daily_traffic` | Traffic | **KEEP** | Raw traffic volume. Independent observable. Corr with `traffic_load_index` = 0.748. | Yes | No |
| 14 | `heavy_vehicle_pct` | Traffic | **KEEP** | Fraction of heavy vehicles; independent observable. Corr with `traffic_load_index` = 0.613. | Yes | No |
| 15 | `traffic_load_index` | Traffic | **REMOVE-DERIVED** | Corr with `avg_daily_traffic × heavy_vehicle_pct` = 0.896. Normalised computed composite. Adds no information beyond its two parents. Range 0–1 (normalized). | Yes | **Yes — derived from ADT × HVP** |
| 16 | `avg_vehicle_speed_kmh` | Traffic | **KEEP** | Independent observable. Negatively correlated with traffic load (r=−0.665 with TLI). Reflects road capacity. | Yes | No |
| 17 | `soil_moisture_pct` | Geotechnical | **KEEP** | Fundamental soil property; directly measurable. Corr with `soil_settlement_mm` = 0.897 (see Part 6). Both are kept because soil moisture is a cause; settlement is an effect. | Yes | No |
| 18 | `soil_density_g_cm3` | Geotechnical | **KEEP** | Independent soil property. Range 1.1–2.4 g/cm³. Corr with `soil_porosity_pct` = −0.849. Both kept; negative correlation is physically expected (denser = less porous). | Yes | No |
| 19 | `soil_bearing_capacity_kpa` | Geotechnical | **KEEP** | Critical structural property; directly tested. Corr with target = −0.472. | Yes | No |
| 20 | `groundwater_depth_m` | Geotechnical | **KEEP** | Directly measured; affects settlement and bearing capacity. Corr with `soil_settlement_mm` = −0.693. | Yes | No |
| 21 | `soil_porosity_pct` | Geotechnical | **KEEP** | Independent of `void_ratio` in a meaningful sense (see Part 3). r=0.781 with void_ratio — high but not deterministic. Both are standard geotechnical properties. | Yes | No |
| 22 | `soil_settlement_mm` | Geotechnical | **KEEP-CAUTION** | Corr with target 0.486. Settlement is a measurable condition (e.g., via survey benchmarks or InSAR). **Caution: settlement can be a consequence of ongoing collapse, not purely a precursor. Must be declared as a condition indicator, not a leading predictor.** Also strongly correlated with `soil_moisture_pct` (r=0.897). | Yes — if measured before event | No |
| 23 | `void_ratio` | Geotechnical | **KEEP** | Standard geotechnical measure; r=0.781 with soil_porosity (not deterministic). Both reflect different aspects of soil structure. | Yes | No |
| 24 | `annual_rainfall_mm` | Climatic | **KEEP** | Independent climatic variable. Corr with `max_daily_rainfall_mm` = 0.788. Both kept; annual total and daily max capture different aspects of rainfall exposure. | Yes | No |
| 25 | `max_daily_rainfall_mm` | Climatic | **KEEP** | Peak event intensity; distinct from annual total. Range 10–362mm. | Yes | No |
| 26 | `flood_frequency_per_year` | Hydrological | **KEEP** | Historical frequency observable at segment level. Integer range 0–15/year. | Yes | No |
| 27 | `temperature_variation_c` | Climatic | **KEEP** | Freeze-thaw stress indicator. Range 2–45°C variation. | Yes | No |
| 28 | `drainage_efficiency` | Infrastructure | **KEEP** | Measures drainage capability; range 0.018–0.999 (normalized). Corr with `waterlogging_duration_hr` = −0.845. Both kept; efficiency is a property, duration is an outcome. | Yes | No |
| 29 | `waterlogging_duration_hr` | Hydrological | **KEEP** | Hours of waterlogging observed per event/year. Corr with `flood_frequency` = 0.801 and `drainage_efficiency` = −0.845. Keep as a distinct outcome measure. | Yes | No |
| 30 | `underground_pipe_density` | Infrastructure | **KEEP** | Density of subsurface utilities. Range 0.001–0.982 (normalized). Independent observable. | Yes | No |
| 31 | `pipe_age_years` | Infrastructure | **KEEP** | Primary causal factor for pipe deterioration. Corr with `pipe_leakage_index` = 0.897. Keep `pipe_age_years` as the raw observable. | Yes | No |
| 32 | `pipe_leakage_index` | Infrastructure | **REMOVE-DERIVED** | Corr with `pipe_age_years` = 0.897. R² of linear fit = 0.804. Computed proxy for leakage risk based on pipe age. Adds minimal independent information. Range 0–1 (normalized). | Yes | **Yes — primarily derived from `pipe_age_years`** |
| 33 | `distance_to_pipeline_m` | Infrastructure | **KEEP** | Proximity to pipes; independent observable. Range 0.004–100m. | Yes | No |
| 34 | `utility_excavation_count` | Infrastructure | **KEEP** | Count of excavation events; independent observable. Integer range 0–22. | Yes | No |
| 35 | `sewer_condition_index` | Infrastructure | **KEEP-CAUTION** | Strongly correlated with `pipe_age_years` (r=−0.939). Almost certainly computed from pipe age. **However, sewer condition is a distinct engineering assessment category** (separate from leakage probability). Keep as a condition indicator; document its strong correlation with `pipe_age_years`. | Yes | Likely — very high correlation with `pipe_age_years` |
| 36 | `historical_collapse_count` | Event history | **REMOVE-LEAKAGE** | Count of past collapses in/near this segment. Would not be available for new, never-collapsed segments. Corr with target 0.178. At prediction time, a new segment has count=0 by definition, making this feature constant for the primary use case. | **No — unavailable for never-collapsed segments** | No |
| 37 | `distance_to_previous_collapse_m` | Event history | **KEEP-CAUTION** | Proximity to nearest known past collapse. Corr with target = −0.078 (weak). **Caution: requires a maintained collapse database.** If no prior collapse exists, value is undefined/maximum. Keep because spatial collapse clustering is a real phenomenon and the feature is weak (low leakage risk). Document dependency on external collapse records. | Conditionally — only if collapse DB exists | No |
| 38 | `nearby_construction_intensity` | Urban | **KEEP** | Normalized construction activity level near segment. Range 0.001–0.977. Independent observable. | Yes | No |
| 39 | `building_density_per_km2` | Urban | **KEEP** | Structural load pressure from surrounding buildings. Integer range 106–20,000/km². | Yes | No |
| 40 | `land_subsidence_rate_mm_year` | Geotechnical | **KEEP-CAUTION** | Rate of annual land sinking. Measurable by satellite InSAR or leveling surveys. Corr with target 0.492. **Caution: high-subsidence areas may have already partially collapsed; this can be a consequence indicator.** Document as a condition indicator that requires temporal context for full defensibility. | Yes — if monitoring data exists | No |
| 41 | `spatial_vulnerability_index` | Composite | **REMOVE-LEAKAGE** | Named "vulnerability index" — almost certainly a pre-computed risk proxy used directly in target generation. Non-zero lower bound (0.117). Highest single correlation with `building_density_per_km2` (r=0.681) but also correlates with many risk features. Removes 0 predictive power from the remaining feature set (depth-5 tree accuracy unchanged after removal). | Technically yes, but constitutes leakage | **Yes — composite of multiple features** |
| 42 | `collapse_risk_level` | Target | **TARGET** | The classification target. 4 classes, perfectly balanced. | N/A | N/A |

---

## PART 3 — Feature Dependency Analysis

### Summary of Investigated Dependencies

#### 3.1 `traffic_load_index` ← `avg_daily_traffic × heavy_vehicle_pct`

| Property | Value |
|---|---|
| Evidence | Corr(TLI, ADT×HVP) = **0.896** |
| TLI range | 0.000–1.000 (normalized) |
| Interpretation | TLI is a normalized product of traffic count and heavy vehicle fraction |
| Both features needed? | No — `avg_daily_traffic` and `heavy_vehicle_pct` are kept as parents |
| Recommendation | **Remove `traffic_load_index`; keep `avg_daily_traffic` and `heavy_vehicle_pct`** |

#### 3.2 `pipe_leakage_index` ← `pipe_age_years`

| Property | Value |
|---|---|
| Evidence | Corr(PLI, pipe_age) = **0.897**; Linear R² = **0.804** |
| PLI range | 0.000–1.000 (normalized) |
| Interpretation | PLI ≈ linear function of pipe age |
| Both features needed? | No — `pipe_age_years` is kept as the raw observable |
| Recommendation | **Remove `pipe_leakage_index`; keep `pipe_age_years`** |

#### 3.3 `sewer_condition_index` ← `pipe_age_years`

| Property | Value |
|---|---|
| Evidence | Corr(SCI, pipe_age) = **−0.939** (strongest pairwise correlation in dataset) |
| SCI range | 0–100 |
| Interpretation | SCI almost certainly computed as inverse linear function of pipe age |
| Both features needed? | **Domain question.** SCI represents a different assessment category than pipe age. Decision: **Keep SCI with caution** as a distinct engineering category despite derivation. |
| Recommendation | **Keep `sewer_condition_index` with caution; document correlation with `pipe_age_years`** |

#### 3.4 `pavement_condition_index` ← `road_age_years`, `surface_crack_density_pct`, `rut_depth_mm`, `surface_deformation_mm`

| Property | Value |
|---|---|
| Evidence | Corr(PCI, road_age) = **−0.895**; Corr(PCI, crack_density) = **−0.802**; Corr(PCI, rut_depth) = **−0.783**; Corr(PCI, surface_deformation) = **−0.768** |
| PCI range | 9.83–100 |
| Interpretation | PCI is a composite pavement condition score computed from multiple distress indicators — standard in road engineering (ASTM D6433) |
| Both features needed? | Yes — PCI is a first-class engineering KPI used independently by practitioners. Keep despite derivation. |
| Recommendation | **Keep `pavement_condition_index` with caution; document as composite metric** |

#### 3.5 `void_ratio` ← `soil_porosity_pct`

| Property | Value |
|---|---|
| Evidence | Corr(VR, porosity) = **0.781**; Theoretical: VR = e/(1−e) where e=porosity/100 |
| Evidence for exact formula | Corr with theoretical derivation sp/(1−sp) = **0.775** (not significantly higher than raw corr) |
| Interpretation | Correlation is high but not deterministic. The standard soil mechanics formula is not perfectly reproduced, suggesting independent generation |
| Both features needed? | **Yes** — both are independent geotechnical measurements in practice. High correlation is physically expected, not evidence of derivation. |
| Recommendation | **Keep both `void_ratio` and `soil_porosity_pct`** |

#### 3.6 `surface_deformation_mm` ← `rut_depth_mm`

| Property | Value |
|---|---|
| Evidence | Corr(SDM, rut_depth) = **0.894** |
| Interpretation | Both measure physical road surface degradation but are conceptually distinct (rut_depth = lateral wheel-path depression; surface_deformation = vertical settlement under load) |
| Both features needed? | Yes — keep both as they represent different physical failure modes |
| Recommendation | **Keep both; document the high correlation** |

#### 3.7 `soil_settlement_mm` ← `soil_moisture_pct`

| Property | Value |
|---|---|
| Evidence | Corr(settlement, moisture) = **0.897** |
| Interpretation | High correlation physically expected — wetter soil settles more. Both are independent observables (moisture measured by probe; settlement by leveling survey). |
| Both needed? | Yes — moisture is a cause; settlement is an effect. Both are predictive of risk at different temporal scales. |
| Recommendation | **Keep both; document the causal relationship** |

#### 3.8 `spatial_vulnerability_index` — Multi-parent composite

The SVI shows moderate correlations with many features:
- `building_density_per_km2`: r = 0.681
- `nearby_construction_intensity`: r = 0.483
- `historical_collapse_count`: r = 0.427
- `land_subsidence_rate_mm_year`: r = 0.402
- `distance_to_previous_collapse_m`: r = −0.294

No single parent explains SVI sufficiently (max R² from any single parent < 0.47). It is a weighted composite of multiple risk indicators. Because it directly proxies the composite used in target generation, **it is removed regardless of derivation status**.

---

## PART 4 — Identifier and Geographic Feature Policy

### `segment_id`

- Format: `URC_XXXXX` — purely sequential integer suffix (1 → 15000)
- Correlation with numeric target: r = **0.007** (negligible)
- **Decision: EXCLUDE from ML features. Use as row identifier only.**
- Rationale: Sequential ID encodes no road segment information. Including it would cause the model to memorize row order.

### `latitude` and `longitude`

**Analysis results:**

| Metric | Value |
|---|---|
| Latitude span | 12.80°–13.25°N (49.9 km N–S) |
| Longitude span | 80.05°–80.35°E (32.4 km E–W) |
| Area covered | ~49.9 × 32.4 km (Chennai Metro) |
| Corr(lat, target) | **−0.002** (negligible) |
| Corr(lon, target) | **+0.004** (negligible) |
| Nearest-neighbor same-class rate | **0.222** (vs. random baseline 0.250) |

**Key findings:**
1. Coordinates have effectively **zero correlation with the target**. They were used for cosmetic realism only.
2. Nearest-neighbor same-class rate (0.222) is **below random chance (0.250)**, confirming there is no spatial risk clustering in the dataset. Spatially proximate segments are no more likely to share a risk class than random pairs.
3. Consecutive CSV rows are spatially scrambled (mean lat diff = 0.15° = 16.7 km between adjacent rows), confirming no spatial ordering in the file.

**Decision: EXCLUDE from ML features. RETAIN in dataset for map/dashboard visualization.**

Rationale: Including coordinates would cause the model to memorize the synthetic generation bounding box with no generalization value. For real-world deployment, geographic coordinates would not generalize to different cities.

---

## PART 5 — Leakage and Temporal Availability Audit

> **Leakage definition used:** Information that (a) was used to directly/indirectly generate the target label, or (b) would not be available at the intended prediction time for a new road segment.

| Feature | Corr with Target | Temporally Available? | Conclusion | Decision |
|---|---|---|---|---|
| `historical_collapse_count` | 0.178 | **No** — A new or previously uncollapsed segment has count=0 by definition. This feature is only non-zero for segments with documented collapse history, which is post-event. | Not available at prediction time for new segments | **REMOVE — LEAKAGE** |
| `distance_to_previous_collapse_m` | −0.078 | **Conditionally** — Available only if a collapse database exists. For segments in an uninspected area, undefined. | Weak leakage risk; weak predictive signal. Requires documented assumption about collapse database availability. | **KEEP WITH CAUTION** |
| `soil_settlement_mm` | 0.486 | **Yes** — Measured by benchmark surveys or InSAR prior to assessment. | Condition indicator available pre-event; high correlation is expected, not leakage. | **KEEP WITH CAUTION** (document as condition indicator) |
| `land_subsidence_rate_mm_year` | 0.492 | **Yes** — Measured by satellite leveling. Available as a standalone spatial dataset. | Same reasoning as soil_settlement_mm. | **KEEP WITH CAUTION** |
| `pipe_leakage_index` | 0.272 | Yes | Derived from `pipe_age_years` — removed on redundancy grounds. Leakage risk is low. | **REMOVE — DERIVED** (not leakage) |
| `sewer_condition_index` | −0.241 | **Yes** — Infrastructure condition surveys. | Legitimate infrastructure metric. High correlation with `pipe_age_years` is redundancy, not leakage. | **KEEP WITH CAUTION** |
| `flood_frequency_per_year` | 0.267 | **Yes** — Historical record, available before prediction. | Standard hydrological input. | **KEEP** |
| `waterlogging_duration_hr` | 0.270 | **Yes** — Observable and recordable prior to risk assessment. | Standard hydrological input. | **KEEP** |
| `surface_deformation_mm` | 0.548 | **Yes** — Measured at road inspection. | Strongest legitimate predictor; condition measurement. | **KEEP** |
| `spatial_vulnerability_index` | 0.283 | Technically yes | Almost certainly used in generating the target. Named "vulnerability index." Removing changes depth-5 accuracy by 0.000%. | **REMOVE — LEAKAGE** |
| `pavement_condition_index` | −0.539 | **Yes** — Standard road survey output. | Computed from distress measurements taken at inspection time. Legitimate. | **KEEP WITH CAUTION** (composite metric) |

---

## PART 6 — Redundancy and Multicollinearity Analysis

### High Correlation Pairs (|r| ≥ 0.80)

| Pair | r | Domain Distinction | Action |
|---|---|---|---|
| `pipe_age_years` ↔ `sewer_condition_index` | −0.939 | Age vs. assessed condition — different engineering categories | Keep `pipe_age_years`; keep `sewer_condition_index` with caution |
| `soil_moisture_pct` ↔ `soil_settlement_mm` | 0.897 | Cause (moisture) vs. effect (settlement) — distinct temporal meaning | Keep both |
| `pipe_age_years` ↔ `pipe_leakage_index` | 0.897 | PLI is derived; `pipe_age_years` is raw | Remove `pipe_leakage_index`; keep `pipe_age_years` |
| `road_age_years` ↔ `pavement_condition_index` | −0.895 | Age vs. composite condition score | Keep both; PCI is a standard KPI |
| `rut_depth_mm` ↔ `surface_deformation_mm` | 0.894 | Lateral deformation vs. vertical settlement — different failure modes | Keep both |
| `avg_daily_traffic` ↔ `avg_vehicle_speed_kmh` | −0.890 | Volume vs. speed — both are independent observables | Keep both |
| `pipe_leakage_index` ↔ `sewer_condition_index` | −0.884 | Both derived from pipe age; PLI removed on derivation grounds | Resolved by PLI removal |
| `soil_moisture_pct` ↔ `soil_bearing_capacity_kpa` | −0.858 | Cause (moisture weakens bearing capacity) vs. structural outcome | Keep both |
| `soil_bearing_capacity_kpa` ↔ `soil_settlement_mm` | −0.855 | Different measurements; both important structural indicators | Keep both |
| `soil_density_g_cm3` ↔ `soil_porosity_pct` | −0.849 | Inverse relationship is physically expected; both are measured properties | Keep both |
| `drainage_efficiency` ↔ `waterlogging_duration_hr` | −0.845 | Property vs. outcome; both are distinct observables | Keep both |
| `road_age_years` ↔ `rut_depth_mm` | 0.839 | Age vs. distress outcome — different temporal status | Keep both |
| `surface_crack_density_pct` ↔ `surface_deformation_mm` | 0.805 | Different pavement failure modes | Keep both |
| `surface_crack_density_pct` ↔ `pavement_condition_index` | −0.802 | PCI component vs. PCI composite — redundant but PCI is a KPI | Keep both with documentation |
| `flood_frequency_per_year` ↔ `waterlogging_duration_hr` | 0.801 | Frequency vs. duration — conceptually distinct | Keep both |

**Key principle applied:** High correlation between two features does not mandate removal if both represent independently meaningful engineering measurements or distinct physical phenomena. Removal is only justified where one feature is mathematically derived from another with negligible residual (TLI, PLI).

**Multicollinearity concern for linear models:** Linear models (Logistic Regression) will be affected by the many correlated pairs. Regularization (L2/L1) is required. Tree-based models (Random Forest, XGBoost) are unaffected and will handle this naturally.

---

## PART 7 — Data Quality Audit

### Full Quality Table

| Column | Min | Max | Missing | Unique | Type | Quality Issue | Action |
|---|---|---|---|---|---|---|---|
| `segment_id` | — | — | 0 | 15,000 | str | Sequential, no predictive value | Use as ID only |
| `latitude` | 12.8000 | 13.2499 | 0 | 4,328 | float64 | Very low CV (0.010) — narrow range | Exclude from ML |
| `longitude` | 80.0500 | 80.3500 | 0 | 2,987 | float64 | Very low CV (0.001) — narrow range | Exclude from ML |
| `elevation_m` | 1.0 | 69.7 | 0 | 14,414 | float64 | None | Keep |
| `road_length_m` | 50.0 | 1999.9 | 0 | 14,996 | float64 | None | Keep |
| `distance_to_water_body_m` | 24.7 | 5000.0 | 0 | 14,976 | float64 | Hard cap at 5000m | Keep; note cap |
| `road_age_years` | 1 | 60 | 0 | 60 | int64 | Hard cap at 60 years | Keep; consistent with urban road lifespan |
| `pavement_thickness_cm` | 10.0 | 60.0 | 0 | 14,482 | float64 | None | Keep |
| `surface_crack_density_pct` | 0.0 | 60.0 | 0 | 14,194 | float64 | 557 zeros (3.7%) — plausible for new roads | Keep; zeros are valid |
| `pavement_condition_index` | 9.83 | 100.0 | 0 | 14,303 | float64 | Hard cap at 100 | Keep; standard PCI scale |
| `rut_depth_mm` | 0.0 | 46.6 | 0 | 14,235 | float64 | None | Keep |
| `surface_deformation_mm` | 0.0 | 100.0 | 0 | 14,567 | float64 | Hard cap at 100mm | Keep; note cap |
| `avg_daily_traffic` | 500 | 100,000 | 0 | 13,931 | int64 | Hard caps (500, 100000) | Keep; synthetic bounds |
| `heavy_vehicle_pct` | 0.001 | 40.0 | 0 | 14,751 | float64 | Hard cap near 40% | Keep |
| `traffic_load_index` | 0.0 | 1.0 | 0 | 6,897 | float64 | **REMOVE — DERIVED** | Remove |
| `avg_vehicle_speed_kmh` | 10.0 | 100.0 | 0 | 14,792 | float64 | Hard caps | Keep |
| `soil_moisture_pct` | 5.0 | 60.0 | 0 | 14,800 | float64 | Hard caps | Keep |
| `soil_density_g_cm3` | 1.1 | 2.4 | 0 | 8,914 | float64 | None | Keep |
| `soil_bearing_capacity_kpa` | 97.2 | 500.0 | 0 | 13,708 | float64 | Hard cap at 500 | Keep; consistent with bearing capacity range |
| `groundwater_depth_m` | 0.5 | 30.0 | 0 | 14,376 | float64 | Hard caps | Keep |
| `soil_porosity_pct` | 20.0 | 58.2 | 0 | 14,372 | float64 | None | Keep |
| `soil_settlement_mm` | 0.0 | 130.4 | 0 | 14,477 | float64 | 435 zeros (2.9%) — plausible | Keep |
| `void_ratio` | 0.2 | 1.5 | 0 | 6,513 | float64 | Hard caps | Keep |
| `annual_rainfall_mm` | 200.0 | 3000.0 | 0 | 14,866 | float64 | Hard caps | Keep |
| `max_daily_rainfall_mm` | 10.0 | 361.8 | 0 | 14,889 | float64 | None | Keep |
| `flood_frequency_per_year` | 0 | 15 | 0 | 16 | int64 | 121 zeros (0.8%) — valid | Keep |
| `temperature_variation_c` | 2.0 | 45.0 | 0 | 14,769 | float64 | Hard caps | Keep |
| `drainage_efficiency` | 0.018 | 0.999 | 0 | 6,970 | float64 | Normalized [0,1] — likely derived | Keep; no clear parent identified |
| `waterlogging_duration_hr` | 0.0 | 72.0 | 0 | 14,711 | float64 | Hard cap at 72hr | Keep |
| `underground_pipe_density` | 0.001 | 0.982 | 0 | 6,889 | float64 | Normalized — likely density per km² normalized | Keep |
| `pipe_age_years` | 0 | 80 | 0 | 81 | int64 | Hard cap at 80; 0 = new pipe | Keep |
| `pipe_leakage_index` | 0.0 | 1.0 | 0 | 7,282 | float64 | **REMOVE — DERIVED** | Remove |
| `distance_to_pipeline_m` | 0.004 | 100.0 | 0 | 14,534 | float64 | Hard cap at 100m | Keep |
| `utility_excavation_count` | 0 | 22 | 0 | 23 | int64 | 61 zeros (0.4%) — valid | Keep |
| `sewer_condition_index` | 0.0 | 100.0 | 0 | 14,425 | float64 | Hard cap at 100 | Keep with caution |
| `historical_collapse_count` | 0 | 14 | 0 | 15 | int64 | **REMOVE — LEAKAGE** | Remove |
| `distance_to_previous_collapse_m` | 0.031 | 5000.0 | 0 | 14,972 | float64 | Hard cap at 5000m | Keep with caution |
| `nearby_construction_intensity` | 0.001 | 0.977 | 0 | 6,657 | float64 | Normalized | Keep |
| `building_density_per_km2` | 106 | 20,000 | 0 | 10,482 | int64 | Hard caps | Keep |
| `land_subsidence_rate_mm_year` | 0.0 | 43.6 | 0 | 14,368 | float64 | 201 zeros (1.3%) — plausible | Keep with caution |
| `spatial_vulnerability_index` | 0.117 | 0.720 | 0 | 3,647 | float64 | **REMOVE — LEAKAGE** | Remove |
| `collapse_risk_level` | — | — | 0 | 4 | str | Target | Target |

**Summary of quality findings:**
- **Zero missing values** across all 42 columns.
- **Zero duplicate rows** (feature space).
- **No negative values** where physically impossible.
- **No constant or near-constant columns** (latitude and longitude have low CV but sufficient variance for visualization).
- **Hard caps/bounds** observed in many columns — consistent with synthetic generation with bounded ranges.
- **Outlier policy:** No outliers require removal. All extreme values fall within stated ranges (e.g., rut_depth up to 46.6mm is physically plausible for severely deteriorated roads). Do not remove any rows based on feature values.

---

## PART 8 — Spatial Leakage and Split Strategy

### Spatial Analysis Results

| Metric | Value |
|---|---|
| Dataset geographic span | 49.9 km (N–S) × 32.4 km (E–W) |
| Nearest-neighbor mean distance | **0.91 km** |
| Nearest-neighbor median distance | **0.84 km** |
| Nearest-neighbor minimum distance | **24 m** (extremely close pair exists) |
| 5th percentile NN distance | **0.20 km** |
| NN same-class rate | **0.222** (random baseline = 0.250) |
| Corr(lat, target) | −0.002 |
| Corr(lon, target) | +0.004 |
| CSV spatial ordering | Random — mean consecutive row distance = 16.7 km |

### Assessment

The nearest-neighbor same-class rate (0.222) is **slightly below** the random chance baseline (0.250). This means that spatially proximate road segments are **no more likely to share a risk class than random pairs**. There is no spatial autocorrelation in risk labels.

This confirms that the dataset was generated with **independent random feature sampling per row**, not spatially correlated block generation. The coordinates are decorative rather than predictive.

Therefore:
- **Spatial autocorrelation does not exist in this dataset.**
- **A spatial/group-aware split would not provide robustness benefits** because there is no spatial structure to protect against.
- **There is no spatial leakage risk** in a random train/test split.

However, 24m nearest-neighbor minimum distance means some segment pairs are effectively co-located. Since their features appear to be independently sampled (no spatial autocorrelation in labels), this does not constitute a leakage risk in this dataset.

### Split Recommendation

**→ Option A: Random stratified split (70/15/15)**

Rationale:
- No spatial autocorrelation detected (NN same-class rate below random).
- Coordinates have zero predictive value and are excluded from the model.
- Stratification preserves the 25/25/25/25 class balance across splits.
- A spatial split would be unnecessarily complex without any robustness benefit for this specific dataset.

**Note for real-world deployment:** If this system were ever applied to real geographic road data, a spatial split (e.g., leave-one-district-out) would be mandatory. This recommendation is specific to this synthetic dataset.

---

## PART 9 — Final Canonical Schema

### Schema Definition

```
=================================================================
TARGET
=================================================================
Name:    collapse_risk_level
Type:    str (categorical)
Classes: Low | Moderate | High | Critical
Encoding for training: LabelEncoder or OrdinalEncoder
Output:  4-class classifier

=================================================================
MODEL FEATURES  (34 features after removals)
=================================================================

--- Road Infrastructure (6 features) ---
road_age_years                 int64    [1, 60]         years
road_length_m                  float64  [50, 2000]      meters
pavement_thickness_cm          float64  [10, 60]        centimeters
surface_crack_density_pct      float64  [0, 60]         percent
pavement_condition_index       float64  [9.83, 100]     PCI score (0=worst, 100=best)  ⚠ CAUTION: composite
rut_depth_mm                   float64  [0, 46.6]       millimeters

--- Traffic (4 features) ---
avg_daily_traffic              int64    [500, 100000]   vehicles/day
heavy_vehicle_pct              float64  [0.001, 40]     percent
avg_vehicle_speed_kmh          float64  [10, 100]       km/h
surface_deformation_mm         float64  [0, 100]        millimeters   (note: traffic + structural effect)

--- Geotechnical / Soil (7 features) ---
soil_moisture_pct              float64  [5, 60]         percent
soil_density_g_cm3             float64  [1.1, 2.4]      g/cm³
soil_bearing_capacity_kpa      float64  [97.2, 500]     kPa
groundwater_depth_m            float64  [0.5, 30]       meters
soil_porosity_pct              float64  [20, 58.2]      percent
void_ratio                     float64  [0.2, 1.5]      dimensionless
soil_settlement_mm             float64  [0, 130.4]      millimeters   ⚠ CAUTION: condition indicator

--- Climatic / Hydrological (6 features) ---
elevation_m                    float64  [1, 69.7]       meters above sea level
annual_rainfall_mm             float64  [200, 3000]     mm/year
max_daily_rainfall_mm          float64  [10, 361.8]     mm/day
flood_frequency_per_year       int64    [0, 15]         events/year
temperature_variation_c        float64  [2, 45]         degrees Celsius
waterlogging_duration_hr       float64  [0, 72]         hours

--- Drainage / Hydrology (2 features) ---
drainage_efficiency            float64  [0.018, 0.999]  normalized [0,1]
distance_to_water_body_m       float64  [24.7, 5000]    meters

--- Underground Infrastructure (6 features) ---
underground_pipe_density       float64  [0.001, 0.982]  normalized density
pipe_age_years                 int64    [0, 80]         years
distance_to_pipeline_m         float64  [0.004, 100]    meters
utility_excavation_count       int64    [0, 22]         count
sewer_condition_index          float64  [0, 100]        index (0=worst, 100=best)  ⚠ CAUTION: derived from pipe_age
land_subsidence_rate_mm_year   float64  [0, 43.6]       mm/year                    ⚠ CAUTION: condition indicator

--- Urban / Environmental (3 features) ---
nearby_construction_intensity  float64  [0.001, 0.977]  normalized [0,1]
building_density_per_km2       int64    [106, 20000]    buildings/km²
distance_to_previous_collapse_m float64 [0.031, 5000]  meters                      ⚠ CAUTION: requires collapse DB

=================================================================
METADATA (retained in dataset, excluded from ML)
=================================================================
segment_id                     str      URC_00001…URC_15000   Row identifier
latitude                       float64  [12.80, 13.25]        Geographic (visualization only)
longitude                      float64  [80.05, 80.35]        Geographic (visualization only)

=================================================================
REMOVED — LEAKAGE
=================================================================
spatial_vulnerability_index    Composite vulnerability proxy; likely used in target generation
historical_collapse_count      Post-event data; unavailable for uncollapsed segments

=================================================================
REMOVED — DERIVED (REDUNDANT)
=================================================================
traffic_load_index             Derived from avg_daily_traffic × heavy_vehicle_pct (r=0.896)
pipe_leakage_index             Derived from pipe_age_years (r=0.897, R²=0.804)

=================================================================
FEATURE COUNTS
=================================================================
Total original columns:        42
TARGET:                        1
ID/METADATA:                   3  (segment_id, latitude, longitude)
Model features (KEEP):         29
Model features (KEEP-CAUTION): 5  (pavement_condition_index, soil_settlement_mm,
                                    land_subsidence_rate_mm_year, sewer_condition_index,
                                    distance_to_previous_collapse_m)
REMOVED — LEAKAGE:             2  (spatial_vulnerability_index, historical_collapse_count)
REMOVED — DERIVED:             2  (traffic_load_index, pipe_leakage_index)
                               ─────────────────────────────────────────
TOTAL MODEL FEATURES:          34  (29 KEEP + 5 KEEP-CAUTION)
```

> **⚠ Caution features note:** The 5 KEEP-CAUTION features are retained in the feature set. Their assumptions must be documented in the ML pipeline code, data schema documentation, and the application disclaimer. They are not removed; their inclusion is a deliberate, documented engineering decision.

---

## PART 10 — Locked Project Decisions

These decisions are **final for Phase 1+ implementation**. Changes require explicit documentation and architectural review.

### Decision Registry

| # | Decision Topic | Decision |
|---|---|---|
| **D01** | **Dataset** | Dataset 2 — Urban Road Collapse Risk Assessment (Kaggle, CC0). Provisionally selected. Treated as synthetic with major caveats. Source file: `storage/datasets/urban_road_collapse_risk_dataset.csv` |
| **D02** | **Project Scope** | "Urban Road Collapse Risk Prediction" — the system predicts road segment collapse risk using ground and infrastructure condition indicators. Not "general soil risk" or "ground stability." All documentation must use this scope. |
| **D03** | **Target Variable** | `collapse_risk_level` — string. Exact classes: `Low`, `Moderate`, `High`, `Critical`. Case-sensitive. Must be validated at ingestion. |
| **D04** | **Number of Classes** | **4 classes.** Do NOT merge Critical into High. PRD must be updated to reflect 4 classes. |
| **D05** | **Feature Policy** | 34 model features as specified in Part 9 (29 KEEP + 5 KEEP-CAUTION). Remove `spatial_vulnerability_index`, `historical_collapse_count`, `traffic_load_index`, `pipe_leakage_index`. Retain 5 caution features (`pavement_condition_index`, `soil_settlement_mm`, `land_subsidence_rate_mm_year`, `sewer_condition_index`, `distance_to_previous_collapse_m`) with documented assumptions. |
| **D06** | **Geographic Feature Policy** | `latitude` and `longitude` are **excluded from ML model features**. They are retained in the dataset and database for map visualization only. `segment_id` is an identifier, excluded from ML. |
| **D07** | **Leakage Policy** | `spatial_vulnerability_index` removed as leakage. `historical_collapse_count` removed as temporally unavailable. All KEEP-CAUTION features must have documented assumptions in ML pipeline code. |
| **D08** | **Split Strategy** | Random stratified 70/15/15 split (stratify by `collapse_risk_level`). Fixed random seed (from `RANDOM_SEED` environment variable). No spatial grouping required — no spatial autocorrelation detected in this dataset. |
| **D09** | **Missing Value Policy** | Current dataset has zero missing values. Pipeline must handle missing values for future uploaded datasets: numeric columns → median imputation (fitted on training set only); categorical → mode imputation. |
| **D10** | **Encoding Policy** | `collapse_risk_level` → LabelEncoder for tree-based models; OrdinalEncoder with fixed ordering [Low=0, Moderate=1, High=2, Critical=3] for linear models. All input features are numeric — no categorical encoding required for this dataset. |
| **D11** | **Scaling Policy** | StandardScaler for linear models (Logistic Regression, SVM). No scaling required for tree-based models (Random Forest, XGBoost, Decision Tree). Scaler must be fitted on training data only; applied via `transform()` to validation and test sets. |
| **D12** | **Outlier Policy** | **No rows removed for outliers.** All feature ranges are physically plausible within the stated synthetic bounds. Hard caps (e.g., distance_to_water_body_m max=5000, avg_daily_traffic max=100,000) are documented. Future uploaded data: flag values outside [Q1−3×IQR, Q3+3×IQR] as warnings in the validation report; do not auto-remove. |
| **D13** | **Reproducibility** | All train/test splits and model training operations must use `RANDOM_SEED` from environment. Random seed = 42 as default. All ML experiments must be reproducible with the same seed. |

### Required Documentation Updates

The following project files must be updated as a consequence of these decisions:

| File | Required Update |
|---|---|
| `PRD.md` | Update target classes from Low/Moderate/High → Low/Moderate/High/Critical |
| `PRD.md` | Update project scope wording to "Urban Road Collapse Risk Prediction" |
| `TODO.md` | Close Phase 0 checklist items (dataset selected, target defined, features documented) |
| `ARCHITECTURE.md` | Reference this specification as the canonical feature schema |
| `docs/` | Create `dataset_specification.md` (this document) |

---

*This specification was produced from actual statistical analysis of the raw dataset file. All correlation values, ranges, and counts are verified from data. No values were fabricated.*

*Signed off for Phase 1 progression: Phase 0 — Dataset Research & Selection*

---

### Changelog
- **2026-08-28 (Phase 1 start):** Corrected bookkeeping inconsistencies in feature counts (removed duplicate `temperature_variation_c` from Urban section, removed duplicate `pavement_condition_index` from KEEP-CAUTION list). Correct total feature count verified as 34.
- **2026-08-31 (Phase 3.1 Audit):** Reconciled D05 decision text from '6 caution features' to the exact verified count of '5 caution features' (29 KEEP + 5 KEEP-CAUTION = 34 total model features).
