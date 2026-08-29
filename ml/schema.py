"""
Canonical Dataset Schema Definitions.
This module is the single source of truth for all feature and target definitions.
"""
from dataclasses import dataclass

# =================================================================
# TARGET DEFINITION
# =================================================================
TARGET_COLUMN = "collapse_risk_level"

# The target classes in nominal order (safe default)
TARGET_CLASSES = ("Low", "Moderate", "High", "Critical")

# For linear models that require ordinal encoding of the target
ORDINAL_ENCODING = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
    "Critical": 3
}

# =================================================================
# FEATURE DEFINITIONS
# =================================================================

# Metadata columns to retain for visualization/DB, but exclude from ML features
METADATA_COLUMNS = frozenset({"segment_id", "latitude", "longitude"})

# Explicitly removed features (leakage or redundant)
REMOVED_COLUMNS = frozenset({
    "spatial_vulnerability_index",  # Leakage
    "historical_collapse_count",    # Leakage
    "traffic_load_index",           # Redundant (Derived)
    "pipe_leakage_index"            # Redundant (Derived)
})

# The 34 definitive model features, ordered logically
MODEL_FEATURES = (
    # Road Infrastructure
    "road_age_years",
    "road_length_m",
    "pavement_thickness_cm",
    "surface_crack_density_pct",
    "pavement_condition_index",      # CAUTION
    "rut_depth_mm",
    
    # Traffic
    "avg_daily_traffic",
    "heavy_vehicle_pct",
    "avg_vehicle_speed_kmh",
    "surface_deformation_mm",
    
    # Geotechnical / Soil
    "soil_moisture_pct",
    "soil_density_g_cm3",
    "soil_bearing_capacity_kpa",
    "groundwater_depth_m",
    "soil_porosity_pct",
    "void_ratio",
    "soil_settlement_mm",            # CAUTION
    
    # Climatic / Hydrological
    "elevation_m",
    "annual_rainfall_mm",
    "max_daily_rainfall_mm",
    "flood_frequency_per_year",
    "temperature_variation_c",
    "waterlogging_duration_hr",
    
    # Drainage / Hydrology
    "drainage_efficiency",
    "distance_to_water_body_m",
    
    # Underground Infrastructure
    "underground_pipe_density",
    "pipe_age_years",
    "distance_to_pipeline_m",
    "utility_excavation_count",
    "sewer_condition_index",         # CAUTION
    "land_subsidence_rate_mm_year",  # CAUTION
    
    # Urban / Environmental
    "nearby_construction_intensity",
    "building_density_per_km2",
    "distance_to_previous_collapse_m" # CAUTION
)

# Subset of MODEL_FEATURES that have documented assumptions
CAUTION_FEATURES = frozenset({
    "pavement_condition_index",
    "soil_settlement_mm",
    "sewer_condition_index",
    "land_subsidence_rate_mm_year",
    "distance_to_previous_collapse_m"
})

# =================================================================
# FEATURE RANGES (For Data Validation)
# =================================================================
# Format: {feature_name: (min_val, max_val)}
FEATURE_RANGES = {
    "road_age_years": (1, 60),
    "road_length_m": (50, 2000),
    "pavement_thickness_cm": (10, 60),
    "surface_crack_density_pct": (0, 60),
    "pavement_condition_index": (9.83, 100),
    "rut_depth_mm": (0, 46.6),
    "avg_daily_traffic": (500, 100000),
    "heavy_vehicle_pct": (0.001, 40),
    "avg_vehicle_speed_kmh": (10, 100),
    "surface_deformation_mm": (0, 100),
    "soil_moisture_pct": (5, 60),
    "soil_density_g_cm3": (1.1, 2.4),
    "soil_bearing_capacity_kpa": (97.2, 500),
    "groundwater_depth_m": (0.5, 30),
    "soil_porosity_pct": (20, 58.2),
    "void_ratio": (0.2, 1.5),
    "soil_settlement_mm": (0, 130.4),
    "elevation_m": (1, 69.7),
    "annual_rainfall_mm": (200, 3000),
    "max_daily_rainfall_mm": (10, 361.8),
    "flood_frequency_per_year": (0, 15),
    "temperature_variation_c": (2, 45),
    "waterlogging_duration_hr": (0, 72),
    "drainage_efficiency": (0.018, 0.999),
    "distance_to_water_body_m": (24.7, 5000),
    "underground_pipe_density": (0.001, 0.982),
    "pipe_age_years": (0, 80),
    "distance_to_pipeline_m": (0.004, 100),
    "utility_excavation_count": (0, 22),
    "sewer_condition_index": (0, 100),
    "land_subsidence_rate_mm_year": (0, 43.6),
    "nearby_construction_intensity": (0.001, 0.977),
    "building_density_per_km2": (106, 20000),
    "distance_to_previous_collapse_m": (0.031, 5000),
}

# =================================================================
# ALGORITHM FAMILIES
# =================================================================
LINEAR_MODEL_FAMILIES = {"logistic_regression", "svm"}
TREE_MODEL_FAMILIES = {"decision_tree", "random_forest", "xgboost"}

# =================================================================
# SCHEMA BUNDLE
# =================================================================
@dataclass
class FeatureSchema:
    """Bundles all schema metadata for a dataset."""
    target_column: str = TARGET_COLUMN
    target_classes: tuple[str, ...] = TARGET_CLASSES
    model_features: tuple[str, ...] = MODEL_FEATURES
    metadata_columns: frozenset[str] = METADATA_COLUMNS
    removed_columns: frozenset[str] = REMOVED_COLUMNS
    
    @property
    def numerical_features(self) -> tuple[str, ...]:
        """All features are numerical in this dataset."""
        return self.model_features

    @property
    def categorical_features(self) -> tuple[str, ...]:
        """No categorical features in this dataset."""
        return tuple()
