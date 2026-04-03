"""
config.py
─────────
Central configuration hub for the Silent Data Debugger pipeline.

All statistical thresholds, RL task parameters, and feature-engineering
settings live here so that callers never embed magic numbers.

Dataclasses
───────────
GlobalConfig            — statistical thresholds shared across every module
ScalingTaskConfig       — Task 1: KNN scale-mismatch synthetic dataset
AttritionTaskConfig     — Task 2: HR attrition imputation/encoding dataset
SkewedTaskConfig        — Task 3: Log-normal regression dataset
EnvConfig               — master RL-environment config (bundles all above)
FeatureEngineeringConfig— settings for feature_engineer.py
CorrelationConfig       — thresholds for correlation.py
HyperparamConfig        — placeholder for hyperparameter tuning module
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ── Shared statistical thresholds ──────────────────────────────────────────

@dataclass
class GlobalConfig:
    """
    Statistical thresholds shared by state.py, evaluate.py, and the env.

    All values can be overridden at construction time to tune sensitivity.
    """
    skew_threshold: float    = 3.0    # |skew| above this → flagged as high skewness
    scale_ratio: float       = 1000.0 # column-range ratio above this → scale mismatch
    imbalance_threshold: float = 0.15 # minority-class ratio below this → imbalanced
    outlier_iqr_factor: float  = 3.0  # IQR multiplier for fence computation
    outlier_min_count: int     = 1    # min outlier rows required to report a column
    random_state: int          = 42   # global random seed for reproducibility


# ── RL task configs ─────────────────────────────────────────────────────────

@dataclass
class ScalingTaskConfig:
    """
    Task 1 — The Scaling Failure (KNN Classification).

    Generates a synthetic classification dataset where one feature is
    inflated by ``scale_factor``, causing KNN accuracy to collapse.
    The agent must apply a scaler to recover accuracy.
    """
    n_samples: int              = 500
    n_features: int             = 5
    n_classes: int              = 2
    scale_factor: float         = 1500.0  # inflation multiple (must exceed GlobalConfig.scale_ratio)
    random_state: int           = 42
    test_size: float            = 0.2
    improvement_threshold: float = 0.15   # min accuracy gain for full reward


@dataclass
class AttritionTaskConfig:
    """
    Task 2 — The Attrition Imputation Trap (HR Dataset).

    Simulates an HR dataset with injected NaNs in numeric columns and
    unencoded categorical columns that block model training.
    """
    n_samples: int   = 1000
    null_rate: float = 0.1         # fraction of each null_col to set to NaN
    null_cols: list  = field(
        default_factory=lambda: ["Age", "MonthlyIncome", "YearsAtCompany"]
    )
    cat_cols: list   = field(
        default_factory=lambda: ["Department", "JobRole", "OverTime"]
    )
    target_col: str  = "Attrition"
    random_state: int = 42
    test_size: float  = 0.2


@dataclass
class SkewedTaskConfig:
    """
    Task 3 — The Skewed Forecasting Nightmare (Regression).

    Numpy log-normal target with heavy right skew.  A naive linear model
    produces very high RMSE; the agent must apply log1p/expm1.
    """
    n_samples: int                = 1000
    n_features: int               = 4
    skew_param: float             = 3.0    # log-normal sigma (higher → more skew)
    noise_level: float            = 0.1    # Gaussian noise on features
    random_state: int             = 42
    test_size: float              = 0.2
    rmse_improvement_factor: float = 0.5   # agent must beat baseline_rmse × this


# ── Master RL environment config ────────────────────────────────────────────

@dataclass
class EnvConfig:
    """
    Master configuration bundle passed to ``SilentDebuggerEnv``.

    ``task`` selects which synthetic dataset to generate.
    All sub-configs are instantiated with sensible defaults and can be
    overridden individually.
    """
    task: str = "scaling"   # "scaling" | "attrition" | "skewed"

    global_cfg:    GlobalConfig         = field(default_factory=GlobalConfig)
    scaling_cfg:   ScalingTaskConfig    = field(default_factory=ScalingTaskConfig)
    attrition_cfg: AttritionTaskConfig  = field(default_factory=AttritionTaskConfig)
    skewed_cfg:    SkewedTaskConfig     = field(default_factory=SkewedTaskConfig)


# ── Feature engineering & pipeline configs ──────────────────────────────────

@dataclass
class FeatureEngineeringConfig:
    """
    Settings for ``feature_engineer.py``.

    Controls how aggressively the iterative feature construction loop
    explores new features versus stopping early.
    """
    max_feature_ratio: float  = 0.5  # max new features as a fraction of original count
    early_stop_patience: int  = 5    # stop after this many non-improving iterations


@dataclass
class CorrelationConfig:
    """
    Thresholds for ``correlation.py``'s feature-filtering logic.

    * ``target_corr_threshold``: drop features whose |correlation with target|
      is below this value (too weak to be useful).
    * ``feature_corr_threshold``: drop one of any pair of features whose
      mutual |correlation| exceeds this value (redundant).
    """
    target_corr_threshold:  float = 0.1   # min |corr(feature, target)| to keep
    feature_corr_threshold: float = 0.9   # max |corr(f1, f2)| before dropping one


@dataclass
class HyperparamConfig:
    """Placeholder for the hyperparameter-tuning module (populated by tune.py)."""
    pass