"""
Central configuration hub for the data-aware AutoML pipeline.
Holds all statistical thresholds and pipeline settings.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Union


@dataclass
class GlobalConfig:
    """Global statistical thresholds for data profiling."""
    skew_threshold: float = 3.0
    scale_ratio: float = 1000.0
    imbalance_threshold: float = 0.15
    outlier_iqr_factor: float = 3.0
    outlier_min_count: int = 1
    random_state: int = 42


@dataclass
class FeatureEngineeringConfig:
    """Settings for feature engineering and selection."""
    max_feature_ratio: float = 0.5
    early_stop_patience: int = 5


@dataclass
class CorrelationConfig:
    """Thresholds for correlation-based feature filtering."""
    target_corr_threshold: float = 0.1
    feature_corr_threshold: float = 0.9


@dataclass
class HyperparamConfig:
    """Owned by hyperparameter_tuning.py — populated by teammate."""
    pass


# --------------- Demo Task Configs ---------------

@dataclass
class ScalingTaskConfig:
    """Demo config for scaling-related tasks."""
    n_samples: int = 500
    n_features: int = 10
    random_state: int = 42


@dataclass
class AttritionTaskConfig:
    """Demo config for attrition-related tasks."""
    n_samples: int = 500
    imbalance_ratio: float = 0.15
    random_state: int = 42


@dataclass
class SkewedTaskConfig:
    """Demo config for skewed-data tasks."""
    n_samples: int = 500
    skew_factor: float = 3.0
    random_state: int = 42


# --------------- Master Environment Config ---------------

@dataclass
class EnvConfig:
    """
    Master configuration bundle.
    Aggregates all sub-configs and controls the pipeline mode.
    """
    mode: Literal["upload", "demo"] = "upload"
    task: Optional[Union[ScalingTaskConfig, AttritionTaskConfig, SkewedTaskConfig]] = None

    global_cfg: GlobalConfig = field(default_factory=GlobalConfig)
    feature_eng_cfg: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    correlation_cfg: CorrelationConfig = field(default_factory=CorrelationConfig)
    hyperparam_cfg: HyperparamConfig = field(default_factory=HyperparamConfig)

    def __post_init__(self) -> None:
        if self.mode == "demo" and self.task is None:
            raise ValueError("Demo mode requires a task config.")
