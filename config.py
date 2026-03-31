from dataclasses import dataclass, field


# ====================================================================== #
#  TASK CONFIG DATACLASSES                                                #
#  All hardcoded values in state.py / evaluate.py / env.py live here.   #
#  Frontend sliders map directly to these fields.                        #
# ====================================================================== #


@dataclass
class GlobalConfig:
    """
    Thresholds used by get_state() to detect data issues.
    Shared across all tasks.
    """
    skew_threshold: float = 3.0       # |skew| above this → flagged as high skewness
    scale_ratio: float = 1000.0       # column range ratio above this → flagged as scale mismatch


@dataclass
class ScalingTaskConfig:
    """
    Task 1 — The Scaling Failure (KNN).
    Synthetic classification dataset with one badly scaled feature.
    """
    n_samples: int = 500              # number of rows
    n_features: int = 5               # total number of features
    n_classes: int = 2                # number of target classes
    scale_factor: float = 1500.0      # how much to inflate the outlier feature (must exceed scale_ratio threshold)
    random_state: int = 42
    test_size: float = 0.2
    improvement_threshold: float = 0.15  # min accuracy gain to earn full reward


@dataclass
class AttritionTaskConfig:
    """
    Task 2 — The Attrition Imputation Trap (HR Dataset).
    Simulated HR dataset with injected nulls and unencoded categoricals.
    """
    n_samples: int = 1000             # number of employee rows
    null_rate: float = 0.1            # fraction of values to null out per null_col
    null_cols: list = field(          # columns where nulls will be injected
        default_factory=lambda: ["Age", "MonthlyIncome", "YearsAtCompany"]
    )
    cat_cols: list = field(           # columns that remain unencoded (agent must encode)
        default_factory=lambda: ["Department", "JobRole", "OverTime"]
    )
    target_col: str = "Attrition"
    random_state: int = 42
    test_size: float = 0.2


@dataclass
class SkewedTaskConfig:
    """
    Task 3 — The Skewed Forecasting Nightmare (Regression).
    Numpy log-normal synthetic data with heavy right skew.
    """
    n_samples: int = 1000             # number of rows
    n_features: int = 4               # number of input features
    skew_param: float = 3.0           # log-normal sigma — higher = more skew
    noise_level: float = 0.1          # gaussian noise added to features
    random_state: int = 42
    test_size: float = 0.2
    rmse_improvement_factor: float = 0.5   # agent must beat baseline_rmse * this factor


@dataclass
class EnvConfig:
    """
    Master config passed to the environment at init.
    Bundles all sub-configs together.
    """
    task: str = "scaling"             # "scaling" | "attrition" | "skewed"
    global_cfg: GlobalConfig = field(default_factory=GlobalConfig)
    scaling_cfg: ScalingTaskConfig = field(default_factory=ScalingTaskConfig)
    attrition_cfg: AttritionTaskConfig = field(default_factory=AttritionTaskConfig)
    skewed_cfg: SkewedTaskConfig = field(default_factory=SkewedTaskConfig)
