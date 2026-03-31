import pandas as pd
import numpy as np
import json
from config import GlobalConfig


def get_state(df: pd.DataFrame, cfg: GlobalConfig | None = None) -> str:
    """
    Analyzes a Pandas DataFrame and returns a Data Health JSON string.
    This acts as the 'Observation' for the AI Agent in the OpenEnv loop.

    Args:
        df  : The DataFrame to analyze.
        cfg : GlobalConfig with configurable thresholds. Defaults to
              GlobalConfig() if not provided.

    Returns a JSON string with the following possible keys:
        - shape                  : {rows, columns} of the DataFrame
        - missing_values         : {col: count} for columns with nulls
        - unencoded_categoricals : [list of object/category dtype columns]
        - high_skewness          : {col: skew_value} for |skew| > cfg.skew_threshold
        - scale_mismatch         : {col: {min, max, range}} for columns whose range
                                   exceeds cfg.scale_ratio x smallest column range
        - class_imbalance        : {col: {majority, minority, minority_ratio}} for
                                   binary/categorical columns below cfg.imbalance_threshold
        - outliers               : {col: {count, lower_fence, upper_fence}} for numeric
                                   columns with values beyond cfg.outlier_iqr_factor x IQR
        - duplicate_rows         : int count of exact duplicate rows
        - wrong_dtypes           : {col: {current_dtype, suggested_dtype}} for columns
                                   that appear numeric but are stored as strings
    """
    if cfg is None:
        cfg = GlobalConfig()

    state_dict = {}
    numeric_df = df.select_dtypes(include=[np.number])

    # ------------------------------------------------------------------ #
    # 0. DataFrame Shape                                                  #
    # ------------------------------------------------------------------ #
    state_dict["shape"] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1])
    }

    # ------------------------------------------------------------------ #
    # 1. Detect Missing Values                                            #
    # ------------------------------------------------------------------ #
    missing_counts = df.isnull().sum()
    missing_dict = missing_counts[missing_counts > 0].to_dict()
    if missing_dict:
        state_dict["missing_values"] = {
            col: int(count) for col, count in missing_dict.items()
        }

    # ------------------------------------------------------------------ #
    # 2. Detect Unencoded Categoricals                                    #
    # ------------------------------------------------------------------ #
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        state_dict["unencoded_categoricals"] = cat_cols

    # ------------------------------------------------------------------ #
    # 3. Detect Heavy Skewness                                            #
    # ------------------------------------------------------------------ #
    if not numeric_df.empty:
        skewness = numeric_df.skew().dropna()
        high_skew = skewness[
            (skewness > cfg.skew_threshold) | (skewness < -cfg.skew_threshold)
        ]
        if not high_skew.empty:
            state_dict["high_skewness"] = {
                col: round(float(val), 4) for col, val in high_skew.items()
            }

    # ------------------------------------------------------------------ #
    # 4. Detect Scale Mismatches                                          #
    # ------------------------------------------------------------------ #
    if not numeric_df.empty:
        col_stats = {}
        for col in numeric_df.columns:
            col_min = float(numeric_df[col].min())
            col_max = float(numeric_df[col].max())
            col_range = col_max - col_min
            col_stats[col] = {
                "min": round(col_min, 4),
                "max": round(col_max, 4),
                "range": round(col_range, 4)
            }

        ranges = {col: stats["range"] for col, stats in col_stats.items()}
        min_range = min(ranges.values())

        if min_range > 1e-5:
            mismatched = {
                col: col_stats[col]
                for col, rng in ranges.items()
                if rng / min_range > cfg.scale_ratio
            }
            if mismatched:
                state_dict["scale_mismatch"] = mismatched

    # ------------------------------------------------------------------ #
    # 5. Detect Class Imbalance                                           #
    #    Checks all low-cardinality columns (<=20 unique values).        #
    #    Flags if the smallest class ratio < cfg.imbalance_threshold.    #
    # ------------------------------------------------------------------ #
    imbalance_report = {}
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        if n_unique < 2 or n_unique > 20:
            continue
        counts = df[col].value_counts(dropna=True)
        total = counts.sum()
        minority_count = int(counts.min())
        majority_count = int(counts.max())
        minority_ratio = minority_count / total if total > 0 else 0.0
        if minority_ratio < cfg.imbalance_threshold:
            imbalance_report[col] = {
                "majority_class": str(counts.idxmax()),
                "majority_count": majority_count,
                "minority_class": str(counts.idxmin()),
                "minority_count": minority_count,
                "minority_ratio": round(float(minority_ratio), 4),
            }
    if imbalance_report:
        state_dict["class_imbalance"] = imbalance_report

    # ------------------------------------------------------------------ #
    # 6. Detect Outliers (IQR method)                                    #
    #    Flags numeric columns where values fall beyond                  #
    #    Q1 - factor*IQR  or  Q3 + factor*IQR.                          #
    # ------------------------------------------------------------------ #
    if not numeric_df.empty:
        outlier_report = {}
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr < 1e-5:
                continue
            lower_fence = q1 - cfg.outlier_iqr_factor * iqr
            upper_fence = q3 + cfg.outlier_iqr_factor * iqr
            outlier_mask = (series < lower_fence) | (series > upper_fence)
            outlier_count = int(outlier_mask.sum())
            if outlier_count >= cfg.outlier_min_count:
                outlier_report[col] = {
                    "count": outlier_count,
                    "lower_fence": round(lower_fence, 4),
                    "upper_fence": round(upper_fence, 4),
                }
        if outlier_report:
            state_dict["outliers"] = outlier_report

    # ------------------------------------------------------------------ #
    # 7. Detect Duplicate Rows                                           #
    # ------------------------------------------------------------------ #
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        state_dict["duplicate_rows"] = duplicate_count

    # ------------------------------------------------------------------ #
    # 8. Detect Wrong Dtypes                                             #
    #    Finds object columns where >=90% of non-null values can be     #
    #    parsed as numeric — should be float/int, not string.           #
    # ------------------------------------------------------------------ #
    wrong_dtype_report = {}
    for col in df.select_dtypes(include=["object"]).columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        numeric_convertible = pd.to_numeric(series, errors="coerce").notna().sum()
        ratio = numeric_convertible / len(series)
        if ratio >= 0.9:
            sample_converted = pd.to_numeric(series, errors="coerce").dropna()
            suggested = "int" if (sample_converted % 1 == 0).all() else "float"
            wrong_dtype_report[col] = {
                "current_dtype": "object",
                "suggested_dtype": suggested,
                "convertible_ratio": round(float(ratio), 4),
            }
    if wrong_dtype_report:
        state_dict["wrong_dtypes"] = wrong_dtype_report

    return json.dumps(state_dict, indent=4)