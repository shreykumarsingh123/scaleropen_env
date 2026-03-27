import pandas as pd
import numpy as np
import json


def get_state(df: pd.DataFrame) -> str:
    """
    Analyzes a Pandas DataFrame and returns a Data Health JSON string.
    This acts as the 'Observation' for the AI Agent in the OpenEnv loop.

    Returns a JSON string with the following possible keys:
        - shape               : {rows, columns} of the DataFrame
        - missing_values      : {col: count} for columns with nulls
        - unencoded_categoricals : [list of object/category columns]
        - high_skewness       : {col: skew_value} for |skew| > 3.0
        - scale_mismatch      : {col: {"min": x, "max": y, "range": z}}
                                for columns whose range is 1000x larger
                                than the smallest-range numeric column
    """
    state_dict = {}

    # ------------------------------------------------------------------ #
    # 0. DataFrame Shape — helps agent reason about data size             #
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
        # Cast to plain int so JSON serializer doesn't complain
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
    # 3. Detect Heavy Skewness  (threshold: |skew| > 3.0)                #
    # ------------------------------------------------------------------ #
    numeric_df = df.select_dtypes(include=[np.number])

    if not numeric_df.empty:
        skewness = numeric_df.skew().dropna()
        high_skew = skewness[(skewness > 3.0) | (skewness < -3.0)]
        if not high_skew.empty:
            state_dict["high_skewness"] = {
                col: round(float(val), 4) for col, val in high_skew.items()
            }

    # ------------------------------------------------------------------ #
    # 4. Detect Scale Mismatches                                          #
    #                                                                     #
    # Fix from v1: instead of comparing global max/min (which breaks on  #
    # negative values), we compute the range (max - min) per column and  #
    # flag any column whose range is > 1000x the smallest observed range. #
    # This correctly identifies which columns are the outliers in scale.  #
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

        # Avoid division by zero for constant columns
        if min_range > 1e-5:
            mismatched = {
                col: col_stats[col]
                for col, rng in ranges.items()
                if rng / min_range > 1000
            }
            if mismatched:
                state_dict["scale_mismatch"] = mismatched

    return json.dumps(state_dict, indent=4)