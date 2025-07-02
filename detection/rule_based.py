import pandas as pd
import numpy as np

def detect_z_score_outliers(df, threshold=3.0, return_only_outliers=False, include_columns=None, exclude_columns=None):
    """
    Detect outliers in numeric columns using Z-score.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Z-score threshold (default=3.0).
        return_only_outliers (bool): If True, return only the outlier rows.
        include_columns (list): Columns to include in detection. If None, use all numeric columns.
        exclude_columns (list): Columns to exclude from detection.

    Returns:
        pd.DataFrame: DataFrame with `is_outlier` and `zscore_*` columns.
    """
    df = df.copy()
    df["is_outlier"] = False

    if include_columns is not None:
        numeric_cols = [col for col in include_columns if pd.api.types.is_numeric_dtype(df[col])]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if exclude_columns is not None:
        numeric_cols = [col for col in numeric_cols if col not in exclude_columns]

    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            df[f"zscore_{col}"] = 0
            continue
        z_scores = (df[col] - mean) / std
        df[f"zscore_{col}"] = z_scores
        df["is_outlier"] |= z_scores.abs() > threshold

    return df[df["is_outlier"]] if return_only_outliers else df


def detect_iqr_outliers(df, k=1.5, return_only_outliers=False, include_columns=None, exclude_columns=None):
    """
    Detect outliers using the IQR (Interquartile Range) method.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        k (float): Multiplier for IQR (default=1.5).
        return_only_outliers (bool): If True, return only the outlier rows.
        include_columns (list): Columns to include in detection. If None, use all numeric columns.
        exclude_columns (list): Columns to exclude from detection.

    Returns:
        pd.DataFrame: DataFrame with `is_outlier` and `iqr_flag_*` columns.
    """
    df = df.copy()
    df["is_outlier"] = False

    if include_columns is not None:
        numeric_cols = [col for col in include_columns if pd.api.types.is_numeric_dtype(df[col])]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if exclude_columns is not None:
        numeric_cols = [col for col in numeric_cols if col not in exclude_columns]

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        flag_col = f"iqr_flag_{col}"
        df[flag_col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        df["is_outlier"] |= df[flag_col]

    return df[df["is_outlier"]] if return_only_outliers else df
