from detection.ml_based import detect_robust_isolation_forest_outliers
from detection.rule_based import detect_z_score_outliers, detect_iqr_outliers
import pandas as pd

def detect_ensemble_outliers(df, id_column="id", voting="majority"):
    z_df = detect_z_score_outliers(df, return_only_outliers=False, exclude_columns=[id_column])
    iqr_df = detect_iqr_outliers(df, return_only_outliers=False, exclude_columns=[id_column])
    iso_df = detect_robust_isolation_forest_outliers(
        df, exclude_columns=[id_column], return_only_outliers=False, id_column=id_column
    )

    #makes sure id columns are strings
    df_result = df.copy()
    df_result[id_column] = df_result[id_column].astype(str)
    z_df[id_column] = z_df[id_column].astype(str)
    iqr_df[id_column] = iqr_df[id_column].astype(str)
    iso_df[id_column] = iso_df[id_column].astype(str)

    df_result["zscore_flag"] = df_result[id_column].isin(z_df[z_df["is_outlier"]][id_column])
    df_result["iqr_flag"] = df_result[id_column].isin(iqr_df[iqr_df["is_outlier"]][id_column])
    iso_flag_df = iso_df[[id_column, "is_outlier"]].rename(columns={"is_outlier": "iso_flag"})

    df_result = df_result.merge(iso_flag_df, on=id_column, how="left")
    df_result["iso_flag"] = df_result["iso_flag"].fillna(False).astype(bool)

    df_result["vote_count"] = df_result[["zscore_flag", "iqr_flag", "iso_flag"]].sum(axis=1)

    #outlier decision making
    if voting == "consensus":
        df_result["is_outlier"] = df_result["vote_count"] == 3
    else:
        df_result["is_outlier"] = df_result["vote_count"] >= 2

    return df_result
