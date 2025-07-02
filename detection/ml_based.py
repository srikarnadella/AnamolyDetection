import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def log_transform(x):
    x = np.where(x < -0.99, 0, x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.log1p(x)

def detect_robust_isolation_forest_outliers(
    df, exclude_columns=None, return_only_outliers=False, id_column="id", random_state=42
):
    exclude_columns = exclude_columns or []
    df_model = df.drop(columns=exclude_columns, errors="ignore").copy()

    for col in df_model.columns:
        if "time" in col.lower() or "date" in col.lower():
            parsed = pd.to_datetime(df_model[col], errors="coerce")
            if parsed.notna().sum() > 0:
                df_model[col] = parsed.astype("int64") // 10**9

    numeric_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_model.select_dtypes(include=["object", "category"]).columns.tolist()

    from sklearn.preprocessing import OneHotEncoder

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])


    num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(log_transform, validate=False)),
            ("scaler", StandardScaler()),
        ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols),
    ])

    

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, numeric_cols)
    ])

    X = preprocessor.fit_transform(df_model)

    model = IsolationForest(contamination=0.05, random_state=random_state)
    outlier_preds = model.fit_predict(X)

    df_result = df.copy()
    df_result["is_outlier"] = (outlier_preds == -1).astype(int)

    if return_only_outliers:
        return df_result[df_result["is_outlier"] == 1][id_column].tolist()

    return df_result
