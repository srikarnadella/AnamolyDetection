import os
import json
import pandas as pd
from detection.rule_based import detect_z_score_outliers, detect_iqr_outliers
from detection.ml_based import detect_robust_isolation_forest_outliers
from detection.ensemble import detect_ensemble_outliers

GROUND_TRUTH_PATH = "data/_ground_truth_log.json"

def load_ground_truth():
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)

def validate_dataset(file_path, ground_truth):
    file_name = os.path.basename(file_path)
    if file_name not in ground_truth:
        print(f"❌ Skipping {file_name} — no ground truth.")
        return

    df = pd.read_csv(file_path)
    true_outliers = set(ground_truth[file_name]["injected_ids"])
    id_column = ground_truth[file_name]["id_column"]

    print(f"\n📂 Validating: {file_name}")

    # Z-SCORE
    z_outliers = detect_z_score_outliers(df, exclude_columns=[id_column], return_only_outliers=True)
    z_ids = set(z_outliers[id_column])
    report_detection("Z-Score", z_ids, true_outliers)

    # IQR
    iqr_outliers = detect_iqr_outliers(df, exclude_columns=[id_column], return_only_outliers=True)
    iqr_ids = set(iqr_outliers[id_column])
    report_detection("IQR", iqr_ids, true_outliers)

    # ISOLATION FOREST
    iso_outliers = detect_robust_isolation_forest_outliers(
        df, 
        exclude_columns=[id_column], 
        return_only_outliers=True
    )
    iso_ids = set(iso_outliers[id_column])
    report_detection("Isolation Forest", iso_ids, true_outliers)

    # ENSEMBLE
    ensemble_outliers = detect_ensemble_outliers(
        df,
        id_column=id_column,
        exclude_columns=[id_column],
        return_only_outliers=True,
        voting="majority"  # "consensus" is stricter
    )

    ensemble_ids = set(ensemble_outliers[id_column])
    report_detection("Ensemble", ensemble_ids, true_outliers)

def report_detection(method_name, predicted_ids, true_ids):
    tp = len(predicted_ids & true_ids)
    total = len(true_ids)
    symbol = "✅" if tp == total else "⚠️"
    print(f"  {symbol} {method_name}: Detected {tp} / {total} injected outliers")
    print(f"     Detected IDs: {sorted(predicted_ids & true_ids)}")

if __name__ == "__main__":
    ground_truth = load_ground_truth()
    data_dir = "data"

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            validate_dataset(os.path.join(data_dir, file), ground_truth)
