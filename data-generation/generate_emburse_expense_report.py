import os
import json
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

GROUND_TRUTH_LOG = {}

def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

def save_dataframe_and_log(df, outliers_df, output_path, id_column, descriptions):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    GROUND_TRUTH_LOG[os.path.basename(output_path)] = {
        "id_column": id_column,
        "injected_ids": outliers_df[id_column].tolist(),
        "descriptions": descriptions
    }

def generate_emburse_expense_report(n=500, num_outliers=50, base_outlier_id=99900, output_path="data/emburse_expense_report.csv"):
    df = pd.DataFrame({
        "expense_id": np.arange(100000, 100000 + n),
        "employee_id": np.random.randint(1000, 1100, n),
        "department": np.random.choice(["Sales", "HR", "Engineering", "Finance"], n),
        "expense_date": [random_date(datetime(2024, 1, 1), datetime(2025, 1, 1)).date() for _ in range(n)],
        "merchant": np.random.choice(["Delta Airlines", "Hilton", "Uber", "Amazon", "WeWork"], n),
        "expense_amount": np.abs(np.random.normal(250, 100, n)).round(2),
        "currency": "USD",
        "category": np.random.choice(["Travel", "Lodging", "Meals", "Supplies", "Software"], n),
        "payment_type": np.random.choice(["Corporate Card", "Reimbursed"], n),
        "project_code": np.random.choice(["CLIENT001", "CLIENT002", "INTERNAL"], n),
        "notes": np.random.choice(["Client meeting", "Quarterly planning", "Office supplies", "Conference trip"], n)
    })

    outliers = []
    descriptions = []

    for i in range(num_outliers):
        eid = base_outlier_id + i
        if i % 3 == 0:
            row = [eid, 9999, "Sales", datetime(2024, 12, 30).date(), "Uber", 9000.00, "USD", "Travel", "Corporate Card", "CLIENT001", "Luxury ride to airport"]
            desc = "Excessively high ride fare"
        elif i % 3 == 1:
            row = [eid, 9998, "HR", datetime(2024, 2, 15).date(), "DarkWeb VPN", 49.99, "USD", "Software", "Reimbursed", "INTERNAL", "Unusual vendor"]
            desc = "Suspicious merchant for VPN service"
        else:
            row = [eid, 9997, "Finance", datetime(2024, 6, 1).date(), "Hilton", 0.01, "USD", "Lodging", "Corporate Card", "CLIENT002", "Invalid minimal hotel stay"]
            desc = "Suspiciously small amount for lodging"
        outliers.append(row)
        descriptions.append(desc)

    outliers_df = pd.DataFrame(outliers, columns=df.columns)
    full_df = pd.concat([df, outliers_df], ignore_index=True)

    save_dataframe_and_log(full_df, outliers_df, output_path, "expense_id", descriptions)

    with open("data/_ground_truth_log.json", "w") as f:
        json.dump(GROUND_TRUTH_LOG, f, indent=2)

# Example usage
if __name__ == "__main__":
    generate_emburse_expense_report()
