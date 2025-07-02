import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

GROUND_TRUTH_LOG = {}

def save_dataframe_and_log(df, outliers_df, output_path, id_column, descriptions):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    GROUND_TRUTH_LOG[os.path.basename(output_path)] = {
        "id_column": id_column,
        "injected_ids": outliers_df[id_column].tolist(),
        "descriptions": descriptions
    }

def generate_employee_expense_reports(n=500, num_outliers=50, base_outlier_id=9999, output_path="data/employee_expense_reports.csv"):
    df = pd.DataFrame({
        "employee_id": np.random.randint(1000, 1100, n),
        "department": np.random.choice(["Sales", "HR", "Engineering", "Finance"], n),
        "expense_amount": np.random.exponential(scale=300, size=n).round(2),
        "category": np.random.choice(["Travel", "Meals", "Software", "Supplies"], n),
        "timestamp": [random_date(datetime(2024, 1, 1), datetime(2025, 1, 1)) for _ in range(n)]
    })

    outliers = []
    descriptions = []
    for i in range(num_outliers):
        eid = base_outlier_id + i
        if i % 3 == 0:
            row = [eid, "Sales", 17000 + i * 5, "Meals", datetime(2024, 12, 25, 3, 0)]
            desc = "Massive expense disguised as meal in Sales"
        elif i % 3 == 1:
            row = [eid, "Finance", 0.99, "Travel", datetime(2024, 5, 20, 2, 30)]
            desc = "Suspiciously tiny expense"
        else:
            row = [eid, "HR", 400, "Crypto Consulting", random_date(datetime(2024, 1, 1), datetime(2025, 1, 1))]
            desc = "Rare category never seen in dataset"
        outliers.append(row)
        descriptions.append(desc)

    outliers_df = pd.DataFrame(outliers, columns=df.columns)
    full_df = pd.concat([df, outliers_df], ignore_index=True)
    save_dataframe_and_log(full_df, outliers_df, output_path, "employee_id", descriptions)

def generate_corporate_card_logs(n=500, num_outliers=50, base_outlier_id=8888, output_path="data/corporate_card_logs.csv"):
    df = pd.DataFrame({
        "card_id": np.random.randint(2000, 3000, n),
        "employee_id": np.random.randint(1000, 1100, n),
        "merchant": np.random.choice(["Amazon", "Uber", "Airbnb", "Office Depot", "Staples"], n),
        "amount": np.abs(np.random.normal(200, 100, n)).round(2),
        "location": np.random.choice(["NY", "CA", "TX", "FL", "WA"], n),
        "timestamp": [random_date(datetime(2024, 1, 1), datetime(2025, 1, 1)) for _ in range(n)]
    })

    outliers = []
    descriptions = []
    for i in range(num_outliers):
        cid = base_outlier_id + i
        if i % 2 == 0:
            row = [cid, cid, "Uber", np.random.uniform(5000, 15000), "NY", datetime(2024, 11, 11, 2, 0)]
            desc = "High amount ride service in middle of night"
        else:
            row = [cid, cid, "DarkMarket", 420.69, "XX", datetime(2024, 6, 6, 6, 6)]
            desc = "Merchant and location unknown to normal logs"
        outliers.append(row)
        descriptions.append(desc)

    outliers_df = pd.DataFrame(outliers, columns=df.columns)
    full_df = pd.concat([df, outliers_df], ignore_index=True)
    save_dataframe_and_log(full_df, outliers_df, output_path, "employee_id", descriptions)

def generate_employee_productivity_logs(n=500, num_outliers=50, base_outlier_id=7777, output_path="data/employee_productivity_logs.csv"):
    df = pd.DataFrame({
        "employee_id": np.random.randint(1000, 1100, n),
        "tasks_completed": np.random.poisson(10, n),
        "hours_logged": np.random.normal(40, 5, n).round(1),
        "department": np.random.choice(["Sales", "HR", "Engineering", "Finance"], n),
        "efficiency_score": np.random.normal(75, 10, n).round(2)
    })

    outliers = []
    descriptions = []
    for i in range(num_outliers):
        eid = base_outlier_id + i
        if i % 4 == 0:
            row = [eid, 0, 90, "Engineering", 5]
            desc = "Worked 90 hrs with 0 tasks and terrible score"
        elif i % 4 == 1:
            row = [eid, 80, 10, "Finance", 99.9]
            desc = "Extreme overperformance in short time"
        elif i % 4 == 2:
            row = [eid, 5, 5, "AI Division", 99.9]
            desc = "Unknown department with perfect score"
        else:
            row = [eid, 1, 1, "Sales", -15]
            desc = "Negative efficiency score"
        outliers.append(row)
        descriptions.append(desc)

    outliers_df = pd.DataFrame(outliers, columns=df.columns)
    full_df = pd.concat([df, outliers_df], ignore_index=True)
    save_dataframe_and_log(full_df, outliers_df, output_path, "employee_id", descriptions)

def generate_login_audit_logs(n=500, num_outliers=50, base_outlier_id=6666, output_path="data/login_audit_logs.csv"):
    df = pd.DataFrame({
        "user_id": np.random.randint(1000, 1100, n),
        "ip_address": [f"192.168.{random.randint(0,255)}.{random.randint(0,255)}" for _ in range(n)],
        "login_time": [random_date(datetime(2024, 1, 1), datetime(2025, 1, 1)) for _ in range(n)],
        "auth_method": np.random.choice(["Password", "2FA", "SSO"], n),
        "failed_attempts": np.random.poisson(1, n)
    })

    outliers = []
    descriptions = []
    for i in range(num_outliers):
        uid = base_outlier_id + i
        if i % 3 == 0:
            ip = "10.0.0.1"
            fail = 35
            desc = "Too many failed attempts from internal IP"
        elif i % 3 == 1:
            ip = "0.0.0.0"
            fail = 0
            desc = "Login from null IP"
        else:
            ip = "255.255.255.255"
            fail = random.randint(20, 40)
            desc = "Broadcast IP with many failures"
        row = [uid, ip, datetime(2024, 10, 13, 2, 50), "Password", fail]
        outliers.append(row)
        descriptions.append(desc)

    outliers_df = pd.DataFrame(outliers, columns=df.columns)
    full_df = pd.concat([df, outliers_df], ignore_index=True)
    save_dataframe_and_log(full_df, outliers_df, output_path, "user_id", descriptions)

if __name__ == "__main__":
    generate_employee_expense_reports()
    generate_corporate_card_logs()
    generate_employee_productivity_logs()
    generate_login_audit_logs()

    with open("data/_ground_truth_log.json", "w") as f:
        json.dump(GROUND_TRUTH_LOG, f, indent=2)
