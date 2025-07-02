import streamlit as st
import pandas as pd
from detection.ensemble import detect_ensemble_outliers
import io

st.set_page_config(page_title="Finance Outlier Detector", layout="wide")
st.title("💼 Outlier Detection Dashboard")

#CSV and excel input
uploaded_file = st.file_uploader("📤 Upload your expense report (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    preview_rows = st.selectbox("🔍 Preview how many rows?", options=[5, 10, 20, 50, 100], index=0)
    st.write("### 📄 Data Preview", df.head(preview_rows))

    id_col = st.selectbox("🆔 Select unique ID column", options=df.columns, index=0)
    voting = st.radio("🗳️ Voting Strategy", ["majority", "consensus"], horizontal=True)

    if st.button("🚨 Detect Outliers"):
        with st.spinner("Running ensemble detection..."):
            result_df = detect_ensemble_outliers(df, id_column=id_col, voting=voting)

            #define numeric columns before using them
            numeric_cols = df.select_dtypes(include="number").columns.tolist()

            num_outliers = result_df["is_outlier"].sum()
            st.success(f"✅ Detected {num_outliers} outliers.")

            st.write("### 🧾 Detection Results (Outliers Only)")
            outliers = result_df[result_df["is_outlier"]]
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            raw_cols = [id_col, "zscore_flag", "iqr_flag", "iso_flag", "vote_count", "is_outlier"] + numeric_cols
            display_cols = list(dict.fromkeys([col for col in raw_cols if col in outliers.columns]))
            st.dataframe(outliers[display_cols])


            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                result_df.to_excel(writer, index=False, sheet_name="OutlierResults")
                summary = pd.DataFrame({
                    "Voting Strategy": [voting],
                    "Total Rows": [len(result_df)],
                    "Outliers Found": [num_outliers]
                })
                summary.to_excel(writer, sheet_name="Summary", index=False)

            st.download_button(
                label="📥 Download Outlier Report (Excel)",
                data=output.getvalue(),
                file_name="outlier_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
