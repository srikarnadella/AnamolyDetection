import streamlit as st
import pandas as pd
from detection.ensemble import detect_ensemble_outliers
import io

st.set_page_config(page_title="Finance Outlier Detector", layout="wide")
st.title("💼 Outlier Detection Toolkit")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your expense report (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preview section
    preview_rows = st.selectbox("Preview how many rows?", options=[5, 10, 20, 50, 100], index=0)
    st.write("### 🔍 Data Preview", df.head(preview_rows))

    # User input for ID column and voting strategy
    id_col = st.selectbox("Select unique ID column", options=df.columns, index=0)
    voting = st.radio("Voting Strategy", ["majority", "consensus"])

    # Detect outliers
    if st.button("🚨 Detect Outliers"):
        with st.spinner("Running ensemble detection..."):
            result_df = detect_ensemble_outliers(df, id_column=id_col, voting=voting)

            # Save in session for reuse
            st.session_state["result_df"] = result_df
            st.session_state["numeric_cols"] = df.select_dtypes(include="number").columns.tolist()
            st.session_state["id_col"] = id_col

            num_outliers = result_df["is_outlier"].sum()
            st.success(f"✅ Detected {num_outliers} outliers.")

            st.write("### 🧾 Detection Results (Outliers Only)")
            outliers = result_df[result_df["is_outlier"]]
            numeric_cols = st.session_state["numeric_cols"]
            display_cols = [id_col, "zscore_flag", "iqr_flag", "iso_flag", "vote_count"] + numeric_cols
            # Remove duplicates from display_cols while preserving order
            seen = set()
            unique_display_cols = [col for col in display_cols if not (col in seen or seen.add(col))]

            st.dataframe(outliers[unique_display_cols])

            # Export to Excel with metadata
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

            # Filter by category if exists
            st.write("### 🎛️ Filter Detected Outliers")
            if "category" in result_df.columns:
                categories = result_df["category"].dropna().unique().tolist()
                selected = st.multiselect("Filter by Category", categories, default=categories)
                filtered = result_df[result_df["category"].isin(selected) & result_df["is_outlier"]]
                st.dataframe(filtered)

# Reset logic
if st.button("🔄 Reset"):
    for key in ["result_df", "numeric_cols", "id_col"]:
        st.session_state.pop(key, None)
    st.experimental_rerun()
