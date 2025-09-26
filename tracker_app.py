# streamlit_app.py

import pandas as pd
import streamlit as st
import plotly.express as px

# --- Load only the 'FY25' sheet from the Excel file ---
df = pd.read_excel("FY25.xlsx", sheet_name="FY25", header=0, engine="openpyxl")

# --- Clean columns ---
df.columns = df.columns.str.strip()

# Drop rows with missing Schools or Fiscal Year
df = df.dropna(subset=["Schools", "Fiscal Year"])

# Ensure Fiscal Year is string and clean spaces
df["Fiscal Year"] = df["Fiscal Year"].astype(str).str.strip()

# --- Melt data into long format ---
value_vars = [c for c in df.columns if c not in ["Schools", "Fiscal Year"]]
df_long = df.melt(
    id_vars=["Schools", "Fiscal Year"],
    value_vars=value_vars,
    var_name="Metric",
    value_name="Value"
)

# --- Sorting Fiscal Year/Quarter properly ---
def sort_fy(x):
    try:
        year = int(x.split()[0][2:])   # FY23 -> 23
        quarter = int(x.split()[1][1:])  # Q1 -> 1
        return (year, quarter)
    except:
        return (999, 9)

# Collect all Fiscal Year/Quarter values BEFORE filtering
fiscal_options = sorted(df["Fiscal Year"].dropna().unique(), key=sort_fy)

# --- Streamlit App Title ---
st.title("📊 NOLA Schools Financial Tracker")
st.markdown("### Built by Emmanuel Igbokwe")

# --- Sidebar Filters ---
st.sidebar.header("🔎 Filters")

# Schools
school_options = sorted(df_long["Schools"].dropna().unique())
selected_schools = st.sidebar.multiselect("Select School(s):", school_options)
if st.sidebar.checkbox("Select All Schools"):
    selected_schools = school_options

# Fiscal Year + Quarter
selected_fy = st.sidebar.multiselect(
    "Select Fiscal Year and Quarter:",
    fiscal_options,
    default=fiscal_options
)

# Metrics
metric_options = sorted(df_long["Metric"].dropna().unique())
selected_metric = st.sidebar.selectbox("Select Metric:", metric_options)

# --- Filtered Data ---
filtered = df_long[
    (df_long["Schools"].isin(selected_schools)) &
    (df_long["Fiscal Year"].isin(selected_fy)) &
    (df_long["Metric"] == selected_metric)
]

# --- Visualization ---
if not filtered.empty:
    # Extract FY group (FY22, FY23...) for coloring
    filtered["FY Group"] = filtered["Fiscal Year"].str.split().str[0]

    fig = px.bar(
        filtered,
        x="Fiscal Year",
        y="Value",
        color="FY Group",
        barmode="group",
        text="Value",
        title=f"{selected_metric} across Fiscal Years"
    )

    fig.update_traces(
        marker_line_width=0,
        width=0.6,
        textposition="outside"
    )
    fig.update_layout(
        bargap=0.1,
        bargroupgap=0.05
    )

    # ✅ Metrics that need $
    dollar_metrics = [
        "Restricted Cash", "Unrestricted Cash & Equivalents",
        "Current Assets", "Fixed Assets", "Total Assets",
        "Current Liabilities", "Long term liabilities", "Total Liabilities",
        "Unrestricted FB", "Restricted FB", "FB",
        "Total Expenses", "Depreciation", "Accu Depreciatn",
        "Local Revenue", "State Rev", "Federa Rev", "Total Revenue",
        "Salaries", "Employee Benefits", "Purchased professional",
        "Purchased Property", "Other Purchased", "Supplies",
        "Property", "Other Objects", "Other Uses of Fund"
    ]

    if selected_metric in dollar_metrics:
        fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",")
        fig.update_traces(texttemplate="$%{text:,.0f}")
    elif selected_metric in ["FB Ratio", "Liabilities to Assets", "Current Ratio"]:
        fig.update_traces(texttemplate="%{text:.1%}")
    else:
        fig.update_traces(texttemplate="%{text:,.0f}")

    st.plotly_chart(fig, use_container_width=True)

    # --- Format table ---
    df_display = filtered.copy()
    if selected_metric in dollar_metrics:
        df_display["Value"] = df_display["Value"].apply(lambda x: f"${x:,.0f}")
    elif selected_metric in ["FB Ratio", "Liabilities to Assets", "Current Ratio"]:
        df_display["Value"] = df_display["Value"].apply(lambda x: f"{x:.1%}")
    else:
        df_display["Value"] = df_display["Value"].apply(lambda x: f"{x:,.0f}")

    st.dataframe(df_display)

else:
    st.warning("⚠️ No data matches your selection. Try adjusting filters.")
