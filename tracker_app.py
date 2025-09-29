# streamlit_app.py

import pandas as pd
import streamlit as st
import plotly.express as px

# --- Load only the 'FY25' sheet from the Excel file ---
df = pd.read_excel("FY25.xlsx", sheet_name="FY25", header=0, engine="openpyxl")

# --- Clean columns ---
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Schools", "Fiscal Year"])
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

fiscal_options = sorted(df["Fiscal Year"].dropna().unique(), key=sort_fy)

# --- Metric Groups ---
csaf_metrics = ["FB Ratio", "Liabilities to Assets", "Current Ratio", "Unrestricted Days COH"]

csaf_descriptions = {
    "FB Ratio": {
        "desc": "Fund Balance Ratio: Will an unforeseen event result in fiscal crisis?<br>"
                "Unrestricted Fund Balance ÷ Total Exp. (Best practice ≥ 10%)",
        "threshold": 0.10
    },
    "Liabilities to Assets": {
        "desc": "Liabilities to Assets Ratio: What % of Liabilities are financed by Assets?<br>"
                "A lower ratio is best. (Best practice ≤ 0.9)",
        "threshold": 0.90
    },
    "Current Ratio": {
        "desc": "Current Ratio: Can bills be paid? Positive numbers mean enough assets to pay bills.<br>"
                "Current Assets ÷ Current Liabilities (Best practice ≥ 1.5)",
        "threshold": 1.50
    },
    "Unrestricted Days COH": {
        "desc": "Unrestricted Cash on Hand: Enough cash to pay bills for 60+ days if no income?<br>"
                "Unrestricted Cash ÷ ((Total Exp. - Depreciation) ÷ 365) (Best practice ≥ 60)",
        "threshold": 60
    }
}

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

other_metrics = sorted([m for m in df_long["Metric"].dropna().unique() if m not in csaf_metrics])

# --- Streamlit UI ---
st.title("📊 NOLA Schools Financial Tracker")
st.markdown("<p style='font-size:14px; color:gray;'>Built by Emmanuel Igbokwe</p>", unsafe_allow_html=True)

# --- Sidebar Filters ---
st.sidebar.header("🔎 Filters")

# Schools
school_options = sorted(df_long["Schools"].dropna().unique())
selected_schools = st.sidebar.multiselect("Select School(s):", options=school_options)
if st.sidebar.checkbox("Select All Schools"):
    selected_schools = school_options

# Fiscal Years
selected_fy = st.sidebar.multiselect("Select Fiscal Year and Quarter:", options=fiscal_options)
if st.sidebar.checkbox("Select All Fiscal Years"):
    selected_fy = fiscal_options

# Metrics
metric_group = st.sidebar.radio("Choose Metric Group:", ["CSAF Metrics", "Other Metrics"])
if metric_group == "CSAF Metrics":
    selected_metrics = st.sidebar.multiselect("Select CSAF Metric(s):", options=csaf_metrics)
else:
    selected_metrics = st.sidebar.multiselect("Select Other Metric(s):", options=other_metrics)

# Visualization type
viz_type = st.sidebar.selectbox("📊 Select Visualization Type:", ["Bar Chart", "Line Chart"])

# --- Filter Data ---
filtered = df_long[
    (df_long["Schools"].isin(selected_schools)) &
    (df_long["Fiscal Year"].isin(selected_fy)) &
    (df_long["Metric"].isin(selected_metrics))
]

# --- Visualization ---
if not filtered.empty:
    filtered = filtered.copy()
    filtered["sort_key"] = filtered["Fiscal Year"].apply(sort_fy)
    filtered = filtered.sort_values("sort_key")
    filtered["FY Group"] = filtered["Fiscal Year"].str.split().str[0]

    color_map = {"FY22": "purple", "FY23": "red", "FY24": "blue", "FY25": "green"}

    # Show CSAF description
    if len(selected_metrics) == 1 and selected_metrics[0] in csaf_descriptions:
        desc = csaf_descriptions[selected_metrics[0]]["desc"]
        st.markdown(f"**{selected_metrics[0]}**<br><span style='font-size:13px'>{desc}</span>", unsafe_allow_html=True)
        chart_title = ""
    else:
        chart_title = f"{', '.join(selected_metrics)} across Fiscal Years"

    # --- Chart handling ---
    if len(selected_schools) > 8:  # ✅ Too many schools → group them in one chart
        if viz_type == "Bar Chart":
            fig = px.bar(
                filtered, x="Fiscal Year", y="Value",
                color="Schools", barmode="group", text="Value", title=chart_title
            )
        else:
            fig = px.line(
                filtered, x="Fiscal Year", y="Value",
                color="Schools", markers=True, title=chart_title
            )
        fig.update_xaxes(tickangle=45, automargin=True)

    else:  # ✅ Normal faceting if ≤ 8 schools
        facet_args = {}
        if len(selected_schools) > 1 and len(selected_metrics) > 1:
            facet_args = {"facet_row": "Schools", "facet_col": "Metric", "facet_col_wrap": 2}
        elif len(selected_schools) > 1:
            facet_args = {"facet_col": "Schools", "facet_col_wrap": 2}
        elif len(selected_metrics) > 1:
            facet_args = {"facet_col": "Metric", "facet_col_wrap": 2}

        if viz_type == "Bar Chart":
            fig = px.bar(
                filtered, x="Fiscal Year", y="Value",
                color="FY Group", color_discrete_map=color_map,
                barmode="group", text="Value", title=chart_title, **facet_args
            )
        else:
            fig = px.line(
                filtered, x="Fiscal Year", y="Value",
                color="FY Group", color_discrete_map=color_map,
                markers=True, title=chart_title, **facet_args
            )

    # --- Order axis ---
    fiscal_order = filtered["Fiscal Year"].unique().tolist()
    fig.update_xaxes(categoryorder="array", categoryarray=fiscal_order)

    # Bar chart tweaks
    if viz_type == "Bar Chart":
        fig.update_traces(marker_line_width=0, width=0.6, textposition="outside")
        fig.update_layout(bargap=0.1, bargroupgap=0.05)

    # --- Metric formatting ---
    if all(m in dollar_metrics for m in selected_metrics):
        fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",")
        if viz_type == "Bar Chart":
            fig.update_traces(texttemplate="$%{text:,.0f}")
    elif selected_metrics == ["FB Ratio"]:
        if viz_type == "Bar Chart":
            fig.update_traces(texttemplate="%{text:.0%}")
        fig.update_layout(yaxis_tickformat=".0%")
        fig.add_hline(y=csaf_descriptions["FB Ratio"]["threshold"], line_dash="dot", line_color="blue")
    elif selected_metrics == ["Liabilities to Assets"]:
        if viz_type == "Bar Chart":
            fig.update_traces(texttemplate="%{text:.2f}")
        fig.add_hline(y=csaf_descriptions["Liabilities to Assets"]["threshold"], line_dash="dot", line_color="blue")
    elif selected_metrics == ["Current Ratio"]:
        if viz_type == "Bar Chart":
            fig.update_traces(texttemplate="%{text:.2f}")
        fig.add_hline(y=csaf_descriptions["Current Ratio"]["threshold"], line_dash="dot", line_color="blue")
    elif selected_metrics == ["Unrestricted Days COH"]:
        if viz_type == "Bar Chart":
            fig.update_traces(texttemplate="%{text:,.0f}")
        fig.add_hline(y=csaf_descriptions["Unrestricted Days COH"]["threshold"], line_dash="dot", line_color="blue")
    else:
        if viz_type == "Bar Chart":
            fig.update_traces(texttemplate="%{text:,.0f}")

    # --- Show chart ---
    st.plotly_chart(fig, use_container_width=True)

    # --- Data Table ---
    df_display = filtered.copy()
    def format_value(val, metric):
        try:
            if metric in dollar_metrics:
                return f"${val:,.0f}"
            elif metric == "FB Ratio":
                return f"{val:.0%}"
            elif metric in ["Liabilities to Assets", "Current Ratio"]:
                return f"{val:.2f}"
            else:
                return f"{val:,.0f}"
        except:
            return val

    df_display["Value"] = df_display.apply(lambda row: format_value(row["Value"], row["Metric"]), axis=1)
    st.markdown("### 📑 Data Table")
    st.dataframe(df_display)

else:
    st.warning("⚠️ Welcome To Finance Accountability Real-Time Dashboard. Try Adjusting your Left filters.")
