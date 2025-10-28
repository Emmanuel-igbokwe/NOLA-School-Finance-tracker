import pandas as pd
import streamlit as st
import plotly.express as px
import os

st.set_page_config(page_title="NOLA Financial Tracker", layout="wide")

# =========================
# Helper functions
# =========================
def sort_fy(x):
    try:
        parts = str(x).split()
        year = int(parts[0][2:]) if parts[0].startswith("FY") else 999
        q = int(parts[1][1:]) if len(parts) > 1 and parts[1].startswith("Q") else 9
        return (year, q)
    except:
        return (999, 9)

def load_excel_safe(path_candidates, **read_kwargs):
    for p in path_candidates:
        try:
            if os.path.exists(p):
                return pd.read_excel(p, engine="openpyxl", **read_kwargs)
        except Exception:
            pass
    return None

# =========================
# Load FY25 Dataset
# =========================
fy25_path = "FY25.xlsx"  # File in repo root
try:
    df = pd.read_excel(fy25_path, sheet_name="FY25", header=0)
except Exception as e:
    st.error(f"❌ Could not load {fy25_path}: {e}")
    st.stop()

df.columns = df.columns.str.strip()
df = df.dropna(subset=["Schools", "Fiscal Year"])
df["Fiscal Year"] = df["Fiscal Year"].astype(str).str.strip()

value_vars = [c for c in df.columns if c not in ["Schools", "Fiscal Year"]]
df_long = df.melt(id_vars=["Schools", "Fiscal Year"], value_vars=value_vars,
                  var_name="Metric", value_name="Value")
fiscal_options = sorted(df["Fiscal Year"].dropna().unique(), key=sort_fy)

# ===== Metric groups & descriptions (FY25)
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

# FY color map (FY26 reserved)
fy_color_map = {"FY22": "purple", "FY23": "red", "FY24": "blue", "FY25": "green", "FY26": "orange"}

# =========================
# Load FY26 Budget-to-Enrollment
# =========================
fy26_path = "Enrollment FY26.xlsx"  # File in repo root
try:
    df_budget_raw = pd.read_excel(fy26_path, sheet_name="FY26 Student enrollment", header=1)
except Exception as e:
    st.warning(f"⚠️ Could not load {fy26_path}: {e}")
    df_budget_raw = None


    expected_cols = ["Schools", "Fiscal Year", "Budgetted", "October 1 Count",
                     "Variance", "%Variance", "Budget to Enrollment Ratio"]
    df_budget_raw = df_budget_raw.dropna(subset=["Schools", "Fiscal Year"])

    df_budget_long = df_budget_raw.melt(
        id_vars=["Schools", "Fiscal Year"],
        value_vars=[c for c in expected_cols if c in df_budget_raw.columns],
        var_name="Metric",
        value_name="Value"
    )

    fiscal_options_budget = sorted(df_budget_long["Fiscal Year"].dropna().unique(), key=sort_fy)
    school_options_budget = sorted(df_budget_long["Schools"].dropna().unique())
else:
    st.warning("⚠️ FY26 file not found or sheet 'FY26 Student enrollment' missing.")
    df_budget_long = pd.DataFrame()
    fiscal_options_budget, school_options_budget = [], []

# === Metric color map for Budget-to-Enrollment (per metric, as requested)
budget_metric_color_map = {
    "Budgetted": "#1f77b4",              # blue
    "October 1 Count": "#2ca02c",        # green
    "Variance": "#d62728",               # red
    "%Variance": "#9467bd",              # purple
    "Budget to Enrollment Ratio": "#ff7f0e",  # orange
}

percent_metrics_budget = {"%Variance", "Budget to Enrollment Ratio"}

# =========================
# UI
# =========================
st.title("📊 NOLA Schools Financial Tracker")
st.markdown("<p style='font-size:14px;color:gray;'>Built by Emmanuel Igbokwe</p>", unsafe_allow_html=True)
st.sidebar.header("🔎 Filters")

modes = ["CSAF Metrics", "Other Metrics"]
if not df_budget_long.empty:
    modes.append("Budget to Enrollment")
metric_group = st.sidebar.radio("Choose Dashboard:", modes)
viz_type = st.sidebar.selectbox("📈 Visualization Type:", ["Bar Chart", "Line Chart"])

# =========================
# BUDGET TO ENROLLMENT
# =========================
if metric_group == "Budget to Enrollment":
    selected_schools = st.sidebar.multiselect("Select School(s):", school_options_budget)
    if st.sidebar.checkbox("Select All Budget Schools"):
        selected_schools = school_options_budget
    selected_fy = st.sidebar.multiselect("Select Fiscal Year(s):", fiscal_options_budget)
    if st.sidebar.checkbox("Select All Budget Fiscal Years"):
        selected_fy = fiscal_options_budget

    metrics_list = ["Budgetted", "October 1 Count", "Variance", "%Variance", "Budget to Enrollment Ratio"]
    metrics_list = [m for m in metrics_list if m in df_budget_long["Metric"].unique()]
    selected_metrics = st.sidebar.multiselect("Select Metrics:", metrics_list)

    df_f = df_budget_long[
        (df_budget_long["Schools"].isin(selected_schools)) &
        (df_budget_long["Fiscal Year"].isin(selected_fy)) &
        (df_budget_long["Metric"].isin(selected_metrics))
    ]

    if not df_f.empty:
        df_f = df_f.copy()
        df_f["sort_key"] = df_f["Fiscal Year"].apply(sort_fy)
        df_f = df_f.sort_values("sort_key")

        # If multiple schools, aggregate to a single line per metric
        if len(selected_schools) > 1:
            agg_fn = st.sidebar.selectbox("Aggregate Multiple Schools by:", ["Average", "Sum"])
            agg_func = "mean" if agg_fn == "Average" else "sum"
            df_f = df_f.groupby(["Fiscal Year", "Metric"], as_index=False).agg({"Value": agg_func})

        title = f"Budget to Enrollment Comparison ({', '.join(selected_metrics)})"

        if viz_type == "Line Chart":
            fig = px.line(
                df_f, x="Fiscal Year", y="Value",
                color="Metric", color_discrete_map=budget_metric_color_map,
                markers=True, title=title
            )
        else:
            fig = px.bar(
                df_f, x="Fiscal Year", y="Value",
                color="Metric", color_discrete_map=budget_metric_color_map,
                barmode="group", text="Value", title=title
            )
            # Per-trace text formatting (keep original decimals/units)
            # We'll adjust texttemplate by metric name
            for tr in fig.data:
                name = tr.name
                if name in percent_metrics_budget:
                    # If values look like fractions (<=1.2), show 0% style; else append % to raw numbers
                    subset = df_f[df_f["Metric"] == name]["Value"]
                    if subset.max() <= 1.2:
                        tr.texttemplate = "%{text:.0%}"
                    else:
                        tr.texttemplate = "%{text:,.2f}%"
                elif name in {"Budgetted", "October 1 Count", "Variance"}:
                    # Integers with commas
                    tr.texttemplate = "%{text:,.0f}"
                else:
                    tr.texttemplate = "%{text}"

            fig.update_traces(textposition="outside")

        # Axis formatting for percent metrics if ALL selected are percents and are fractional
        if selected_metrics and all(m in percent_metrics_budget for m in selected_metrics):
            if df_f["Value"].max() <= 1.2:
                fig.update_layout(yaxis_tickformat=".0%")

        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=600, legend_title="Metrics")
        st.plotly_chart(fig, use_container_width=True)

        # ---------- Data Table w/ formatted values ----------
        def fmt_budget(row):
            m, v = row["Metric"], row["Value"]
            try:
                if m in percent_metrics_budget:
                    # Match the logic used for chart: fraction vs whole-number percent
                    if df_f[df_f["Metric"] == m]["Value"].max() <= 1.2:
                        return f"{v:.0%}"
                    else:
                        return f"{v:,.2f}%"
                else:
                    return f"{v:,.0f}"
            except:
                return v

        df_show = df_f.copy()
        df_show["Value"] = df_show.apply(fmt_budget, axis=1)
        st.markdown("### 📋 Budget to Enrollment Data")
        st.dataframe(df_show, use_container_width=True)
    else:
        st.warning("⚠️ No Budget to Enrollment data matches your filters.")

# =========================
# FY25 (CSAF + Other) — with restored colors & formatting
# =========================
else:
    school_options = sorted(df_long["Schools"].dropna().unique())
    selected_schools = st.sidebar.multiselect("Select School(s):", school_options)
    if st.sidebar.checkbox("Select All Schools"):
        selected_schools = school_options

    selected_fy = st.sidebar.multiselect("Select Fiscal Year and Quarter:", fiscal_options)
    if st.sidebar.checkbox("Select All Fiscal Years"):
        selected_fy = fiscal_options

    if metric_group == "CSAF Metrics":
        selected_metrics = st.sidebar.multiselect("Select CSAF Metric(s):", csaf_metrics)
    else:
        selected_metrics = st.sidebar.multiselect("Select Other Metric(s):", other_metrics)

    filtered = df_long[
        (df_long["Schools"].isin(selected_schools)) &
        (df_long["Fiscal Year"].isin(selected_fy)) &
        (df_long["Metric"].isin(selected_metrics))
    ]

    if not filtered.empty:
        filtered = filtered.copy()
        filtered["sort_key"] = filtered["Fiscal Year"].apply(sort_fy)
        filtered = filtered.sort_values("sort_key")
        filtered["FY Group"] = filtered["Fiscal Year"].str.split().str[0]

        # Show CSAF description if a single metric
        if len(selected_metrics) == 1 and selected_metrics[0] in csaf_descriptions:
            desc = csaf_descriptions[selected_metrics[0]]["desc"]
            st.markdown(f"**{selected_metrics[0]}**<br><span style='font-size:13px'>{desc}</span>", unsafe_allow_html=True)
            chart_title = ""
        else:
            chart_title = f"{', '.join(selected_metrics)} across Fiscal Years"

        # Chart handling (as your original)
        if len(selected_schools) > 8:
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
        else:
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
                    color="FY Group", color_discrete_map=fy_color_map,
                    barmode="group", text="Value", title=chart_title, **facet_args
                )
            else:
                fig = px.line(
                    filtered, x="Fiscal Year", y="Value",
                    color="FY Group", color_discrete_map=fy_color_map,
                    markers=True, title=chart_title, **facet_args
                )

        # X order
        fiscal_order = filtered["Fiscal Year"].unique().tolist()
        fig.update_xaxes(categoryorder="array", categoryarray=fiscal_order)

        # Bar tweaks
        if viz_type == "Bar Chart":
            fig.update_traces(marker_line_width=0, width=0.6, textposition="outside")
            fig.update_layout(bargap=0.1, bargroupgap=0.05)

        # ===== Metric formatting (restored) =====
        if selected_metrics and all(m in dollar_metrics for m in selected_metrics):
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

        st.plotly_chart(fig, use_container_width=True)

        # Data Table (restored formatting)
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

        df_display = filtered.copy()
        df_display["Value"] = df_display.apply(lambda row: format_value(row["Value"], row["Metric"]), axis=1)
        st.markdown("### 📑 Data Table")
        st.dataframe(df_display, use_container_width=True)
    else:
        st.warning("⚠️ Welcome To Finance Accountability Real-Time Dashboard. Try Adjusting your Left filters.")

