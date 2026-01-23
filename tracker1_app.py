import pandas as pd
import streamlit as st
import plotly.express as px
import os
import numpy as np
import re
from sklearn.linear_model import HuberRegressor, LinearRegression
import base64

# =========================
# PAGE CONFIG + HEADER
# =========================
st.set_page_config(page_title="NOLA Financial Tracker", layout="wide")

logo_path = "nola_parish_logo.png"
col1, col2 = st.columns([1, 5])

with col1:
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            encoded_logo = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <div style="text-align:center;">
                <img src="data:image/png;base64,{encoded_logo}" width="100"
                     style="animation: spin 5s linear infinite; border-radius: 50%;">
            </div>
            <style>
                @keyframes spin {{
                    from {{ transform: rotate(0deg); }}
                    to {{ transform: rotate(360deg); }}
                }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Logo not found in app directory.")

with col2:
    st.markdown(
        """
        <h1 style="
            color:#003366;
            font-size:28px;
            margin-bottom:0;
            animation: moveText 3s infinite alternate;">
            Welcome to NOLA Public Schools Finance Accountability App
        </h1>
        <style>
            @keyframes moveText {
                from {transform: translateX(0px);}
                to {transform: translateX(15px);}
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# =========================
# HELPERS
# =========================
def sort_fy(x):
    """For labels like 'FY25 Q1', returns (25, 1)."""
    try:
        s = str(x).strip().upper()
        m = re.search(r"FY\s*(\d{2,4})", s)
        y = int(m.group(1)) % 100 if m else 999
        qm = re.search(r"Q\s*(\d)", s)
        q = int(qm.group(1)) if qm else 9
        return (y, q)
    except Exception:
        return (999, 9)

def sort_fy_only(x):
    """For labels like FY22 / FY 2022 / FY22 Q1 -> 22."""
    try:
        s = str(x).replace("\u00A0", " ").replace("\n", " ").replace("\r", " ")
        s = s.strip().upper()
        m = re.search(r"FY\s*(\d{2,4})", s)
        if not m:
            return 999
        return int(m.group(1)) % 100
    except Exception:
        return 999

def standardize_fy_label(x):
    y = sort_fy_only(x)
    return f"FY{y:02d}" if y != 999 else str(x).strip()

def normalize_col(c):
    return re.sub(r"\s+", " ", str(c).strip()).lower()

def clean_series(y):
    return pd.to_numeric(pd.Series(y), errors="coerce").values.astype(float)

# =========================
# LOAD FY25 (CSAF + OTHER)
# =========================
fy25_path = "FY25.xlsx"
try:
    df = pd.read_excel(fy25_path, sheet_name="FY25", header=0)
except Exception as e:
    st.error(f"❌ Could not load {fy25_path}: {e}")
    st.stop()

df.columns = df.columns.str.strip()
df = df.dropna(subset=["Schools", "Fiscal Year"]).copy()
df["Fiscal Year"] = df["Fiscal Year"].astype(str).str.strip()

value_vars = [c for c in df.columns if c not in ["Schools", "Fiscal Year"]]
df_long = df.melt(
    id_vars=["Schools", "Fiscal Year"],
    value_vars=value_vars,
    var_name="Metric",
    value_name="Value"
)

fiscal_options = sorted(df["Fiscal Year"].dropna().unique(), key=sort_fy)

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

fy_color_map = {"FY22": "purple", "FY23": "red", "FY24": "blue", "FY25": "green", "FY26": "orange"}

# =========================
# ✅ ROBUST LOADER: Enrollment FY26.xlsx
# =========================
def load_budget_enrollment_excel(
    file_path: str,
    preferred_sheet: str = "FY26 Student enrollment",
    header_candidates=(0, 1, 2, 3, 4, 5, 6),
):
    """
    Loads Enrollment FY26.xlsx even if:
    - sheet name is slightly different
    - header row is not 0/1
    - columns are named differently

    Returns: (df_budget_long, fiscal_options_budget, school_options_budget)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}. Put it in the repo root (same folder as this app)."
        )

    xl = pd.ExcelFile(file_path, engine="openpyxl")
    sheet_names = xl.sheet_names

    # Pick sheet: exact match OR closest contains "student" and "enroll"
    if preferred_sheet in sheet_names:
        sheet = preferred_sheet
    else:
        candidates = []
        for s in sheet_names:
            sn = s.lower()
            score = 0
            if "enroll" in sn:
                score += 2
            if "student" in sn:
                score += 2
            if "fy26" in sn or "26" in sn:
                score += 1
            candidates.append((score, s))
        candidates.sort(reverse=True)
        sheet = candidates[0][1] if candidates else sheet_names[0]

    # Try multiple header rows until we detect School + FY columns
    df_raw = None
    used_header = None
    for hdr in header_candidates:
        tmp = pd.read_excel(file_path, sheet_name=sheet, header=hdr, engine="openpyxl")
        tmp.columns = [str(c).strip() for c in tmp.columns]
        cols_norm = [normalize_col(c) for c in tmp.columns]
        has_school = any("school" in c for c in cols_norm)
        has_fy = any(("fiscal" in c and "year" in c) or c in ["fy", "fiscal year"] or "fiscal year" in c for c in cols_norm)
        if has_school and has_fy:
            df_raw = tmp
            used_header = hdr
            break

    if df_raw is None:
        # last attempt: header=None then promote a row (rare files)
        tmp = pd.read_excel(file_path, sheet_name=sheet, header=None, engine="openpyxl")
        # find a row that contains "school" and "fiscal"
        best_row = None
        for i in range(min(30, len(tmp))):
            row = tmp.iloc[i].astype(str).str.lower().tolist()
            if any("school" in v for v in row) and any("fiscal" in v and "year" in v for v in row):
                best_row = i
                break
        if best_row is None:
            raise ValueError(
                f"Could not detect header row in sheet '{sheet}'. "
                f"Sheets found: {sheet_names}"
            )
        df_raw = pd.read_excel(file_path, sheet_name=sheet, header=best_row, engine="openpyxl")
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        used_header = best_row

    # Rename columns (flexible)
    rename_map = {}
    for c in df_raw.columns:
        cn = normalize_col(c)

        # Schools
        if cn in {"school", "schools", "campus", "site"} or "school" in cn:
            rename_map[c] = "Schools"

        # Fiscal Year
        if cn in {"fy", "fiscal year", "fiscal_year"} or ("fiscal" in cn and "year" in cn):
            rename_map[c] = "Fiscal Year"

        # Budgeted
        if ("budget" in cn) and ("enroll" not in cn) and ("ratio" not in cn) and ("%" not in cn):
            rename_map[c] = "Budgetted"

        # Oct count
        if ("oct" in cn) and (("count" in cn) or ("enroll" in cn) or ("enrollment" in cn)):
            rename_map[c] = "October 1 Count"

        # Feb count
        if ("feb" in cn or "february" in cn) and (("count" in cn) or ("enroll" in cn) or ("enrollment" in cn)):
            rename_map[c] = "February 1 Count"

        # Ratio
        if ("budget" in cn) and ("enroll" in cn) and ("ratio" in cn or "%" in cn):
            rename_map[c] = "Budget to Enrollment Ratio"

    df_raw = df_raw.rename(columns=rename_map)

    # Drop typical junk columns
    for junk in ["CMO", "Network", "Notes"]:
        if junk in df_raw.columns:
            df_raw = df_raw.drop(columns=[junk])

    required = ["Schools", "Fiscal Year"]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        # give debug info in Streamlit and fail with details
        raise ValueError(
            f"Loaded sheet '{sheet}' (header={used_header}) but missing required columns: {missing}. "
            f"Columns detected: {list(df_raw.columns)}"
        )

    # Standardize FY label + numeric conversion
    df_raw = df_raw.dropna(subset=["Schools", "Fiscal Year"]).copy()
    df_raw["Schools"] = df_raw["Schools"].astype(str).str.strip()
    df_raw["FY"] = df_raw["Fiscal Year"].apply(standardize_fy_label)

    metric_cols = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    for c in metric_cols:
        if c in df_raw.columns:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

    # IMPORTANT: do NOT convert blanks to 0; also avoid zero-lines if your file uses 0 for missing
    for c in metric_cols:
        if c in df_raw.columns:
            df_raw.loc[df_raw[c] <= 0, c] = np.nan

    # Long format
    use_cols = [c for c in metric_cols if c in df_raw.columns]
    df_long_budget = df_raw.melt(
        id_vars=["Schools", "FY"],
        value_vars=use_cols,
        var_name="Metric",
        value_name="Value"
    ).dropna(subset=["Value"]).copy()

    fiscal_opts = sorted(df_long_budget["FY"].dropna().unique(), key=sort_fy_only)
    school_opts = sorted(df_long_budget["Schools"].dropna().unique())

    return df_long_budget, fiscal_opts, school_opts, sheet, used_header, list(df_raw.columns)

# ---- Load it (with a nice debug panel)
fy26_path = "Enrollment FY26.xlsx"
df_budget_long = pd.DataFrame()
fiscal_options_budget, school_options_budget = [], []
_budget_debug = None

try:
    df_budget_long, fiscal_options_budget, school_options_budget, sheet_used, header_used, cols_used = load_budget_enrollment_excel(
        fy26_path,
        preferred_sheet="FY26 Student enrollment",
    )
    _budget_debug = {
        "file": fy26_path,
        "sheet_used": sheet_used,
        "header_used": header_used,
        "columns_after_rename": cols_used,
        "rows_long": int(len(df_budget_long)),
        "fiscal_options_budget": fiscal_options_budget[:20],
    }
except Exception as e:
    st.error(f"❌ Enrollment file failed to load: {e}")
    # show file list so you can confirm name on Streamlit Cloud
    st.write("📁 Files in app folder:", sorted(os.listdir(".")))
    st.stop()

budget_metric_color_map = {
    "Budgetted": "#1f77b4",
    "October 1 Count": "#2ca02c",
    "February 1 Count": "#d62728",
    "Budget to Enrollment Ratio": "#ff7f0e",
}

# =========================
# UI
# =========================
st.title("📊 NOLA Schools Financial Tracker")
st.markdown("<p style='font-size:14px;color:gray;'>Built by Emmanuel Igbokwe</p>", unsafe_allow_html=True)
st.sidebar.header("🔎 Filters")

with st.expander("✅ Enrollment FY26 Loader Debug (open if something looks wrong)", expanded=False):
    st.json(_budget_debug)

modes = ["CSAF Metrics", "CSAF Predicted", "Other Metrics"]
if not df_budget_long.empty:
    modes += ["Budget to Enrollment", "Budget to Enrollment Predicted"]

metric_group = st.sidebar.radio("Choose Dashboard:", modes)
viz_type = st.sidebar.selectbox("📈 Visualization Type:", ["Bar Chart", "Line Chart"])

smooth_lines = st.sidebar.checkbox("✨ Smooth lines (Budget charts)", value=True)
show_line_labels = st.sidebar.checkbox("🔤 Show point labels on Line Chart", value=False)

# =========================
# CSAF Predicted (keep your existing CSAF Predicted block here)
# =========================
# NOTE: I’m not rewriting CSAF Predicted again to keep this response focused on fixing Enrollment loading.
# Paste your CSAF Predicted section exactly as you already have it.

# =========================
# BUDGET TO ENROLLMENT (COMPARISON)
# =========================
if metric_group == "Budget to Enrollment":
    selected_schools = st.sidebar.multiselect("Select School(s):", school_options_budget, default=school_options_budget[:1])
    if st.sidebar.checkbox("Select All Budget Schools"):
        selected_schools = school_options_budget

    selected_fy = st.sidebar.multiselect("Select Fiscal Year(s):", fiscal_options_budget, default=fiscal_options_budget)
    if st.sidebar.checkbox("Select All Budget Fiscal Years"):
        selected_fy = fiscal_options_budget

    metrics_list = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    metrics_list = [m for m in metrics_list if m in df_budget_long["Metric"].unique()]
    selected_metrics = st.sidebar.multiselect("Select Metrics:", metrics_list, default=metrics_list)

    df_f = df_budget_long[
        (df_budget_long["Schools"].isin(selected_schools)) &
        (df_budget_long["FY"].isin(selected_fy)) &
        (df_budget_long["Metric"].isin(selected_metrics))
    ].copy()

    if df_f.empty:
        st.warning("⚠️ No Budget to Enrollment data matches your filters.")
        st.stop()

    df_f["sort_key"] = df_f["FY"].apply(sort_fy_only)
    df_f = df_f.sort_values("sort_key")
    fy_order = df_f["FY"].unique().tolist()

    title = f"Budget to Enrollment Comparison — {', '.join(selected_metrics)}"

    if viz_type == "Line Chart":
        fig = px.line(
            df_f,
            x="FY", y="Value",
            color="Metric",
            color_discrete_map=budget_metric_color_map,
            markers=True,
            facet_col="Schools",
            facet_col_wrap=2,
            title=title,
            line_shape="spline" if smooth_lines else "linear",
        )
        fig.update_traces(connectgaps=False)
        if show_line_labels:
            for tr in fig.data:
                met = tr.name
                if met == "Budget to Enrollment Ratio":
                    tr.texttemplate = "%{y:.0%}"
                else:
                    tr.texttemplate = "%{y:,.0f}"
            fig.update_traces(textposition="top center")
        else:
            fig.update_traces(text=None)

    else:
        fig = px.bar(
            df_f,
            x="FY", y="Value",
            color="Metric",
            color_discrete_map=budget_metric_color_map,
            barmode="group",
            text="Value",
            facet_col="Schools",
            facet_col_wrap=2,
            title=title
        )
        for tr in fig.data:
            met = tr.name
            if met == "Budget to Enrollment Ratio":
                tr.texttemplate = "%{text:.0%}"
            else:
                tr.texttemplate = "%{text:,.0f}"
        fig.update_traces(textposition="outside")

    fig.update_xaxes(categoryorder="array", categoryarray=fy_order, tickangle=45)
    fig.update_layout(height=700, legend_title="Metric", title_x=0.5, bargap=0.15, bargroupgap=0.05)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# BUDGET TO ENROLLMENT PREDICTED (multi-metric frozen)
# =========================
elif metric_group == "Budget to Enrollment Predicted":
    st.markdown("## 🔮 Budget to Enrollment Predicted (Frozen Forecast)")

    schools_b = sorted(df_budget_long["Schools"].dropna().unique())
    selected_school_b = st.sidebar.selectbox("🏫 Select School:", schools_b)

    metrics_all = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    metrics_all = [m for m in metrics_all if m in df_budget_long["Metric"].unique()]
    selected_metrics_b = st.sidebar.multiselect("📊 Choose Metric(s):", metrics_all, default=metrics_all)

    fiscal_years_b = sorted(df_budget_long["FY"].dropna().unique(), key=sort_fy_only)
    selected_fy_hist_b = st.sidebar.multiselect("📅 History FY (training):", fiscal_years_b, default=fiscal_years_b)

    train_through_b = st.sidebar.selectbox("🧊 Forecast Origin (freeze at):", fiscal_years_b, index=max(0, len(fiscal_years_b) - 1))

    forecast_method_b = st.sidebar.selectbox(
        "🧠 Forecast Method",
        ["Ensemble (Robust Trend + Linear Trend)", "Robust Trend (Huber log1p)", "Linear Trend (log1p)"],
        index=0
    )

    n_future_b = st.sidebar.slider("🔮 Forecast horizon (years)", 1, 6, 3)
    viz_type_b = st.sidebar.selectbox("📈 Budget Prediction Chart Type:", ["Line Chart", "Bar Chart"])
    run_pred_b = st.sidebar.button("▶ Run Budget Prediction")

    if not run_pred_b:
        st.info("Pick School + Metrics + History + Origin + Method, then click **Run Budget Prediction**.")
        st.stop()

    if not selected_metrics_b:
        st.warning("⚠️ Select at least one metric.")
        st.stop()

    def forecast_series(values, n_future):
        y = clean_series(values)
        y = y[~np.isnan(y)]
        if len(y) < 3:
            return None

        t = np.arange(len(y)).reshape(-1, 1)
        y_log = np.log1p(np.clip(y, 0, None))

        def pred_huber():
            model = HuberRegressor().fit(t, y_log)
            tf = np.arange(len(y_log), len(y_log) + n_future).reshape(-1, 1)
            return np.expm1(model.predict(tf))

        def pred_linear():
            model = LinearRegression().fit(t, y_log)
            tf = np.arange(len(y_log), len(y_log) + n_future).reshape(-1, 1)
            return np.expm1(model.predict(tf))

        if forecast_method_b.startswith("Ensemble"):
            p1 = pred_huber()
            p2 = pred_linear()
            return np.clip(0.5 * p1 + 0.5 * p2, 0, None)
        if forecast_method_b.startswith("Robust"):
            return np.clip(pred_huber(), 0, None)
        return np.clip(pred_linear(), 0, None)

    origin_year = sort_fy_only(train_through_b)
    future_labels = [f"FY{origin_year + i:02d}" for i in range(1, n_future_b + 1)]

    st.session_state.setdefault("forecast_store_budget", {})

    actual_now = df_budget_long[
        (df_budget_long["Schools"] == selected_school_b) &
        (df_budget_long["Metric"].isin(selected_metrics_b))
    ].copy()
    actual_now["sort_key"] = actual_now["FY"].apply(sort_fy_only)
    actual_now = actual_now.sort_values("sort_key")
    actual_now = actual_now.rename(columns={"FY": "Period"})
    actual_now["Type"] = "Actual"

    frozen_frames = []
    for met in selected_metrics_b:
        dfh = df_budget_long[
            (df_budget_long["Schools"] == selected_school_b) &
            (df_budget_long["FY"].isin(selected_fy_hist_b)) &
            (df_budget_long["Metric"] == met)
        ].copy()
        dfh["sort_key"] = dfh["FY"].apply(sort_fy_only)
        dfh = dfh[dfh["sort_key"] <= origin_year].sort_values("sort_key")

        key = f"BUDGET__{selected_school_b}__{met}__{train_through_b}__{forecast_method_b}__{n_future_b}"
        if key not in st.session_state["forecast_store_budget"]:
            y_future = forecast_series(dfh["Value"].values, n_future_b)
            if y_future is None:
                continue
            st.session_state["forecast_store_budget"][key] = pd.DataFrame({
                "Period": future_labels,
                "Value": y_future,
                "Metric": met,
                "Type": "Forecast (Frozen)"
            })
        frozen_frames.append(st.session_state["forecast_store_budget"][key].copy())

    if not frozen_frames:
        st.warning("⚠️ Not enough data to forecast the selected metric(s). Need ≥ 3 points per metric.")
        st.stop()

    frozen_all = pd.concat(frozen_frames, ignore_index=True)
    combined = pd.concat(
        [actual_now[["Period", "Value", "Metric", "Type"]], frozen_all[["Period", "Value", "Metric", "Type"]]],
        ignore_index=True
    )
    combined["sort_key"] = combined["Period"].apply(sort_fy_only)
    combined = combined.sort_values(["sort_key", "Metric", "Type"]).drop(columns="sort_key")
    period_order = combined["Period"].unique().tolist()

    if viz_type_b == "Line Chart":
        fig = px.line(
            combined,
            x="Period", y="Value",
            color="Metric",
            line_dash="Type",
            color_discrete_map=budget_metric_color_map,
            markers=True,
            line_shape="spline" if smooth_lines else "linear",
            title=f"{selected_school_b} — Budget Metrics (Actual vs Frozen Forecast)"
        )
        fig.update_traces(connectgaps=False)
        fig.update_traces(text=None)

        if show_line_labels:
            for tr in fig.data:
                met = tr.name
                if met == "Budget to Enrollment Ratio":
                    tr.texttemplate = "%{y:.0%}"
                else:
                    tr.texttemplate = "%{y:,.0f}"
            fig.update_traces(textposition="top center")

    else:
        fig = px.bar(
            combined,
            x="Period", y="Value",
            color="Metric",
            barmode="group",
            text="Value",
            color_discrete_map=budget_metric_color_map,
            facet_row="Type",
            title=f"{selected_school_b} — Budget Metrics (Actual vs Frozen Forecast)"
        )
        for tr in fig.data:
            met = tr.name
            if met == "Budget to Enrollment Ratio":
                tr.texttemplate = "%{text:.0%}"
            else:
                tr.texttemplate = "%{text:,.0f}"
        fig.update_traces(textposition="outside")

    try:
        x0 = frozen_all["Period"].iloc[0]
        x1 = frozen_all["Period"].iloc[-1]
        fig.add_vrect(x0=x0, x1=x1, fillcolor="orange", opacity=0.08, line_width=0)
    except Exception:
        pass

    fig.update_xaxes(categoryorder="array", categoryarray=period_order, tickangle=45)
    fig.update_layout(height=560, legend_title="Metric")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Paste your existing CSAF Metrics / Other Metrics blocks here (unchanged).")
