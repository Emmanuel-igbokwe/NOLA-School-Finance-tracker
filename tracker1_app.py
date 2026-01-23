
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
        st.image(logo_path, width=100)
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
# VISUAL CONSTANTS + YEAR RANGES
# =========================
BASE_FONT_SIZE = 18
BASE_LABEL_FONT_SIZE = 17
BASE_TEXT_FONT_SIZE = 17
UNIFORM_TEXT_MIN = 16
BASE_HEIGHT_TALL = 820
BASE_HEIGHT_TALLER = 900

START_FY = 22        # FY22
END_ACTUAL_FY = 26   # FY26 (actuals end)
END_FORECAST_FY = 28 # FY28 (forecast end)

def fy_label(y: int) -> str:
    return f"FY{int(y):02d}"

# Robust FY parser: FY22, FY 22, FY2022, 2022 → 22
def fy_num(fy_str: str):
    s = str(fy_str)
    digits = re.sub(r"[^0-9]", "", s)
    if digits == "":
        return None
    n = int(digits)
    if n >= 2000:
        n = n % 100
    elif n > 100:
        n = n % 100
    return n

def fy_in_range(fy_str, start_y, end_y):
    y = fy_num(fy_str)
    return y is not None and start_y <= y <= end_y

def full_fy_range(start_y, end_y):
    return [fy_label(y) for y in range(start_y, end_y + 1)]

FY22_TO_FY26 = full_fy_range(START_FY, END_ACTUAL_FY)
FY22_TO_FY28 = full_fy_range(START_FY, END_FORECAST_FY)

# =========================
# HELPERS
# =========================
def sort_fy(x):
    """ For 'FY25 Q1' → (25,1). """
    try:
        parts = str(x).split()
        year = fy_num(parts[0]) if parts else None
        if year is None:
            return (999, 9)
        q = int(parts[1].replace("Q", "").strip()) if len(parts) > 1 and str(parts[1]).upper().startswith("Q") else 9
        return (year, q)
    except:
        return (999, 9)

def sort_fy_only(x):
    """ For 'FY22' or 'FY2022' → 22 """
    n = fy_num(x)
    return n if n is not None else 999

def normalize_col(c):
    return re.sub(r"\s+", " ", str(c).strip()).lower()

def clean_series(y):
    return pd.to_numeric(pd.Series(y), errors="coerce").values.astype(float)

def thicken_and_enlarge(fig, height=None):
    """Thicker bars + larger fonts/labels."""
    fig.update_traces(textposition="outside", textfont_size=BASE_TEXT_FONT_SIZE, marker_line_width=0)
    fig.update_layout(
        height=height or BASE_HEIGHT_TALL,
        font=dict(size=BASE_FONT_SIZE),
        legend_font=dict(size=BASE_FONT_SIZE),
        bargap=0.05,
        bargroupgap=0.03,
        uniformtext_minsize=UNIFORM_TEXT_MIN,
        uniformtext_mode="show",
        margin=dict(t=80, b=90)
    )
    return fig

def guard_growth(y_future, last_val, max_up=1.35, max_down=0.70, lower=0.0, upper=None):
    """Stepwise cap to prevent runaway spikes or collapses."""
    out = []
    prev = float(last_val)
    for v in y_future:
        vhat = float(v)
        if upper is not None:
            vhat = min(vhat, upper)
        vhat = max(vhat, lower)
        if prev > 0:
            vhat = min(vhat, prev * max_up)
            vhat = max(vhat, prev * max_down)
        out.append(vhat)
        prev = vhat
    return np.array(out, dtype=float)

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
df = df.dropna(subset=["Schools", "Fiscal Year"])
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
# LOAD BUDGET TO ENROLLMENT FY26 (ROBUST)
# =========================
fy26_path = "Enrollment FY26.xlsx"

try:
    df_budget_raw = None

    for hdr in [0, 1, 2, 3, 4]:
        tmp = pd.read_excel(fy26_path, sheet_name="FY26 Student enrollment", header=hdr)
        tmp.columns = [str(c).strip() for c in tmp.columns]
        cols_norm = [normalize_col(c) for c in tmp.columns]

        if any("school" in c for c in cols_norm) and any(
            ("fiscal" in c and "year" in c) or c in ["fy", "fiscal year"] for c in cols_norm
        ):
            df_budget_raw = tmp
            break

    if df_budget_raw is None:
        df_budget_raw = pd.read_excel(fy26_path, sheet_name="FY26 Student enrollment", header=1)
        df_budget_raw.columns = df_budget_raw.columns.str.strip()

    rename_map = {}
    for c in df_budget_raw.columns:
        cn = normalize_col(c)

        if cn in ["school", "schools", "site", "campus"] or "school" in cn:
            rename_map[c] = "Schools"

        if cn in ["fy", "fiscal year", "fiscal_year"] or ("fiscal" in cn and "year" in cn):
            rename_map[c] = "Fiscal Year"

        if "budget" in cn and "enroll" not in cn:
            rename_map[c] = "Budgetted"

        if "oct" in cn and ("count" in cn or "enroll" in cn):
            rename_map[c] = "October 1 Count"

        if ("feb" in cn or "february" in cn) and ("count" in cn or "enroll" in cn):
            rename_map[c] = "February 1 Count"

        if "budget" in cn and ("enrollment" in cn or "enrol" in cn) and ("ratio" in cn or "%" in cn):
            rename_map[c] = "Budget to Enrollment Ratio"

    df_budget_raw.rename(columns=rename_map, inplace=True)

    if "CMO" in df_budget_raw.columns:
        df_budget_raw.drop(columns=["CMO"], inplace=True)

    required = ["Schools", "Fiscal Year"]
    missing = [c for c in required if c not in df_budget_raw.columns]
    if missing:
        st.error(f"FY26 sheet loaded, but missing columns: {missing}")
        st.write("Columns found:", list(df_budget_raw.columns))
        st.stop()

    expected_cols = [
        "Schools", "Fiscal Year",
        "Budgetted", "October 1 Count", "February 1 Count",
        "Budget to Enrollment Ratio"
    ]

    df_budget_raw = df_budget_raw.dropna(subset=["Schools", "Fiscal Year"]).copy()
    df_budget_raw["Fiscal Year"] = df_budget_raw["Fiscal Year"].astype(str).str.strip()

    df_budget_long = df_budget_raw.melt(
        id_vars=["Schools", "Fiscal Year"],
        value_vars=[c for c in expected_cols if c in df_budget_raw.columns],
        var_name="Metric",
        value_name="Value"
    )

    fiscal_options_budget = sorted(df_budget_long["Fiscal Year"].dropna().unique(), key=sort_fy_only)
    school_options_budget = sorted(df_budget_long["Schools"].dropna().unique())

except Exception as e:
    st.warning(f"⚠️ Could not load {fy26_path} or sheet 'FY26 Student enrollment': {e}")
    df_budget_long = pd.DataFrame()
    fiscal_options_budget, school_options_budget = [], []

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

modes = ["CSAF Metrics", "CSAF Predicted", "Other Metrics"]
if not df_budget_long.empty:
    modes += ["Budget to Enrollment", "Budget to Enrollment Predicted"]

metric_group = st.sidebar.radio("Choose Dashboard:", modes)

# =========================
# CSAF PREDICTED (unchanged structurally; enlarged bars/labels)
# =========================
if metric_group == "CSAF Predicted":
    st.markdown("## 🔮 CSAF Predicted Metrics (FY22–FY28)")

    csaf_formulas = {
        "FB Ratio": ("Fund Balance Ratio", "Unrestricted Fund Balance ÷ Total Expenses", 0.10, "≥ 10%"),
        "Liabilities to Assets": ("Liabilities to Assets Ratio", "Total Liabilities ÷ Total Assets", 0.90, "≤ 0.90"),
        "Current Ratio": ("Current Ratio", "Current Assets ÷ Current Liabilities", 1.50, "≥ 1.5"),
        "Unrestricted Days COH": ("Unrestricted Days Cash on Hand", "Unrestricted Cash ÷ ((Total Exp. - Depreciation) ÷ 365)", 60, "≥ 60 days"),
    }

    schools = sorted(df["Schools"].unique())
    selected_school = st.sidebar.selectbox("🏫 Select School:", schools, index=0 if schools else None)
    selected_metric = st.sidebar.selectbox("📊 Choose Metric:", list(csaf_formulas.keys()))

    fiscal_years = sorted(df["Fiscal Year"].dropna().unique(), key=sort_fy)
    selected_fy_hist = st.sidebar.multiselect("📅 History Fiscal Years (training):", fiscal_years, default=fiscal_years)
    if st.sidebar.checkbox("Select All Fiscal Years (training)"):
        selected_fy_hist = fiscal_years

    all_quarters = sorted(df["Fiscal Year"].dropna().astype(str).unique(), key=sort_fy)
    train_through = st.sidebar.selectbox("🧊 Forecast Origin (freeze at):", all_quarters, index=max(0, len(all_quarters) - 1))

    forecast_method = st.sidebar.selectbox(
        "🧠 Forecast Method",
        [
            "Ensemble (Seasonal Naive + Robust Seasonal + Trend×Seasonality)",
            "Seasonal Naive (same quarter last year)",
            "Robust Seasonal Regression (Huber + quarter dummies, log1p)",
            "Trend × Seasonal Index (linear trend on de-seasonalized)",
        ],
        index=0,
    )

    n_future = st.sidebar.slider("🔮 Forecast horizon (quarters)", 3, 9, 6)
    run_pred = st.sidebar.button("▶ Run Prediction")

    fyq_re = re.compile(r"FY\s*(\d{2,4})\s*Q\s*(\d)")

    def parse_fyq(label: str):
        m = fyq_re.search(str(label))
        if not m:
            return None, None
        fy_raw = m.group(1)
        fy = fy_num(f"FY{fy_raw}")
        return fy, int(m.group(2)) if fy is not None else (None, None)

    def make_future_labels(last_label: str, n: int, quarters_per_year=3):
        fy, q = parse_fyq(last_label)
        if fy is None:
            fy, q = END_ACTUAL_FY, 0
        out = []
        for _ in range(n):
            q += 1
            if q > quarters_per_year:
                fy += 1
                q = 1
            out.append(f"FY{fy:02d} Q{q}")
        return out

    def quarter_index(labels):
        qs = []
        for lbl in labels:
            _, q = parse_fyq(lbl)
            qs.append(q if q is not None else np.nan)
        return np.array(qs, dtype=float)

    def seasonal_groups(q_arr):
        Q2 = (q_arr == 2).astype(int)
        Q3 = (q_arr == 3).astype(int)
        return np.column_stack([Q2, Q3])

    if not run_pred:
        st.info("Use the sidebar to pick School, Metric, History Years, Forecast Origin, Method, then click **Run Prediction**.")
        st.stop()

    hist_df = df[(df["Schools"] == selected_school) & (df["Fiscal Year"].isin(selected_fy_hist))].copy()
    if hist_df.empty:
        st.warning("⚠️ No rows for the selected school and history years.")
        st.stop()
    if selected_metric not in hist_df.columns:
        st.warning(f"⚠️ {selected_metric} not found for {selected_school}.")
        st.stop()

    hist_df["sort_key"] = hist_df["Fiscal Year"].apply(sort_fy)
    cut_key = sort_fy(train_through)
    hist_df = hist_df[hist_df["sort_key"].apply(lambda k: k <= cut_key)]
    hist_df = hist_df.sort_values("sort_key").drop(columns="sort_key")

    y_hist = clean_series(hist_df[selected_metric])
    fy_labels_hist = hist_df["Fiscal Year"].astype(str).tolist()
    valid_mask = ~np.isnan(y_hist)
    y_hist = y_hist[valid_mask]
    fy_labels_hist = [fy_labels_hist[i] for i, m in enumerate(valid_mask) if m]

    if len(y_hist) < 4:
        st.warning("⚠️ Not enough historical points to produce a reliable forecast (need ≥ 4).")
        st.stop()

    # Simple CSAF forecasters (same as prior version)
    def forecast_naive_same_q(y, n):
        if len(y) < 3:
            return np.repeat(y[-1], n)
        return np.array([y[max(0, len(y)-3 + (i % 3))] for i in range(n)], dtype=float)

    def forecast_robust(y, n):
        t = np.arange(len(y)).reshape(-1, 1)
        mdl = HuberRegressor().fit(t, np.log1p(np.clip(y, 0, None)))
        tf = np.arange(len(y), len(y)+n).reshape(-1, 1)
        return np.expm1(mdl.predict(tf))

    def forecast_linear(y, n):
        t = np.arange(len(y)).reshape(-1, 1)
        mdl = LinearRegression().fit(t, np.log1p(np.clip(y, 0, None)))
        tf = np.arange(len(y), len(y)+n).reshape(-1, 1)
        return np.expm1(mdl.predict(tf))

    if forecast_method.startswith("Ensemble"):
        y_future = 0.5*forecast_robust(y_hist, n_future) + 0.5*forecast_linear(y_hist, n_future)
    elif forecast_method.startswith("Seasonal Naive"):
        y_future = forecast_naive_same_q(y_hist, n_future)
    elif forecast_method.startswith("Robust"):
        y_future = forecast_robust(y_hist, n_future)
    else:
        y_future = forecast_linear(y_hist, n_future)

    future_labels = make_future_labels(train_through, n_future, quarters_per_year=3)

    actual_now = df[df["Schools"] == selected_school].copy()
    actual_now["sort_key"] = actual_now["Fiscal Year"].apply(sort_fy)
    actual_now = actual_now.sort_values("sort_key")

    actual_series = pd.DataFrame({
        "Quarter": actual_now["Fiscal Year"].astype(str),
        "Value": pd.to_numeric(actual_now[selected_metric], errors="coerce"),
        "Type": "Actual"
    }).dropna(subset=["Value"])

    frozen_pred = pd.DataFrame({"Quarter": future_labels, "Value": y_future, "Type": "Forecast (Frozen)"})
    combined = pd.concat([actual_series, frozen_pred], ignore_index=True)

    fig = px.bar(
        combined, x="Quarter", y="Value", color="Type",
        color_discrete_map={"Actual": "blue", "Forecast (Frozen)": "red"},
        barmode="group", text="Value",
        title=f"{selected_school} — {selected_metric} (Actual vs Frozen Forecast)"
    )
    if selected_metric == "FB Ratio":
        fig.update_traces(texttemplate="%{y:.1%}")
    elif selected_metric in ("Liabilities to Assets", "Current Ratio"):
        fig.update_traces(texttemplate="%{y:.2f}")
    else:
        fig.update_traces(texttemplate="%{y:,.0f}")

    thresh = csaf_descriptions[selected_metric]["threshold"]
    if selected_metric in ["FB Ratio", "Current Ratio", "Unrestricted Days COH"]:
        fig.add_hline(y=thresh, line_dash="dot", line_color="green",
                      annotation_text="Best Practice", annotation_position="top left")
    else:
        fig.add_hline(y=thresh, line_dash="dot", line_color="red",
                      annotation_text="Best Practice", annotation_position="top left")

    fig.update_xaxes(tickangle=45, tickfont=dict(size=BASE_LABEL_FONT_SIZE))
    fig = thicken_and_enlarge(fig, height=BASE_HEIGHT_TALL)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# BUDGET TO ENROLLMENT (ACTUALS) — BAR ONLY
# =========================
elif metric_group == "Budget to Enrollment":
    st.markdown("## 📊 Budget to Enrollment (Actuals FY22–FY26)")

    selected_schools = st.sidebar.multiselect("Select School(s):", school_options_budget)

    # Fiscal year filter + Select All
    fy_all_b = sorted(df_budget_long["Fiscal Year"].dropna().astype(str).unique(), key=sort_fy_only)
    selected_fy_b = st.sidebar.multiselect("Select Fiscal Year(s):", fy_all_b, default=[fy for fy in fy_all_b if fy_in_range(fy, START_FY, END_ACTUAL_FY)])
    if st.sidebar.checkbox("Select All Fiscal Years (Actuals)"):
        selected_fy_b = fy_all_b

    metrics_list = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    metrics_list = [m for m in metrics_list if m in df_budget_long["Metric"].unique()]
    selected_metrics = st.sidebar.multiselect("Select Metrics:", metrics_list, default=metrics_list)

    # Filter ACTUAL rows by chosen years (robust FY parsing)
    df_f = df_budget_long[
        (df_budget_long["Schools"].isin(selected_schools)) &
        (df_budget_long["Metric"].isin(selected_metrics)) &
        (df_budget_long["Fiscal Year"].isin(selected_fy_b))
    ].copy()

    if df_f.empty:
        st.warning("⚠️ No Budget to Enrollment data matches your filters.")
        st.stop()

    # Labels (hide zeros)
    df_f["Fiscal Year"] = df_f["Fiscal Year"].astype(str).str.strip()
    df_f["ValueNum"] = pd.to_numeric(df_f["Value"], errors="coerce")
    def fmt_actual(row):
        m = row["Metric"]; v = row["ValueNum"]
        if pd.isna(v) or v == 0:
            return ""
        return f"{v:.0%}" if "Ratio" in m and v <= 1.5 else (f"{v:,.2f}" if "Ratio" in m else f"{v:,.0f}")
    df_f["TextLabel"] = df_f.apply(fmt_actual, axis=1)

    # Order FY as canonical ascending
    df_f["sort_key"] = df_f["Fiscal Year"].apply(sort_fy_only)
    df_f = df_f.sort_values("sort_key")

    fig = px.bar(
        df_f, x="Fiscal Year", y="ValueNum",
        color="Metric",
        color_discrete_map=budget_metric_color_map,
        barmode="group",
        text="TextLabel",
        facet_col="Schools", facet_col_wrap=2,
        title=f"Budget to Enrollment — {', '.join(selected_metrics)}"
    )

    # Force FY order left→right
    axis_order = [fy for fy in FY22_TO_FY28 if fy in df_f["Fiscal Year"].unique()]
    fig.update_xaxes(categoryorder="array", categoryarray=axis_order, tickangle=45, tickfont=dict(size=BASE_LABEL_FONT_SIZE))
    fig = thicken_and_enlarge(fig, height=BASE_HEIGHT_TALLER)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# BUDGET TO ENROLLMENT PREDICTED — MULTI-METRIC, CSAF-LIKE VIEW
# =========================
elif metric_group == "Budget to Enrollment Predicted":
    st.markdown("## 🔮 Budget to Enrollment Predicted (Actual vs Frozen Forecast to FY28)")

    if df_budget_long.empty:
        st.warning("⚠️ Budget dataset not loaded.")
        st.stop()

    # School, metrics (multi), history years, freeze origin
    selected_school_b = st.sidebar.selectbox("🏫 Select School:", sorted(df_budget_long["Schools"].dropna().unique()))
    metrics_all = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    metrics_all = [m for m in metrics_all if m in df_budget_long["Metric"].unique()]
    selected_metrics_b = st.sidebar.multiselect("📊 Choose Metric(s):", metrics_all, default=metrics_all)

    fy_all = sorted(df_budget_long["Fiscal Year"].dropna().astype(str).unique(), key=sort_fy_only)
    selected_fy_hist_b = st.sidebar.multiselect("📅 History Fiscal Years (training):", fy_all, default=[fy for fy in fy_all if fy_in_range(fy, START_FY, END_ACTUAL_FY)] or fy_all)
    if st.sidebar.checkbox("Select All Fiscal Years (training)"):
        selected_fy_hist_b = fy_all

    default_origin = fy_label(END_ACTUAL_FY)
    origin_index = fy_all.index(default_origin) if default_origin in fy_all else max(0, len(fy_all) - 1)
    train_through_b = st.sidebar.selectbox("🧊 Forecast Origin (freeze at):", fy_all, index=origin_index)

    forecast_method_b = st.sidebar.selectbox(
        "🧠 Forecast Method",
        [
            "Ensemble (Huber + Linear, log1p)",
            "Robust Trend (Huber, log1p)",
            "Linear Trend (log1p)",
        ],
        index=0,
    )

    run_pred_b = st.sidebar.button("▶ Run Budget Prediction")
    if not run_pred_b:
        st.info("Pick School, Metric(s), History, Freeze Origin, Method, then click **Run Budget Prediction**.")
        st.stop()

    origin_year = sort_fy_only(train_through_b)
    n_future_b = max(0, END_FORECAST_FY - origin_year)
    future_labels = [fy_label(y) for y in range(origin_year + 1, END_FORECAST_FY + 1)]

    def pred_huber(t, ylog, n_future):
        model = HuberRegressor().fit(t, ylog)
        tf = np.arange(len(ylog), len(ylog) + n_future).reshape(-1, 1)
        return np.expm1(model.predict(tf))

    def pred_linear(t, ylog, n_future):
        model = LinearRegression().fit(t, ylog)
        tf = np.arange(len(ylog), len(ylog) + n_future).reshape(-1, 1)
        return np.expm1(model.predict(tf))

    frames = []
    for met in selected_metrics_b:
        # HISTORY (FYs selected by user) up to freeze origin
        dfh = df_budget_long[
            (df_budget_long["Schools"] == selected_school_b) &
            (df_budget_long["Metric"] == met) &
            (df_budget_long["Fiscal Year"].isin(selected_fy_hist_b))
        ].copy()
        dfh["sort_key"] = dfh["Fiscal Year"].apply(sort_fy_only)
        dfh = dfh[dfh["sort_key"] <= origin_year].sort_values("sort_key").dropna(subset=["Value"])

        y = clean_series(dfh["Value"])
        # FORECAST to FY28
        if n_future_b > 0:
            if len(y) >= 3:
                t = np.arange(len(y)).reshape(-1, 1)
                y_log = np.log1p(np.clip(y, 0, None))
                if forecast_method_b.startswith("Ensemble"):
                    y_future = 0.6*pred_huber(t, y_log, n_future_b) + 0.4*pred_linear(t, y_log, n_future_b)
                elif forecast_method_b.startswith("Robust"):
                    y_future = pred_huber(t, y_log, n_future_b)
                else:
                    y_future = pred_linear(t, y_log, n_future_b)
            elif len(y) == 2:
                slope = (y[-1] - y[-2])
                last = y[-1]
                y_future = np.array([max(0.0, last + slope * (i + 1)) for i in range(n_future_b)], dtype=float)
            elif len(y) == 1:
                last = max(0.0, y[0])
                y_future = np.array([last for _ in range(n_future_b)], dtype=float)
            else:
                y_future = np.zeros(n_future_b, dtype=float)

            last_val = y[-1] if len(y) else 0.0
            if met == "Budget to Enrollment Ratio":
                y_future = np.clip(y_future, 0.0, 1.5)
                y_future = guard_growth(y_future, last_val, max_up=1.25, max_down=0.75, lower=0.0, upper=1.5)
            else:
                hist_max = np.nanmax(y) if len(y) else 0.0
                cap = max(hist_max * 1.8, last_val * 1.6, 1.0)
                y_future = guard_growth(y_future, last_val, max_up=1.35, max_down=0.70, lower=0.0, upper=cap)
        else:
            y_future = np.array([])

        # BUILD ACTUAL (FY<=freeze) AND FORECAST (freeze+1..FY28)
        actual_now = df_budget_long[
            (df_budget_long["Schools"] == selected_school_b) &
            (df_budget_long["Metric"] == met)
        ].copy()
        actual_now["sort_key"] = actual_now["Fiscal Year"].apply(sort_fy_only)
        actual_now = actual_now[actual_now["sort_key"] <= origin_year].sort_values("sort_key")
        actual_part = pd.DataFrame({
            "FY": actual_now["Fiscal Year"].astype(str),
            "Value": pd.to_numeric(actual_now["Value"], errors="coerce"),
            "Metric": met,
            "Type": "Actual"
        }).dropna(subset=["Value"])

        forecast_part = pd.DataFrame({
            "FY": future_labels,
            "Value": y_future,
            "Metric": met,
            "Type": "Forecast (Frozen)"
        })

        frames.append(actual_part)
        frames.append(forecast_part)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["FY","Value","Metric","Type"])
    if combined.empty:
        st.warning("⚠️ Not enough data to build forecast for the selected metric(s).")
        st.stop()

    # Labels
    combined["ValueNum"] = pd.to_numeric(combined["Value"], errors="coerce")
    def fmt_pred(row):
        m = row["Metric"]; v = row["ValueNum"]
        if pd.isna(v) or v == 0:
            return ""
        if "Ratio" in m:
            return f"{v:.0%}" if v <= 1.5 else f"{v:,.2f}"
        return f"{v:,.0f}"
    combined["TextLabel"] = combined.apply(fmt_pred, axis=1)

    # Sort FY; force x-axis to FY22..FY28 with FY22 on the left
    combined["sort_key"] = combined["FY"].apply(sort_fy_only)
    combined = combined.sort_values(["sort_key", "Metric", "Type"]).drop(columns="sort_key")

    fig = px.bar(
        combined,
        x="FY", y="ValueNum",
        color="Metric",
        text="TextLabel",
        pattern_shape="Type",
        pattern_shape_map={"Actual": "", "Forecast (Frozen)": "/"},
        color_discrete_map=budget_metric_color_map,
        barmode="group",
        title=f"{selected_school_b} — Budget Metrics (Actual vs Frozen Forecast to FY{END_FORECAST_FY})"
    )

    # Enforce canonical FY order so FY22 is first
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=FY22_TO_FY28,  # <— fixes FY22 appearing at the end
        tickangle=45,
        tickfont=dict(size=BASE_LABEL_FONT_SIZE)
    )
    fig = thicken_and_enlarge(fig, height=BASE_HEIGHT_TALLER)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# FY25 (CSAF METRICS + OTHER METRICS) — BAR ONLY
# =========================
else:
    school_options = sorted(df_long["Schools"].dropna().unique())
    selected_schools = st.sidebar.multiselect("Select School(s):", school_options)

    selected_fy = st.sidebar.multiselect("Select Fiscal Year and Quarter:", fiscal_options, default=fiscal_options)
    if st.sidebar.checkbox("Select All Fiscal Years"):
        selected_fy = fiscal_options

    if metric_group == "CSAF Metrics":
        selected_metric_single = st.sidebar.selectbox("Select CSAF Metric:", csaf_metrics)
        selected_metrics = [selected_metric_single]
    else:
        selected_metrics = st.sidebar.multiselect("Select Other Metric(s):", other_metrics)

    filtered = df_long[
        (df_long["Schools"].isin(selected_schools)) &
        (df_long["Fiscal Year"].isin(selected_fy)) &
        (df_long["Metric"].isin(selected_metrics))
    ]

    if filtered.empty:
        st.warning("⚠️ Welcome To Finance Accountability Real-Time Dashboard. Try adjusting your filters.")
        st.stop()

    filtered = filtered.copy()
    filtered["sort_key"] = filtered["Fiscal Year"].apply(sort_fy)
    filtered = filtered.sort_values("sort_key")
    filtered["FY Group"] = filtered["Fiscal Year"].str.split().str[0]

    # Formatted labels; hide 0s
    filtered["ValueNum"] = pd.to_numeric(filtered["Value"], errors="coerce")
    def fmt_general(row):
        v = row["ValueNum"]
        if pd.isna(v) or v == 0:
            return ""
        m = row["Metric"]
        if m in dollar_metrics:
            return f"${v:,.0f}"
        if m == "FB Ratio":
            return f"{v:.0%}"
        if m in ("Liabilities to Assets", "Current Ratio"):
            return f"{v:.2f}"
        if m == "Unrestricted Days COH":
            return f"{v:,.0f}"
        return f"{v:,.0f}"
    filtered["TextLabel"] = filtered.apply(fmt_general, axis=1)

    # Facets
    facet_args = {}
    if len(selected_schools) > 1 and len(selected_metrics) > 1:
        facet_args = {"facet_row": "Schools", "facet_col": "Metric", "facet_col_wrap": 2}
    elif len(selected_schools) > 1:
        facet_args = {"facet_col": "Schools", "facet_col_wrap": 2}
    elif len(selected_metrics) > 1:
        facet_args = {"facet_col": "Metric", "facet_col_wrap": 2}

    fig = px.bar(
        filtered, x="Fiscal Year", y="ValueNum",
        color="FY Group", color_discrete_map=fy_color_map,
        barmode="group", text="TextLabel",
        title=", ".join(selected_metrics) if selected_metrics else "Metrics",
        **facet_args
    )

    fiscal_order = sorted(filtered["Fiscal Year"].unique(), key=sort_fy)
    fig.update_xaxes(categoryorder="array", categoryarray=fiscal_order, tickangle=45, tickfont=dict(size=BASE_LABEL_FONT_SIZE))

    # CSAF thresholds when single CSAF metric
    if selected_metrics == ["FB Ratio"]:
        fig.add_hline(y=csaf_descriptions["FB Ratio"]["threshold"], line_dash="dot", line_color="blue")
    elif selected_metrics == ["Liabilities to Assets"]:
        fig.add_hline(y=csaf_descriptions["Liabilities to Assets"]["threshold"], line_dash="dot", line_color="blue")
    elif selected_metrics == ["Current Ratio"]:
        fig.add_hline(y=csaf_descriptions["Current Ratio"]["threshold"], line_dash="dot", line_color="blue")
    elif selected_metrics == ["Unrestricted Days COH"]:
        fig.add_hline(y=csaf_descriptions["Unrestricted Days COH"]["threshold"], line_dash="dot", line_color="blue")

    fig = thicken_and_enlarge(fig, height=BASE_HEIGHT_TALL)
    st.plotly_chart(fig, use_container_width=True)
