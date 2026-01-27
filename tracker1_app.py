import pandas as pd
import streamlit as st
import plotly.express as px
import os
import numpy as np
import re

from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

# =========================
# PAGE CONFIG + COMPACT SIDEBAR UI
# =========================
st.set_page_config(page_title="NOLA Financial Tracker", layout="wide")

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] * { font-size: 13px !important; }
    section[data-testid="stSidebar"] label { font-size: 13px !important; }
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stMultiSelect,
    section[data-testid="stSidebar"] .stRadio,
    section[data-testid="stSidebar"] .stCheckbox,
    section[data-testid="stSidebar"] .stSlider {
        margin-bottom: 0.40rem !important;
    }
    section[data-testid="stSidebar"] div[role="listbox"] div { font-size: 12px !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# HEADER
# =========================
logo_path = "nola_parish_logo.png"
col1, col2 = st.columns([1, 5])
with col1:
    if os.path.exists(logo_path):
        st.image(logo_path, width=96)
with col2:
    st.markdown(
        """
        <h1 style="color:#003366; font-size:28px; margin-bottom:0;">
            Welcome to NOLA Public Schools Finance Accountability App
        </h1>
        """,
        unsafe_allow_html=True
    )

# =========================
# VISUAL CONSTANTS + YEAR RANGES
# =========================
BASE_FONT_SIZE = 18
BASE_LABEL_FONT_SIZE = 16
BASE_TEXT_FONT_SIZE = 16
UNIFORM_TEXT_MIN = 14
BASE_HEIGHT_TALL = 780
BASE_HEIGHT_TALLER = 860

START_FY = 22        # FY22
END_ACTUAL_FY = 26   # FY26 (end of actuals)
END_FORECAST_FY = 28 # FY28 (end of forecast)

def fy_label(y: int) -> str:
    return f"FY{int(y):02d}"

# Robust FY parser: handles FY22, FY 22, FY2022, 2022 → 22
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

def fy_std(val) -> str:
    """
    Normalize anything like FY22, FY 2022, 2022, '22' => 'FY22'
    Returns original string if it can't parse.
    """
    n = fy_num(val)
    return fy_label(n) if n is not None else str(val).strip()

def sort_fy(x):
    """ 'FY25 Q1' → (25,1) """
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
    n = fy_num(x)
    return n if n is not None else 999

def full_fy_range(start_y, end_y):
    return [fy_label(y) for y in range(start_y, end_y + 1)]

# Canonical FY sequences (FORCE FY22 first)
FY22_TO_FY26 = full_fy_range(START_FY, END_ACTUAL_FY)
FY22_TO_FY28 = full_fy_range(START_FY, END_FORECAST_FY)

def thicken_and_enlarge(fig, height=None):
    """Bigger fonts/labels."""
    fig.update_traces(textposition="top center", textfont_size=BASE_TEXT_FONT_SIZE)
    fig.update_layout(
        height=height or BASE_HEIGHT_TALL,
        font=dict(size=BASE_FONT_SIZE),
        legend_font=dict(size=BASE_FONT_SIZE),
        uniformtext_minsize=UNIFORM_TEXT_MIN,
        uniformtext_mode="show",
        margin=dict(t=70, b=90)
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
# BEST PRACTICE / THRESHOLDS
# =========================
csaf_metrics = ["FB Ratio", "Liabilities to Assets", "Current Ratio", "Unrestricted Days COH"]
csaf_descriptions = {
    "FB Ratio": {"desc": "Unrestricted FB ÷ Total Exp. (Best practice ≥ 10%)", "threshold": 0.10, "direction": "gte"},
    "Liabilities to Assets": {"desc": "Total Liabilities ÷ Total Assets (Best practice ≤ 0.90)", "threshold": 0.90, "direction": "lte"},
    "Current Ratio": {"desc": "Current Assets ÷ Current Liabilities (Best practice ≥ 1.5)", "threshold": 1.50, "direction": "gte"},
    "Unrestricted Days COH": {"desc": "Unrestricted Cash ÷ ((Total Exp. - Depreciation) ÷ 365) (Best practice ≥ 60)", "threshold": 60, "direction": "gte"},
}

# Budget-to-Enrollment ratio best practice band (you can adjust)
BUDGET_RATIO_TARGET = 1.00
BUDGET_RATIO_BAND_LOW = 0.90
BUDGET_RATIO_BAND_HIGH = 1.10

def add_best_practice_csaf(fig, metric):
    if metric not in csaf_descriptions:
        return fig
    thr = csaf_descriptions[metric]["threshold"]
    direction = csaf_descriptions[metric]["direction"]
    line_color = "green" if direction == "gte" else "red"
    fig.add_hline(
        y=thr,
        line_dash="dot",
        line_color=line_color,
        annotation_text="Best Practice",
        annotation_position="top left"
    )
    return fig

def add_best_practice_budget_ratio(fig):
    # Band + center line
    fig.add_hrect(
        y0=BUDGET_RATIO_BAND_LOW, y1=BUDGET_RATIO_BAND_HIGH,
        line_width=0,
        fillcolor="rgba(0,200,0,0.10)",
        annotation_text="Best Practice Band (0.90–1.10)",
        annotation_position="top left"
    )
    fig.add_hline(
        y=BUDGET_RATIO_TARGET,
        line_dash="dot",
        line_color="green",
        annotation_text="Target = 1.00",
        annotation_position="top left"
    )
    return fig

# =========================
# LOAD DATA (CSAF + OTHER)
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

# Standardize FY labels so order works consistently
df["Fiscal Year"] = df["Fiscal Year"].apply(lambda x: f"{fy_std(str(x).split()[0])} " + " ".join(str(x).split()[1:]) if len(str(x).split()) > 1 else str(x).strip())

value_vars = [c for c in df.columns if c not in ["Schools", "Fiscal Year"]]
df_long = df.melt(
    id_vars=["Schools", "Fiscal Year"],
    value_vars=value_vars,
    var_name="Metric",
    value_name="Value"
)

fiscal_options = sorted(df["Fiscal Year"].dropna().unique(), key=sort_fy)

dollar_metrics = [
    "Restricted Cash", "Unrestricted Cash & Equivalents", "Current Assets", "Fixed Assets", "Total Assets",
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
# LOAD DATA (BUDGET TO ENROLLMENT)
# =========================
fy26_path = "Enrollment FY26.xlsx"
try:
    df_budget_raw = None
    for hdr in [0, 1, 2, 3, 4]:
        tmp = pd.read_excel(fy26_path, sheet_name="FY26 Student enrollment", header=hdr)
        tmp.columns = [str(c).strip() for c in tmp.columns]
        cols_norm = [re.sub(r"\s+", " ", str(c).strip()).lower() for c in tmp.columns]
        if any("school" in c for c in cols_norm) and any(("fiscal" in c and "year" in c) or c in ["fy", "fiscal year"] for c in cols_norm):
            df_budget_raw = tmp
            break
    if df_budget_raw is None:
        df_budget_raw = pd.read_excel(fy26_path, sheet_name="FY26 Student enrollment", header=1)
        df_budget_raw.columns = df_budget_raw.columns.str.strip()

    # Canonical names
    rename_map = {}
    for c in df_budget_raw.columns:
        cn = re.sub(r"\s+", " ", str(c).strip()).lower()
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
        st.error(f"Enrollment sheet missing required columns: {missing}")
        st.stop()

    expected_cols = ["Schools", "Fiscal Year", "Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    df_budget_raw = df_budget_raw.dropna(subset=["Schools", "Fiscal Year"]).copy()

    # Standardize FY
    df_budget_raw["Fiscal Year"] = df_budget_raw["Fiscal Year"].astype(str).str.strip()
    df_budget_raw["Fiscal Year"] = df_budget_raw["Fiscal Year"].apply(fy_std)

    df_budget_long = df_budget_raw.melt(
        id_vars=["Schools", "Fiscal Year"],
        value_vars=[c for c in expected_cols if c in df_budget_raw.columns],
        var_name="Metric",
        value_name="Value"
    )

    fiscal_options_budget = sorted(df_budget_long["Fiscal Year"].dropna().unique(), key=sort_fy_only)
    school_options_budget = sorted(df_budget_long["Schools"].dropna().unique())

except Exception as e:
    st.warning(f"⚠️ Could not load '{fy26_path}' / sheet 'FY26 Student enrollment': {e}")
    df_budget_long = pd.DataFrame()
    fiscal_options_budget, school_options_budget = [], []

budget_metric_color_map = {
    "Budgetted": "#1f77b4",
    "October 1 Count": "#2ca02c",
    "February 1 Count": "#d62728",
    "Budget to Enrollment Ratio": "#ff7f0e",
}

# =========================
# UI — DASHBOARD SWITCHER
# =========================
st.title("📊 NOLA Schools Financial Tracker")
st.markdown("<p style='font-size:14px;color:gray;'>Built by Emmanuel Igbokwe</p>", unsafe_allow_html=True)
st.sidebar.header("🔎 Filters")

modes = ["CSAF Metrics", "CSAF Predicted", "Other Metrics"]
if not df_budget_long.empty:
    modes += ["Budget to Enrollment", "Budget to Enrollment Predicted"]
metric_group = st.sidebar.radio("Choose Dashboard:", modes)

# =========================
# CSAF PREDICTED
# =========================
if metric_group == "CSAF Predicted":
    st.markdown("## 🔮 CSAF Predicted Metrics (FY22–FY28)")

    schools = sorted(df["Schools"].unique())
    selected_school = st.sidebar.selectbox("🏫 Select School:", schools, index=0 if schools else None)
    selected_metric = st.sidebar.selectbox("📊 Choose Metric:", csaf_metrics)

    fiscal_years = sorted(df["Fiscal Year"].dropna().unique(), key=sort_fy)
    selected_fy_hist = st.sidebar.multiselect("📅 History Fiscal Years (training):", fiscal_years, default=fiscal_years)
    if st.sidebar.checkbox("Select All Fiscal Years (training)"):
        selected_fy_hist = fiscal_years

    all_quarters = sorted(df["Fiscal Year"].dropna().astype(str).unique(), key=sort_fy)
    train_through = st.sidebar.selectbox("🧊 Forecast Origin (freeze at):", all_quarters, index=max(0, len(all_quarters) - 1))

    forecast_method = st.sidebar.selectbox(
        "🧠 Forecast Method",
        [
            "Ensemble (Robust Seasonal + Trend×Seasonality)",
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
        if fy is None:
            return None, None
        return fy, int(m.group(2))

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
    hist_df = hist_df[hist_df["sort_key"].apply(lambda k: k <= cut_key)].sort_values("sort_key")

    y_hist = pd.to_numeric(hist_df[selected_metric], errors="coerce").values.astype(float)
    fy_labels_hist = hist_df["Fiscal Year"].astype(str).tolist()
    q_hist = quarter_index(fy_labels_hist)

    valid_mask = ~np.isnan(y_hist) & ~np.isnan(q_hist)
    y_hist = y_hist[valid_mask]
    q_hist = q_hist[valid_mask]
    fy_labels_hist = [fy_labels_hist[i] for i, m in enumerate(valid_mask) if m]

    if len(y_hist) < 4:
        st.warning("⚠️ Not enough historical points to produce a reliable forecast (need ≥ 4).")
        st.stop()

    def forecast_seasonal_naive(y, q, n):
        if len(y) < 3:
            return np.repeat(y[-1], n)
        return np.array([y[max(0, len(y)-3 + (i % 3))] for i in range(n)], dtype=float)

    def forecast_robust_seasonal(y, q, n):
        t = np.arange(len(y)).reshape(-1, 1)
        Q2 = (q == 2).astype(int)
        Q3 = (q == 3).astype(int)
        X = np.hstack([t, Q2.reshape(-1, 1), Q3.reshape(-1, 1)])
        mdl = HuberRegressor().fit(X, np.log1p(np.clip(y, 0, None)))

        tf = np.arange(len(y), len(y) + n).reshape(-1, 1)
        qf = ((q[-1] + np.arange(1, n+1) - 1) % 3) + 1
        Xf = np.hstack([tf, (qf == 2).astype(int).reshape(-1, 1), (qf == 3).astype(int).reshape(-1, 1)])
        return np.expm1(mdl.predict(Xf))

    def forecast_trend_times_seasonal(y, q, n):
        season_means = {s: np.nanmean(y[q == s]) for s in [1, 2, 3]}
        global_mean = np.nanmean(y)
        for s in [1, 2, 3]:
            if not np.isfinite(season_means.get(s, np.nan)):
                season_means[s] = global_mean
        core = y / np.array([season_means[int(s)] for s in q])
        core = np.where(np.isfinite(core), core, global_mean)

        t = np.arange(len(core)).reshape(-1, 1)
        mdl = LinearRegression().fit(t, np.log1p(np.clip(core, 0, None)))

        tf = np.arange(len(core), len(core) + n).reshape(-1, 1)
        qf = ((q[-1] + np.arange(1, n+1) - 1) % 3) + 1
        core_pred = np.expm1(mdl.predict(tf))
        return core_pred * np.array([season_means[int(s)] for s in qf])

    if forecast_method.startswith("Ensemble"):
        y_future = 0.5 * forecast_robust_seasonal(y_hist, q_hist, n_future) + 0.5 * forecast_trend_times_seasonal(y_hist, q_hist, n_future)
    elif forecast_method.startswith("Seasonal Naive"):
        y_future = forecast_seasonal_naive(y_hist, q_hist, n_future)
    elif forecast_method.startswith("Robust"):
        y_future = forecast_robust_seasonal(y_hist, q_hist, n_future)
    else:
        y_future = forecast_trend_times_seasonal(y_hist, q_hist, n_future)

    future_labels = make_future_labels(train_through, n_future, quarters_per_year=3)

    actual_now = df[df["Schools"] == selected_school].copy()
    actual_now["sort_key"] = actual_now["Fiscal Year"].apply(sort_fy)
    actual_now = actual_now.sort_values("sort_key")

    combined = pd.concat(
        [
            pd.DataFrame({
                "Quarter": actual_now["Fiscal Year"].astype(str),
                "Value": pd.to_numeric(actual_now[selected_metric], errors="coerce"),
                "Type": "Actual"
            }).dropna(subset=["Value"]),
            pd.DataFrame({
                "Quarter": future_labels,
                "Value": y_future,
                "Type": "Forecast (Frozen)"
            }),
        ],
        ignore_index=True
    )

    fig = px.line(
        combined, x="Quarter", y="Value", color="Type",
        markers=True,
        title=f"{selected_school} — {selected_metric} (Actual vs Frozen Forecast)"
    )
    fig.update_traces(connectgaps=True)

    # Format y labels
    if selected_metric == "FB Ratio":
        fig.update_traces(hovertemplate="%{y:.1%}<extra></extra>")
    elif selected_metric in ("Liabilities to Assets", "Current Ratio"):
        fig.update_traces(hovertemplate="%{y:.2f}<extra></extra>")
    else:
        fig.update_traces(hovertemplate="%{y:,.0f}<extra></extra>")

    # Best practice line
    fig = add_best_practice_csaf(fig, selected_metric)

    fig.update_xaxes(tickangle=45, tickfont=dict(size=BASE_LABEL_FONT_SIZE))
    fig = thicken_and_enlarge(fig, height=BASE_HEIGHT_TALL)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# BUDGET TO ENROLLMENT (ACTUALS) — LINE + FY ORDER FIX + BEST PRACTICE BAND
# =========================
elif metric_group == "Budget to Enrollment":
    st.markdown("## 📈 Budget to Enrollment (Actuals — Line Chart)")

    selected_schools = st.sidebar.multiselect("Select School(s):", school_options_budget)

    fy_all_b = sorted(df_budget_long["Fiscal Year"].dropna().astype(str).unique(), key=sort_fy_only)
    default_fy_actuals = [fy for fy in fy_all_b if START_FY <= sort_fy_only(fy) <= END_ACTUAL_FY]
    selected_fy_b = st.sidebar.multiselect("Select Fiscal Year(s):", fy_all_b, default=default_fy_actuals or fy_all_b)
    if st.sidebar.checkbox("Select All Fiscal Years (Actuals)"):
        selected_fy_b = fy_all_b

    metrics_list = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    metrics_list = [m for m in metrics_list if m in df_budget_long["Metric"].unique()]
    selected_metrics = st.sidebar.multiselect("Select Metrics:", metrics_list, default=metrics_list)

    df_f = df_budget_long[
        (df_budget_long["Schools"].isin(selected_schools)) &
        (df_budget_long["Metric"].isin(selected_metrics)) &
        (df_budget_long["Fiscal Year"].isin(selected_fy_b))
    ].copy()

    if df_f.empty:
        st.warning("⚠️ No Budget to Enrollment data matches your filters.")
        st.stop()

    df_f["Fiscal Year"] = df_f["Fiscal Year"].astype(str).str.strip().apply(fy_std)
    df_f["ValueNum"] = pd.to_numeric(df_f["Value"], errors="coerce")
    df_f["sort_key"] = df_f["Fiscal Year"].apply(sort_fy_only)
    df_f = df_f.sort_values("sort_key")

    fig = px.line(
        df_f,
        x="Fiscal Year", y="ValueNum",
        color="Metric",
        markers=True,
        facet_col="Schools", facet_col_wrap=2,
        title=f"Budget to Enrollment — {', '.join(selected_metrics)}"
    )
    fig.update_traces(connectgaps=True)

    axis_order = [fy for fy in FY22_TO_FY28 if fy in df_f["Fiscal Year"].unique()]
    fig.update_xaxes(categoryorder="array", categoryarray=axis_order, tickangle=45, tickfont=dict(size=BASE_LABEL_FONT_SIZE))

    # Best practice band/line ONLY if ratio is included
    if "Budget to Enrollment Ratio" in selected_metrics:
        fig = add_best_practice_budget_ratio(fig)

    fig.update_layout(
        height=BASE_HEIGHT_TALLER,
        font=dict(size=BASE_FONT_SIZE),
        legend_font=dict(size=BASE_FONT_SIZE),
        margin=dict(t=70, b=90)
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# BUDGET TO ENROLLMENT PREDICTED — ADVANCED (MIN ERROR BY TIME-SERIES CV) + LINE + BEST PRACTICE BAND
# =========================
elif metric_group == "Budget to Enrollment Predicted":
    st.markdown("## 🔮 Budget to Enrollment Predicted (Actual ≤ Freeze • Forecast → FY28)")

    if df_budget_long.empty:
        st.warning("⚠️ Budget dataset not loaded.")
        st.stop()

    selected_school_b = st.sidebar.selectbox("🏫 Select School:", sorted(df_budget_long["Schools"].dropna().unique()))
    metrics_all = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    metrics_all = [m for m in metrics_all if m in df_budget_long["Metric"].unique()]
    selected_metrics_b = st.sidebar.multiselect("📊 Choose Metric(s):", metrics_all, default=metrics_all)

    fy_all = sorted(df_budget_long["Fiscal Year"].dropna().astype(str).unique(), key=sort_fy_only)
    default_hist = [fy for fy in fy_all if START_FY <= sort_fy_only(fy) <= END_ACTUAL_FY]
    selected_fy_hist_b = st.sidebar.multiselect("📅 History Fiscal Years (training):", fy_all, default=default_hist or fy_all)
    if st.sidebar.checkbox("Select All Fiscal Years (training)"):
        selected_fy_hist_b = fy_all

    default_origin = fy_label(END_ACTUAL_FY)
    origin_index = fy_all.index(default_origin) if default_origin in fy_all else max(0, len(fy_all) - 1)
    train_through_b = st.sidebar.selectbox("🧊 Forecast Origin (freeze at):", fy_all, index=origin_index)

    show_model_info = st.sidebar.checkbox("Show model + error details", value=True)

    run_pred_b = st.sidebar.button("▶ Run Budget Prediction")
    if not run_pred_b:
        st.info("Pick School, Metric(s), History, Freeze Origin, then click **Run Budget Prediction**.")
        st.stop()

    origin_year = sort_fy_only(train_through_b)
    n_future_b = max(0, END_FORECAST_FY - origin_year)
    future_labels = [fy_label(y) for y in range(origin_year + 1, END_FORECAST_FY + 1)]

    # --- ADVANCED: choose lowest-error model (time-series CV) ---
    def make_supervised(n):
        # time index features
        t = np.arange(n).reshape(-1, 1).astype(float)
        return np.hstack([t, t**2])

    def ts_cv_mae(model, X, y_log, splits=3):
        if len(y_log) < 4:
            return np.inf
        n_splits = min(splits, max(2, len(y_log) - 1))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        maes = []
        for tr, te in tscv.split(X):
            model.fit(X[tr], y_log[tr])
            pred = model.predict(X[te])
            maes.append(mean_absolute_error(y_log[te], pred))
        return float(np.mean(maes)) if maes else np.inf

    def best_forecast(y, n_future, metric_name):
        if len(y) == 0:
            return np.zeros(n_future), "None", np.inf

        y_clip = np.clip(y, 0, None)
        if metric_name == "Budget to Enrollment Ratio":
            y_clip = np.clip(y_clip, 0.0, 1.5)

        y_log = np.log1p(y_clip)
        X = make_supervised(len(y_log))

        candidates = {
            "Huber": HuberRegressor(),
            "Linear": LinearRegression(),
            "HGBR": HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.08,
                max_iter=500,
                random_state=42
            ),
            "MLP": make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(16, 8),
                    activation="relu",
                    solver="adam",
                    max_iter=8000,
                    random_state=42,
                    learning_rate="adaptive",
                    early_stopping=True,
                    n_iter_no_change=120
                )
            )
        }

        scores = {}
        for name, mdl in candidates.items():
            try:
                scores[name] = ts_cv_mae(mdl, X, y_log, splits=3)
            except Exception:
                scores[name] = np.inf

        best_name = min(scores, key=scores.get)
        best_model = candidates[best_name]
        best_err = scores[best_name]

        best_model.fit(X, y_log)

        Xf = make_supervised(len(y_log) + n_future)[-n_future:]
        y_pred = np.expm1(best_model.predict(Xf))
        y_pred = np.clip(y_pred, 0, None)

        last_val = float(y_clip[-1]) if len(y_clip) else 0.0
        if metric_name == "Budget to Enrollment Ratio":
            y_pred = np.clip(y_pred, 0.0, 1.5)
            y_pred = guard_growth(y_pred, last_val, max_up=1.20, max_down=0.80, lower=0.0, upper=1.5)
        else:
            hist_max = float(np.nanmax(y_clip)) if len(y_clip) else 0.0
            cap = max(hist_max * 1.6, last_val * 1.5, 1.0)
            y_pred = guard_growth(y_pred, last_val, max_up=1.25, max_down=0.75, lower=0.0, upper=cap)

        return y_pred, best_name, best_err

    frames = []
    model_rows = []

    for met in selected_metrics_b:
        dfh = df_budget_long[
            (df_budget_long["Schools"] == selected_school_b) &
            (df_budget_long["Metric"] == met) &
            (df_budget_long["Fiscal Year"].isin(selected_fy_hist_b))
        ].copy()

        dfh["Fiscal Year"] = dfh["Fiscal Year"].astype(str).str.strip().apply(fy_std)
        dfh["sort_key"] = dfh["Fiscal Year"].apply(sort_fy_only)
        dfh = dfh[dfh["sort_key"] <= origin_year].sort_values("sort_key").dropna(subset=["Value"])

        y = pd.to_numeric(dfh["Value"], errors="coerce").values.astype(float)

        if n_future_b > 0:
            if len(y) >= 3:
                y_future, best_name, best_err = best_forecast(y, n_future_b, met)
            elif len(y) == 2:
                slope = (y[-1] - y[-2])
                last = y[-1]
                y_future = np.array([max(0.0, last + slope * (i + 1)) for i in range(n_future_b)], dtype=float)
                best_name, best_err = "2pt-trend", np.nan
            elif len(y) == 1:
                y_future = np.array([max(0.0, y[0]) for _ in range(n_future_b)], dtype=float)
                best_name, best_err = "carry-forward", np.nan
            else:
                y_future = np.zeros(n_future_b, dtype=float)
                best_name, best_err = "none", np.nan
        else:
            y_future = np.array([])
            best_name, best_err = "none", np.nan

        model_rows.append({
            "Metric": met,
            "Model Selected": best_name,
            "CV MAE (log-scale)": (None if not np.isfinite(best_err) else float(best_err)),
            "History Points": int(len(y))
        })

        actual_now = df_budget_long[
            (df_budget_long["Schools"] == selected_school_b) &
            (df_budget_long["Metric"] == met)
        ].copy()
        actual_now["Fiscal Year"] = actual_now["Fiscal Year"].astype(str).str.strip().apply(fy_std)
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

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["FY", "Value", "Metric", "Type"])
    if combined.empty:
        st.warning("⚠️ Not enough data to build forecast for the selected metric(s).")
        st.stop()

    combined["FY"] = combined["FY"].astype(str).str.strip().apply(fy_std)
    combined["ValueNum"] = pd.to_numeric(combined["Value"], errors="coerce")

    combined["sort_key"] = combined["FY"].apply(sort_fy_only)
    combined = combined.sort_values(["sort_key", "Metric", "Type"]).drop(columns="sort_key")

    fig = px.line(
        combined,
        x="FY", y="ValueNum",
        color="Metric",
        line_dash="Type",
        markers=True,
        title=f"{selected_school_b} — Budget Metrics (Actual ≤ {train_through_b} • Forecast → FY{END_FORECAST_FY})"
    )
    fig.update_traces(connectgaps=True)

    fig.update_xaxes(
        categoryorder="array",
        categoryarray=FY22_TO_FY28,
        tickangle=45,
        tickfont=dict(size=BASE_LABEL_FONT_SIZE)
    )

    # Best practice band/line if ratio included
    if "Budget to Enrollment Ratio" in selected_metrics_b:
        fig = add_best_practice_budget_ratio(fig)

    fig.update_layout(
        height=BASE_HEIGHT_TALLER,
        font=dict(size=BASE_FONT_SIZE),
        legend_font=dict(size=BASE_FONT_SIZE),
        margin=dict(t=70, b=90)
    )
    st.plotly_chart(fig, use_container_width=True)

    if show_model_info:
        st.markdown("### 🧠 Model Selection & Error (by Metric)")
        st.dataframe(pd.DataFrame(model_rows))

# =========================
# CSAF METRICS + OTHER METRICS (LINE + BEST PRACTICES)
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
        st.warning("⚠️ Try adjusting your filters.")
        st.stop()

    filtered = filtered.copy()
    filtered["sort_key"] = filtered["Fiscal Year"].apply(sort_fy)
    filtered = filtered.sort_values("sort_key")
    filtered["FY Group"] = filtered["Fiscal Year"].str.split().str[0]
    filtered["ValueNum"] = pd.to_numeric(filtered["Value"], errors="coerce")

    facet_args = {}
    if len(selected_schools) > 1 and len(selected_metrics) > 1:
        facet_args = {"facet_row": "Schools", "facet_col": "Metric", "facet_col_wrap": 2}
    elif len(selected_schools) > 1:
        facet_args = {"facet_col": "Schools", "facet_col_wrap": 2}
    elif len(selected_metrics) > 1:
        facet_args = {"facet_col": "Metric", "facet_col_wrap": 2}

    # LINE chart (continuous)
    fig = px.line(
        filtered,
        x="Fiscal Year", y="ValueNum",
        color="FY Group",
        markers=True,
        title=", ".join(selected_metrics) if selected_metrics else "Metrics",
        **facet_args
    )
    fig.update_traces(connectgaps=True)

    fiscal_order = sorted(filtered["Fiscal Year"].unique(), key=sort_fy)
    fig.update_xaxes(categoryorder="array", categoryarray=fiscal_order, tickangle=45, tickfont=dict(size=BASE_LABEL_FONT_SIZE))

    # Best practice for CSAF when one metric is selected
    if metric_group == "CSAF Metrics" and len(selected_metrics) == 1:
        fig = add_best_practice_csaf(fig, selected_metrics[0])

    fig = thicken_and_enlarge(fig, height=BASE_HEIGHT_TALL)
    st.plotly_chart(fig, use_container_width=True)
