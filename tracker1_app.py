import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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
# THEME / BACKGROUND (match your screenshot look)
# =========================
APP_BG = "#dfe7df"     # soft green/gray
PLOT_BG = "#dfe7df"
GRID_CLR = "rgba(0,0,0,0.12)"

st.set_page_config(page_title="NOLA Financial Tracker", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {APP_BG};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {APP_BG};
    }}
    section[data-testid="stSidebar"] * {{ font-size: 13px !important; }}
    section[data-testid="stSidebar"] label {{ font-size: 13px !important; }}
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stMultiSelect,
    section[data-testid="stSidebar"] .stRadio,
    section[data-testid="stSidebar"] .stCheckbox,
    section[data-testid="stSidebar"] .stSlider {{
        margin-bottom: 0.40rem !important;
    }}
    section[data-testid="stSidebar"] div[role="listbox"] div {{ font-size: 12px !important; }}
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
# CONSTANTS
# =========================
BASE_FONT_SIZE = 18
BASE_LABEL_FONT_SIZE = 16
BASE_TEXT_FONT_SIZE = 16
BASE_HEIGHT_TALL = 880        # ⬅ bigger overall
BASE_HEIGHT_TALLER = 980      # ⬅ bigger overall
CSAF_PRED_HEIGHT = 980        # ⬅ bigger CSAF predicted bars

START_FY = 22
END_ACTUAL_FY = 26
END_FORECAST_FY = 28

def fy_label(y: int) -> str:
    return f"FY{int(y):02d}"

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

FY22_TO_FY28 = full_fy_range(START_FY, END_FORECAST_FY)

def std_fyq_label(x):
    s = str(x).strip()
    parts = s.split()
    if not parts:
        return s
    parts[0] = fy_std(parts[0])
    return " ".join(parts)

def apply_plot_style(fig, height):
    fig.update_layout(
        height=height,
        font=dict(size=BASE_FONT_SIZE),
        legend_font=dict(size=BASE_FONT_SIZE),
        margin=dict(t=70, b=90),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
    )
    fig.update_xaxes(tickfont=dict(size=BASE_LABEL_FONT_SIZE), showgrid=False)
    fig.update_yaxes(tickfont=dict(size=BASE_LABEL_FONT_SIZE), gridcolor=GRID_CLR)
    return fig

def guard_growth(y_future, last_val, max_up=1.25, max_down=0.75, lower=0.0, upper=None):
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
# CSAF BEST PRACTICES (ONLY CSAF)
# =========================
csaf_metrics = ["FB Ratio", "Liabilities to Assets", "Current Ratio", "Unrestricted Days COH"]
csaf_descriptions = {
    "FB Ratio": {"desc": "Unrestricted FB ÷ Total Exp. (Best practice ≥ 10%)", "threshold": 0.10, "direction": "gte"},
    "Liabilities to Assets": {"desc": "Total Liabilities ÷ Total Assets (Best practice ≤ 0.90)", "threshold": 0.90, "direction": "lte"},
    "Current Ratio": {"desc": "Current Assets ÷ Current Liabilities (Best practice ≥ 1.5)", "threshold": 1.50, "direction": "gte"},
    "Unrestricted Days COH": {"desc": "Unrestricted Cash ÷ ((Total Exp. - Depreciation) ÷ 365) (Best practice ≥ 60)", "threshold": 60, "direction": "gte"},
}

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

# =========================
# LOAD DATA (FINANCIALS)
# =========================
fy25_path = "FY25.xlsx"
try:
    df = pd.read_excel(fy25_path, sheet_name="FY25", header=0)
except Exception as e:
    st.error(f"❌ Could not load {fy25_path}: {e}")
    st.stop()

df.columns = df.columns.str.strip()
df = df.dropna(subset=["Schools", "Fiscal Year"])
df["Fiscal Year"] = df["Fiscal Year"].astype(str).str.strip().apply(std_fyq_label)

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
# LOAD DATA (BUDGET / ENROLLMENT)
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
    df_budget_raw["Fiscal Year"] = df_budget_raw["Fiscal Year"].astype(str).str.strip().apply(fy_std)

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
# MODEL UTILITIES (minimal error selection)
# =========================
def make_feats_quarters(q_idx):
    # q_idx: 1..3
    Q2 = (q_idx == 2).astype(int)
    Q3 = (q_idx == 3).astype(int)
    return np.column_stack([Q2, Q3])

def ts_cv_mae(model, X, y, splits=3):
    if len(y) < 5:
        return np.inf
    n_splits = min(splits, max(2, len(y) - 2))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    for tr, te in tscv.split(X):
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        maes.append(mean_absolute_error(y[te], pred))
    return float(np.mean(maes)) if maes else np.inf

def best_model_forecast_yearly(y, n_future, is_ratio=False):
    """
    Yearly data forecaster (Budget/Enrollment): choose best model by time-series CV on log1p.
    """
    if len(y) == 0:
        return np.zeros(n_future), "None", np.inf

    y_clip = np.clip(y, 0, None)
    if is_ratio:
        y_clip = np.clip(y_clip, 0.0, 1.5)

    y_log = np.log1p(y_clip)
    t = np.arange(len(y_log)).reshape(-1, 1).astype(float)
    X = np.hstack([t, t**2])

    candidates = {
        "Huber": HuberRegressor(),
        "Linear": LinearRegression(),
        "HGBR": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.08, max_iter=600, random_state=42),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(16, 8),
                activation="relu",
                solver="adam",
                max_iter=9000,
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
    best_mdl = candidates[best_name]
    best_err = scores[best_name]

    best_mdl.fit(X, y_log)
    tf = np.arange(len(y_log), len(y_log) + n_future).reshape(-1, 1).astype(float)
    Xf = np.hstack([tf, tf**2])

    y_pred = np.expm1(best_mdl.predict(Xf))
    y_pred = np.clip(y_pred, 0, None)

    last_val = float(y_clip[-1]) if len(y_clip) else 0.0
    if is_ratio:
        y_pred = np.clip(y_pred, 0.0, 1.5)
        y_pred = guard_growth(y_pred, last_val, max_up=1.20, max_down=0.80, lower=0.0, upper=1.5)
    else:
        hist_max = float(np.nanmax(y_clip)) if len(y_clip) else 0.0
        cap = max(hist_max * 1.6, last_val * 1.5, 1.0)
        y_pred = guard_growth(y_pred, last_val, max_up=1.25, max_down=0.75, lower=0.0, upper=cap)

    return y_pred, best_name, best_err

# =========================
# CSAF PREDICTED (BAR, BIG)
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

    def quarter_index(labels):
        qs = []
        for lbl in labels:
            _, q = parse_fyq(lbl)
            qs.append(q if q is not None else np.nan)
        return np.array(qs, dtype=float)

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

    if not run_pred:
        st.info("Pick School, Metric, History, Freeze Origin, then click **Run Prediction**.")
        st.stop()

    hist_df = df[(df["Schools"] == selected_school) & (df["Fiscal Year"].isin(selected_fy_hist))].copy()
    if hist_df.empty or selected_metric not in hist_df.columns:
        st.warning("⚠️ Missing data for selection.")
        st.stop()

    hist_df["sort_key"] = hist_df["Fiscal Year"].apply(sort_fy)
    cut_key = sort_fy(train_through)
    hist_df = hist_df[hist_df["sort_key"].apply(lambda k: k <= cut_key)].sort_values("sort_key")

    y = pd.to_numeric(hist_df[selected_metric], errors="coerce").values.astype(float)
    labs = hist_df["Fiscal Year"].astype(str).tolist()
    q = quarter_index(labs)

    m = ~np.isnan(y) & ~np.isnan(q)
    y, q = y[m], q[m]
    labs = [labs[i] for i, ok in enumerate(m) if ok]

    if len(y) < 5:
        st.warning("⚠️ Not enough points for reliable forecast (need ≥ 5).")
        st.stop()

    # ---- Advanced CSAF prediction: pick best model by time-series CV ----
    # Features: time + season dummies (Q2/Q3)
    t = np.arange(len(y)).reshape(-1, 1).astype(float)
    seas = make_feats_quarters(q.astype(int))
    X = np.hstack([t, t**2, seas])

    y_clip = np.clip(y, 0, None)
    y_log = np.log1p(y_clip)

    candidates = {
        "Huber": HuberRegressor(),
        "Linear": LinearRegression(),
        "HGBR": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.08, max_iter=700, random_state=42),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(24, 12),
                activation="relu",
                solver="adam",
                max_iter=12000,
                random_state=42,
                learning_rate="adaptive",
                early_stopping=True,
                n_iter_no_change=140
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
    best_mdl = candidates[best_name]
    best_mdl.fit(X, y_log)

    future_labels = make_future_labels(train_through, n_future, quarters_per_year=3)

    # future quarter index
    q_last = int(q[-1])
    qf = ((q_last + np.arange(1, n_future + 1) - 1) % 3) + 1
    tf = np.arange(len(y), len(y) + n_future).reshape(-1, 1).astype(float)
    Xf = np.hstack([tf, tf**2, make_feats_quarters(qf)])

    y_future = np.expm1(best_mdl.predict(Xf))
    y_future = np.clip(y_future, 0, None)

    # ---- Combine Actual + Forecast ----
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

    # labels
    def fmt_csaf(v):
        if pd.isna(v):
            return ""
        if selected_metric == "FB Ratio":
            return f"{v:.0%}"
        if selected_metric in ("Liabilities to Assets", "Current Ratio"):
            return f"{v:.2f}"
        return f"{v:,.0f}"

    combined["TextLabel"] = combined["Value"].apply(fmt_csaf)

    fig = px.bar(
        combined,
        x="Quarter", y="Value",
        color="Type",
        barmode="group",
        text="TextLabel",
        title=f"{selected_school} — {selected_metric} (Best model: {best_name})"
    )
    fig.update_traces(textposition="outside", textfont_size=BASE_TEXT_FONT_SIZE)
    fig.update_xaxes(tickangle=45)
    fig = add_best_practice_csaf(fig, selected_metric)
    fig = apply_plot_style(fig, CSAF_PRED_HEIGHT)
    fig.update_layout(bargap=0.05, bargroupgap=0.03)  # thicker bars look
    st.plotly_chart(fig, use_container_width=True)

# =========================
# BUDGET TO ENROLLMENT (ACTUAL) — selector Bar/Line/Combo (NO best practice)
# =========================
elif metric_group == "Budget to Enrollment":
    st.markdown("## 📊 Budget to Enrollment (Actuals)")

    chart_mode = st.sidebar.selectbox("📈 Enrollment Chart Type:", ["Line (markers)", "Bar", "Combo (Bar + Line + markers)"], index=0)

    selected_schools = st.sidebar.multiselect("🏫 Select School(s):", school_options_budget)

    fy_all_b = sorted(df_budget_long["Fiscal Year"].dropna().astype(str).unique(), key=sort_fy_only)
    default_fy_actuals = [fy for fy in fy_all_b if START_FY <= sort_fy_only(fy) <= END_ACTUAL_FY]
    selected_fy_b = st.sidebar.multiselect("📅 Select Fiscal Year(s):", fy_all_b, default=default_fy_actuals or fy_all_b)

    metrics_list = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    metrics_list = [m for m in metrics_list if m in df_budget_long["Metric"].unique()]
    selected_metrics = st.sidebar.multiselect("📌 Select Metrics:", metrics_list, default=["October 1 Count", "February 1 Count"])

    if not selected_schools:
        st.warning("⚠️ Select at least one school.")
        st.stop()

    axis_order = [fy for fy in FY22_TO_FY28 if fy in selected_fy_b]

    # one chart per school (avoids facet-axis crashes)
    for school in selected_schools:
        df_f = df_budget_long[
            (df_budget_long["Schools"] == school) &
            (df_budget_long["Metric"].isin(selected_metrics)) &
            (df_budget_long["Fiscal Year"].isin(selected_fy_b))
        ].copy()

        df_f["Fiscal Year"] = df_f["Fiscal Year"].astype(str).str.strip().apply(fy_std)
        df_f["ValueNum"] = pd.to_numeric(df_f["Value"], errors="coerce")
        df_f = df_f.dropna(subset=["ValueNum"])

        if df_f.empty:
            st.info(f"ℹ️ No enrollment data for {school} with current filters.")
            continue

        df_f["sort_key"] = df_f["Fiscal Year"].apply(sort_fy_only)
        df_f = df_f.sort_values(["Metric", "sort_key"])

        # split ratio vs others (ratio can cause “zigzag” scale issues)
        is_ratio_selected = "Budget to Enrollment Ratio" in selected_metrics
        df_ratio = df_f[df_f["Metric"] == "Budget to Enrollment Ratio"].copy()
        df_main = df_f[df_f["Metric"] != "Budget to Enrollment Ratio"].copy()

        fig = go.Figure()

        # bars (main metrics)
        if chart_mode in ["Bar", "Combo (Bar + Line + markers)"]:
            for met in df_main["Metric"].unique():
                d = df_main[df_main["Metric"] == met].copy()
                d = d.sort_values("sort_key")
                fig.add_trace(go.Bar(
                    x=d["Fiscal Year"],
                    y=d["ValueNum"],
                    name=met,
                    marker_color=budget_metric_color_map.get(met, None),
                    text=[f"{v:,.0f}" for v in d["ValueNum"]],
                    textposition="outside"
                ))

        # lines (main metrics) — clean connected straight lines (no zigzag: sorted)
        if chart_mode in ["Line (markers)", "Combo (Bar + Line + markers)"]:
            for met in df_main["Metric"].unique():
                d = df_main[df_main["Metric"] == met].copy()
                d = d.sort_values("sort_key")
                fig.add_trace(go.Scatter(
                    x=d["Fiscal Year"],
                    y=d["ValueNum"],
                    mode="lines+markers+text",
                    name=f"{met} (Line)",
                    text=[f"{v:,.0f}" for v in d["ValueNum"]],
                    textposition="top center",
                    line=dict(width=3),
                    connectgaps=True,
                    showlegend=False if chart_mode == "Combo (Bar + Line + markers)" else True
                ))

        # ratio on secondary axis (optional)
        if is_ratio_selected and not df_ratio.empty:
            df_ratio = df_ratio.sort_values("sort_key")
            fig.add_trace(go.Scatter(
                x=df_ratio["Fiscal Year"],
                y=df_ratio["ValueNum"],
                mode="lines+markers+text",
                name="Budget to Enrollment Ratio",
                text=[f"{v:.0%}" for v in df_ratio["ValueNum"]],
                textposition="bottom center",
                line=dict(width=3),
                connectgaps=True,
                yaxis="y2"
            ))

        fig.update_layout(
            title=f"Budget / Enrollment Trend: {school}",
            barmode="group",
            yaxis=dict(title="Budget / Counts"),
            yaxis2=dict(
                title="Ratio",
                overlaying="y",
                side="right",
                range=[0, 1.5]
            ),
        )
        fig.update_xaxes(categoryorder="array", categoryarray=axis_order, tickangle=0)
        fig = apply_plot_style(fig, BASE_HEIGHT_TALLER)
        st.plotly_chart(fig, use_container_width=True)

# =========================
# BUDGET TO ENROLLMENT PREDICTED — selector Bar/Line/Combo (NO best practice)
# =========================
elif metric_group == "Budget to Enrollment Predicted":
    st.markdown("## 🔮 Budget to Enrollment Predicted (Actual ≤ Freeze • Forecast → FY28)")

    if df_budget_long.empty:
        st.warning("⚠️ Budget dataset not loaded.")
        st.stop()

    chart_mode = st.sidebar.selectbox("📈 Enrollment Prediction Chart Type:", ["Line (markers)", "Bar", "Combo (Bar + Line + markers)"], index=2)

    selected_school_b = st.sidebar.selectbox("🏫 Select School:", sorted(df_budget_long["Schools"].dropna().unique()))

    metrics_all = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    metrics_all = [m for m in metrics_all if m in df_budget_long["Metric"].unique()]
    selected_metrics_b = st.sidebar.multiselect("📌 Choose Metric(s):", metrics_all, default=["October 1 Count", "February 1 Count"])

    fy_all = sorted(df_budget_long["Fiscal Year"].dropna().astype(str).unique(), key=sort_fy_only)
    default_hist = [fy for fy in fy_all if START_FY <= sort_fy_only(fy) <= END_ACTUAL_FY]
    selected_fy_hist_b = st.sidebar.multiselect("📅 History FYs (training):", fy_all, default=default_hist or fy_all)

    default_origin = fy_label(END_ACTUAL_FY)
    origin_index = fy_all.index(default_origin) if default_origin in fy_all else max(0, len(fy_all) - 1)
    train_through_b = st.sidebar.selectbox("🧊 Freeze at:", fy_all, index=origin_index)

    show_model_info = st.sidebar.checkbox("Show model + error details", value=True)

    run_pred_b = st.sidebar.button("▶ Run Budget Prediction")
    if not run_pred_b:
        st.info("Pick School, Metrics, History, Freeze FY, then click **Run Budget Prediction**.")
        st.stop()

    origin_year = sort_fy_only(train_through_b)
    n_future_b = max(0, END_FORECAST_FY - origin_year)
    future_labels = [fy_label(y) for y in range(origin_year + 1, END_FORECAST_FY + 1)]
    axis_order = FY22_TO_FY28

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

        is_ratio = (met == "Budget to Enrollment Ratio")
        if n_future_b > 0 and len(y) >= 3:
            y_future, best_name, best_err = best_model_forecast_yearly(y, n_future_b, is_ratio=is_ratio)
        elif n_future_b > 0 and len(y) == 2:
            slope = (y[-1] - y[-2])
            last = y[-1]
            y_future = np.array([max(0.0, last + slope * (i + 1)) for i in range(n_future_b)], dtype=float)
            best_name, best_err = "2pt-trend", np.nan
        elif n_future_b > 0 and len(y) == 1:
            y_future = np.array([max(0.0, y[0]) for _ in range(n_future_b)], dtype=float)
            best_name, best_err = "carry-forward", np.nan
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
    combined = combined.dropna(subset=["ValueNum"])

    combined["sort_key"] = combined["FY"].apply(sort_fy_only)
    combined = combined.sort_values(["Metric", "Type", "sort_key"])

    axis_order = FY22_TO_FY28

    # ---- Create combo chart: Bar + Line ----
    fig = go.Figure()

    # Bar traces (grouped by Metric, with pattern via Type)
    # To keep it readable: bars grouped by Metric, and Type shown via opacity + legend label.
    # (You can change this later if you want side-by-side Actual vs Forecast bars.)
    for met in selected_metrics_b:
        dmet = combined[combined["Metric"] == met].copy()
        if dmet.empty:
            continue
        # Bar per Type
        for tname in ["Actual", "Forecast (Frozen)"]:
            dt = dmet[dmet["Type"] == tname].copy()
            if dt.empty:
                continue
            fig.add_trace(
                go.Bar(
                    x=dt["FY"],
                    y=dt["ValueNum"],
                    name=f"{met} — {tname}",
                    legendgroup=f"{met}-{tname}",
                    opacity=0.85 if tname == "Actual" else 0.55,
                    text=[("" if pd.isna(v) else (f"{v:.0%}" if met == "Budget to Enrollment Ratio" else f"{v:,.0f}")) for v in dt["ValueNum"]],
                    textposition="outside",
                )
            )

    # Line traces (comparison lines) — one per Metric, with dash by Type
    for met in selected_metrics_b:
        dmet = combined[combined["Metric"] == met].copy()
        if dmet.empty:
            continue
        for tname, dash in [("Actual", "solid"), ("Forecast (Frozen)", "dash")]:
            dt = dmet[dmet["Type"] == tname].copy()
            if dt.empty:
                continue
            dt = dt.sort_values("sort_key")
            fig.add_trace(
                go.Scatter(
                    x=dt["FY"],
                    y=dt["ValueNum"],
                    mode="lines+markers",
                    name=f"{met} Line — {tname}",
                    line=dict(dash=dash),
                    connectgaps=True,
                    showlegend=False,  # keep legend clean (bars already label types)
                )
            )

    fig.update_layout(
        barmode="group",
        title=f"{selected_school_b} — Budget Metrics (Actual ≤ {train_through_b} • Forecast → FY{END_FORECAST_FY})",
        height=BASE_HEIGHT_TALLER,
        font=dict(size=BASE_FONT_SIZE),
        legend_font=dict(size=BASE_FONT_SIZE),
        margin=dict(t=70, b=90)
    )

    fig.update_xaxes(
        categoryorder="array",
        categoryarray=axis_order,
        tickangle=45,
        tickfont=dict(size=BASE_LABEL_FONT_SIZE)
    )

    st.plotly_chart(fig, use_container_width=True)

    if show_model_info:
        st.markdown("### 🧠 Model Selection & Error (by Metric)")
        st.dataframe(pd.DataFrame(model_rows))

# =========================
# CSAF METRICS + OTHER METRICS (BAR) + CSAF BEST PRACTICE
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

    def fmt_general(metric, v):
        if pd.isna(v) or v == 0:
            return ""
        if metric in dollar_metrics:
            return f"${v:,.0f}"
        if metric == "FB Ratio":
            return f"{v:.0%}"
        if metric in ("Liabilities to Assets", "Current Ratio"):
            return f"{v:.2f}"
        if metric == "Unrestricted Days COH":
            return f"{v:,.0f}"
        return f"{v:,.0f}"

    filtered["TextLabel"] = [fmt_general(m, v) for m, v in zip(filtered["Metric"], filtered["ValueNum"])]

    facet_args = {}
    if len(selected_schools) > 1 and len(selected_metrics) > 1:
        facet_args = {"facet_row": "Schools", "facet_col": "Metric", "facet_col_wrap": 2}
    elif len(selected_schools) > 1:
        facet_args = {"facet_col": "Schools", "facet_col_wrap": 2}
    elif len(selected_metrics) > 1:
        facet_args = {"facet_col": "Metric", "facet_col_wrap": 2}

    fig = px.bar(
        filtered,
        x="Fiscal Year",
        y="ValueNum",
        color="FY Group",
        color_discrete_map=fy_color_map,
        barmode="group",
        text="TextLabel",
        title=", ".join(selected_metrics) if selected_metrics else "Metrics",
        **facet_args
    )

    fiscal_order = sorted(filtered["Fiscal Year"].unique(), key=sort_fy)
    fig.update_xaxes(categoryorder="array", categoryarray=fiscal_order, tickangle=45, tickfont=dict(size=BASE_LABEL_FONT_SIZE))
    fig.update_traces(textposition="outside", textfont_size=BASE_TEXT_FONT_SIZE)

    # CSAF best practice line only when one CSAF metric is selected
    if metric_group == "CSAF Metrics" and len(selected_metrics) == 1:
        fig = add_best_practice_csaf(fig, selected_metrics[0])

    fig = thicken_and_enlarge(fig, height=BASE_HEIGHT_TALL)
    st.plotly_chart(fig, use_container_width=True)

