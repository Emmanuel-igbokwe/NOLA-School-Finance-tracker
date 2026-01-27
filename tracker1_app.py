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

# ============================================================
# THEME / BACKGROUND
# ============================================================
APP_BG = "#dfe7df"      # soft green/gray
PLOT_BG = "#dfe7df"
GRID_CLR = "rgba(0,0,0,0.10)"

st.set_page_config(page_title="NOLA Financial Tracker", layout="wide")

# ---- HARD FIX: remove Streamlit default top padding + wide whitespace ----
st.markdown(
    f"""
    <style>
      .stApp {{
        background-color: {APP_BG};
      }}

      /* REMOVE TOP WHITE SPACE */
      header[data-testid="stHeader"] {{
        background: {APP_BG};
      }}
      .block-container {{
        padding-top: 0.6rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;   /* stops the super-wide layout */
      }}

      /* Sidebar styling */
      section[data-testid="stSidebar"] {{
        background-color: {APP_BG};
      }}
      section[data-testid="stSidebar"] * {{
        font-size: 13px !important;
      }}
      section[data-testid="stSidebar"] .stSelectbox,
      section[data-testid="stSidebar"] .stMultiSelect,
      section[data-testid="stSidebar"] .stRadio,
      section[data-testid="stSidebar"] .stCheckbox,
      section[data-testid="stSidebar"] .stSlider {{
        margin-bottom: 0.45rem !important;
      }}

      /* Reduce empty space under title blocks */
      h1, h2, h3 {{
        margin-bottom: 0.4rem !important;
      }}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================
logo_path = "nola_parish_logo.png"
col1, col2 = st.columns([1, 10], vertical_alignment="center")

with col1:
    if os.path.exists(logo_path):
        st.image(logo_path, width=70)

with col2:
    st.markdown(
        """
        <div style="padding-top:6px;">
          <div style="color:#003366; font-size:24px; font-weight:800; line-height:1.2;">
            Welcome to NOLA Public Schools Finance Accountability App
          </div>
          <div style="color:#2b2b2b; font-size:14px; margin-top:4px;">
            NOLA Schools Financial Tracker • Built by Emmanuel Igbokwe
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# ============================================================
# CONSTANTS
# ============================================================
BASE_FONT_SIZE = 18
AXIS_FONT = 16
TEXT_FONT = 16

CHART_HEIGHT = 760
CHART_HEIGHT_TALL = 860

# Thicker bars + bigger labels
BARGAP = 0.14
BARGROUPGAP = 0.08

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

def apply_plot_style(fig, height=CHART_HEIGHT):
    fig.update_layout(
        height=height,
        font=dict(size=BASE_FONT_SIZE),
        legend_font=dict(size=16),
        margin=dict(t=80, b=100, l=30, r=30),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(tickfont=dict(size=AXIS_FONT), showgrid=False)
    fig.update_yaxes(tickfont=dict(size=AXIS_FONT), gridcolor=GRID_CLR)
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

# ============================================================
# CSAF METRICS
# ============================================================
csaf_metrics = ["FB Ratio", "Liabilities to Assets", "Current Ratio", "Unrestricted Days COH"]
csaf_descriptions = {
    "FB Ratio": {"threshold": 0.10, "direction": "gte"},
    "Liabilities to Assets": {"threshold": 0.90, "direction": "lte"},
    "Current Ratio": {"threshold": 1.50, "direction": "gte"},
    "Unrestricted Days COH": {"threshold": 60, "direction": "gte"},
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

def fmt_csaf(metric, v):
    if pd.isna(v):
        return ""
    if metric == "FB Ratio":
        return f"{v:.0%}"
    if metric in ("Liabilities to Assets", "Current Ratio"):
        return f"{v:.2f}"
    return f"{v:,.0f}"

# ============================================================
# LOAD DATA (FINANCIALS)
# ============================================================
fy25_path = "FY25.xlsx"
try:
    df = pd.read_excel(fy25_path, sheet_name="FY25", header=0)
except Exception as e:
    st.error(f"❌ Could not load {fy25_path}: {e}")
    st.stop()

df.columns = df.columns.str.strip()
df = df.dropna(subset=["Schools", "Fiscal Year"])
df["Fiscal Year"] = df["Fiscal Year"].astype(str).str.strip().apply(std_fyq_label)

fiscal_options = sorted(df["Fiscal Year"].dropna().unique(), key=sort_fy)

# ============================================================
# LOAD DATA (BUDGET / ENROLLMENT)
# ============================================================
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

    df_budget_raw = df_budget_raw.dropna(subset=["Schools", "Fiscal Year"]).copy()
    df_budget_raw["Fiscal Year"] = df_budget_raw["Fiscal Year"].astype(str).str.strip().apply(fy_std)

    expected_cols = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    df_budget_long = df_budget_raw.melt(
        id_vars=["Schools", "Fiscal Year"],
        value_vars=[c for c in expected_cols if c in df_budget_raw.columns],
        var_name="Metric",
        value_name="Value"
    )

    school_options_budget = sorted(df_budget_long["Schools"].dropna().unique())
    fy_options_budget = sorted(df_budget_long["Fiscal Year"].dropna().unique(), key=sort_fy_only)

except Exception as e:
    st.warning(f"⚠️ Could not load '{fy26_path}' / sheet 'FY26 Student enrollment': {e}")
    df_budget_long = pd.DataFrame()
    school_options_budget, fy_options_budget = [], []

# Bright, readable colors
budget_color_map = {
    "Budgetted": "#0057B8",         # bright blue
    "October 1 Count": "#00A676",   # bright green
    "February 1 Count": "#E53935",  # bright red
    "Budget to Enrollment Ratio": "#FB8C00",  # bright orange
}

# ============================================================
# SIDEBAR / UI
# ============================================================
st.sidebar.header("🔎 Filters")

modes = ["CSAF Metrics", "CSAF Predicted", "Other Metrics"]
if not df_budget_long.empty:
    modes += ["Budget to Enrollment (Bar)", "Budget to Enrollment Predicted (Bar)"]

metric_group = st.sidebar.radio("Choose Dashboard:", modes)

# ============================================================
# MODEL UTILITIES
# ============================================================
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

# ============================================================
# CSAF PREDICTED (BIG BAR + ML TYPES BACK)
# ============================================================
if metric_group == "CSAF Predicted":
    st.markdown("## 🔮 CSAF Predicted Metrics (Big Bars + ML Models)")

    schools = sorted(df["Schools"].dropna().unique())
    selected_school = st.sidebar.selectbox("🏫 Select School:", schools)

    selected_metric = st.sidebar.selectbox("📊 Select CSAF Metric:", csaf_metrics)

    all_quarters = sorted(df["Fiscal Year"].dropna().astype(str).unique(), key=sort_fy)
    train_through = st.sidebar.selectbox("🧊 Freeze at (Forecast Origin):", all_quarters, index=len(all_quarters)-1)

    n_future = st.sidebar.slider("🔮 Forecast horizon (quarters)", 3, 12, 6)
    run_pred = st.sidebar.button("▶ Run CSAF Prediction")

    if not run_pred:
        st.info("Pick School + Metric + Freeze + Horizon, then click **Run CSAF Prediction**.")
        st.stop()

    # ---- Build training series (up to freeze) ----
    hist_df = df[df["Schools"] == selected_school].copy()
    hist_df["sort_key"] = hist_df["Fiscal Year"].apply(sort_fy)
    cut_key = sort_fy(train_through)
    hist_df = hist_df[hist_df["sort_key"].apply(lambda k: k <= cut_key)].sort_values("sort_key")

    y = pd.to_numeric(hist_df[selected_metric], errors="coerce").values.astype(float)
    labs = hist_df["Fiscal Year"].astype(str).tolist()

    # quarter parsing
    fyq_re = re.compile(r"FY\s*(\d{2,4})\s*Q\s*(\d)", re.IGNORECASE)
    def parse_fyq(label: str):
        m = fyq_re.search(str(label))
        if not m:
            return None, None
        fy = fy_num("FY" + m.group(1))
        q = int(m.group(2))
        return fy, q

    q_arr = []
    for lbl in labs:
        _, q = parse_fyq(lbl)
        q_arr.append(q if q is not None else np.nan)
    q_arr = np.array(q_arr, dtype=float)

    mask = ~np.isnan(y) & ~np.isnan(q_arr)
    y = y[mask]
    q_arr = q_arr[mask].astype(int)
    labs = [labs[i] for i, ok in enumerate(mask) if ok]

    if len(y) < 5:
        st.warning("⚠️ Not enough points for reliable forecast (need ≥ 5).")
        st.stop()

    # features: time + quadratic + seasonal dummies
    def feats_quarter_dummies(q_int):
        Q2 = (q_int == 2).astype(int).reshape(-1, 1)
        Q3 = (q_int == 3).astype(int).reshape(-1, 1)
        return np.hstack([Q2, Q3])

    y_clip = np.clip(y, 0, None)
    y_log = np.log1p(y_clip)

    t = np.arange(len(y_log)).reshape(-1, 1).astype(float)
    X = np.hstack([t, t**2, feats_quarter_dummies(q_arr)])

    # ---- ML TYPES BACK (Robust / Trend / Sesame(SES) + extra models) ----
    candidates = {
        "Robust (Huber)": HuberRegressor(),
        "Trend (Linear)": LinearRegression(),
        "HGBR": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.08, max_iter=900, random_state=42),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(28, 14),
                activation="relu",
                solver="adam",
                max_iter=15000,
                random_state=42,
                learning_rate="adaptive",
                early_stopping=True,
                n_iter_no_change=180
            )
        ),
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

    # future labels
    def make_future_labels(last_label: str, n: int, q_per_year=3):
        fy, q = parse_fyq(last_label)
        if fy is None:
            fy, q = END_ACTUAL_FY, 0
        out = []
        for _ in range(n):
            q += 1
            if q > q_per_year:
                fy += 1
                q = 1
            out.append(f"FY{fy:02d} Q{q}")
        return out

    future_labels = make_future_labels(train_through, n_future, q_per_year=3)

    q_last = int(q_arr[-1])
    qf = ((q_last + np.arange(1, n_future + 1) - 1) % 3) + 1
    tf = np.arange(len(y_log), len(y_log) + n_future).reshape(-1, 1).astype(float)
    Xf = np.hstack([tf, tf**2, feats_quarter_dummies(qf)])

    y_future = np.expm1(best_mdl.predict(Xf))
    y_future = np.clip(y_future, 0, None)

    # guard growth
    hist_max = float(np.nanmax(y_clip))
    last_val = float(y_clip[-1])
    cap = max(hist_max * 1.6, last_val * 1.5, 1.0)
    y_future = guard_growth(y_future, last_val, max_up=1.25, max_down=0.75, lower=0.0, upper=cap)

    # combined display (Actual all + Forecast)
    actual_now = df[df["Schools"] == selected_school].copy()
    actual_now["sort_key"] = actual_now["Fiscal Year"].apply(sort_fy)
    actual_now = actual_now.sort_values("sort_key")

    combined = pd.concat(
        [
            pd.DataFrame({
                "Period": actual_now["Fiscal Year"].astype(str),
                "Value": pd.to_numeric(actual_now[selected_metric], errors="coerce"),
                "Type": "Actual"
            }).dropna(subset=["Value"]),
            pd.DataFrame({
                "Period": future_labels,
                "Value": y_future,
                "Type": "Forecast (Frozen)"
            }),
        ],
        ignore_index=True
    )

    combined["Label"] = combined["Value"].apply(lambda v: fmt_csaf(selected_metric, v))

    # ---- BIG BAR CHART (no tiny bars) ----
    fig = px.bar(
        combined,
        x="Period", y="Value",
        color="Type",
        barmode="group",
        text="Label",
        title=f"{selected_school} — {selected_metric}"
    )
    fig.update_traces(
        textposition="outside",
        textfont_size=18
    )
    fig.update_layout(
        bargap=BARGAP,
        bargroupgap=BARGROUPGAP
    )
    fig.update_xaxes(tickangle=35)

    fig = add_best_practice_csaf(fig, selected_metric)
    fig = apply_plot_style(fig, height=CHART_HEIGHT_TALL)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Show ML model types + freeze + horizon clearly ----
    freeze_fy = train_through
    horizon_years = round(n_future / 3.0, 2)  # 3 quarters per FY in your data
    st.markdown(
        f"""
        **Prediction Details**  
        - **Freeze at:** {freeze_fy}  
        - **Horizon:** {n_future} quarters (≈ **{horizon_years} FY**)  
        - **Best model selected (min CV MAE):** **{best_name}**  
        """
    )

    info_df = pd.DataFrame({
        "Model Type": list(scores.keys()),
        "CV MAE (log-scale, lower is better)": [None if not np.isfinite(scores[k]) else float(scores[k]) for k in scores]
    }).sort_values("CV MAE (log-scale, lower is better)")
    st.dataframe(info_df, use_container_width=True)

# ============================================================
# BUDGET / ENROLLMENT (BAR ONLY — CLEAN)
# ============================================================
elif metric_group == "Budget to Enrollment (Bar)":
    st.markdown("## 📊 Budget to Enrollment (Actuals — Bar Only)")

    if df_budget_long.empty:
        st.warning("⚠️ Enrollment dataset not loaded.")
        st.stop()

    selected_school = st.sidebar.selectbox("🏫 Select School:", school_options_budget)

    metrics_list = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    available = sorted(df_budget_long["Metric"].dropna().unique())
    metrics_list = [m for m in metrics_list if m in available]

    selected_metrics = st.sidebar.multiselect(
        "📌 Select Metrics (Bar):",
        metrics_list,
        default=["October 1 Count", "February 1 Count"]
    )

    selected_fy = st.sidebar.multiselect(
        "📅 Select Fiscal Years:",
        fy_options_budget,
        default=[fy for fy in fy_options_budget if START_FY <= sort_fy_only(fy) <= END_ACTUAL_FY] or fy_options_budget
    )

    df_f = df_budget_long[
        (df_budget_long["Schools"] == selected_school) &
        (df_budget_long["Metric"].isin(selected_metrics)) &
        (df_budget_long["Fiscal Year"].isin(selected_fy))
    ].copy()

    df_f["ValueNum"] = pd.to_numeric(df_f["Value"], errors="coerce")
    df_f = df_f.dropna(subset=["ValueNum"])
    df_f["sort_key"] = df_f["Fiscal Year"].apply(sort_fy_only)
    df_f = df_f.sort_values("sort_key")

    if df_f.empty:
        st.warning("⚠️ No data for current filters.")
        st.stop()

    # Labels
    def fmt_enroll(metric, v):
        if metric == "Budget to Enrollment Ratio":
            return f"{v:.0%}"
        return f"{v:,.0f}"

    df_f["Label"] = [fmt_enroll(m, v) for m, v in zip(df_f["Metric"], df_f["ValueNum"])]

    fig = px.bar(
        df_f,
        x="Fiscal Year",
        y="ValueNum",
        color="Metric",
        barmode="group",
        text="Label",
        color_discrete_map=budget_color_map,
        title=f"{selected_school} — Budget & Enrollment"
    )
    fig.update_traces(textposition="outside", textfont_size=18)
    fig.update_layout(bargap=BARGAP, bargroupgap=BARGROUPGAP)
    fig.update_xaxes(categoryorder="array", categoryarray=sorted(df_f["Fiscal Year"].unique(), key=sort_fy_only))

    fig = apply_plot_style(fig, height=CHART_HEIGHT_TALL)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# BUDGET / ENROLLMENT PREDICTED (BAR ONLY)
# ============================================================
elif metric_group == "Budget to Enrollment Predicted (Bar)":
    st.markdown("## 🔮 Budget to Enrollment Predicted (Bar Only)")

    if df_budget_long.empty:
        st.warning("⚠️ Enrollment dataset not loaded.")
        st.stop()

    selected_school = st.sidebar.selectbox("🏫 Select School:", school_options_budget)

    metrics_all = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    metrics_all = [m for m in metrics_all if m in df_budget_long["Metric"].unique()]

    selected_metric = st.sidebar.selectbox("📌 Select One Metric:", metrics_all, index=1)

    freeze_at = st.sidebar.selectbox(
        "🧊 Freeze at:",
        fy_options_budget,
        index=fy_options_budget.index(fy_label(END_ACTUAL_FY)) if fy_label(END_ACTUAL_FY) in fy_options_budget else len(fy_options_budget)-1
    )

    run_pred = st.sidebar.button("▶ Run Enrollment Forecast")
    if not run_pred:
        st.info("Pick School + Metric + Freeze, then click **Run Enrollment Forecast**.")
        st.stop()

    origin_year = sort_fy_only(freeze_at)
    future_years = [fy_label(y) for y in range(origin_year + 1, END_FORECAST_FY + 1)]
    n_future = len(future_years)

    # history <= freeze
    dfh = df_budget_long[
        (df_budget_long["Schools"] == selected_school) &
        (df_budget_long["Metric"] == selected_metric)
    ].copy()

    dfh["sort_key"] = dfh["Fiscal Year"].apply(sort_fy_only)
    dfh = dfh[dfh["sort_key"] <= origin_year].sort_values("sort_key")

    y_hist = pd.to_numeric(dfh["Value"], errors="coerce").dropna().values.astype(float)
    if len(y_hist) < 3:
        st.warning("⚠️ Not enough history points for forecasting (need ≥ 3).")
        st.stop()

    # yearly model selection (simple & stable)
    y_clip = np.clip(y_hist, 0, None)
    is_ratio = (selected_metric == "Budget to Enrollment Ratio")
    if is_ratio:
        y_clip = np.clip(y_clip, 0.0, 1.5)

    y_log = np.log1p(y_clip)
    t = np.arange(len(y_log)).reshape(-1, 1).astype(float)
    X = np.hstack([t, t**2])

    candidates = {
        "Robust (Huber)": HuberRegressor(),
        "Trend (Linear)": LinearRegression(),
        "Sesame (HGBR)": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.08, max_iter=700, random_state=42),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(24, 12),
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

    tf = np.arange(len(y_log), len(y_log) + n_future).reshape(-1, 1).astype(float)
    Xf = np.hstack([tf, tf**2])
    y_future = np.expm1(best_mdl.predict(Xf))
    y_future = np.clip(y_future, 0, None)

    # guard
    last_val = float(y_clip[-1])
    if is_ratio:
        y_future = guard_growth(y_future, last_val, max_up=1.20, max_down=0.80, lower=0.0, upper=1.5)
    else:
        cap = max(np.nanmax(y_clip) * 1.6, last_val * 1.5, 1.0)
        y_future = guard_growth(y_future, last_val, max_up=1.25, max_down=0.75, lower=0.0, upper=cap)

    # actual + forecast dataframe
    actual_df = pd.DataFrame({
        "FY": dfh["Fiscal Year"].astype(str),
        "Value": pd.to_numeric(dfh["Value"], errors="coerce"),
        "Type": "Actual"
    }).dropna()

    forecast_df = pd.DataFrame({
        "FY": future_years,
        "Value": y_future,
        "Type": "Forecast (Frozen)"
    })

    combo = pd.concat([actual_df, forecast_df], ignore_index=True)
    combo["sort_key"] = combo["FY"].apply(sort_fy_only)
    combo = combo.sort_values("sort_key")

    def fmt_pred(v):
        return f"{v:.0%}" if selected_metric == "Budget to Enrollment Ratio" else f"{v:,.0f}"

    combo["Label"] = combo["Value"].apply(fmt_pred)

    fig = px.bar(
        combo,
        x="FY", y="Value",
        color="Type",
        barmode="group",
        text="Label",
        title=f"{selected_school} — {selected_metric} (Freeze at {freeze_at})"
    )
    fig.update_traces(textposition="outside", textfont_size=18)
    fig.update_layout(bargap=BARGAP, bargroupgap=BARGROUPGAP)
    fig.update_xaxes(categoryorder="array", categoryarray=FY22_TO_FY28)

    fig = apply_plot_style(fig, height=CHART_HEIGHT_TALL)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        **Prediction Details**  
        - **Freeze at:** {freeze_at}  
        - **Forecast through:** FY{END_FORECAST_FY:02d} (**{n_future} year(s)**)  
        - **Best model selected:** **{best_name}**
        """
    )

    info_df = pd.DataFrame({
        "Model Type": list(scores.keys()),
        "CV MAE (log-scale, lower is better)": [None if not np.isfinite(scores[k]) else float(scores[k]) for k in scores]
    }).sort_values("CV MAE (log-scale, lower is better)")
    st.dataframe(info_df, use_container_width=True)

# ============================================================
# CSAF METRICS (BAR ONLY + BEST PRACTICE) — BIG
# ============================================================
elif metric_group == "CSAF Metrics":
    st.markdown("## 📌 CSAF Metrics (Actuals — Big Bars)")

    school_options = sorted(df["Schools"].dropna().unique())
    selected_school = st.sidebar.selectbox("🏫 Select School:", school_options)

    selected_metric = st.sidebar.selectbox("📊 Select CSAF Metric:", csaf_metrics)

    selected_fy = st.sidebar.multiselect("📅 Select Fiscal Year + Quarter:", fiscal_options, default=fiscal_options)

    filtered = df[
        (df["Schools"] == selected_school) &
        (df["Fiscal Year"].isin(selected_fy))
    ].copy()

    if filtered.empty:
        st.warning("⚠️ Try adjusting your filters.")
        st.stop()

    filtered["sort_key"] = filtered["Fiscal Year"].apply(sort_fy)
    filtered = filtered.sort_values("sort_key")
    filtered["ValueNum"] = pd.to_numeric(filtered[selected_metric], errors="coerce")
    filtered = filtered.dropna(subset=["ValueNum"])

    filtered["Label"] = filtered["ValueNum"].apply(lambda v: fmt_csaf(selected_metric, v))

    fig = px.bar(
        filtered,
        x="Fiscal Year",
        y="ValueNum",
        text="Label",
        title=f"{selected_school} — {selected_metric}"
    )
    fig.update_traces(textposition="outside", textfont_size=18)
    fig.update_layout(bargap=BARGAP, bargroupgap=BARGROUPGAP)
    fig.update_xaxes(tickangle=35)

    fig = add_best_practice_csaf(fig, selected_metric)
    fig = apply_plot_style(fig, height=CHART_HEIGHT_TALL)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# OTHER METRICS (KEEP SIMPLE)
# ============================================================
else:
    st.markdown("## 📌 Other Metrics (Actuals)")

    # If you still want this section, keep it simple (one school, multi-metric)
    school_options = sorted(df["Schools"].dropna().unique())
    selected_school = st.sidebar.selectbox("🏫 Select School:", school_options)

    selected_fy = st.sidebar.multiselect("📅 Select Fiscal Year + Quarter:", fiscal_options, default=fiscal_options)

    # detect all other metrics from df columns
    all_metrics = [c for c in df.columns if c not in ["Schools", "Fiscal Year"]]
    other_metrics = [m for m in all_metrics if m not in csaf_metrics]
    selected_metrics = st.sidebar.multiselect("📊 Select Metrics:", sorted(other_metrics), default=sorted(other_metrics)[:3])

    filtered = df[
        (df["Schools"] == selected_school) &
        (df["Fiscal Year"].isin(selected_fy))
    ].copy()

    if filtered.empty or not selected_metrics:
        st.warning("⚠️ Try adjusting your filters.")
        st.stop()

    melted = filtered.melt(
        id_vars=["Fiscal Year"],
        value_vars=selected_metrics,
        var_name="Metric",
        value_name="Value"
    )
    melted["ValueNum"] = pd.to_numeric(melted["Value"], errors="coerce")
    melted = melted.dropna(subset=["ValueNum"])
    melted["sort_key"] = melted["Fiscal Year"].apply(sort_fy)
    melted = melted.sort_values("sort_key")

    melted["Label"] = melted["ValueNum"].apply(lambda v: f"{v:,.0f}")

    fig = px.bar(
        melted,
        x="Fiscal Year",
        y="ValueNum",
        color="Metric",
        barmode="group",
        text="Label",
        title=f"{selected_school} — Other Metrics"
    )
    fig.update_traces(textposition="outside", textfont_size=16)
    fig.update_layout(bargap=BARGAP, bargroupgap=BARGROUPGAP)
    fig.update_xaxes(tickangle=35)

    fig = apply_plot_style(fig, height=CHART_HEIGHT_TALL)
    st.plotly_chart(fig, use_container_width=True)
