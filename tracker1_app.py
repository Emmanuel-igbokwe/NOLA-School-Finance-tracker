import os
import re
import math
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

# ============================================================
# THEME / BACKGROUND
# ============================================================
APP_BG = "#dfe7df"
PLOT_BG = "#dfe7df"
GRID_CLR = "rgba(0,0,0,0.10)"

st.set_page_config(page_title="NOLA Financial Tracker", layout="wide")

# NOTE: If you prefer to keep your escaped HTML (&lt;style&gt;), revert the block below.
st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {APP_BG}; }}
      section[data-testid="stSidebar"] {{ background-color: {APP_BG}; }}

      /* remove top whitespace + keep content centered */
      header[data-testid="stHeader"] {{ background: {APP_BG}; }}
      .block-container {{
        padding-top: 1.15rem !important;     /* prevents top header/logo cut */
        padding-bottom: 2rem !important;
        max-width: 1250px !important;        /* stop ultra-wide scrolling */
      }}

      /* sidebar readability */
      section[data-testid="stSidebar"] * {{ font-size: 13px !important; }}
      section[data-testid="stSidebar"] .stSelectbox,
      section[data-testid="stSidebar"] .stMultiSelect,
      section[data-testid="stSidebar"] .stRadio,
      section[data-testid="stSidebar"] .stCheckbox,
      section[data-testid="stSidebar"] .stSlider {{
        margin-bottom: 0.45rem !important;
      }}

      /* ---- Executive KPI Cards (inspired by your HTML dashboard) ---- */
      .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 6px 0 18px 0;
      }}
      @media (max-width: 1100px) {{
        .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
      }}
      .kpi-card {{
        background: #1a1f2e10;  /* faint contrast */
        border: 1px solid rgba(0,0,0,0.10);
        border-radius: 14px;
        padding: 16px 16px 14px 16px;
        transition: all .2s ease;
      }}
      .kpi-card:hover {{
        border-color: #7B61FF33;
        box-shadow: 0 8px 26px rgba(0,0,0,0.06);
        transform: translateY(-1px);
      }}
      .kpi-label {{
        font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
        color: #485569; margin-bottom: 8px; font-weight: 700;
      }}
      .kpi-value {{
        font-size: 28px; font-weight: 800; color:#003366; line-height: 1.0;
      }}
      .kpi-sub {{
        font-size: 12px; color:#1f2937; margin-top: 6px;
      }}
      .kpi-good {{ color:#16a34a; font-weight:700; }}
      .kpi-warn {{ color:#f59e0b; font-weight:700; }}
      .kpi-bad  {{ color:#dc2626; font-weight:700; }}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER (ROTATING LOGO + VISIBLE TITLE)
# ============================================================
logo_path = "nola_parish_logo.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        encoded_logo = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
          .nola-wrap {{
            background: {APP_BG};
            border-bottom: 1px solid rgba(0,0,0,0.10);
            margin-bottom: 8px;
          }}
          .nola-header {{
            display:flex;
            align-items:center;
            gap:14px;
            padding:12px 8px 12px 8px;
          }}
          .nola-title {{
            color:#003366;
            font-size:26px;
            font-weight:900;
            line-height:1.1;
          }}
          .nola-sub {{
            color:#1f1f1f;
            font-size:14px;
            margin-top:4px;
          }}
          .spin {{
            width:74px; height:74px;
            border-radius:50%;
            animation: spin 6s linear infinite;
          }}
          @keyframes spin {{
            from {{ transform: rotate(0deg); }}
            to   {{ transform: rotate(360deg); }}
          }}
        </style>

        <div class="nola-wrap">
          <div class="nola-header">
            <img class="spin" src="data:image/png;base64,{encoded_logo}">
            <div>
              <div class="nola-title">Welcome to NOLA Public Schools Finance Accountability App</div>
              <div class="nola-sub">NOLA Schools Financial Tracker • Built by Emmanuel Igbokwe</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown("### Welcome to NOLA Public Schools Finance Accountability App")
    st.caption("NOLA Schools Financial Tracker • Built by Emmanuel Igbokwe")

st.divider()

# ============================================================
# CONSTANTS
# ============================================================
BASE_FONT_SIZE = 18
AXIS_FONT = 16
TEXT_FONT = 18

CHART_H = 760
CHART_H_TALL = 860

# thicker bars (global)
BARGAP = 0.08
BARGROUPGAP = 0.04

START_FY = 22
END_ACTUAL_FY = 26
END_FORECAST_FY = 28

# FY colors
fy_color_map = {
    "FY22": "#2E6B3C",  # green
    "FY23": "#E15759",  # red
    "FY24": "#1F77B4",  # blue
    "FY25": "#7B61FF",  # purple
    "FY26": "#FF4FA3",  # pink
}

# CSAF prediction colors
TYPE_COLOR_CSAF_PRED = {
    "Actual": "#1F77B4",
    "Forecast (Frozen)": "#E15759"
}

# Enrollment colors
ENROLL_COLORS = {
    ("October 1 Count", "Actual"): "#1F4ED8",
    ("October 1 Count", "Forecast (Frozen)"): "#93C5FD",
    ("February 1 Count", "Actual"): "#D97706",
    ("February 1 Count", "Forecast (Frozen)"): "#FCD34D",
    ("Budgetted", "Actual"): "#166534",
    ("Budgetted", "Forecast (Frozen)"): "#86EFAC",
    ("Budget to Enrollment Ratio", "Actual"): "#7C3AED",
    ("Budget to Enrollment Ratio", "Forecast (Frozen)"): "#C4B5FD",
}

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

def apply_plot_style(fig, height=CHART_H):
    fig.update_layout(
        height=height,
        font=dict(size=BASE_FONT_SIZE),
        legend_font=dict(size=16),
        margin=dict(t=90, b=110, l=25, r=25),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
    )
    fig.update_xaxes(tickfont=dict(size=AXIS_FONT), showgrid=False)
    fig.update_yaxes(tickfont=dict(size=AXIS_FONT), gridcolor=GRID_CLR)
    return fig

# ============================================================
# CSAF METRICS + THRESHOLDS
# ============================================================
csaf_metrics = ["FB Ratio", "Liabilities to Assets", "Current Ratio", "Unrestricted Days COH"]
csaf_desc = {
    "FB Ratio": "Fund Balance Ratio: Unrestricted Fund Balance / Total Exp.",
    "Liabilities to Assets": "Liabilities to Assets Ratio: Total Liabilities / Total Assets",
    "Current Ratio": "Current Ratio: Current Assets / Current Liabilities",
    "Unrestricted Days COH": "Unrestricted Cash on Hand: Cash / ((Exp.-Depreciation)/365)",
}
csaf_best = {
    "FB Ratio": {"threshold": 0.10, "direction": "gte"},
    "Liabilities to Assets": {"threshold": 0.90, "direction": "lte"},
    "Current Ratio": {"threshold": 1.50, "direction": "gte"},
    "Unrestricted Days COH": {"threshold": 60.0, "direction": "gte"},
}

def add_best_practice_csaf(fig, metric, row=None, col=None):
    if metric not in csaf_best:
        return fig
    thr = csaf_best[metric]["threshold"]
    if metric == "FB Ratio":
        label = f"{thr:.0%}"
    elif metric in ("Liabilities to Assets", "Current Ratio"):
        label = f"{thr:.2f}"
    else:
        label = f"{thr:.0f}"
    kwargs = dict(
        y=thr,
        line_dash="dot",
        line_color="#005A9C",
        line_width=3,
        annotation_text=label,
        annotation_position="top left",
        annotation_font=dict(size=16, color="#0066cc"),
    )
    if row is not None and col is not None:
        fig.add_hline(row=row, col=col, **kwargs)
    else:
        fig.add_hline(**kwargs)
    return fig

def fmt_csaf(metric, v):
    if pd.isna(v):
        return ""
    if metric == "FB Ratio":
        return f"{v:.0%}"
    if metric in ("Liabilities to Assets", "Current Ratio"):
        return f"{v:.2f}"
    if metric == "Unrestricted Days COH":
        return f"{v:,.0f}"
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
school_options = sorted(df["Schools"].dropna().unique())

# Long form
value_vars = [c for c in df.columns if c not in ["Schools", "Fiscal Year"]]
df_long = df.melt(id_vars=["Schools", "Fiscal Year"], value_vars=value_vars, var_name="Metric", value_name="Value")
df_long["ValueNum"] = pd.to_numeric(df_long["Value"], errors="coerce")
df_long["sort_key"] = df_long["Fiscal Year"].apply(sort_fy)

# ============================================================
# LOAD DATA (BUDGET / ENROLLMENT)
# ============================================================
fy26_path = "Enrollment FY26.xlsx"
df_budget_long = pd.DataFrame()
school_options_budget, fy_options_budget = [], []

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
        st.warning(f"⚠️ Enrollment sheet missing required columns: {missing}")
    else:
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

# ============================================================
# FORECAST ENGINE (kept as in your original)
# ============================================================
def _safe_log1p(y):
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0, None)
    return np.log1p(y)

def _safe_expm1(ylog):
    ylog = np.asarray(ylog, dtype=float)
    return np.clip(np.exp(ylog) - 1.0, 0, None)

def _guard_growth(y_future, last_val, max_up=1.25, max_down=0.75, lower=0.0, upper=None):
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

def _quarter_dummies(q_int, season_period=3):
    q = np.asarray(q_int, dtype=int).reshape(-1)
    if season_period == 3:
        Q2 = (q == 2).astype(int).reshape(-1, 1)
        Q3 = (q == 3).astype(int).reshape(-1, 1)
        return np.hstack([Q2, Q3])
    if season_period == 4:
        Q2 = (q == 2).astype(int).reshape(-1, 1)
        Q3 = (q == 3).astype(int).reshape(-1, 1)
        Q4 = (q == 4).astype(int).reshape(-1, 1)
        return np.hstack([Q2, Q3, Q4])
    return np.zeros((len(q), 0))

def _seasonal_index(y, q_int):
    y = np.asarray(y, dtype=float)
    q = np.asarray(q_int, dtype=int)
    overall = np.nanmedian(y) if np.nanmedian(y) > 0 else (np.nanmean(y) if np.nanmean(y) > 0 else 1.0)
    idx = {}
    for qq in np.unique(q):
        m = np.nanmedian(y[q == qq])
        if not np.isfinite(m) or m <= 0:
            m = overall
        idx[int(qq)] = float(m / overall) if overall > 0 else 1.0
    return idx

def _tscv_mae(fit_predict_fn, y, min_train=5, splits=3):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < max(min_train + 2, 6):
        return np.inf
    n_splits = min(splits, max(2, n - min_train))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    for tr_idx, te_idx in tscv.split(np.arange(n)):
        if len(tr_idx) < min_train:
            continue
        te0 = np.array([te_idx[0]], dtype=int)
        yhat = fit_predict_fn(tr_idx, te0)
        maes.append(mean_absolute_error(y[te0], yhat))
    return float(np.mean(maes)) if maes else np.inf

def _make_lag_features(y, q_int=None, n_lags=4, use_time_poly=True, use_season=True, season_period=3):
    y = np.asarray(y, dtype=float)
    n = len(y)
    rows, targets, qs = [], [], []
    for t in range(n_lags, n):
        rows.append(y[t-n_lags:t][::-1])
        targets.append(y[t])
        if q_int is not None:
            qs.append(int(q_int[t]))
    X = np.asarray(rows, dtype=float)
    y_target = np.asarray(targets, dtype=float)
    feats = [X]
    if use_time_poly:
        tt = np.arange(n_lags, n).astype(float)
        feats.append(tt.reshape(-1, 1))
        feats.append((tt**2).reshape(-1, 1))
    if use_season and q_int is not None:
        feats.append(_quarter_dummies(np.asarray(qs, dtype=int), season_period=season_period))
    return np.hstack(feats), y_target

def _iterative_forecast_supervised(
    model, y_hist, q_hist, horizon,
    n_lags=None, use_time_poly=True, use_season=True,
    log_target=True, season_period=3,
    is_ratio=False
):
    y_hist = np.asarray(y_hist, dtype=float)
    y_hist = np.clip(y_hist, 0, None)
    last = float(y_hist[-1])
    if n_lags is None:
        n_lags = min(6, max(3, len(y_hist) // 3))
    Xtr, ytr = _make_lag_features(
        y_hist, q_hist,
        n_lags=n_lags,
        use_time_poly=use_time_poly,
        use_season=use_season,
        season_period=season_period
    )
    yfit = _safe_log1p(ytr) if log_target else ytr.copy()
    Xtr = np.asarray(Xtr, dtype=float)
    yfit = np.asarray(yfit, dtype=float)
    m = np.isfinite(Xtr).all(axis=1) & np.isfinite(yfit)
    Xtr, yfit = Xtr[m], yfit[m]
    if len(yfit) < 6:
        return np.array([last] * horizon, dtype=float)
    model.fit(Xtr, yfit)
    y_all = y_hist.copy()
    q_all = None if q_hist is None else np.asarray(q_hist, dtype=int).copy()
    preds = []
    for _ in range(horizon):
        t_next = len(y_all)
        lag_vec = y_all[-n_lags:][::-1].reshape(1, -1)
        parts = [lag_vec]
        if use_time_poly:
            parts.append(np.array([[float(t_next), float(t_next**2)]]))
        if use_season and q_all is not None:
            q_next = int(((q_all[-1] % season_period) + 1))
            parts.append(_quarter_dummies(np.array([q_next], dtype=int), season_period=season_period))
        Xf = np.hstack(parts)
        yhat_fit = model.predict(Xf)[0]
        yhat = _safe_expm1([yhat_fit])[0] if log_target else float(yhat_fit)
        preds.append(yhat)
        y_all = np.append(y_all, yhat)
        if q_all is not None:
            q_all = np.append(q_all, q_next)
    if is_ratio:
        max_up, max_down = 1.15, 0.85
    else:
        max_up, max_down = 1.30, 0.70
    hist_max = float(np.nanmax(np.clip(y_hist, 0, None))) if len(y_hist) else 1.0
    cap = max(hist_max * 1.6, last * 1.5, 1.0)
    preds = _guard_growth(preds, last, max_up=max_up, max_down=max_down, lower=0.0, upper=cap)
    return np.array(preds, dtype=float)

def forecast_timeseries(y, q=None, horizon=6, model_choice="Auto (min error)", season_period=3, is_ratio=False):
    y = np.asarray(y, dtype=float); y = np.clip(y, 0, None)
    if is_ratio: y = np.clip(y, 0.0, 1.5)
    if q is None: season_period = 1

    def seasonal_naive(y_hist, q_hist, h):
        y_hist = np.asarray(y_hist, dtype=float)
        preds = []; sp = season_period
        for _ in range(h):
            idx = len(y_hist) - sp
            if idx >= 0:
                preds.append(float(y_hist[idx])); y_hist = np.append(y_hist, y_hist[idx])
            else:
                preds.append(float(y_hist[-1])); y_hist = np.append(y_hist, y_hist[-1])
        return np.array(preds, dtype=float)

    def seasonal_naive_drift(y_hist, q_hist, h):
        y_hist = np.asarray(y_hist, dtype=float); sp = season_period
        base = []; y_tmp = y_hist.copy()
        for _ in range(h):
            idx = len(y_tmp) - sp
            val = float(y_tmp[idx]) if idx >= 0 else float(y_tmp[-1])
            base.append(val); y_tmp = np.append(y_tmp, val)
        base = np.asarray(base, dtype=float)
        if len(y_hist) > sp + 1:
            diffs = y_hist[sp:] - y_hist[:-sp]; recent = diffs[-min(len(diffs), 6):]
            med = np.nanmedian(recent); drift = float(med) if np.isfinite(med) else 0.0
        else:
            drift = 0.0
        damp = np.linspace(1.0, 0.6, h)
        out = base + drift * damp
        out = np.clip(out, 0, None)
        if is_ratio: out = np.clip(out, 0.0, 1.5)
        return out

    def robust_seasonal_regression(y_hist, q_hist, h):
        ylog = _safe_log1p(y_hist)
        t = np.arange(len(ylog)).reshape(-1, 1).astype(float)
        X = t if q_hist is None else np.hstack([t, _quarter_dummies(q_hist, season_period=season_period)])
        mdl = HuberRegressor(); mdl.fit(X, ylog)
        tf = np.arange(len(ylog), len(ylog) + h).reshape(-1, 1).astype(float)
        if q_hist is None: Xf = tf
        else:
            q_last = int(q_hist[-1]); qf = []
            for _ in range(h):
                q_last = (q_last % season_period) + 1; qf.append(q_last)
            qf = np.asarray(qf, dtype=int)
            Xf = np.hstack([tf, _quarter_dummies(qf, season_period=season_period)])
        return _safe_expm1(mdl.predict(Xf))

    def trend_times_seasonal_index(y_hist, q_hist, h):
        y_hist = np.asarray(y_hist, dtype=float)
        t = np.arange(len(y_hist)).reshape(-1, 1).astype(float)
        if q_hist is None:
            mdl = LinearRegression(); mdl.fit(t, _safe_log1p(y_hist))
            tf = np.arange(len(y_hist), len(y_hist) + h).reshape(-1, 1).astype(float)
            return _safe_expm1(mdl.predict(tf))
        idx_map = _seasonal_index(y_hist, q_hist)
        deseason = np.array([y_hist[i] / max(idx_map[int(q_hist[i])], 1e-6) for i in range(len(y_hist))], dtype=float)
        mdl = LinearRegression(); mdl.fit(t, _safe_log1p(deseason))
        tf = np.arange(len(deseason), len(deseason) + h).reshape(-1, 1).astype(float)
        base = _safe_expm1(mdl.predict(tf))
        q_last = int(q_hist[-1]); preds = []
        for i in range(h):
            q_last = (q_last % season_period) + 1
            preds.append(float(base[i] * idx_map.get(int(q_last), 1.0)))
        return np.asarray(preds, dtype=float)

    def ensemble_3(y_hist, q_hist, h):
        p1 = seasonal_naive_drift(y_hist, q_hist, h)
        p2 = robust_seasonal_regression(y_hist, q_hist, h)
        p3 = trend_times_seasonal_index(y_hist, q_hist, h)
        w = np.array([0.55, 0.30, 0.15], dtype=float)
        return np.average(np.vstack([p1, p2, p3]), axis=0, weights=w)

    def hgbr_lag(y_hist, q_hist, h):
        mdl = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.08, max_iter=900, random_state=42)
        return _iterative_forecast_supervised(
            mdl, y_hist, q_hist, h,
            n_lags=None, use_time_poly=True, use_season=(q_hist is not None),
            log_target=True, season_period=season_period,
            is_ratio=is_ratio
        )

    def neural_mlp_lag(y_hist, q_hist, h):
        mdl = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(48, 24), activation="relu", solver="adam", max_iter=7000, random_state=42)
        )
        return _iterative_forecast_supervised(
            mdl, y_hist, q_hist, h,
            n_lags=None, use_time_poly=True, use_season=(q_hist is not None),
            log_target=True, season_period=season_period,
            is_ratio=is_ratio
        )

    def ridge_lag(y_hist, q_hist, h):
        mdl = Ridge(alpha=1.0)
        return _iterative_forecast_supervised(
            mdl, y_hist, q_hist, h,
            n_lags=None, use_time_poly=True, use_season=(q_hist is not None),
            log_target=True, season_period=season_period,
            is_ratio=is_ratio
        )

    models = {
        "Ensemble (Seasonal Naive + Drift + Robust Seasonal + Trend×Seasonality)": ensemble_3,
        "Seasonal Naive + Drift (recommended baseline)": seasonal_naive_drift,
        "Seasonal Naive (same quarter last year)": seasonal_naive,
        "Robust Seasonal Regression (Huber + quarter dummies, log1p)": robust_seasonal_regression,
        "Trend × Seasonal Index (linear trend on de-seasonalized)": trend_times_seasonal_index,
        "HGBR Lag + Trend + Season (recommended)": hgbr_lag,
        "Neural MLP Lag + Season": neural_mlp_lag,
        "Ridge Lag + Season": ridge_lag,
    }

    def score_model(name):
        fn = models[name]
        def fit_pred(tr_idx, te_idx):
            y_tr = y[tr_idx]
            q_tr = None if q is None else np.asarray(q, dtype=int)[tr_idx]
            pred = fn(y_tr, q_tr, 1)
            pred = np.asarray(pred, dtype=float)
            if is_ratio: pred = np.clip(pred, 0.0, 1.5)
            return pred
        return _tscv_mae(fit_pred, y, min_train=5, splits=3)

    scores = {}
    if model_choice == "Auto (min error)":
        for name in models:
            try: scores[name] = score_model(name)
            except Exception: scores[name] = np.inf
        chosen = min(scores, key=scores.get)
    else:
        chosen = model_choice
        for name in models:
            try: scores[name] = score_model(name)
            except Exception: scores[name] = np.inf

    # ---- (Fix) predict with the chosen function
    y_future = models[chosen](y, None if q is None else np.asarray(q, dtype=int), horizon)
    y_future = np.clip(y_future, 0, None)
    if is_ratio: y_future = np.clip(y_future, 0.0, 1.5)

    # final guard
    if is_ratio: max_up, max_down = 1.15, 0.85
    else:        max_up, max_down = 1.30, 0.70
    last = float(y[-1])
    hist_max = float(np.nanmax(y)) if len(y) else 1.0
    cap = max(hist_max * 1.6, last * 1.5, 1.0)
    y_future = _guard_growth(y_future, last, max_up=max_up, max_down=max_down, lower=0.0, upper=cap)
    if is_ratio: y_future = np.clip(y_future, 0.0, 1.5)
    return y_future, chosen, scores, models[chosen]

def bootstrap_intervals(y_hist, q_hist, horizon, model_fn, season_period=3, n_sims=300, p_lo=10, p_hi=90, is_ratio=False, seed=42):
    rng = np.random.default_rng(seed)
    y_hist = np.asarray(y_hist, dtype=float)
    y_hist = np.clip(y_hist, 0, None)
    if is_ratio: y_hist = np.clip(y_hist, 0.0, 1.5)

    residuals = []
    min_train = max(6, min(10, len(y_hist) - 1))
    for t in range(min_train, len(y_hist)):
        y_tr = y_hist[:t]
        q_tr = None if q_hist is None else np.asarray(q_hist[:t], dtype=int)
        pred1 = model_fn(y_tr, q_tr, 1)[0]
        residuals.append(float(y_hist[t] - pred1))
    if len(residuals) < 5:
        residuals = [0.0]
    residuals = np.asarray(residuals, dtype=float)

    damp = np.linspace(1.0, 0.65, horizon)
    sims = np.zeros((n_sims, horizon), dtype=float)
    base = model_fn(y_hist, None if q_hist is None else np.asarray(q_hist, dtype=int), horizon)

    for s in range(n_sims):
        noise = rng.choice(residuals, size=horizon, replace=True) * damp
        sim = base + noise
        sim = np.clip(sim, 0, None)
        if is_ratio: sim = np.clip(sim, 0.0, 1.5)
        sims[s, :] = sim

    p10 = np.percentile(sims, p_lo, axis=0)
    p50 = np.percentile(sims, 50, axis=0)
    p90 = np.percentile(sims, p_hi, axis=0)
    return p10, p50, p90

# ============================================================
# SIDEBAR NAV  (Portfolio option removed)
# ============================================================
st.sidebar.header("🔎 Filters")

modes = [
    "Executive Summary (School Report Card)",  # KPI-first executive view
    "CSAF Metrics (4-panel)",
    "CSAF Predicted",
    "Other Metrics",
]
if not df_budget_long.empty:
    modes += ["Budget/Enrollment (Bar)", "Budget/Enrollment Predicted (Bar)"]

metric_group = st.sidebar.radio("Choose Dashboard:", modes)

# ============================================================
# HELPERS: parse quarter from "FY25 Q1"
# ============================================================
fyq_re = re.compile(r"FY\s*(\d{2,4})\s*Q\s*(\d)", re.IGNORECASE)

def parse_q(label: str):
    m = fyq_re.search(str(label))
    return int(m.group(2)) if m else None

# ============================================================
# 0) EXECUTIVE SUMMARY — SCHOOL REPORT CARD (KPI bar on top)
#     KPI cards styled similar to your attached HTML stats bar.
# ============================================================
if metric_group == "Executive Summary (School Report Card)":
    st.markdown("## 🏫 Executive Summary — School Report Card")

    selected_school = st.sidebar.selectbox("Select School", school_options)

    # --- Helpers (local) ---
    def _risk_status(metric_name: str, v: float) -> str:
        if v is None or not np.isfinite(v):
            return "No Data"
        thr = csaf_best[metric_name]["threshold"]
        direction = csaf_best[metric_name]["direction"]
        band = 0.05 * thr if thr > 1 else 0.02  # tolerance band
        if direction == "gte":
            if v >= thr: return "On Track"
            if v >= (thr - band): return "Monitor"
            return "At Risk"
        else:
            if v <= thr: return "On Track"
            if v <= (thr + band): return "Monitor"
            return "At Risk"

    def _latest_for_school(school: str) -> dict:
        out = {}
        d0 = df[df["Schools"] == school].copy()
        if d0.empty:
            return {m: np.nan for m in csaf_metrics}
        d0["sort_key"] = d0["Fiscal Year"].apply(sort_fy)
        d0 = d0.sort_values("sort_key")
        for met in csaf_metrics:
            vals = pd.to_numeric(d0[met], errors="coerce").dropna()
            out[met] = float(vals.iloc[-1]) if len(vals) else np.nan
        return out) -> pd.DataFrame:
        d1 = df[df["Schools"] == school].copy()
        d1["ValueNum_FB"]   = pd.to_numeric(d1["FB Ratio"], errors="coerce")
        d1["ValueNum_LA"]   = pd.to_numeric(d1["Liabilities to Assets"], errors="coerce")
        d1["ValueNum_CR"]   = pd.to_numeric(d1["Current Ratio"], errors="coerce")
        d1["ValueNum_DCOH"] = pd.to_numeric(d1["Unrestricted Days COH"], errors="coerce")
        d1["sort_key"] = d1["Fiscal Year"].apply(sort_fy)
        d1 = d1.sort_values("sort_key")
        d1["FYQ"] = d1["Fiscal Year"].astype(str)
        return d1[["FYQ","ValueNum_FB","ValueNum_LA","ValueNum_CR","ValueNum_DCOH","sort_key"]]

    # Build latest snapshot + series
    latest = _latest_for_school(selected_school)
    ts = _trend_series(selected_school)

    # ---------- KPI BAR (cards) ----------
    # Style inspired by your HTML "stats bar": label, bold value, small status text. [1](https://opsbus-my.sharepoint.com/personal/eigbokwe_nolapublicschools_com).html)
    def kpi_html(label, value_txt, status_txt, status_cls):
        return f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value_txt}</div>
          <div class="kpi-sub"><span class="{status_cls}">{status_txt}</span></div>
        </div>
        """

    fb  = latest.get("FB Ratio", np.nan)
    cr  = latest.get("Current Ratio", np.nan)
    dch = latest.get("Unrestricted Days COH", np.nan)
    la  = latest.get("Liabilities to Assets", np.nan)

    def cls_for(metric, v):
        s = _risk_status(metric, v)
        return "kpi-good" if s == "On Track" else ("kpi-warn" if s == "Monitor" else "kpi-bad")

    kpi_cards = f"""
      <div class="kpi-grid">
        {kpi_html("Fund Balance Ratio", f"{fb:.1%}" if np.isfinite(fb) else "—", _risk_status("FB Ratio", fb), cls_for("FB Ratio", fb))}
        {kpi_html("Current Ratio", f"{cr:.2f}" if np.isfinite(cr) else "—", _risk_status("Current Ratio", cr), cls_for("Current Ratio", cr))}
        {kpi_html("Days Cash on Hand", f"{dch:,.0f}" if np.isfinite(dch) else "—", _risk_status("Unrestricted Days COH", dch), cls_for("Unrestricted Days COH", dch))}
        {kpi_html("Liabilities / Assets", f"{la:.2f}" if np.isfinite(la) else "—", _risk_status("Liabilities to Assets", la), cls_for("Liabilities to Assets", la))}
      </div>
    """
    st.markdown(kpi_cards, unsafe_allow_html=True)

    # ---------- Health Gauges (after KPI bar) ----------
    st.markdown("### Health Gauges")
    g1, g2, g3, g4 = st.columns(4)
    try:
        def _gauge(ax_range, steps, threshold, value, number_fmt, title):
            return go.Figure(go.Indicator(
                mode="gauge+number",
                value=value if np.isfinite(value) else 0,
                number={'valueformat': number_fmt},
                title={'text': title},
                gauge={
                    "axis": {"range": ax_range},
                    "bar": {"color": "#2563eb"},
                    "steps": steps,
                    "threshold": {"line": {"color": "#0ea5e9", "width": 3}, "thickness": 0.75, "value": threshold},
                }
            ))

        fig_fb = _gauge([0, 0.40],
                        [{"range": [0, 0.10], "color": "#fecaca"},
                         {"range": [0.10, 0.20], "color": "#fef08a"},
                         {"range": [0.20, 0.40], "color": "#bbf7d0"}],
                        0.10, fb, ".0%", "FB Ratio (≥10%)")
        fig_cr = _gauge([0, 3.0],
                        [{"range": [0.0, 1.5], "color": "#fecaca"},
                         {"range": [1.5, 2.2], "color": "#fef08a"},
                         {"range": [2.2, 3.0], "color": "#bbf7d0"}],
                        1.50, cr, ".2f", "Current Ratio (≥1.50×)")
        fig_dcoh = _gauge([0, 180],
                          [{"range": [0, 60], "color": "#fecaca"},
                           {"range": [60, 90], "color": "#fef08a"},
                           {"range": [90, 180], "color": "#bbf7d0"}],
                          60, dch, ",.0f", "Days COH (≥60)")
        fig_la = _gauge([0, 1.5],
                        [{"range": [0.0, 0.90], "color": "#bbf7d0"},
                         {"range": [0.90, 1.10], "color": "#fef08a"},
                         {"range": [1.10, 1.50], "color": "#fecaca"}],
                        0.90, la, ".2f", "Liab/Assets (≤0.90)")

        for col, fig in zip([g1, g2, g3, g4], [fig_fb, fig_cr, fig_dcoh, fig_la]):
            fig = apply_plot_style(fig, height=240)
            col.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Gauges unavailable due to missing values.")

    # ---------- Trend Lines ----------
    st.markdown("### Trend Lines (Latest Quarters)")
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Fund Balance Ratio", "Liabilities to Assets", "Current Ratio", "Unrestricted Days COH"],
        horizontal_spacing=0.08, vertical_spacing=0.12
    )
    x_order = ts["FYQ"].tolist()
    panels = [
        ("ValueNum_FB",  "FB Ratio",                 1,1),
        ("ValueNum_LA",  "Liabilities to Assets",    1,2),
        ("ValueNum_CR",  "Current Ratio",            2,1),
        ("ValueNum_DCOH","Unrestricted Days COH",    2,2),
    ]
    for col_name, met, r, c in panels:
        sub = ts[["FYQ", col_name]].dropna()
        if sub.empty: 
            continue
        fig.add_trace(go.Scatter(
            x=sub["FYQ"], y=sub[col_name], mode="lines+markers",
            name=met, showlegend=False, line=dict(width=3), marker=dict(size=8)
        ), row=r, col=c)
        fig.update_xaxes(categoryorder="array", categoryarray=x_order, tickangle=25, row=r, col=c)
        fig = add_best_practice_csaf(fig, met, row=r, col=c)
    fig = apply_plot_style(fig, height=700)
    fig.update_layout(margin=dict(t=90, r=30, b=80, l=30))
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Explanations & Advice ----------
    st.markdown("### 📄 Financial Analysis & Advice")
    def bullet(metric, v):
        return f"- **{metric}**: {fmt_csaf(metric, v) if np.isfinite(v) else '—'} — **{_risk_status(metric, v)}**"
    bullets = [
        bullet("FB Ratio", fb),
        bullet("Current Ratio", cr),
        bullet("Unrestricted Days COH", dch),
        bullet("Liabilities to Assets", la),
    ]
    st.markdown("\n".join(bullets))

    advice = []
    if np.isfinite(fb) and fb < csaf_best["FB Ratio"]["threshold"]:
        gap = csaf_best["FB Ratio"]["threshold"] - fb
        advice += [
            f"Fund Balance below target by ~{gap:.1%}. Drive structural balance: align staffing to enrollment, pause non‑essential spend.",
            "Accelerate reimbursements/receivables (grants/state) and improve cash conversion.",
        ]
    if np.isfinite(cr) and cr < csaf_best["Current Ratio"]["threshold"]:
        advice += [
            "Liquidity risk: defer non‑critical capex, smooth payables, strengthen A/R collections.",
            "Consider contingency credit/LOC for seasonal tightness.",
        ]
    if np.isfinite(dch) and dch < csaf_best["Unrestricted Days COH"]["threshold"]:
        advice += [
            "Cash runway thin: target ≥60‑day reserve; expedite reimbursements; adjust disbursement timing; review vendor terms.",
        ]
    if np.isfinite(la) and la > csaf_best["Liabilities to Assets"]["threshold"]:
        advice += [
            "Leverage elevated: reduce short‑term liabilities; schedule pay‑downs; avoid new obligations until recovery.",
        ]
    if not advice:
        advice = ["Overall on track. Maintain controls, monitor quarterly, and continue enrollment retention."]

    st.markdown("#### Recommendations")
    st.markdown("\n".join([f"- {a}" for a in advice]))

    # Download (Markdown)
    md_lines = [
        f"# Executive Summary — {selected_school}",
        "## Current Status",
        *[li.replace("- ", "* ") for li in bullets],
        "## Recommendations",
        *[f"* {a}" for a in advice],
        "## Next‑Quarter Targets",
        "* FB Ratio ≥ 10%",
        "* Current Ratio ≥ 1.50×",
        "* Unrestricted Days COH ≥ 60",
        "* Liabilities to Assets ≤ 0.90",
    ]
    st.download_button(
        "⬇ Download Report Card (Markdown)",
        data="\n".join(md_lines),
        file_name=f"{selected_school.replace(' ','_').lower()}_executive_report_card.md",
        mime="text/markdown"
    )

# ============================================================
# 1) CSAF METRICS — 4 PANEL
# ============================================================
elif metric_group == "CSAF Metrics (4-panel)":
    st.markdown("## 📌 CSAF Metrics (4-panel)")

    selected_school = st.sidebar.selectbox("🏫 Select School:", school_options)
    selected_fy = st.sidebar.multiselect(
        "📅 Select Fiscal Year + Quarter:",
        fiscal_options,
        default=fiscal_options
    )

    d = df[(df["Schools"] == selected_school) & (df["Fiscal Year"].isin(selected_fy))].copy()
    if d.empty:
        st.warning("⚠️ No data for selection.")
        st.stop()

    d["sort_key"] = d["Fiscal Year"].apply(sort_fy)
    d = d.sort_values("sort_key")
    d["FY Group"] = d["Fiscal Year"].astype(str).str.split().str[0]
    x_order = d["Fiscal Year"].tolist()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"{csaf_desc['FB Ratio']}",
            f"{csaf_desc['Liabilities to Assets']}",
            f"{csaf_desc['Current Ratio']}",
            f"{csaf_desc['Unrestricted Days COH']}",
        ],
        horizontal_spacing=0.08, vertical_spacing=0.12
    )

    metric_positions = {
        "FB Ratio": (1, 1),
        "Liabilities to Assets": (1, 2),
        "Current Ratio": (2, 1),
        "Unrestricted Days COH": (2, 2),
    }

    for met, (r, c) in metric_positions.items():
        dd = d.copy()
        dd["ValueNum"] = pd.to_numeric(dd[met], errors="coerce")
        dd = dd.dropna(subset=["ValueNum"])
        dd["FY Group"] = dd["FY Group"].astype(str)
        fy_groups_ordered = (
            dd.drop_duplicates("FY Group")[["FY Group", "sort_key"]]
            .sort_values("sort_key")["FY Group"]
            .tolist()
        )
        for fygrp in fy_groups_ordered:
            sub = dd[dd["FY Group"] == fygrp]
            if sub.empty: continue
            show_leg = (r == 1 and c == 1)
            fig.add_trace(
                go.Bar(
                    x=sub["Fiscal Year"],
                    y=sub["ValueNum"],
                    name=fygrp,
                    legendgroup=fygrp,
                    showlegend=show_leg,
                    marker_color=fy_color_map.get(fygrp, None),
                    text=[fmt_csaf(met, v) for v in sub["ValueNum"]],
                    textposition="outside",
                    cliponaxis=False
                ),
                row=r, col=c
            )
        add_best_practice_csaf(fig, met, row=r, col=c)
        fig.update_xaxes(
            row=r, col=c, categoryorder="array", categoryarray=x_order,
            tickangle=-35, tickfont=dict(size=10), automargin=True
        )
        fig.update_yaxes(row=r, col=c, tickfont=dict(size=11), automargin=True)

    fig.update_layout(
        barmode="group",
        bargap=BARGAP,
        bargroupgap=BARGROUPGAP,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        height=980,
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.14, xanchor="left", x=0, font=dict(size=12), tracegroupgap=10),
        margin=dict(t=140, b=80, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

# ============================================================
# 2) CSAF PREDICTED
# ============================================================
elif metric_group == "CSAF Predicted":
    st.markdown("## 🔮 CSAF Predicted (Freeze or Unfrozen Forecast)")

    selected_school = st.sidebar.selectbox("🏫 Select School:", school_options)
    selected_metric = st.sidebar.selectbox("📊 Select CSAF Metric:", csaf_metrics)

    forecast_mode = st.sidebar.radio(
        "🧊 Forecast Mode",
        ["Freeze at selected quarter", "Unfrozen (use all actuals)"],
        index=0
    )

    all_quarters = sorted(df["Fiscal Year"].dropna().astype(str).unique(), key=sort_fy)
    last_actual_qtr = all_quarters[-1] if all_quarters else None

    if forecast_mode == "Freeze at selected quarter":
        freeze_at = st.sidebar.selectbox("🧊 Freeze at:", all_quarters, index=max(0, len(all_quarters) - 1))
        train_end_label = freeze_at
    else:
        train_end_label = last_actual_qtr
        st.sidebar.info(f"Unfrozen mode uses all actuals up to: **{train_end_label}**")

    horizon_q = st.sidebar.slider("🔮 Forecast horizon (quarters)", 3, 12, 6)

    csaf_model_choice = st.sidebar.selectbox(
        "🧠 CSAF Forecast Model:",
        ["Auto (min error)"] + [
            "Ensemble (Seasonal Naive + Drift + Robust Seasonal + Trend×Seasonality)",
            "Seasonal Naive + Drift (recommended baseline)",
            "Seasonal Naive (same quarter last year)",
            "Robust Seasonal Regression (Huber + quarter dummies, log1p)",
            "Trend × Seasonal Index (linear trend on de-seasonalized)",
            "HGBR Lag + Trend + Season (recommended)",
            "Neural MLP Lag + Season",
            "Ridge Lag + Season",
        ],
        index=0
    )

    show_intervals = st.sidebar.checkbox("📊 Show interval forecast (Bootstrap P10–P50–P90)", value=False)
    n_sims = st.sidebar.slider("🎲 Bootstrap simulations", 200, 800, 300) if show_intervals else 300
    p_lo = st.sidebar.slider("📉 Lower percentile", 5, 25, 10) if show_intervals else 10
    p_hi = st.sidebar.slider("📈 Upper percentile", 75, 95, 90) if show_intervals else 90

    show_model_table = st.sidebar.checkbox("Show model error table", value=False)
    run = st.sidebar.button("▶ Run CSAF Prediction")
    if not run:
        st.info("Choose options in the sidebar, then click **Run CSAF Prediction**.")
        st.stop()

    hist = df[df["Schools"] == selected_school].copy()
    hist["sort_key"] = hist["Fiscal Year"].apply(sort_fy)
    hist = hist.sort_values("sort_key")

    if train_end_label is not None:
        cut_key = sort_fy(train_end_label)
        hist = hist[hist["sort_key"].apply(lambda k: k <= cut_key)].sort_values("sort_key")

    y = pd.to_numeric(hist[selected_metric], errors="coerce").values.astype(float)
    labels = hist["Fiscal Year"].astype(str).tolist()
    q = np.array([parse_q(x) if parse_q(x) is not None else np.nan for x in labels], dtype=float)

    mask = ~np.isnan(y) & ~np.isnan(q)
    y = y[mask]; q = q[mask].astype(int)

    if len(y) < 5:
        st.warning("⚠️ Not enough points for a reliable forecast (need ≥ 5).")
        st.stop()

    is_ratio = (selected_metric == "FB Ratio")
    y_future, chosen_model, scores, chosen_fn = forecast_timeseries(
        y=y, q=q, horizon=horizon_q,
        model_choice=csaf_model_choice, season_period=3, is_ratio=is_ratio
    )

    def make_future_labels(last_label: str, n: int, q_per_year=3):
        m = fyq_re.search(str(last_label))
        if not m:
            fy, qq = END_ACTUAL_FY, 0
        else:
            fy = fy_num("FY" + m.group(1)) or END_ACTUAL_FY
            qq = int(m.group(2))
        out = []
        for _ in range(n):
            qq += 1
            if qq > q_per_year:
                fy += 1; qq = 1
            out.append(f"FY{fy:02d} Q{qq}")
        return out
    future_labels = make_future_labels(train_end_label, horizon_q, q_per_year=3)

    actual_full = df[df["Schools"] == selected_school].copy()
    actual_full["sort_key"] = actual_full["Fiscal Year"].apply(sort_fy)
    actual_full = actual_full.sort_values("sort_key")
    actual_vals = pd.to_numeric(actual_full[selected_metric], errors="coerce")
    actual_part = pd.DataFrame({"Period": actual_full["Fiscal Year"].astype(str), "Value": actual_vals, "Type": "Actual"}).dropna(subset=["Value"])

    pred_label = "Forecast (Frozen)" if forecast_mode == "Freeze at selected quarter" else "Forecast (Unfrozen)"
    if "Forecast (Unfrozen)" not in TYPE_COLOR_CSAF_PRED:
        TYPE_COLOR_CSAF_PRED["Forecast (Unfrozen)"] = TYPE_COLOR_CSAF_PRED.get("Forecast (Frozen)", "#e15759")
    pred_part = pd.DataFrame({"Period": future_labels, "Value": y_future, "Type": pred_label})

    combined = pd.concat([actual_part, pred_part], ignore_index=True)
    combined["Label"] = combined["Value"].apply(lambda v: fmt_csaf(selected_metric, v))

    fig = px.bar(
        combined, x="Period", y="Value", color="Type", barmode="group",
        text="Label", color_discrete_map=TYPE_COLOR_CSAF_PRED,
        title=f"{selected_school} — {selected_metric}"
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside", cliponaxis=False, textfont=dict(size=18))
    fig.update_layout(uniformtext_mode="show", uniformtext_minsize=12, bargap=0.12, bargroupgap=0.06)
    fig.update_xaxes(tickangle=30)
    fig = add_best_practice_csaf(fig, selected_metric)

    if show_intervals:
        p10, p50, p90 = bootstrap_intervals(
            y_hist=y, q_hist=q, horizon=horizon_q, model_fn=chosen_fn,
            season_period=3, n_sims=n_sims, p_lo=p_lo, p_hi=p_hi, is_ratio=is_ratio
        )
        fig.add_trace(go.Scatter(
            x=future_labels + future_labels[::-1], y=list(p90) + list(p10[::-1]),
            fill="toself", mode="lines", line=dict(width=0),
            name=f"Interval P{p_lo}–P{p_hi}", showlegend=True, opacity=0.20
        ))
        fig.add_trace(go.Scatter(x=future_labels, y=p50, mode="lines+markers", name="P50 (median)", line=dict(width=2)))

    fig = apply_plot_style(fig, height=700)
    fig.update_layout(
        title=dict(text=f"{selected_school} — {selected_metric}", x=0.01, y=0.985),
        legend=dict(title="Type", orientation="h", yanchor="bottom", y=1.25, xanchor="left", x=0.01),
        margin=dict(t=210, r=40, b=90, l=60)
    )
    fig.update_traces(cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 3) BUDGET/ENROLLMENT (ACTUAL)
# ============================================================
elif metric_group == "Budget/Enrollment (Bar)":
    st.markdown("## 📊 Budget/Enrollment (Actuals)")

    if df_budget_long.empty:
        st.warning("⚠️ Enrollment dataset not loaded.")
        st.stop()

    selected_school = st.sidebar.selectbox("🏫 Select School:", school_options_budget)
    metrics_all = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    available = sorted(df_budget_long["Metric"].dropna().unique())
    metrics_all = [m for m in metrics_all if m in available]

    selected_metrics = st.sidebar.multiselect("📌 Select Metric(s):", metrics_all, default=["October 1 Count", "February 1 Count"])
    selected_fy = st.sidebar.multiselect(
        "📅 Select Fiscal Years:",
        fy_options_budget,
        default=[fy for fy in fy_options_budget if START_FY <= sort_fy_only(fy) <= END_ACTUAL_FY] or fy_options_budget
    )

    d = df_budget_long[
        (df_budget_long["Schools"] == selected_school) &
        (df_budget_long["Metric"].isin(selected_metrics)) &
        (df_budget_long["Fiscal Year"].isin(selected_fy))
    ].copy()
    d["ValueNum"] = pd.to_numeric(d["Value"], errors="coerce")
    d = d.dropna(subset=["ValueNum"])
    d["sort_key"] = d["Fiscal Year"].apply(sort_fy_only)
    d = d.sort_values("sort_key")

    if d.empty:
        st.warning("⚠️ No data for current filters.")
        st.stop()

    def fmt_enroll(metric, v):
        return f"{v:.0%}" if metric == "Budget to Enrollment Ratio" else f"{v:,.0f}"

    fig = go.Figure()
    for met in selected_metrics:
        sub = d[d["Metric"] == met]
        fig.add_trace(go.Bar(
            x=sub["Fiscal Year"], y=sub["ValueNum"], name=f"{met}",
            marker_color=ENROLL_COLORS.get((met, "Actual"), None),
            text=[fmt_enroll(met, v) for v in sub["ValueNum"]], textposition="outside",
        ))
    fig.update_layout(
        title=dict(text=f"{selected_school} — Enrollment (Actuals)", x=0.01, y=0.98),
        barmode="group", bargap=BARGAP, bargroupgap=BARGROUPGAP,
        legend=dict(orientation="h", yanchor="bottom", y=1.18, xanchor="left", x=0.01),
        margin=dict(t=170, r=40, b=80, l=60)
    )
    fig.update_xaxes(categoryorder="array", categoryarray=sorted(d["Fiscal Year"].unique(), key=sort_fy_only), tickangle=0)
    fig = apply_plot_style(fig, height=CHART_H_TALL)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 4) BUDGET/ENROLLMENT PREDICTED
# ============================================================
elif metric_group == "Budget/Enrollment Predicted (Bar)":
    st.markdown("## 🔮 Enrollment Predicted (Stable Forecast)")

    if df_budget_long.empty:
        st.warning("⚠️ Enrollment dataset not loaded.")
        st.stop()

    selected_school = st.sidebar.selectbox("🏫 Select School:", school_options_budget)
    metrics_all = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    available = sorted(df_budget_long["Metric"].dropna().unique())
    metrics_all = [m for m in metrics_all if m in available]

    selected_metrics = st.sidebar.multiselect(
        "📌 Select Metric(s) to forecast:",
        metrics_all,
        default=[m for m in ["October 1 Count", "February 1 Count"] if m in metrics_all] or metrics_all[:2]
    )

    freeze_at = st.sidebar.selectbox(
        "🧊 Freeze at (use actuals up to this FY):",
        fy_options_budget,
        index=fy_options_budget.index(fy_label(END_ACTUAL_FY)) if fy_label(END_ACTUAL_FY) in fy_options_budget else len(fy_options_budget) - 1
    )

    st.sidebar.markdown("### ⚙️ Enrollment realism controls")
    max_oct_growth = st.sidebar.slider("Max Oct YoY growth", 0.00, 0.12, 0.03, 0.01)
    max_oct_drop   = st.sidebar.slider("Max Oct YoY drop",   0.00, 0.12, 0.02, 0.01)
    ret_lo = st.sidebar.slider("Min Feb/Oct retention", 0.70, 1.00, 0.92, 0.01)
    ret_hi = st.sidebar.slider("Max Feb/Oct retention", 0.80, 1.10, 1.02, 0.01)

    show_intervals = st.sidebar.checkbox("📊 Show simple bands (±1.5% on counts)", value=False)
    show_model_table = st.sidebar.checkbox("Show model info table", value=True)

    run = st.sidebar.button("▶ Run Enrollment Prediction")
    if not run:
        st.info("Choose options in the sidebar, then click **Run Budget/Enrollment Prediction**.")
        st.stop()

    def _get_series(metric_name, up_to_fy_label):
        dh = df_budget_long[(df_budget_long["Schools"] == selected_school) & (df_budget_long["Metric"] == metric_name)].copy()
        dh["FY"] = dh["Fiscal Year"].astype(str)
        dh["sort_key"] = dh["FY"].apply(sort_fy_only)
        cut = sort_fy_only(up_to_fy_label)
        dh = dh[dh["sort_key"] <= cut].sort_values("sort_key")
        dh["ValueNum"] = pd.to_numeric(dh["Value"], errors="coerce")
        dh = dh.dropna(subset=["ValueNum"])
        return dh[["FY", "sort_key", "ValueNum"]].copy()

    def _robust_pct_change(y):
        y = np.asarray(y, dtype=float)
        if len(y) < 3: return 0.0
        prev = y[:-1]; nxt  = y[1:]
        mask = (prev > 0) & np.isfinite(prev) & np.isfinite(nxt)
        if mask.sum() < 2: return 0.0
        pct = (nxt[mask] - prev[mask]) / prev[mask]
        return float(np.median(pct))

    def _forecast_oct_history(oct_hist_vals, horizon, max_g, max_d):
        y = np.asarray(oct_hist_vals, dtype=float)
        last = float(y[-1])
        g = _robust_pct_change(y); g = float(np.clip(g, -max_d, max_g))
        out = []; cur = last
        for _ in range(horizon):
            cur = float(cur * (1.0 + g)); out.append(cur)
        return np.asarray(out, dtype=float), g

    def _estimate_retention_ratio(freeze_fy_label):
        oct_df = _get_series("October 1 Count", freeze_fy_label)
        feb_df = _get_series("February 1 Count", freeze_fy_label)
        if oct_df.empty or feb_df.empty: return None
        merged = pd.merge(
            oct_df[["FY", "sort_key", "ValueNum"]],
            feb_df[["FY", "ValueNum"]],
            on="FY", how="inner", suffixes=("_Oct", "_Feb")
        ).dropna()
        merged = merged[(merged["ValueNum_Oct"] > 0) & (merged["ValueNum_Feb"] > 0)]
        if merged.empty: return None
        merged = merged.sort_values("sort_key").tail(3)
        ratio = float(np.median(merged["ValueNum_Feb"].values / merged["ValueNum_Oct"].values))
        ratio = float(np.clip(ratio, ret_lo, ret_hi))
        return ratio

    def fmt_val(met_name, v):
        if met_name == "Budget to Enrollment Ratio": return f"{v:.0%}"
        return f"{v:,.0f}"

    origin_year = sort_fy_only(freeze_at)
    future_years = [fy_label(y) for y in range(origin_year + 1, END_FORECAST_FY + 1)]
    horizon_y = len(future_years)
    if horizon_y <= 0:
        st.warning("⚠️ Freeze year is already at/after forecast end.")
        st.stop()

    combined_frames = []
    model_info_rows = []
    need_oct = ("October 1 Count" in selected_metrics)
    need_feb = ("February 1 Count" in selected_metrics)
    oct_hist_df = _get_series("October 1 Count", freeze_at) if need_oct else pd.DataFrame()
    feb_hist_df = _get_series("February 1 Count", freeze_at) if need_feb else pd.DataFrame()
    retention = _estimate_retention_ratio(freeze_at) if need_feb else None

    oct_future = None; oct_growth_used = None
    if need_oct:
        if len(oct_hist_df) < 3:
            st.warning("⚠️ Not enough October history (need ≥ 3) for stable forecast.")
        else:
            oct_vals = oct_hist_df["ValueNum"].values.astype(float)
            oct_future, oct_growth_used = _forecast_oct_history(
                oct_vals, horizon=horizon_y, max_g=max_oct_growth, max_d=max_oct_drop
            )
            model_info_rows.append({
                "Metric": "October 1 Count", "Method": "Conservative bounded YoY",
                "YoY % used": f"{oct_growth_used*100:.2f}%",
                "Notes": f"Clipped to [-{max_oct_drop*100:.0f}%, +{max_oct_growth*100:.0f}%]"
            })
            combined_frames.append(pd.DataFrame({"FY": oct_hist_df["FY"], "ValueNum": oct_hist_df["ValueNum"], "Metric": "October 1 Count", "Type": "Actual"}))
            combined_frames.append(pd.DataFrame({"FY": future_years, "ValueNum": oct_future, "Metric": "October 1 Count", "Type": "Forecast (Frozen)"}))

    if need_feb:
        if retention is None:
            st.warning("⚠️ Could not compute Feb/Oct retention (insufficient paired history). Feb forecast disabled.")
        else:
            model_info_rows.append({
                "Metric": "February 1 Count", "Method": "Derived from October",
                "YoY % used": "", "Notes": f"Feb = Oct × retention, retention={retention:.3f} (clipped {ret_lo:.2f}–{ret_hi:.2f})"
            })
            if not feb_hist_df.empty:
                combined_frames.append(pd.DataFrame({"FY": feb_hist_df["FY"], "ValueNum": feb_hist_df["ValueNum"], "Metric": "February 1 Count", "Type": "Actual"}))
            freeze_fy_str = str(freeze_at)
            has_feb_freeze = (not feb_hist_df.empty) and (freeze_fy_str in feb_hist_df["FY"].values)
            if (not has_feb_freeze) and need_oct and (not oct_hist_df.empty):
                oct_freeze_row = oct_hist_df[oct_hist_df["FY"] == freeze_fy_str]
                if not oct_freeze_row.empty and np.isfinite(oct_freeze_row["ValueNum"].iloc[0]):
                    feb_est = float(oct_freeze_row["ValueNum"].iloc[0] * retention)
                    combined_frames.append(pd.DataFrame([{"FY": freeze_fy_str, "ValueNum": feb_est, "Metric": "February 1 Count", "Type": "Forecast (Frozen)"}]))
            if need_oct and (oct_future is not None):
                feb_future = (oct_future * retention).astype(float)
                if show_intervals:
                    band = 0.015
                    combined_frames.append(pd.DataFrame({"FY": future_years, "ValueNum": feb_future * (1 - band), "Metric": "February 1 Count (P10)", "Type": "Band"}))
                    combined_frames.append(pd.DataFrame({"FY": future_years, "ValueNum": feb_future * (1 + band), "Metric": "February 1 Count (P90)", "Type": "Band"}))
                combined_frames.append(pd.DataFrame({"FY": future_years, "ValueNum": feb_future, "Metric": "February 1 Count", "Type": "Forecast (Frozen)"}))

    for met in selected_metrics:
        if met in ["October 1 Count", "February 1 Count"]:
            continue
        is_ratio = (met == "Budget to Enrollment Ratio")
        dh = _get_series(met, freeze_at)
        if len(dh) < 3:
            st.warning(f"⚠️ Not enough history for {met} (need ≥ 3).")
            continue
        y_hist = dh["ValueNum"].values.astype(float)
        y_future, chosen_model, scores, chosen_fn = forecast_timeseries(
            y=y_hist, q=None, horizon=horizon_y,
            model_choice="Seasonal Naive + Drift (recommended baseline)", season_period=1, is_ratio=is_ratio
        )
        model_info_rows.append({"Metric": met, "Method": "Time-series", "YoY % used": "", "Notes": f"Model: {chosen_model}"})
        combined_frames.append(pd.DataFrame({"FY": dh["FY"], "ValueNum": dh["ValueNum"], "Metric": met, "Type": "Actual"}))
        combined_frames.append(pd.DataFrame({"FY": future_years, "ValueNum": y_future, "Metric": met, "Type": "Forecast (Frozen)"}))

    if not combined_frames:
        st.warning("⚠️ Nothing to chart for current selections.")
        st.stop()

    combined = pd.concat(combined_frames, ignore_index=True)
    combined["sort_key"] = combined["FY"].apply(sort_fy_only)

    fig = go.Figure()
    metrics_to_plot = [m for m in selected_metrics]
    for met in metrics_to_plot:
        for tname in ["Actual", "Forecast (Frozen)"]:
            dt = combined[(combined["Metric"] == met) & (combined["Type"] == tname)].sort_values("sort_key")
            if dt.empty: continue
            fig.add_trace(go.Bar(
                x=dt["FY"], y=dt["ValueNum"], name=f"{met} — {tname}",
                marker_color=ENROLL_COLORS.get((met, tname), None),
                opacity=0.95 if tname == "Actual" else 0.78,
                text=[fmt_val(met, v) for v in dt["ValueNum"]], textposition="outside"
            ))
    fig.update_layout(barmode="group", bargap=BARGAP, bargroupgap=BARGROUPGAP)
    fig.update_xaxes(categoryorder="array", categoryarray=FY22_TO_FY28, tickangle=0)
    fig = apply_plot_style(fig, height=700)
    fig.update_layout(
        title=dict(text=f"{selected_school} — Enrollment Predicted (Freeze at {freeze_at})", x=0.01, y=0.985),
        legend=dict(orientation="h", yanchor="bottom", y=1.28, xanchor="left", x=0.01),
        margin=dict(t=230, r=40, b=90, l=60),
        uniformtext_mode="show", uniformtext_minsize=11
    )
    fig.update_traces(cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)

    if show_model_table and model_info_rows:
        st.markdown("### 🧠 Forecast Method Summary")
        st.dataframe(pd.DataFrame(model_info_rows), use_container_width=True)

# ============================================================
# 5) OTHER METRICS — FACETED (4 PER ROW)
# ============================================================
else:
    st.markdown("## 📌 Other Metrics (Actuals)")

    selected_school = st.sidebar.selectbox("🏫 Select School:", school_options)
    selected_fy = st.sidebar.multiselect(
        "📅 Select Fiscal Year + Quarter:",
        fiscal_options,
        default=fiscal_options
    )
    other_metrics = sorted([m for m in df_long["Metric"].dropna().unique() if m not in csaf_metrics])

    DEFAULT_METRIC = "Current Assets"
    default_metrics = [DEFAULT_METRIC] if DEFAULT_METRIC in other_metrics else ([other_metrics[0]] if other_metrics else [])

    selected_metrics = st.sidebar.multiselect("📊 Select Metric(s):", other_metrics, default=default_metrics)

    filtered = df_long[
        (df_long["Schools"] == selected_school) &
        (df_long["Fiscal Year"].isin(selected_fy)) &
        (df_long["Metric"].isin(selected_metrics))
    ].copy()

    filtered["ValueNum"] = pd.to_numeric(filtered["Value"], errors="coerce")
    filtered = filtered.dropna(subset=["ValueNum"])
    filtered["sort_key"] = filtered["Fiscal Year"].apply(sort_fy)
    filtered = filtered.sort_values("sort_key")

    if filtered.empty:
        st.warning("⚠️ No data for current filters.")
        st.stop()

    filtered["FY Group"] = filtered["Fiscal Year"].astype(str).str.split().str[0]
    filtered["Label"] = filtered["ValueNum"].apply(lambda v: f"${v:,.0f}")

    if len(selected_metrics) == 1:
        metric_title = selected_metrics[0]
    elif len(selected_metrics) <= 4:
        metric_title = " | ".join(selected_metrics)
    else:
        metric_title = f"{len(selected_metrics)} Metrics Selected"

    n_metrics = max(1, len(selected_metrics))
    rows = math.ceil(n_metrics / 4)
    fig_height = 360 * rows + 320

    fig = px.bar(
        filtered, x="Fiscal Year", y="ValueNum",
        color="FY Group", color_discrete_map=fy_color_map, barmode="group",
        facet_col="Metric", facet_col_wrap=4, facet_col_spacing=0.06, facet_row_spacing=0.12,
        text="Label", title=f"{selected_school} — {metric_title}"
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_traces(texttemplate="%{text}", textposition="outside", textfont=dict(size=13), cliponaxis=False, width=0.42)
    fig.update_layout(uniformtext_mode="show", uniformtext_minsize=11)
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    fig.update_layout(bargap=0.12, bargroupgap=0.05)
    fig.update_layout(
        title=dict(x=0.01, y=0.985),
        legend=dict(title="FY Group", orientation="v", yanchor="top", y=0.90, xanchor="left", x=1.12, tracegroupgap=10),
        margin=dict(r=340, t=140, b=90)
    )
    fig.update_xaxes(tickangle=30)
    fig = apply_plot_style(fig, height=fig_height)
    st.plotly_chart(fig, use_container_width=True)
