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

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# THEME / BACKGROUND
# ============================================================
APP_BG = "#dfe7df"
PLOT_BG = "#dfe7df"
GRID_CLR = "rgba(0,0,0,0.10)"

st.set_page_config(page_title="NOLA Financial Tracker", layout="wide")

st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {APP_BG}; }}
      section[data-testid="stSidebar"] {{ background-color: {APP_BG}; }}
      header[data-testid="stHeader"] {{ background: {APP_BG}; }}
      .block-container {{
        padding-top: 1.15rem !important;
        padding-bottom: 2rem !important;
        max-width: 1250px !important;
      }}
      section[data-testid="stSidebar"] * {{ font-size: 13px !important; }}
      section[data-testid="stSidebar"] .stSelectbox,
      section[data-testid="stSidebar"] .stMultiSelect,
      section[data-testid="stSidebar"] .stRadio,
      section[data-testid="stSidebar"] .stCheckbox,
      section[data-testid="stSidebar"] .stSlider {{
        margin-bottom: 0.45rem !important;
      }}
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
        background: #1a1f2e10;
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
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================
logo_path = "nola_parish_logo.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        encoded_logo = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
          .nola-wrap {{ background:{APP_BG}; border-bottom:1px solid rgba(0,0,0,0.10); margin-bottom:8px; }}
          .nola-header {{ display:flex; align-items:center; gap:14px; padding:12px 8px; }}
          .nola-title {{ color:#003366; font-size:26px; font-weight:900; line-height:1.1; }}
          .nola-sub {{ color:#1f1f1f; font-size:14px; margin-top:4px; }}
          .spin {{ width:74px; height:74px; border-radius:50%; animation:spin 6s linear infinite; }}
          @keyframes spin {{ from{{transform:rotate(0deg);}} to{{transform:rotate(360deg);}} }}
        </style>
        <div class="nola-wrap"><div class="nola-header">
          <img class="spin" src="data:image/png;base64,{encoded_logo}">
          <div>
            <div class="nola-title">Welcome to NOLA Public Schools Finance Accountability App</div>
            <div class="nola-sub">NOLA Schools Financial Tracker • Built by Emmanuel Igbokwe</div>
          </div>
        </div></div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown("### Welcome to NOLA Public Schools Finance Accountability App")
    st.caption("NOLA Schools Financial Tracker • Built by Emmanuel Igbokwe")

st.divider()

# ============================================================
# CONSTANTS
# ============================================================
BASE_FONT_SIZE  = 18
AXIS_FONT       = 16
CHART_H         = 760
CHART_H_TALL    = 860
BARGAP          = 0.08
BARGROUPGAP     = 0.04
START_FY        = 22
END_ACTUAL_FY   = 26
END_FORECAST_FY = 28

fy_color_map = {
    "FY22": "#2E6B3C", "FY23": "#E15759",
    "FY24": "#1F77B4", "FY25": "#7B61FF", "FY26": "#FF4FA3",
}

TYPE_COLOR_CSAF_PRED = {
    "Actual":              "#1F77B4",
    "Forecast (Frozen)":   "#E15759",
    "Forecast (Unfrozen)": "#E15759",
}

ENROLL_COLORS = {
    ("October 1 Count",           "Actual"):            "#1F4ED8",
    ("October 1 Count",           "Forecast (Frozen)"): "#93C5FD",
    ("February 1 Count",          "Actual"):            "#D97706",
    ("February 1 Count",          "Forecast (Frozen)"): "#FCD34D",
    ("Budgetted",                 "Actual"):            "#166534",
    ("Budgetted",                 "Forecast (Frozen)"): "#86EFAC",
    ("Budget to Enrollment Ratio","Actual"):            "#7C3AED",
    ("Budget to Enrollment Ratio","Forecast (Frozen)"): "#C4B5FD",
}


def fy_label(y: int) -> str:
    return f"FY{int(y):02d}"


def fy_num(fy_str: str):
    s = str(fy_str)
    digits = re.sub(r"[^0-9]", "", s)
    if not digits:
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
    try:
        parts = str(x).split()
        year  = fy_num(parts[0]) if parts else None
        if year is None:
            return (999, 9)
        q = int(parts[1].replace("Q", "").strip()) if len(parts) > 1 and parts[1].upper().startswith("Q") else 9
        return (year, q)
    except Exception:
        return (999, 9)


def sort_fy_only(x):
    n = fy_num(x)
    return n if n is not None else 999


def full_fy_range(start_y, end_y):
    return [fy_label(y) for y in range(start_y, end_y + 1)]


FY22_TO_FY28 = full_fy_range(START_FY, END_FORECAST_FY)


def std_fyq_label(x):
    s     = str(x).strip()
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
csaf_metrics = [
    "FB Ratio", "Liabilities to Assets",
    "Current Ratio", "Unrestricted Days COH",
]
csaf_desc = {
    "FB Ratio":              "Fund Balance Ratio: Unrestricted Fund Balance / Total Exp.",
    "Liabilities to Assets": "Liabilities to Assets Ratio: Total Liabilities / Total Assets",
    "Current Ratio":         "Current Ratio: Current Assets / Current Liabilities",
    "Unrestricted Days COH": "Unrestricted Cash on Hand: Cash / ((Exp.-Depreciation)/365)",
}
csaf_best = {
    "FB Ratio":              {"threshold": 0.10, "direction": "gte"},
    "Liabilities to Assets": {"threshold": 0.90, "direction": "lte"},
    "Current Ratio":         {"threshold": 1.50, "direction": "gte"},
    "Unrestricted Days COH": {"threshold": 60.0, "direction": "gte"},
}


def add_best_practice_csaf(fig, metric, row=None, col=None):
    if metric not in csaf_best:
        return fig
    thr   = csaf_best[metric]["threshold"]
    label = (f"{thr:.0%}" if metric == "FB Ratio"
             else (f"{thr:.2f}" if metric in ("Liabilities to Assets", "Current Ratio")
                   else f"{thr:.0f}"))
    kwargs = dict(
        y=thr, line_dash="dot", line_color="#005A9C", line_width=3,
        annotation_text=label, annotation_position="top left",
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

value_vars = [c for c in df.columns if c not in ["Schools", "Fiscal Year"]]
df_long = df.melt(
    id_vars=["Schools", "Fiscal Year"],
    value_vars=value_vars,
    var_name="Metric",
    value_name="Value",
)
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
    missing  = [c for c in required if c not in df_budget_raw.columns]
    if missing:
        st.warning(f"⚠️ Enrollment sheet missing required columns: {missing}")
    else:
        df_budget_raw = df_budget_raw.dropna(subset=["Schools", "Fiscal Year"]).copy()
        df_budget_raw["Fiscal Year"] = df_budget_raw["Fiscal Year"].astype(str).str.strip().apply(fy_std)
        expected_cols = [
            "Budgetted", "October 1 Count",
            "February 1 Count", "Budget to Enrollment Ratio",
        ]
        df_budget_long = df_budget_raw.melt(
            id_vars=["Schools", "Fiscal Year"],
            value_vars=[c for c in expected_cols if c in df_budget_raw.columns],
            var_name="Metric",
            value_name="Value",
        )
        school_options_budget = sorted(df_budget_long["Schools"].dropna().unique())
        fy_options_budget     = sorted(df_budget_long["Fiscal Year"].dropna().unique(), key=sort_fy_only)
except Exception as e:
    st.warning(f"⚠️ Could not load '{fy26_path}' / sheet 'FY26 Student enrollment': {e}")


# ============================================================
# ── CSAF FORECAST ENGINE ─────────────────────────────────────
#
# Replaced all lag-based ML models (HGBR, MLP, Ridge, HuberRegressor).
# Root cause of old misalignment: treating Q1→Q2 as consecutive time steps
# caused models to learn a "crash" shape instead of the true seasonal sawtooth.
#
# New engine: Two-model seasonal ensemble
#
# Model A (60%) — Per-Quarter YoY Drift
#   Split history into Q1, Q2, Q3 sub-series independently.
#   Compute median of last-2 YoY growth rates within that quarter sub-series.
#   Cap growth by _YOY_CAPS, project n years ahead from last known same-Q value.
#
# Model B (40%) — Q1-Anchored Seasonal Ratio
#   Fit log-linear (OLS in log1p space) trend to Q1 sub-series.
#   Derive Q2 = Q1_forecast × school-specific median(Q2/Q1 history)
#   Derive Q3 = Q1_forecast × school-specific median(Q3/Q1 history)
#   Falls back to global calibrated ratios when < 2 paired observations exist.
#
# Global seasonal ratios (calibrated from 29 schools × 14 quarters,
# winsorised P5-P95 before taking median):
#   FB Ratio:              Q2/Q1=0.464  Q3/Q1=0.318
#   Unrestricted Days COH: Q2/Q1=0.445  Q3/Q1=0.316
#   Current Ratio:         Q2/Q1=0.943  Q3/Q1=0.999
#   Liabilities to Assets: Q2/Q1=1.014  Q3/Q1=0.939
# ============================================================

_GLOBAL_SEASONAL: dict = {
    "FB Ratio":              {2: 0.4640, 3: 0.3177},
    "Unrestricted Days COH": {2: 0.4453, 3: 0.3156},
    "Current Ratio":         {2: 0.9428, 3: 0.9986},
    "Liabilities to Assets": {2: 1.0135, 3: 0.9389},
}

# Max YoY growth / drop allowed per metric
_YOY_CAPS: dict = {
    "FB Ratio":              (-0.30, 0.40),
    "Unrestricted Days COH": (-0.30, 0.40),
    "Current Ratio":         (-0.25, 0.30),
    "Liabilities to Assets": (-0.25, 0.30),
}

_fyq_re = re.compile(r"FY\s*(\d{2,4})\s*Q\s*(\d)", re.IGNORECASE)


def parse_fyq(label: str):
    """Return (fy_2digit, q) or (None, None)."""
    m = _fyq_re.search(str(label))
    if not m:
        return None, None
    fy = int(m.group(1))
    fy = fy % 100 if fy >= 100 else fy
    return fy, int(m.group(2))


def make_future_fyq_labels(last_label: str, n: int, q_per_year: int = 3) -> list:
    """Produce n quarter labels following last_label."""
    fy, qq = parse_fyq(last_label)
    if fy is None:
        fy, qq = END_ACTUAL_FY, 0
    out = []
    for _ in range(n):
        qq += 1
        if qq > q_per_year:
            fy += 1
            qq  = 1
        out.append(f"FY{fy:02d} Q{qq}")
    return out


def _winsorise(arr, lo: int = 10, hi: int = 90) -> np.ndarray:
    arr   = np.asarray(arr, dtype=float)
    valid = arr[np.isfinite(arr)]
    if len(valid) < 4:
        return arr
    return np.clip(arr, np.percentile(valid, lo), np.percentile(valid, hi))


def _log_linear_forecast(x_vals, y_vals, x_future) -> np.ndarray:
    """OLS in log1p space → expm1 predictions (≥ 0)."""
    y_log = np.log1p(np.clip(np.asarray(y_vals, dtype=float), 0, None))
    X     = np.column_stack([np.ones(len(x_vals)), np.asarray(x_vals, dtype=float)])
    Xf    = np.column_stack([np.ones(len(x_future)), np.asarray(x_future, dtype=float)])
    mdl   = LinearRegression(fit_intercept=False)
    mdl.fit(X, y_log)
    return np.expm1(mdl.predict(Xf))


def forecast_csaf_per_school(
    y, labels: list, horizon: int, metric_name: str
):
    """
    Seasonal ensemble forecast for one school / one CSAF metric.

    Returns
    -------
    predictions   : np.ndarray shape (horizon,)
    future_labels : list[str] length horizon
    method_desc   : str
    """
    y      = np.clip(np.asarray(y, dtype=float), 0, None)
    n      = len(y)
    labels = list(labels)
    future_labels = make_future_fyq_labels(labels[-1] if labels else "FY26 Q2", horizon)

    if n < 3:
        return np.full(horizon, float(y[-1]) if n else 0.0), future_labels, "Insufficient history"

    fys = np.array([parse_fyq(l)[0] or 0 for l in labels], dtype=int)
    qs  = np.array([parse_fyq(l)[1] or 1 for l in labels], dtype=int)

    cap_lo, cap_hi  = _YOY_CAPS.get(metric_name, (-0.30, 0.40))
    global_seasonal = _GLOBAL_SEASONAL.get(metric_name, {2: 1.0, 3: 1.0})

    # ── Model A: per-quarter YoY drift ──────────────────────────────────────
    def _modelA(fy_f: int, q_f: int) -> float:
        mask   = qs == q_f
        y_sub  = _winsorise(y[mask])
        fy_sub = fys[mask]
        if len(y_sub) < 2:
            return float(np.nanmedian(_winsorise(y))) if n else 0.0
        yoy = []
        for i in range(1, len(y_sub)):
            prev = y_sub[i - 1]
            if prev > 0 and np.isfinite(prev) and np.isfinite(y_sub[i]):
                yoy.append((y_sub[i] - prev) / prev)
        med_g   = float(np.clip(np.median(yoy[-2:]) if yoy else 0.0, cap_lo, cap_hi))
        n_ahead = max(1, fy_f - int(fy_sub[-1]))
        return max(float(y_sub[-1]) * ((1.0 + med_g) ** n_ahead), 0.0)

    # ── Model B: Q1-anchored seasonal ratio ─────────────────────────────────
    mask_q1 = qs == 1
    y_q1    = _winsorise(y[mask_q1])
    fy_q1   = fys[mask_q1]

    def _school_ratio(q_target: int) -> float:
        ratios = []
        for fy in np.unique(fys):
            m1 = (fys == fy) & (qs == 1)
            mq = (fys == fy) & (qs == q_target)
            if m1.any() and mq.any():
                v1 = float(y[m1][0]); vq = float(y[mq][0])
                if v1 > 0 and np.isfinite(v1) and np.isfinite(vq):
                    ratios.append(vq / v1)
        return float(np.median(ratios)) if len(ratios) >= 2 else float(global_seasonal.get(q_target, 1.0))

    ratio_q2 = _school_ratio(2)
    ratio_q3 = _school_ratio(3)

    def _modelB_q1(fy_f: int) -> float:
        if len(y_q1) < 2:
            return float(np.nanmedian(y_q1)) if len(y_q1) else 0.0
        pred = _log_linear_forecast(fy_q1.astype(float), y_q1, [float(fy_f)])
        return max(float(pred[0]), 0.0)

    def _modelB(fy_f: int, q_f: int) -> float:
        q1v = _modelB_q1(fy_f)
        if q_f == 2: return max(q1v * ratio_q2, 0.0)
        if q_f == 3: return max(q1v * ratio_q3, 0.0)
        return q1v  # Q1

    # ── Ensemble ─────────────────────────────────────────────────────────────
    predictions = []
    for lbl in future_labels:
        fy_f, q_f = parse_fyq(lbl)
        if fy_f is None:
            fy_f, q_f = int(fys[-1]) + 1, 1
        pA   = _modelA(fy_f, q_f)
        pB   = _modelB(fy_f, q_f)
        pred = max(0.60 * pA + 0.40 * pB, 0.0)
        predictions.append(pred)

    method_desc = (
        "Seasonal Ensemble — Per-Quarter YoY Drift (60%) "
        "+ Q1-Anchored Seasonal Ratio (40%)"
    )
    return np.array(predictions, dtype=float), future_labels, method_desc


def csaf_bootstrap_bands(
    y, labels: list, horizon: int, metric_name: str,
    n_sims: int = 300, p_lo: int = 10, p_hi: int = 90, seed: int = 42,
):
    """Bootstrap P10 / P50 / P90 using LOO residuals per quarter sub-series."""
    rng    = np.random.default_rng(seed)
    y      = np.clip(np.asarray(y, dtype=float), 0, None)
    labels = list(labels)
    qs     = np.array([parse_fyq(l)[1] or 1 for l in labels], dtype=int)

    residuals = []
    for q in [1, 2, 3]:
        mask    = qs == q
        y_sub   = y[mask]
        lbl_sub = [l for l, qq in zip(labels, qs) if qq == q]
        for i in range(1, len(y_sub)):
            preds, _, _ = forecast_csaf_per_school(y_sub[:i], lbl_sub[:i], 1, metric_name)
            residuals.append(float(y_sub[i] - preds[0]))

    if len(residuals) < 3:
        residuals = [0.0]
    residuals = np.array(residuals, dtype=float)

    base, _, _ = forecast_csaf_per_school(y, labels, horizon, metric_name)
    damp       = np.linspace(1.0, 0.55, horizon)
    sims       = np.zeros((n_sims, horizon), dtype=float)
    for s in range(n_sims):
        noise      = rng.choice(residuals, size=horizon, replace=True) * damp
        sims[s, :] = np.clip(base + noise, 0, None)

    return (
        np.percentile(sims, p_lo, axis=0),
        np.percentile(sims, 50,   axis=0),
        np.percentile(sims, p_hi, axis=0),
    )


# ============================================================
# ENROLLMENT FORECAST (simple annual YoY drift)
# ============================================================
def forecast_enrollment_series(y, horizon: int, is_ratio: bool = False) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return np.full(horizon, float(y[-1]) if len(y) else 0.0)
    pct_changes = []
    for i in range(1, len(y)):
        if y[i - 1] > 0 and np.isfinite(y[i - 1]) and np.isfinite(y[i]):
            pct_changes.append((y[i] - y[i - 1]) / y[i - 1])
    if not pct_changes:
        g = 0.0
    else:
        g = float(np.median(pct_changes[-3:]))
    g   = float(np.clip(g, -0.10, 0.10) if is_ratio else np.clip(g, -0.05, 0.05))
    out = []; cur = float(y[-1])
    for _ in range(horizon):
        cur = max(cur * (1.0 + g), 0.0); out.append(cur)
    return np.array(out, dtype=float)


# ============================================================
# PREDICTIVE RISK MODEL
# ============================================================
RISK_CONDITIONS = [
    ("FB Ratio",                        "lt", 0.05,    3, "Fund balance ratio is critically low (< 5%)"),
    ("FB Ratio",                        "lt", 0.10,    2, "Fund balance ratio is below the 10% target"),
    ("Liabilities to Assets",           "gt", 0.90,    3, "Liabilities-to-assets ratio critically elevated (> 0.90)"),
    ("Liabilities to Assets",           "gt", 0.75,    2, "Liabilities-to-assets ratio above moderate threshold (> 0.75)"),
    ("Current Ratio",                   "lt", 1.00,    3, "Current ratio below 1.0 — bills may not be payable"),
    ("Current Ratio",                   "lt", 1.50,    2, "Current ratio below 1.50 best-practice floor"),
    ("Unrestricted Days COH",           "lt", 40.0,    3, "Cash on hand critically low (< 40 days)"),
    ("Unrestricted Days COH",           "lt", 60.0,    2, "Cash on hand below 60-day best-practice threshold"),
    ("Unrestricted Cash & Equivalents", "lt", 500_000, 2, "Unrestricted cash below $500,000"),
]

_OPS = {
    "lt":  lambda v, t: v < t,
    "gt":  lambda v, t: v > t,
    "lte": lambda v, t: v <= t,
    "gte": lambda v, t: v >= t,
}

RISK_FEATURE_COLS = [
    "FB Ratio", "Liabilities to Assets", "Current Ratio", "Unrestricted Days COH",
    "Unrestricted Cash & Equivalents", "Total Revenue", "Total Expenses",
    "Current Assets", "Current Liabilities", "Total Liabilities", "Total Assets",
    "Salaries", "Employee Benefits",
]


def _score_row(row: pd.Series) -> int:
    score = 0
    for col, op, thr, pts, _ in RISK_CONDITIONS:
        v = row.get(col, np.nan)
        if pd.isna(v):
            continue
        if _OPS[op](float(v), thr):
            score += pts
    rev = row.get("Total Revenue", np.nan); exp = row.get("Total Expenses", np.nan)
    if pd.notna(rev) and pd.notna(exp) and float(exp) > float(rev):
        score += 2
    return score


def _score_to_label(score: int) -> str:
    return "High" if score >= 8 else ("Medium" if score >= 4 else "Low")


def _build_risk_dataset(df_src: pd.DataFrame) -> pd.DataFrame:
    df_work = df_src.copy()
    feat_cols_present = [c for c in RISK_FEATURE_COLS if c in df_work.columns]
    for c in feat_cols_present:
        df_work[c] = pd.to_numeric(df_work[c], errors="coerce")
    df_work["_sort_key"] = df_work["Fiscal Year"].apply(sort_fy)
    df_work = df_work.sort_values(["Schools", "_sort_key"]).reset_index(drop=True)
    df_work["_risk_score"] = df_work.apply(_score_row, axis=1)
    df_work["_risk_label"] = df_work["_risk_score"].apply(_score_to_label)
    if "Total Revenue" in df_work.columns and "Total Expenses" in df_work.columns:
        df_work["_expense_over_revenue"] = (df_work["Total Expenses"] > df_work["Total Revenue"]).astype(int)
    else:
        df_work["_expense_over_revenue"] = 0

    rows_out = []
    for school, grp in df_work.groupby("Schools", sort=False):
        grp = grp.sort_values("_sort_key").reset_index(drop=True)
        for i in range(len(grp)):
            row      = grp.iloc[i]
            feat_row = {
                "Schools": school, "Fiscal Year": row["Fiscal Year"],
                "_sort_key": row["_sort_key"], "_risk_label": row["_risk_label"],
                "_risk_score": row["_risk_score"],
                "_expense_over_revenue": row.get("_expense_over_revenue", 0),
            }
            for c in feat_cols_present:
                v0 = row.get(c, np.nan)
                v1 = grp.iloc[i - 1][c] if i >= 1 else np.nan
                v2 = grp.iloc[i - 2][c] if i >= 2 else np.nan
                feat_row[c]           = v0
                feat_row[f"{c}_lag1"] = v1
                feat_row[f"{c}_lag2"] = v2
                feat_row[f"{c}_chg"]  = (float(v0) - float(v1)) if (pd.notna(v0) and pd.notna(v1)) else np.nan
            rows_out.append(feat_row)

    df_feat   = pd.DataFrame(rows_out)
    targets   = []
    for school, grp in df_feat.groupby("Schools", sort=False):
        grp     = grp.sort_values("_sort_key").reset_index(drop=True)
        shifted = grp["_risk_label"].shift(-1)
        for idx2 in range(len(grp)):
            targets.append({
                "Schools": school,
                "Fiscal Year": grp.loc[idx2, "Fiscal Year"],
                "_next_q_risk": shifted.iloc[idx2],
            })
    df_targets = pd.DataFrame(targets)
    return pd.merge(df_feat, df_targets, on=["Schools", "Fiscal Year"], how="left")


def _train_risk_model(df_feat: pd.DataFrame):
    df_train = df_feat.dropna(subset=["_next_q_risk"]).copy()
    if df_train.empty:
        return None, [], []
    exclude = {
        "Schools", "Fiscal Year", "_sort_key",
        "_risk_label", "_risk_score", "_next_q_risk", "_expense_over_revenue",
    }
    ml_cols = [c for c in df_train.columns if c not in exclude]
    X = df_train[ml_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y_labels = df_train["_next_q_risk"].values
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=2,
        random_state=42, class_weight="balanced",
    )
    clf.fit(X, y_labels)
    return clf, ml_cols, clf.classes_


def _explain_risk(row: pd.Series, feat_cols_present: list) -> list:
    explanations = []
    for col, op, thr, _, label in RISK_CONDITIONS:
        if col not in row.index:
            continue
        v = row.get(col, np.nan)
        if pd.isna(v):
            continue
        if _OPS[op](float(v), thr):
            explanations.append(label)
    rev = row.get("Total Revenue", np.nan); exp = row.get("Total Expenses", np.nan)
    if pd.notna(rev) and pd.notna(exp) and float(exp) > float(rev):
        explanations.append(f"Total expenses exceed revenue by ${float(exp)-float(rev):,.0f}")
    qoq_checks = [
        ("FB Ratio",             "down", "Fund balance ratio declined from prior quarter"),
        ("Current Ratio",        "down", "Current ratio declined from prior quarter"),
        ("Unrestricted Days COH","down", "Cash on hand declined from prior quarter"),
        ("Total Revenue",        "down", "Revenue declined from prior quarter"),
        ("Total Expenses",       "up",   "Total expenses increased from prior quarter"),
        ("Liabilities to Assets","up",   "Liabilities-to-assets ratio worsened from prior quarter"),
    ]
    for col, direction, msg in qoq_checks:
        chg = row.get(f"{col}_chg", np.nan)
        if pd.isna(chg):
            continue
        if direction == "down" and float(chg) < 0:
            explanations.append(msg)
        if direction == "up" and float(chg) > 0:
            explanations.append(msg)
    return explanations if explanations else ["No specific risk factors flagged for this quarter."]


# ============================================================
# SIDEBAR NAV
# ============================================================
st.sidebar.header("🔎 Filters")
modes = [
    "Executive Summary (School Report Card)",
    "CSAF Metrics (4-panel)",
    "CSAF Predicted",
    "Other Metrics",
    "Predictive Risk Model",
]
if not df_budget_long.empty:
    modes += ["Budget/Enrollment (Bar)", "Budget/Enrollment Predicted (Bar)"]

metric_group = st.sidebar.radio("Choose Dashboard:", modes)


# ============================================================
# GAUGE helper
# ============================================================
def make_gauge(value, vmin, vmax, threshold, good_direction, value_fmt, title_text):
    vmax_eff = max(vmax, float(value) * 1.10) if np.isfinite(value) else vmax
    thr_eff  = min(max(threshold, vmin), vmax_eff)
    if good_direction == "gte":
        steps = [
            {"range": [vmin, thr_eff], "color": "#e11d48"},
            {"range": [thr_eff, vmax_eff], "color": "#15803d"},
        ]
    else:
        steps = [
            {"range": [vmin, thr_eff], "color": "#15803d"},
            {"range": [thr_eff, vmax_eff], "color": "#e11d48"},
        ]
    thr_label = (f"{thr_eff:.0%}" if value_fmt.endswith("%")
                 else (f"{thr_eff:.2f}" if thr_eff < 10 else f"{thr_eff:,.0f}"))
    axis_conf = dict(
        range=[vmin, vmax_eff], tickmode="array",
        tickvals=[thr_eff], ticktext=[thr_label],
        tickcolor="#0ea5e9", ticklen=8, ticks="outside",
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=0 if not np.isfinite(value) else float(value),
        number={"valueformat": value_fmt, "font": {"size": 34, "color": "#14b8a6", "family": "Arial Black"}},
        gauge={
            "shape": "angular", "axis": axis_conf,
            "bar": {"color": "rgba(0,0,0,0)"}, "steps": steps,
            "threshold": {"line": {"color": "#0ea5e9", "width": 3}, "thickness": 0.75, "value": thr_eff},
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig = apply_plot_style(fig, height=260)
    fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG, margin=dict(t=10, r=10, b=10, l=10))
    return fig


# ============================================================
# 0) EXECUTIVE SUMMARY
# ============================================================
if metric_group == "Executive Summary (School Report Card)":
    st.markdown("## 🏫 Executive Summary — School Report Card")
    selected_school = st.sidebar.selectbox("Select School", school_options)

    def _risk_status(metric_name, v):
        if v is None or not np.isfinite(v):
            return "No Data"
        thr = csaf_best[metric_name]["threshold"]
        direction = csaf_best[metric_name]["direction"]
        band = 0.05 * thr if thr > 1 else 0.02
        if direction == "gte":
            return "On Track" if v >= thr else ("Monitor" if v >= thr - band else "At Risk")
        else:
            return "On Track" if v <= thr else ("Monitor" if v <= thr + band else "At Risk")

    def _latest_for_school(school):
        d0 = df[df["Schools"] == school].copy()
        if d0.empty:
            return {m: np.nan for m in csaf_metrics}
        d0["sort_key"] = d0["Fiscal Year"].apply(sort_fy)
        d0 = d0.sort_values("sort_key")
        out = {}
        for met in csaf_metrics:
            vals = pd.to_numeric(d0[met], errors="coerce").dropna()
            out[met] = float(vals.iloc[-1]) if len(vals) else np.nan
        return out

    latest = _latest_for_school(selected_school)
    fb  = latest.get("FB Ratio", np.nan)
    cr  = latest.get("Current Ratio", np.nan)
    dch = latest.get("Unrestricted Days COH", np.nan)
    la  = latest.get("Liabilities to Assets", np.nan)

    def cls_for(metric, v):
        s = _risk_status(metric, v)
        return "kpi-good" if s == "On Track" else ("kpi-warn" if s == "Monitor" else "kpi-bad")

    def kpi_html(label, value_txt, status_txt, status_cls):
        return (f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value_txt}</div>'
                f'<div class="kpi-sub"><span class="{status_cls}">{status_txt}</span></div></div>')

    st.markdown(
        f'<div class="kpi-grid">'
        f'{kpi_html("Fund Balance Ratio", f"{fb:.1%}" if np.isfinite(fb) else "—", _risk_status("FB Ratio",fb), cls_for("FB Ratio",fb))}'
        f'{kpi_html("Current Ratio", f"{cr:.2f}" if np.isfinite(cr) else "—", _risk_status("Current Ratio",cr), cls_for("Current Ratio",cr))}'
        f'{kpi_html("Days Cash on Hand", f"{dch:,.0f}" if np.isfinite(dch) else "—", _risk_status("Unrestricted Days COH",dch), cls_for("Unrestricted Days COH",dch))}'
        f'{kpi_html("Liabilities / Assets", f"{la:.2f}" if np.isfinite(la) else "—", _risk_status("Liabilities to Assets",la), cls_for("Liabilities to Assets",la))}'
        f'</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Fund Balance Ratio: Will an unforeseen event result in fiscal crisis?**  \n_Unrestricted Fund Balance / Total Exp.  •  Best practice ≥ 10%_")
        st.plotly_chart(make_gauge(fb, 0.0, 0.60, 0.10, "gte", ".0%", "Fund Balance Ratio"), use_container_width=True)
    with c2:
        st.markdown("**Liabilities to Assets Ratio: What % of liabilities are financed by assets?**  \n_Total Liabilities / Total Assets.  Lower is better.  •  Best practice ≤ 0.90_")
        st.plotly_chart(make_gauge(la, 0.0, 1.50, 0.90, "lte", ".2f", "Liabilities to Assets"), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Current Ratio: Can bills be paid?**  \n_Current Assets / Current Liabilities.  •  Best practice ≥ 1.50_")
        vmax_cr = max(3.0, cr * 1.1) if np.isfinite(cr) else 3.0
        st.plotly_chart(make_gauge(cr, 0.0, vmax_cr, 1.50, "gte", ".2f", "Current Ratio"), use_container_width=True)
    with c4:
        st.markdown("**Unrestricted Cash on Hand: Enough cash to pay bills for 60+ days?**  \n_Unrestricted Cash / ((Total Exp. − Depreciation)/365).  •  Best practice ≥ 60_")
        vmax_dch = 300.0 if not np.isfinite(dch) else max(300.0, math.ceil(dch / 50.0) * 50.0)
        st.plotly_chart(make_gauge(dch, 0.0, vmax_dch, 60.0, "gte", ",.0f", "Unrestricted Days COH"), use_container_width=True)

    def _fmt(v, metric): return fmt_csaf(metric, v) if np.isfinite(v) else "—"
    def _sb(metric, v): return {"On Track": 2, "Monitor": 1, "At Risk": 0}.get(_risk_status(metric, v), 1)
    def _overall_health(fb, cr, dch, la):
        pts  = 0.35 * _sb("Current Ratio", cr) + 0.35 * _sb("Unrestricted Days COH", dch)
        pts += 0.20 * _sb("FB Ratio", fb) + 0.10 * _sb("Liabilities to Assets", la)
        return "Strong" if pts >= 1.6 else ("Stable" if pts >= 1.1 else ("Watch" if pts >= 0.7 else "At Risk"))

    def _strengths(fb, cr, dch, la):
        s = []
        if np.isfinite(dch) and dch >= 60:   s.append(f"**Cash runway** healthy at **{_fmt(dch,'Unrestricted Days COH')}** (≥ 60 days).")
        if np.isfinite(cr) and cr >= 1.50:   s.append(f"**Short-term liquidity** solid — CR **{_fmt(cr,'Current Ratio')}** (≥ 1.50×).")
        if np.isfinite(fb) and fb >= 0.10:   s.append(f"**Fund balance** at **{_fmt(fb,'FB Ratio')}** provides shock cushion.")
        if np.isfinite(la) and la <= 0.90:   s.append(f"**Leverage** contained at **{_fmt(la,'Liabilities to Assets')}** (≤ 0.90).")
        return s

    def _vulns(fb, cr, dch, la):
        v = []
        if np.isfinite(cr) and cr < 1.50:    v.append(f"**Liquidity tightness** — CR **{_fmt(cr,'Current Ratio')}** < 1.50×.")
        if np.isfinite(dch) and dch < 60:    v.append(f"**Cash runway below floor** — **{_fmt(dch,'Unrestricted Days COH')}** (< 60 days).")
        if np.isfinite(fb) and fb < 0.10:    v.append(f"**Fund balance under target** — **{_fmt(fb,'FB Ratio')}** (< 10%).")
        if np.isfinite(la) and la > 0.90:    v.append(f"**Elevated leverage** — **{_fmt(la,'Liabilities to Assets')}** (> 0.90).")
        return v

    def _plan(fb, cr, dch, la):
        p = []
        if not np.isfinite(cr) or cr < 1.50:
            p += ["Smooth **payables** across months; avoid same-month vendor clusters.",
                  "Defer **non-critical capex** until liquidity normalizes.",
                  "Tighten **A/R collections** (move to weekly grant submissions).",
                  "Pre-arrange **contingency LOC** sized to 1–1.5× avg tight-month outflows."]
        if not np.isfinite(dch) or dch < 60:
            p += ["Target **≥ 60 days COH**: accelerate grant draws; rebalance disbursement calendars.",
                  "Negotiate **vendor terms** (net-45/60) where feasible."]
        if not np.isfinite(fb) or fb < 0.10:
            p += ["Drive **structural balance**: align staffing with live enrollment/retention.",
                  "Freeze **non-essential spend**; enforce purchase order controls."]
        if np.isfinite(la) and la > 0.90:
            p += ["Reduce **short-term liabilities** and schedule principal pay-downs.",
                  "Avoid new obligations until liquidity and reserves meet thresholds."]
        if not p:
            p = ["Maintain **current controls** and monthly cash cadence.",
                 "Review staffing vs. enrollment **each quarter**; keep reserves ≥ policy."]
        return p

    overall   = _overall_health(fb, cr, dch, la)
    strengths = _strengths(fb, cr, dch, la)
    risks     = _vulns(fb, cr, dch, la)
    plan_raw  = _plan(fb, cr, dch, la)
    seen = set(); plan_unique = []
    for item in plan_raw:
        if item not in seen: plan_unique.append(item); seen.add(item)

    st.markdown(
        f"### 📌 {selected_school}: Executive Financial Health — **{overall}**\n"
        f"**FB Ratio:** {_fmt(fb,'FB Ratio')}  •  **CR:** {_fmt(cr,'Current Ratio')}  •  "
        f"**Days COH:** {_fmt(dch,'Unrestricted Days COH')}  •  **Liab/Assets:** {_fmt(la,'Liabilities to Assets')}"
    )
    if strengths: st.markdown("#### ✅ Strengths\n" + "\n".join(f"- {s}" for s in strengths))
    if risks:     st.markdown("#### ⚠️ Vulnerabilities\n" + "\n".join(f"- {r}" for r in risks))
    st.markdown("#### 🗓️ 90-Day CFO Plan (Prioritized)")
    st.markdown("\n".join(f"{i+1}. {p}" for i, p in enumerate(plan_unique)))

    md_lines = [
        f"# Executive Summary — {selected_school}", f"**Overall Health:** {overall}", "",
        "## Current Status",
        f"* FB Ratio: {_fmt(fb,'FB Ratio')}", f"* Current Ratio: {_fmt(cr,'Current Ratio')}",
        f"* Days COH: {_fmt(dch,'Unrestricted Days COH')}", f"* Liabilities/Assets: {_fmt(la,'Liabilities to Assets')}", "",
        "## Strengths", *(["* " + s for s in strengths] if strengths else ["* —"]), "",
        "## Vulnerabilities", *(["* " + r for r in risks] if risks else ["* —"]), "",
        "## 90-Day CFO Plan", *[f"{i+1}. {p}" for i, p in enumerate(plan_unique)], "",
        "## Thresholds", "* FB Ratio ≥ 10%", "* Current Ratio ≥ 1.50×",
        "* Unrestricted Days COH ≥ 60", "* Liabilities to Assets ≤ 0.90",
    ]
    st.download_button(
        "⬇ Download Report Card (Markdown)", data="\n".join(md_lines),
        file_name=f"{selected_school.replace(' ','_').lower()}_report_card.md",
        mime="text/markdown",
    )

# ============================================================
# 1) CSAF METRICS — 4 PANEL
# ============================================================
elif metric_group == "CSAF Metrics (4-panel)":
    st.markdown("## 📌 CSAF Metrics (4-panel)")
    selected_school = st.sidebar.selectbox("🏫 Select School:", school_options)
    selected_fy     = st.sidebar.multiselect("📅 Select Fiscal Year + Quarter:", fiscal_options, default=fiscal_options)

    d = df[(df["Schools"] == selected_school) & (df["Fiscal Year"].isin(selected_fy))].copy()
    if d.empty: st.warning("⚠️ No data for selection."); st.stop()

    d["sort_key"] = d["Fiscal Year"].apply(sort_fy)
    d = d.sort_values("sort_key")
    d["FY Group"] = d["Fiscal Year"].astype(str).str.split().str[0]
    x_order = d["Fiscal Year"].tolist()

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[csaf_desc["FB Ratio"], csaf_desc["Liabilities to Assets"],
                        csaf_desc["Current Ratio"], csaf_desc["Unrestricted Days COH"]],
        horizontal_spacing=0.08, vertical_spacing=0.12)

    metric_positions = {
        "FB Ratio": (1, 1), "Liabilities to Assets": (1, 2),
        "Current Ratio": (2, 1), "Unrestricted Days COH": (2, 2),
    }
    for met, (r, c) in metric_positions.items():
        dd = d.copy()
        dd["ValueNum"] = pd.to_numeric(dd[met], errors="coerce")
        dd = dd.dropna(subset=["ValueNum"])
        dd["FY Group"] = dd["FY Group"].astype(str)
        fy_groups_ordered = (
            dd.drop_duplicates("FY Group")[["FY Group", "sort_key"]]
            .sort_values("sort_key")["FY Group"].tolist()
        )
        for fygrp in fy_groups_ordered:
            sub = dd[dd["FY Group"] == fygrp]
            if sub.empty: continue
            show_leg = r == 1 and c == 1
            fig.add_trace(go.Bar(
                x=sub["Fiscal Year"], y=sub["ValueNum"], name=fygrp,
                legendgroup=fygrp, showlegend=show_leg,
                marker_color=fy_color_map.get(fygrp, None),
                text=[fmt_csaf(met, v) for v in sub["ValueNum"]],
                textposition="outside", cliponaxis=False,
            ), row=r, col=c)
        add_best_practice_csaf(fig, met, row=r, col=c)
        fig.update_xaxes(row=r, col=c, categoryorder="array", categoryarray=x_order,
                         tickangle=-35, tickfont=dict(size=10), automargin=True)
        fig.update_yaxes(row=r, col=c, tickfont=dict(size=11), automargin=True)

    fig.update_layout(barmode="group", bargap=BARGAP, bargroupgap=BARGROUPGAP,
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG, height=980, font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.14, xanchor="left", x=0,
                    font=dict(size=12), tracegroupgap=10),
        margin=dict(t=140, b=80, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

# ============================================================
# 2) CSAF PREDICTED — Seasonal Ensemble
# ============================================================
elif metric_group == "CSAF Predicted":
    st.markdown("## 🔮 CSAF Predicted — Seasonal Ensemble Forecast")
    st.caption(
        "**Engine:** Per-Quarter YoY Drift (60%) + Q1-Anchored Seasonal Ratio (40%).  "
        "Preserves the Q1 > Q2 > Q3 sawtooth pattern observed across all 29 NOLA schools."
    )

    selected_school = st.sidebar.selectbox("🏫 Select School:", school_options)
    selected_metric = st.sidebar.selectbox("📊 Select CSAF Metric:", csaf_metrics)

    forecast_mode = st.sidebar.radio(
        "🧊 Forecast Mode",
        ["Freeze at selected quarter", "Unfrozen (use all actuals)"],
        index=0,
    )

    all_quarters    = sorted(df["Fiscal Year"].dropna().astype(str).unique(), key=sort_fy)
    last_actual_qtr = all_quarters[-1] if all_quarters else None

    if forecast_mode == "Freeze at selected quarter":
        freeze_at       = st.sidebar.selectbox("🧊 Freeze at:", all_quarters,
                                                index=max(0, len(all_quarters) - 1))
        train_end_label = freeze_at
    else:
        train_end_label = last_actual_qtr
        st.sidebar.info(f"Unfrozen mode uses all actuals up to: **{train_end_label}**")

    horizon_q = st.sidebar.slider("🔮 Forecast horizon (quarters)", 3, 12, 6)

    show_intervals = st.sidebar.checkbox("📊 Show Bootstrap Bands (P10–P50–P90)", value=False)
    n_sims = st.sidebar.slider("🎲 Bootstrap simulations", 100, 600, 300) if show_intervals else 300
    p_lo   = st.sidebar.slider("📉 Lower percentile", 5, 25, 10) if show_intervals else 10
    p_hi   = st.sidebar.slider("📈 Upper percentile", 75, 95, 90) if show_intervals else 90

    run = st.sidebar.button("▶ Run CSAF Prediction")
    if not run:
        st.info("Choose options in the sidebar, then click **▶ Run CSAF Prediction**.")
        st.stop()

    # Build training slice
    hist = df[df["Schools"] == selected_school].copy()
    hist["sort_key"] = hist["Fiscal Year"].apply(sort_fy)
    hist = hist.sort_values("sort_key")
    if train_end_label is not None:
        cut_key = sort_fy(train_end_label)
        hist    = hist[hist["sort_key"].apply(lambda k: k <= cut_key)]

    y_raw      = pd.to_numeric(hist[selected_metric], errors="coerce").values.astype(float)
    labels_raw = hist["Fiscal Year"].astype(str).tolist()
    valid      = ~np.isnan(y_raw)
    y_train    = y_raw[valid]
    lbl_train  = [labels_raw[i] for i in range(len(labels_raw)) if valid[i]]

    if len(y_train) < 3:
        st.warning("⚠️ Not enough data points for a reliable forecast (need ≥ 3).")
        st.stop()

    with st.spinner("Running seasonal ensemble forecast…"):
        predictions, future_labels, method_desc = forecast_csaf_per_school(
            y_train, lbl_train, horizon_q, selected_metric
        )

    # Full actual series for chart (not just training slice)
    actual_full = df[df["Schools"] == selected_school].copy()
    actual_full["sort_key"] = actual_full["Fiscal Year"].apply(sort_fy)
    actual_full = actual_full.sort_values("sort_key")
    actual_vals = pd.to_numeric(actual_full[selected_metric], errors="coerce")

    actual_part = pd.DataFrame({
        "Period": actual_full["Fiscal Year"].astype(str),
        "Value":  actual_vals,
        "Type":   "Actual",
    }).dropna(subset=["Value"])

    pred_label = "Forecast (Frozen)" if forecast_mode == "Freeze at selected quarter" else "Forecast (Unfrozen)"
    pred_part  = pd.DataFrame({
        "Period": future_labels,
        "Value":  predictions,
        "Type":   pred_label,
    })

    combined          = pd.concat([actual_part, pred_part], ignore_index=True)
    combined["Label"] = combined["Value"].apply(lambda v: fmt_csaf(selected_metric, v))
    combined["sort_key"] = combined["Period"].apply(sort_fy)
    all_periods_sorted   = sorted(combined["Period"].unique(), key=sort_fy)

    fig = px.bar(
        combined, x="Period", y="Value", color="Type", barmode="group",
        text="Label", color_discrete_map=TYPE_COLOR_CSAF_PRED,
        title=f"{selected_school} — {selected_metric}",
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside",
                      cliponaxis=False, textfont=dict(size=16))
    fig.update_layout(uniformtext_mode="show", uniformtext_minsize=10,
                      bargap=0.12, bargroupgap=0.06)
    fig.update_xaxes(categoryorder="array", categoryarray=all_periods_sorted, tickangle=30)
    fig = add_best_practice_csaf(fig, selected_metric)

    if show_intervals:
        with st.spinner("Computing bootstrap bands…"):
            p10, p50, p90 = csaf_bootstrap_bands(
                y_train, lbl_train, horizon_q, selected_metric,
                n_sims=n_sims, p_lo=p_lo, p_hi=p_hi,
            )
        fig.add_trace(go.Scatter(
            x=future_labels + future_labels[::-1],
            y=list(p90) + list(p10[::-1]),
            fill="toself", mode="lines", line=dict(width=0),
            name=f"Band P{p_lo}–P{p_hi}", fillcolor="rgba(231,87,89,0.18)", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=future_labels, y=p50, mode="lines+markers",
            name="P50 (median)", line=dict(width=2, color="#E15759", dash="dot"),
        ))

    fig = apply_plot_style(fig, height=720)
    fig.update_layout(
        title=dict(text=f"{selected_school} — {selected_metric}", x=0.01, y=0.985),
        legend=dict(title="Type", orientation="h", yanchor="bottom", y=1.25, xanchor="left", x=0.01),
        margin=dict(t=220, r=40, b=90, l=60),
    )
    fig.update_traces(cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    st.markdown("#### 📋 Forecast Values")
    st.dataframe(pd.DataFrame({
        "Quarter": future_labels,
        selected_metric: [fmt_csaf(selected_metric, v) for v in predictions],
    }), use_container_width=True, hide_index=True)

    # Model detail expander
    with st.expander("ℹ️ Forecast Engine Details"):
        st.markdown(f"""
**Method:** {method_desc}

**Why this engine?**

CSAF metrics follow a predictable within-year sawtooth:
- FB Ratio and Days COH: Q1 is typically **2–3× higher** than Q2, and Q2 is higher than Q3
- Liabilities to Assets and Current Ratio: relatively flat within year, small seasonal dip/peak

Prior ML models (HGBR, MLP, Ridge) treated Q1→Q2 as a consecutive drop, misreading seasonality as trend.  
The new engine **splits each metric into per-quarter sub-series** so Q1s are always projected against Q1s.

| Component | Weight | Logic |
|---|---|---|
| Model A: Per-Quarter YoY Drift | 60% | Median of last-2 YoY growth rates within same quarter sub-series, capped at ±30-40% |
| Model B: Q1-Anchored Ratio | 40% | Log-linear trend on Q1 sub-series; Q2/Q3 derived via school-specific or global seasonal ratios |

**Global seasonal ratios (calibrated from all 29 schools, P5-P95 winsorised):**

| Metric | Q2/Q1 ratio | Q3/Q1 ratio |
|---|---|---|
| FB Ratio | 0.464 | 0.318 |
| Unrestricted Days COH | 0.445 | 0.316 |
| Current Ratio | 0.943 | 1.000 |
| Liabilities to Assets | 1.014 | 0.939 |

School-specific ratios override global defaults when ≥ 2 paired observations exist.
""")

# ============================================================
# 3) BUDGET/ENROLLMENT (ACTUAL)
# ============================================================
elif metric_group == "Budget/Enrollment (Bar)":
    st.markdown("## 📊 Budget/Enrollment (Actuals)")
    if df_budget_long.empty:
        st.warning("⚠️ Enrollment dataset not loaded."); st.stop()

    selected_school  = st.sidebar.selectbox("🏫 Select School:", school_options_budget)
    metrics_all      = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    available        = sorted(df_budget_long["Metric"].dropna().unique())
    metrics_all      = [m for m in metrics_all if m in available]
    selected_metrics = st.sidebar.multiselect("📌 Select Metric(s):", metrics_all,
                                               default=["October 1 Count", "February 1 Count"])
    selected_fy      = st.sidebar.multiselect("📅 Select Fiscal Years:", fy_options_budget,
        default=[fy for fy in fy_options_budget if START_FY <= sort_fy_only(fy) <= END_ACTUAL_FY] or fy_options_budget)

    d = df_budget_long[
        (df_budget_long["Schools"] == selected_school) &
        (df_budget_long["Metric"].isin(selected_metrics)) &
        (df_budget_long["Fiscal Year"].isin(selected_fy))
    ].copy()
    d["ValueNum"] = pd.to_numeric(d["Value"], errors="coerce")
    d = d.dropna(subset=["ValueNum"])
    d["sort_key"] = d["Fiscal Year"].apply(sort_fy_only)
    d = d.sort_values("sort_key")
    if d.empty: st.warning("⚠️ No data for current filters."); st.stop()

    def fmt_enroll(metric, v):
        return f"{v:.0%}" if metric == "Budget to Enrollment Ratio" else f"{v:,.0f}"

    fig = go.Figure()
    for met in selected_metrics:
        sub = d[d["Metric"] == met]
        fig.add_trace(go.Bar(
            x=sub["Fiscal Year"], y=sub["ValueNum"], name=met,
            marker_color=ENROLL_COLORS.get((met, "Actual"), None),
            text=[fmt_enroll(met, v) for v in sub["ValueNum"]], textposition="outside",
        ))
    fig.update_layout(
        title=dict(text=f"{selected_school} — Enrollment (Actuals)", x=0.01, y=0.98),
        barmode="group", bargap=BARGAP, bargroupgap=BARGROUPGAP,
        legend=dict(orientation="h", yanchor="bottom", y=1.18, xanchor="left", x=0.01),
        margin=dict(t=170, r=40, b=80, l=60),
    )
    fig.update_xaxes(categoryorder="array",
                     categoryarray=sorted(d["Fiscal Year"].unique(), key=sort_fy_only), tickangle=0)
    fig = apply_plot_style(fig, height=CHART_H_TALL)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 4) BUDGET/ENROLLMENT PREDICTED
# ============================================================
elif metric_group == "Budget/Enrollment Predicted (Bar)":
    st.markdown("## 🔮 Enrollment Predicted (Stable Forecast)")
    if df_budget_long.empty:
        st.warning("⚠️ Enrollment dataset not loaded."); st.stop()

    selected_school  = st.sidebar.selectbox("🏫 Select School:", school_options_budget)
    metrics_all      = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    available        = sorted(df_budget_long["Metric"].dropna().unique())
    metrics_all      = [m for m in metrics_all if m in available]
    selected_metrics = st.sidebar.multiselect("📌 Select Metric(s) to forecast:", metrics_all,
        default=[m for m in ["October 1 Count", "February 1 Count"] if m in metrics_all] or metrics_all[:2])

    freeze_at = st.sidebar.selectbox("🧊 Freeze at (use actuals up to this FY):", fy_options_budget,
        index=fy_options_budget.index(fy_label(END_ACTUAL_FY))
        if fy_label(END_ACTUAL_FY) in fy_options_budget else len(fy_options_budget) - 1)

    st.sidebar.markdown("### ⚙️ Enrollment realism controls")
    max_oct_growth = st.sidebar.slider("Max Oct YoY growth",    0.00, 0.12, 0.03, 0.01)
    max_oct_drop   = st.sidebar.slider("Max Oct YoY drop",      0.00, 0.12, 0.02, 0.01)
    ret_lo         = st.sidebar.slider("Min Feb/Oct retention", 0.70, 1.00, 0.92, 0.01)
    ret_hi         = st.sidebar.slider("Max Feb/Oct retention", 0.80, 1.10, 1.02, 0.01)
    show_intervals  = st.sidebar.checkbox("📊 Show simple bands (±1.5% on counts)", value=False)
    show_model_table= st.sidebar.checkbox("Show model info table", value=True)

    run = st.sidebar.button("▶ Run Enrollment Prediction")
    if not run:
        st.info("Choose options in the sidebar, then click **▶ Run Budget/Enrollment Prediction**.")
        st.stop()

    def _get_series(metric_name, up_to_fy_label):
        dh = df_budget_long[(df_budget_long["Schools"] == selected_school) &
                             (df_budget_long["Metric"] == metric_name)].copy()
        dh["FY"] = dh["Fiscal Year"].astype(str)
        dh["sort_key"] = dh["FY"].apply(sort_fy_only)
        dh = dh[dh["sort_key"] <= sort_fy_only(up_to_fy_label)].sort_values("sort_key")
        dh["ValueNum"] = pd.to_numeric(dh["Value"], errors="coerce")
        return dh.dropna(subset=["ValueNum"])[["FY", "sort_key", "ValueNum"]].copy()

    def _robust_pct_change(y):
        y = np.asarray(y, dtype=float)
        if len(y) < 3: return 0.0
        prev = y[:-1]; nxt = y[1:]
        mask = (prev > 0) & np.isfinite(prev) & np.isfinite(nxt)
        if mask.sum() < 2: return 0.0
        return float(np.median((nxt[mask] - prev[mask]) / prev[mask]))

    def _forecast_oct_history(oct_hist_vals, horizon, max_g, max_d):
        y = np.asarray(oct_hist_vals, dtype=float); last = float(y[-1])
        g = float(np.clip(_robust_pct_change(y), -max_d, max_g))
        out = []; cur = last
        for _ in range(horizon): cur = float(cur * (1.0 + g)); out.append(cur)
        return np.asarray(out, dtype=float), g

    def _estimate_retention_ratio(freeze_fy_label):
        oct_df = _get_series("October 1 Count", freeze_fy_label)
        feb_df = _get_series("February 1 Count", freeze_fy_label)
        if oct_df.empty or feb_df.empty: return None
        merged = pd.merge(oct_df[["FY", "sort_key", "ValueNum"]],
                          feb_df[["FY", "ValueNum"]], on="FY", how="inner",
                          suffixes=("_Oct", "_Feb")).dropna()
        merged = merged[(merged["ValueNum_Oct"] > 0) & (merged["ValueNum_Feb"] > 0)]
        if merged.empty: return None
        merged = merged.sort_values("sort_key").tail(3)
        ratio = float(np.median(merged["ValueNum_Feb"].values / merged["ValueNum_Oct"].values))
        return float(np.clip(ratio, ret_lo, ret_hi))

    def fmt_val(met_name, v):
        return f"{v:.0%}" if met_name == "Budget to Enrollment Ratio" else f"{v:,.0f}"

    origin_year  = sort_fy_only(freeze_at)
    future_years = [fy_label(y) for y in range(origin_year + 1, END_FORECAST_FY + 1)]
    horizon_y    = len(future_years)
    if horizon_y <= 0: st.warning("⚠️ Freeze year is already at/after forecast end."); st.stop()

    combined_frames = []; model_info_rows = []
    need_oct = "October 1 Count" in selected_metrics
    need_feb = "February 1 Count" in selected_metrics
    oct_hist_df = _get_series("October 1 Count", freeze_at) if need_oct else pd.DataFrame()
    feb_hist_df = _get_series("February 1 Count", freeze_at) if need_feb else pd.DataFrame()
    retention   = _estimate_retention_ratio(freeze_at) if need_feb else None

    oct_future = None; oct_growth_used = None
    if need_oct:
        if len(oct_hist_df) < 3:
            st.warning("⚠️ Not enough October history (need ≥ 3).")
        else:
            oct_vals = oct_hist_df["ValueNum"].values.astype(float)
            oct_future, oct_growth_used = _forecast_oct_history(oct_vals, horizon_y, max_oct_growth, max_oct_drop)
            model_info_rows.append({"Metric": "October 1 Count", "Method": "Conservative bounded YoY",
                "YoY % used": f"{oct_growth_used*100:.2f}%",
                "Notes": f"Clipped to [-{max_oct_drop*100:.0f}%, +{max_oct_growth*100:.0f}%]"})
            combined_frames.append(pd.DataFrame({"FY": oct_hist_df["FY"], "ValueNum": oct_hist_df["ValueNum"],
                "Metric": "October 1 Count", "Type": "Actual"}))
            combined_frames.append(pd.DataFrame({"FY": future_years, "ValueNum": oct_future,
                "Metric": "October 1 Count", "Type": "Forecast (Frozen)"}))

    if need_feb:
        if retention is None:
            st.warning("⚠️ Could not compute Feb/Oct retention. Feb forecast disabled.")
        else:
            model_info_rows.append({"Metric": "February 1 Count", "Method": "Derived from October",
                "YoY % used": "", "Notes": f"Feb = Oct × {retention:.3f} (clipped {ret_lo:.2f}–{ret_hi:.2f})"})
            if not feb_hist_df.empty:
                combined_frames.append(pd.DataFrame({"FY": feb_hist_df["FY"], "ValueNum": feb_hist_df["ValueNum"],
                    "Metric": "February 1 Count", "Type": "Actual"}))
            freeze_fy_str  = str(freeze_at)
            has_feb_freeze = (not feb_hist_df.empty) and (freeze_fy_str in feb_hist_df["FY"].values)
            if (not has_feb_freeze) and need_oct and (not oct_hist_df.empty):
                oct_freeze_row = oct_hist_df[oct_hist_df["FY"] == freeze_fy_str]
                if not oct_freeze_row.empty and np.isfinite(oct_freeze_row["ValueNum"].iloc[0]):
                    combined_frames.append(pd.DataFrame([{"FY": freeze_fy_str,
                        "ValueNum": float(oct_freeze_row["ValueNum"].iloc[0] * retention),
                        "Metric": "February 1 Count", "Type": "Forecast (Frozen)"}]))
            if need_oct and oct_future is not None:
                feb_future = (oct_future * retention).astype(float)
                if show_intervals:
                    band = 0.015
                    combined_frames.append(pd.DataFrame({"FY": future_years, "ValueNum": feb_future * (1 - band),
                        "Metric": "February 1 Count (P10)", "Type": "Band"}))
                    combined_frames.append(pd.DataFrame({"FY": future_years, "ValueNum": feb_future * (1 + band),
                        "Metric": "February 1 Count (P90)", "Type": "Band"}))
                combined_frames.append(pd.DataFrame({"FY": future_years, "ValueNum": feb_future,
                    "Metric": "February 1 Count", "Type": "Forecast (Frozen)"}))

    for met in selected_metrics:
        if met in ["October 1 Count", "February 1 Count"]: continue
        is_ratio = met == "Budget to Enrollment Ratio"
        dh = _get_series(met, freeze_at)
        if len(dh) < 3: st.warning(f"⚠️ Not enough history for {met} (need ≥ 3)."); continue
        y_fcast = forecast_enrollment_series(dh["ValueNum"].values.astype(float), horizon_y, is_ratio)
        model_info_rows.append({"Metric": met, "Method": "YoY Drift (capped)", "YoY % used": "",
            "Notes": "Conservative median growth capped ±10%"})
        combined_frames.append(pd.DataFrame({"FY": dh["FY"], "ValueNum": dh["ValueNum"],
            "Metric": met, "Type": "Actual"}))
        combined_frames.append(pd.DataFrame({"FY": future_years, "ValueNum": y_fcast,
            "Metric": met, "Type": "Forecast (Frozen)"}))

    if not combined_frames: st.warning("⚠️ Nothing to chart for current selections."); st.stop()
    combined = pd.concat(combined_frames, ignore_index=True)
    combined["sort_key"] = combined["FY"].apply(sort_fy_only)

    fig = go.Figure()
    for met in selected_metrics:
        for tname in ["Actual", "Forecast (Frozen)"]:
            dt = combined[(combined["Metric"] == met) & (combined["Type"] == tname)].sort_values("sort_key")
            if dt.empty: continue
            fig.add_trace(go.Bar(x=dt["FY"], y=dt["ValueNum"], name=f"{met} — {tname}",
                marker_color=ENROLL_COLORS.get((met, tname), None),
                opacity=0.95 if tname == "Actual" else 0.78,
                text=[fmt_val(met, v) for v in dt["ValueNum"]], textposition="outside"))

    fig.update_layout(barmode="group", bargap=BARGAP, bargroupgap=BARGROUPGAP)
    fig.update_xaxes(categoryorder="array", categoryarray=FY22_TO_FY28, tickangle=0)
    fig = apply_plot_style(fig, height=700)
    fig.update_layout(
        title=dict(text=f"{selected_school} — Enrollment Predicted (Freeze at {freeze_at})", x=0.01, y=0.985),
        legend=dict(orientation="h", yanchor="bottom", y=1.28, xanchor="left", x=0.01),
        margin=dict(t=230, r=40, b=90, l=60), uniformtext_mode="show", uniformtext_minsize=11,
    )
    fig.update_traces(cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)
    if show_model_table and model_info_rows:
        st.markdown("### 🧠 Forecast Method Summary")
        st.dataframe(pd.DataFrame(model_info_rows), use_container_width=True)

# ============================================================
# 5) PREDICTIVE RISK MODEL
# ============================================================
elif metric_group == "Predictive Risk Model":
    st.markdown("## 🤖 Predictive Risk Model — Next-Quarter Financial Risk")
    st.caption("Uses quarter-level lag features from CSAF and balance-sheet metrics to forecast each school's next-quarter risk tier: **Low / Medium / High**.")

    with st.spinner("Building risk features and training model…"):
        df_feat          = _build_risk_dataset(df)
        feat_cols_present = [c for c in RISK_FEATURE_COLS if c in df.columns]
        clf, ml_cols, clf_classes = _train_risk_model(df_feat)

    if clf is None:
        st.error("❌ Not enough historical data to train the risk model."); st.stop()

    latest_rows  = (df_feat.sort_values("_sort_key").groupby("Schools", sort=False).last().reset_index())
    X_latest     = latest_rows[ml_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    pred_labels  = clf.predict(X_latest)
    pred_probas  = clf.predict_proba(X_latest)
    class_order  = list(clf_classes)
    latest_rows["Predicted Next-Q Risk"] = pred_labels
    latest_rows["Confidence"] = [
        float(pred_probas[i, class_order.index(pred_labels[i])]) * 100
        for i in range(len(pred_labels))
    ]

    selected_school_rm = st.sidebar.selectbox("🏫 Select School (detail view):", sorted(latest_rows["Schools"].tolist()))
    school_row         = latest_rows[latest_rows["Schools"] == selected_school_rm].iloc[0]
    school_hist        = df_feat[df_feat["Schools"] == selected_school_rm].sort_values("_sort_key")
    school_latest_hist = school_hist.iloc[-1] if not school_hist.empty else pd.Series(dtype=float)

    pred_risk  = school_row["Predicted Next-Q Risk"]
    confidence = school_row["Confidence"]
    current_fy = school_row["Fiscal Year"]
    risk_color = {"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a"}.get(pred_risk, "#6b7280")
    risk_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(pred_risk, "⚪")

    st.markdown(
        f'<div style="background:{risk_color}18;border:2px solid {risk_color};border-radius:14px;'
        f'padding:18px 22px;margin-bottom:18px;">'
        f'<div style="font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#485569;font-weight:700;">'
        f'{selected_school_rm} — Forecast based on {current_fy}</div>'
        f'<div style="font-size:36px;font-weight:900;color:{risk_color};margin:6px 0 2px 0;">'
        f'{risk_emoji} Next-Quarter Risk: {pred_risk}</div>'
        f'<div style="font-size:14px;color:#374151;">Model confidence: <strong>{confidence:.1f}%</strong></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    def _kv(col): return school_latest_hist.get(col, np.nan)
    fb_v = _kv("FB Ratio"); cr_v = _kv("Current Ratio")
    dch_v = _kv("Unrestricted Days COH"); la_v = _kv("Liabilities to Assets")

    def _metric_tile(col_obj, label, value_str, good):
        color = "#16a34a" if good else "#dc2626"
        col_obj.markdown(
            f'<div style="border:1px solid {color}33;border-radius:10px;padding:12px 14px;background:{color}0d;">'
            f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:#6b7280;font-weight:700;">{label}</div>'
            f'<div style="font-size:26px;font-weight:800;color:{color};">{value_str}</div></div>',
            unsafe_allow_html=True,
        )

    k1, k2, k3, k4 = st.columns(4)
    _metric_tile(k1, "FB Ratio",           f"{fb_v:.1%}"  if np.isfinite(fb_v)  else "—", np.isfinite(fb_v)  and fb_v  >= 0.10)
    _metric_tile(k2, "Current Ratio",      f"{cr_v:.2f}"  if np.isfinite(cr_v)  else "—", np.isfinite(cr_v)  and cr_v  >= 1.50)
    _metric_tile(k3, "Days Cash on Hand",  f"{dch_v:,.0f}" if np.isfinite(dch_v) else "—", np.isfinite(dch_v) and dch_v >= 60)
    _metric_tile(k4, "Liabilities/Assets", f"{la_v:.2f}"  if np.isfinite(la_v)  else "—", np.isfinite(la_v)  and la_v  <= 0.90)
    st.markdown("")

    explanations = _explain_risk(school_latest_hist, feat_cols_present)
    st.markdown("#### 📋 Risk Explanation")
    if pred_risk == "Low":
        st.success("No significant risk factors detected. School appears financially stable heading into next quarter.")
    else:
        for ex in explanations:
            icon = "🔴" if pred_risk == "High" else "🟡"
            st.markdown(f"- {icon} {ex}")

    st.markdown("#### 📊 Current & Change Metrics")
    detail_metrics = [
        "FB Ratio", "Current Ratio", "Liabilities to Assets",
        "Unrestricted Days COH", "Unrestricted Cash & Equivalents",
        "Total Revenue", "Total Expenses",
    ]
    detail_rows = []
    for col in detail_metrics:
        v_curr = school_latest_hist.get(col, np.nan)
        v_chg  = school_latest_hist.get(f"{col}_chg", np.nan)
        v_lag1 = school_latest_hist.get(f"{col}_lag1", np.nan)
        is_dollar = col in ("Unrestricted Cash & Equivalents", "Total Revenue", "Total Expenses")
        is_pct    = col in ("FB Ratio",)
        is_plain  = col in ("Current Ratio", "Liabilities to Assets")
        is_days   = col in ("Unrestricted Days COH",)

        def _fmt_v(v):
            if pd.isna(v): return "—"
            if is_pct:   return f"{v:.1%}"
            if is_plain: return f"{v:.2f}"
            if is_days:  return f"{v:,.0f} days"
            return f"${v:,.0f}"

        def _fmt_chg(v):
            if pd.isna(v): return "—"
            if is_pct:   return f"{v:+.1%}"
            if is_plain: return f"{v:+.2f}"
            if is_days:  return f"{v:+,.0f} days"
            return f"${v:+,.0f}"

        detail_rows.append({
            "Metric": col, "Current Quarter": _fmt_v(v_curr),
            "Prior Quarter": _fmt_v(v_lag1), "Change": _fmt_chg(v_chg),
        })
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
    st.divider()

    st.markdown("#### 🔴 All Schools Forecasted HIGH Risk Next Quarter")
    display_cols  = ["Schools", "Fiscal Year", "Predicted Next-Q Risk", "Confidence",
                     "FB Ratio", "Current Ratio", "Unrestricted Days COH", "Liabilities to Assets"]
    avail_display = [c for c in display_cols if c in latest_rows.columns]
    high_df = (latest_rows[latest_rows["Predicted Next-Q Risk"] == "High"][avail_display].copy()
               .sort_values("Confidence", ascending=False).reset_index(drop=True))
    if "FB Ratio" in high_df.columns:
        high_df["FB Ratio"] = high_df["FB Ratio"].apply(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    for col in ["Current Ratio", "Liabilities to Assets"]:
        if col in high_df.columns:
            high_df[col] = high_df[col].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    if "Unrestricted Days COH" in high_df.columns:
        high_df["Unrestricted Days COH"] = high_df["Unrestricted Days COH"].apply(
            lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
    high_df["Confidence"] = high_df["Confidence"].apply(lambda v: f"{v:.1f}%")
    high_df.rename(columns={"Fiscal Year": "Latest Actual Quarter"}, inplace=True)
    if high_df.empty:
        st.success("✅ No schools are currently forecasted as High Risk next quarter.")
    else:
        st.dataframe(high_df, use_container_width=True, hide_index=True)

    st.markdown("#### 🏫 All Schools — Next-Quarter Risk Ranking")
    risk_order = {"High": 0, "Medium": 1, "Low": 2}
    rank_df = (
        latest_rows[["Schools", "Fiscal Year", "Predicted Next-Q Risk", "Confidence"]].copy()
        .assign(_rank_sort=latest_rows["Predicted Next-Q Risk"].map(risk_order))
        .sort_values(["_rank_sort", "Confidence"], ascending=[True, False])
        .drop(columns="_rank_sort").reset_index(drop=True)
    )
    rank_df["Confidence"] = rank_df["Confidence"].apply(lambda v: f"{v:.1f}%")
    rank_df.rename(columns={"Fiscal Year": "Latest Actual Quarter"}, inplace=True)
    rank_df.index = rank_df.index + 1
    st.dataframe(rank_df, use_container_width=True)

    st.markdown("#### 🧠 Top Feature Importances (Model Drivers)")
    feat_imp_df = (
        pd.DataFrame({"Feature": ml_cols, "Importance": clf.feature_importances_})
        .sort_values("Importance", ascending=False).head(20).reset_index(drop=True)
    )
    feat_imp_df["Importance"] = feat_imp_df["Importance"].apply(lambda v: f"{v:.4f}")
    st.dataframe(feat_imp_df, use_container_width=True, hide_index=True)

    csv_bytes = rank_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Risk Predictions (CSV)", data=csv_bytes,
                       file_name="nola_next_quarter_risk_predictions.csv", mime="text/csv")

# ============================================================
# 6) OTHER METRICS
# ============================================================
else:
    st.markdown("## 📌 Other Metrics (Actuals)")
    selected_school  = st.sidebar.selectbox("🏫 Select School:", school_options)
    selected_fy      = st.sidebar.multiselect("📅 Select Fiscal Year + Quarter:", fiscal_options, default=fiscal_options)
    other_metrics    = sorted([m for m in df_long["Metric"].dropna().unique() if m not in csaf_metrics])
    DEFAULT_METRIC   = "Current Assets"
    default_metrics  = [DEFAULT_METRIC] if DEFAULT_METRIC in other_metrics else ([other_metrics[0]] if other_metrics else [])
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
    if filtered.empty: st.warning("⚠️ No data for current filters."); st.stop()

    filtered["FY Group"] = filtered["Fiscal Year"].astype(str).str.split().str[0]
    filtered["Label"]    = filtered["ValueNum"].apply(lambda v: f"${v:,.0f}")

    if len(selected_metrics) == 1:        metric_title = selected_metrics[0]
    elif len(selected_metrics) <= 4:      metric_title = " | ".join(selected_metrics)
    else:                                 metric_title = f"{len(selected_metrics)} Metrics Selected"

    n_metrics  = max(1, len(selected_metrics))
    rows       = math.ceil(n_metrics / 4)
    fig_height = 360 * rows + 320

    fig = px.bar(
        filtered, x="Fiscal Year", y="ValueNum", color="FY Group",
        color_discrete_map=fy_color_map, barmode="group",
        facet_col="Metric", facet_col_wrap=4,
        facet_col_spacing=0.06, facet_row_spacing=0.12,
        text="Label", title=f"{selected_school} — {metric_title}",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_traces(texttemplate="%{text}", textposition="outside",
                      textfont=dict(size=13), cliponaxis=False, width=0.42)
    fig.update_layout(uniformtext_mode="show", uniformtext_minsize=11)
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    fig.update_layout(
        bargap=0.12, bargroupgap=0.05,
        title=dict(x=0.01, y=0.985),
        legend=dict(title="FY Group", orientation="v", yanchor="top", y=0.90,
                    xanchor="left", x=1.12, tracegroupgap=10),
        margin=dict(r=340, t=140, b=90),
    )
    fig.update_xaxes(tickangle=30)
    fig = apply_plot_style(fig, height=fig_height)
    st.plotly_chart(fig, use_container_width=True)
