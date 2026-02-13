"""
NOLA SCHOOLS FINANCIAL TRACKER - WORLD-CLASS EDITION
Executive Financial Intelligence Platform

Built by Emmanuel Igbokwe
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
import re

from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, RandomForestRegressor

# ============================================================
# THEME / BACKGROUND - Enhanced Professional Look
# ============================================================
APP_BG = "#dfe7df"
PLOT_BG = "#dfe7df"
GRID_CLR = "rgba(0,0,0,0.10)"

# Executive colors
EXEC_EXCELLENT = "#00C853"  # Green
EXEC_GOOD = "#2196F3"       # Blue
EXEC_WARNING = "#FF9800"    # Orange
EXEC_DANGER = "#F44336"     # Red

st.set_page_config(
    page_title="NOLA Financial Intelligence",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {APP_BG}; }}
      section[data-testid="stSidebar"] {{ background-color: {APP_BG}; }}

      /* Enhanced header */
      header[data-testid="stHeader"] {{ background: {APP_BG}; }}
      .block-container {{
        padding-top: 1.15rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
      }}

      /* Sidebar styling */
      section[data-testid="stSidebar"] * {{ 
        font-size: 13px !important;
        color: #1f1f1f !important;
      }}
      
      section[data-testid="stSidebar"] .stSelectbox,
      section[data-testid="stSidebar"] .stMultiSelect,
      section[data-testid="stSidebar"] .stRadio,
      section[data-testid="stSidebar"] .stCheckbox,
      section[data-testid="stSidebar"] .stSlider {{
        margin-bottom: 0.45rem !important;
      }}
      
      /* Executive Cards */
      .exec-card {{
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 4px solid #2196F3;
      }}
      
      .exec-card-excellent {{ border-left-color: #00C853; }}
      .exec-card-good {{ border-left-color: #2196F3; }}
      .exec-card-warning {{ border-left-color: #FF9800; }}
      .exec-card-danger {{ border-left-color: #F44336; }}
      
      .health-score {{
        font-size: 48px;
        font-weight: 700;
        text-align: center;
        margin: 10px 0;
      }}
      
      .health-score-excellent {{ color: #00C853; }}
      .health-score-good {{ color: #2196F3; }}
      .health-score-warning {{ color: #FF9800; }}
      .health-score-danger {{ color: #F44336; }}
      
      .school-name {{
        font-size: 18px;
        font-weight: 600;
        color: #1f1f1f;
        margin-bottom: 5px;
      }}
      
      .health-label {{
        font-size: 14px;
        font-weight: 500;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 1px;
      }}
      
      /* Metrics badge */
      .metric-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin: 2px;
      }}
      
      .badge-excellent {{ background: #E8F5E9; color: #2E7D32; }}
      .badge-good {{ background: #E3F2FD; color: #1565C0; }}
      .badge-warning {{ background: #FFF3E0; color: #E65100; }}
      .badge-danger {{ background: #FFEBEE; color: #C62828; }}
      
      /* Insight boxes */
      .insight-box {{
        background: #F5F5F5;
        border-left: 3px solid #2196F3;
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
      }}
      
      .insight-positive {{ border-left-color: #00C853; }}
      .insight-warning {{ border-left-color: #FF9800; }}
      .insight-critical {{ border-left-color: #F44336; }}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================
logo_path = "nola_parish_logo.png"
if os.path.exists(logo_path):
    import base64
    with open(logo_path, "rb") as f:
        encoded_logo = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
          .nola-wrap {{
            background: {APP_BG};
            border-bottom: 2px solid rgba(0,0,0,0.10);
            margin-bottom: 12px;
          }}
          .nola-header {{
            display:flex;
            align-items:center;
            gap:14px;
            padding:16px 8px;
          }}
          .nola-title {{
            color:#003366;
            font-size:28px;
            font-weight:900;
            line-height:1.1;
          }}
          .nola-sub {{
            color:#1f1f1f;
            font-size:15px;
            margin-top:4px;
            font-weight: 500;
          }}
          .spin {{
            width:74px; height:74px;
            border-radius:50%;
            animation: spin 8s linear infinite;
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
              <div class="nola-title">NOLA Schools Financial Intelligence Platform</div>
              <div class="nola-sub">Executive Financial Analysis & Predictive Analytics • Built by Emmanuel Igbokwe</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown("### NOLA Schools Financial Intelligence Platform")
    st.caption("Executive Financial Analysis • Built by Emmanuel Igbokwe")

st.divider()

# ============================================================
# CONSTANTS
# ============================================================
BASE_FONT_SIZE = 18
AXIS_FONT = 16
TEXT_FONT = 18

CHART_H = 760
CHART_H_TALL = 860

BARGAP = 0.08
BARGROUPGAP = 0.04

START_FY = 22
END_ACTUAL_FY = 26
END_FORECAST_FY = 28

fy_color_map = {
    "FY22": "#2E6B3C",
    "FY23": "#E15759",
    "FY24": "#1F77B4",
    "FY25": "#7B61FF",
    "FY26": "#FF4FA3",
}

TYPE_COLOR_CSAF_PRED = {
    "Actual": "#1F77B4",
    "Forecast (Frozen)": "#E15759",
    "Forecast (Unfrozen)": "#FF4FA3"
}

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

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
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
# FINANCIAL HEALTH ANALYZER
# ============================================================
class FinancialHealthAnalyzer:
    """Calculate financial health scores and generate insights"""
    
    CSAF_THRESHOLDS = {
        'FB Ratio': {'excellent': 0.20, 'good': 0.15, 'adequate': 0.10, 'poor': 0.05},
        'Current Ratio': {'excellent': 2.50, 'good': 2.00, 'adequate': 1.50, 'poor': 1.00},
        'Liabilities to Assets': {'excellent': 0.50, 'good': 0.70, 'adequate': 0.90, 'poor': 1.00},
        'Unrestricted Days COH': {'excellent': 120, 'good': 90, 'adequate': 60, 'poor': 30}
    }
    
    @staticmethod
    def calculate_health_score(school_data):
        """Calculate 0-100 health score"""
        scores = []
        weights = {
            'FB Ratio': 0.30,
            'Current Ratio': 0.25,
            'Liabilities to Assets': 0.25,
            'Unrestricted Days COH': 0.20
        }
        
        for metric, weight in weights.items():
            value = school_data.get(metric)
            if pd.isna(value):
                continue
                
            thresh = FinancialHealthAnalyzer.CSAF_THRESHOLDS[metric]
            
            if metric == 'Liabilities to Assets':
                # Lower is better
                if value <= thresh['excellent']:
                    score = 100
                elif value <= thresh['good']:
                    score = 85
                elif value <= thresh['adequate']:
                    score = 70
                elif value <= thresh['poor']:
                    score = 50
                else:
                    score = max(0, 50 - (value - thresh['poor']) * 50)
            else:
                # Higher is better
                if value >= thresh['excellent']:
                    score = 100
                elif value >= thresh['good']:
                    score = 85
                elif value >= thresh['adequate']:
                    score = 70
                elif value >= thresh['poor']:
                    score = 50
                else:
                    score = max(0, 50 * (value / thresh['poor']))
            
            scores.append(score * weight)
        
        return sum(scores) if scores else 0
    
    @staticmethod
    def get_health_rating(score):
        """Get rating and CSS class from score"""
        if score >= 85:
            return "Excellent", "excellent"
        elif score >= 70:
            return "Good", "good"
        elif score >= 55:
            return "Adequate", "warning"
        else:
            return "At Risk", "danger"
    
    @staticmethod
    def generate_insights(school_data, school_name):
        """Generate automated insights"""
        insights = []
        
        # Fund Balance
        fb_ratio = school_data.get('FB Ratio', 0)
        if fb_ratio >= 0.20:
            insights.append({
                'type': 'positive',
                'icon': '✅',
                'title': 'Strong Fund Balance',
                'message': f"FB Ratio of {fb_ratio:.1%} significantly exceeds 10% threshold"
            })
        elif fb_ratio < 0.10:
            insights.append({
                'type': 'critical',
                'icon': '🚨',
                'title': 'Low Fund Balance',
                'message': f"FB Ratio of {fb_ratio:.1%} below required 10% - build reserves urgently"
            })
        
        # Liquidity
        current_ratio = school_data.get('Current Ratio', 0)
        if current_ratio >= 2.50:
            insights.append({
                'type': 'positive',
                'icon': '💧',
                'title': 'Excellent Liquidity',
                'message': f"Current Ratio of {current_ratio:.2f} indicates strong short-term health"
            })
        elif current_ratio < 1.50:
            insights.append({
                'type': 'warning',
                'icon': '⚠️',
                'title': 'Liquidity Concerns',
                'message': f"Current Ratio of {current_ratio:.2f} below 1.50 - monitor cash flow"
            })
        
        # Leverage
        liab_assets = school_data.get('Liabilities to Assets', 0)
        if liab_assets > 0.90:
            insights.append({
                'type': 'warning',
                'icon': '📊',
                'title': 'High Leverage',
                'message': f"Liabilities at {liab_assets:.1%} of assets - consider debt reduction"
            })
        
        # Cash on Hand
        days_coh = school_data.get('Unrestricted Days COH', 0)
        if days_coh >= 120:
            insights.append({
                'type': 'positive',
                'icon': '💰',
                'title': 'Strong Cash Reserves',
                'message': f"{days_coh:.0f} days cash provides excellent flexibility"
            })
        elif days_coh < 60:
            insights.append({
                'type': 'critical',
                'icon': '🚨',
                'title': 'Cash Flow Risk',
                'message': f"Only {days_coh:.0f} days cash - immediate action needed"
            })
        
        return insights

# ============================================================
# CSAF CONFIGURATION
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
@st.cache_data
def load_financial_data():
    """Load financial data with caching"""
    try:
        df = pd.read_excel("FY25.xlsx", sheet_name="FY25", header=0)
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["Schools", "Fiscal Year"])
        df["Fiscal Year"] = df["Fiscal Year"].astype(str).str.strip().apply(std_fyq_label)
        return df
    except Exception as e:
        st.error(f"❌ Could not load FY25.xlsx: {e}")
        return None

df = load_financial_data()
if df is None:
    st.stop()

fiscal_options = sorted(df["Fiscal Year"].dropna().unique(), key=sort_fy)
school_options = sorted(df["Schools"].dropna().unique())

# Long form for Other Metrics
value_vars = [c for c in df.columns if c not in ["Schools", "Fiscal Year"]]
df_long = df.melt(id_vars=["Schools", "Fiscal Year"], value_vars=value_vars, var_name="Metric", value_name="Value")

# ============================================================
# LOAD DATA (BUDGET / ENROLLMENT)
# ============================================================
@st.cache_data
def load_enrollment_data():
    """Load enrollment data with caching"""
    df_budget_long = pd.DataFrame()
    school_options_budget, fy_options_budget = [], []
    
    try:
        df_budget_raw = None
        for hdr in [0, 1, 2, 3, 4]:
            tmp = pd.read_excel("Enrollment FY26.xlsx", sheet_name="FY26 Student enrollment", header=hdr)
            tmp.columns = [str(c).strip() for c in tmp.columns]
            cols_norm = [re.sub(r"\s+", " ", str(c).strip()).lower() for c in tmp.columns]
            if any("school" in c for c in cols_norm) and any(("fiscal" in c and "year" in c) or c in ["fy", "fiscal year"] for c in cols_norm):
                df_budget_raw = tmp
                break

        if df_budget_raw is None:
            df_budget_raw = pd.read_excel("Enrollment FY26.xlsx", sheet_name="FY26 Student enrollment", header=1)
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
        if not missing:
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
        pass  # Silent fail - enrollment data is optional
    
    return df_budget_long, school_options_budget, fy_options_budget

df_budget_long, school_options_budget, fy_options_budget = load_enrollment_data()

# ============================================================
# FORECAST ENGINE (Keep existing implementation)
# ============================================================
[... INCLUDE ALL THE FORECAST FUNCTIONS FROM ORIGINAL CODE ...]

def _safe_log1p(y):
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0, None)
    return np.log1p(y)

def _safe_expm1(ylog):
    ylog = np.asarray(ylog, dtype=float)
    return np.clip(np.expm1(ylog), 0, None)

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

def _iterative_forecast_supervised(model, y_hist, q_hist, horizon, n_lags=None, use_time_poly=True, use_season=True, log_target=True, season_period=3, is_ratio=False):
    y_hist = np.asarray(y_hist, dtype=float)
    y_hist = np.clip(y_hist, 0, None)
    last = float(y_hist[-1])
    if n_lags is None:
        n_lags = min(6, max(3, len(y_hist) // 3))
    Xtr, ytr = _make_lag_features(y_hist, q_hist, n_lags=n_lags, use_time_poly=use_time_poly, use_season=use_season, season_period=season_period)
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
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0, None)
    if is_ratio:
        y = np.clip(y, 0.0, 1.5)
    if q is None:
        season_period = 1
    
    def seasonal_naive(y_hist, q_hist, h):
        y_hist = np.asarray(y_hist, dtype=float)
        preds = []
        sp = season_period
        for _ in range(h):
            idx = len(y_hist) - sp
            if idx >= 0:
                preds.append(float(y_hist[idx]))
                y_hist = np.append(y_hist, y_hist[idx])
            else:
                preds.append(float(y_hist[-1]))
                y_hist = np.append(y_hist, y_hist[-1])
        return np.array(preds, dtype=float)

    def seasonal_naive_drift(y_hist, q_hist, h):
        y_hist = np.asarray(y_hist, dtype=float)
        sp = season_period
        base = []
        y_tmp = y_hist.copy()
        for _ in range(h):
            idx = len(y_tmp) - sp
            val = float(y_tmp[idx]) if idx >= 0 else float(y_tmp[-1])
            base.append(val)
            y_tmp = np.append(y_tmp, val)
        base = np.asarray(base, dtype=float)
        if len(y_hist) > sp + 1:
            diffs = y_hist[sp:] - y_hist[:-sp]
            recent = diffs[-min(len(diffs), 6):]
            med = np.nanmedian(recent)
            drift = float(med) if np.isfinite(med) else 0.0
        else:
            drift = 0.0
        damp = np.linspace(1.0, 0.6, h)
        out = base + drift * damp
        out = np.clip(out, 0, None)
        if is_ratio:
            out = np.clip(out, 0.0, 1.5)
        return out

    def robust_seasonal_regression(y_hist, q_hist, h):
        ylog = _safe_log1p(y_hist)
        t = np.arange(len(ylog)).reshape(-1, 1).astype(float)
        X = t if q_hist is None else np.hstack([t, _quarter_dummies(q_hist, season_period=season_period)])
        mdl = HuberRegressor()
        mdl.fit(X, ylog)
        tf = np.arange(len(ylog), len(ylog) + h).reshape(-1, 1).astype(float)
        if q_hist is None:
            Xf = tf
        else:
            q_last = int(q_hist[-1])
            qf = []
            for _ in range(h):
                q_last = (q_last % season_period) + 1
                qf.append(q_last)
            qf = np.asarray(qf, dtype=int)
            Xf = np.hstack([tf, _quarter_dummies(qf, season_period=season_period)])
        return _safe_expm1(mdl.predict(Xf))

    def trend_times_seasonal_index(y_hist, q_hist, h):
        y_hist = np.asarray(y_hist, dtype=float)
        t = np.arange(len(y_hist)).reshape(-1, 1).astype(float)
        if q_hist is None:
            mdl = LinearRegression()
            mdl.fit(t, _safe_log1p(y_hist))
            tf = np.arange(len(y_hist), len(y_hist) + h).reshape(-1, 1).astype(float)
            return _safe_expm1(mdl.predict(tf))
        idx_map = _seasonal_index(y_hist, q_hist)
        deseason = np.array([y_hist[i] / max(idx_map[int(q_hist[i])], 1e-6) for i in range(len(y_hist))], dtype=float)
        mdl = LinearRegression()
        mdl.fit(t, _safe_log1p(deseason))
        tf = np.arange(len(deseason), len(deseason) + h).reshape(-1, 1).astype(float)
        base = _safe_expm1(mdl.predict(tf))
        q_last = int(q_hist[-1])
        preds = []
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
        return _iterative_forecast_supervised(mdl, y_hist, q_hist, h, n_lags=None, use_time_poly=True, use_season=(q_hist is not None), log_target=True, season_period=season_period, is_ratio=is_ratio)

    def neural_mlp_lag(y_hist, q_hist, h):
        mdl = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(48, 24), activation="relu", solver="adam", max_iter=7000, random_state=42))
        return _iterative_forecast_supervised(mdl, y_hist, q_hist, h, n_lags=None, use_time_poly=True, use_season=(q_hist is not None), log_target=True, season_period=season_period, is_ratio=is_ratio)

    def ridge_lag(y_hist, q_hist, h):
        mdl = Ridge(alpha=1.0)
        return _iterative_forecast_supervised(mdl, y_hist, q_hist, h, n_lags=None, use_time_poly=True, use_season=(q_hist is not None), log_target=True, season_period=season_period, is_ratio=is_ratio)
    
    # NEW: Add advanced models
    def gradient_boost_lag(y_hist, q_hist, h):
        mdl = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42)
        return _iterative_forecast_supervised(mdl, y_hist, q_hist, h, n_lags=None, use_time_poly=True, use_season=(q_hist is not None), log_target=True, season_period=season_period, is_ratio=is_ratio)
    
    def random_forest_lag(y_hist, q_hist, h):
        mdl = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        return _iterative_forecast_supervised(mdl, y_hist, q_hist, h, n_lags=None, use_time_poly=True, use_season=(q_hist is not None), log_target=True, season_period=season_period, is_ratio=is_ratio)

    models = {
        "Ensemble (Seasonal Naive + Drift + Robust Seasonal + Trend×Seasonality)": ensemble_3,
        "Seasonal Naive + Drift (recommended baseline)": seasonal_naive_drift,
        "Seasonal Naive (same quarter last year)": seasonal_naive,
        "Robust Seasonal Regression (Huber + quarter dummies, log1p)": robust_seasonal_regression,
        "Trend × Seasonal Index (linear trend on de-seasonalized)": trend_times_seasonal_index,
        "HGBR Lag + Trend + Season (recommended)": hgbr_lag,
        "Neural MLP Lag + Season": neural_mlp_lag,
        "Ridge Lag + Season": ridge_lag,
        "Gradient Boosting Lag + Season (NEW)": gradient_boost_lag,
        "Random Forest Lag + Season (NEW)": random_forest_lag,
    }

    def score_model(name):
        fn = models[name]
        def fit_pred(tr_idx, te_idx):
            y_tr = y[tr_idx]
            q_tr = None if q is None else np.asarray(q, dtype=int)[tr_idx]
            pred = fn(y_tr, q_tr, 1)
            pred = np.asarray(pred, dtype=float)
            if is_ratio:
                pred = np.clip(pred, 0.0, 1.5)
            return pred
        return _tscv_mae(fit_pred, y, min_train=5, splits=3)

    scores = {}
    if model_choice == "Auto (min error)":
        for name in models:
            try:
                scores[name] = score_model(name)
            except Exception:
                scores[name] = np.inf
        chosen = min(scores, key=scores.get)
    else:
        chosen = model_choice
        for name in models:
            try:
                scores[name] = score_model(name)
            except Exception:
                scores[name] = np.inf

    y_future = models[chosen](y, None if q is None else np.asarray(q, dtype=int), horizon)
    y_future = np.clip(y_future, 0, None)
    if is_ratio:
        y_future = np.clip(y_future, 0.0, 1.5)
    
    if is_ratio:
        max_up, max_down = 1.15, 0.85
    else:
        max_up, max_down = 1.30, 0.70
    last = float(y[-1])
    hist_max = float(np.nanmax(y)) if len(y) else 1.0
    cap = max(hist_max * 1.6, last * 1.5, 1.0)
    y_future = _guard_growth(y_future, last, max_up=max_up, max_down=max_down, lower=0.0, upper=cap)
    if is_ratio:
        y_future = np.clip(y_future, 0.0, 1.5)
    return y_future, chosen, scores, models[chosen]

def bootstrap_intervals(y_hist, q_hist, horizon, model_fn, season_period=3, n_sims=300, p_lo=10, p_hi=90, is_ratio=False, seed=42):
    rng = np.random.default_rng(seed)
    y_hist = np.asarray(y_hist, dtype=float)
    y_hist = np.clip(y_hist, 0, None)
    if is_ratio:
        y_hist = np.clip(y_hist, 0.0, 1.5)
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
        if is_ratio:
            sim = np.clip(sim, 0.0, 1.5)
        sims[s, :] = sim
    p10 = np.percentile(sims, p_lo, axis=0)
    p50 = np.percentile(sims, 50, axis=0)
    p90 = np.percentile(sims, p_hi, axis=0)
    return p10, p50, p90

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.header("🔎 Navigation")

modes = [
    "📊 Executive Summary",  # NEW - First option!
    "CSAF Metrics (4-panel)",
    "CSAF Predicted",
    "Other Metrics"
]

if not df_budget_long.empty:
    modes += ["Budget/Enrollment (Bar)", "Budget/Enrollment Predicted (Bar)"]

metric_group = st.sidebar.radio("Choose Dashboard:", modes)

fyq_re = re.compile(r"FY\s*(\d{2,4})\s*Q\s*(\d)", re.IGNORECASE)

def parse_q(label: str):
    m = fyq_re.search(str(label))
    return int(m.group(2)) if m else None

# ============================================================
# NEW: EXECUTIVE SUMMARY DASHBOARD
# ============================================================
if metric_group == "📊 Executive Summary":
    st.markdown("## 📊 Executive Financial Summary - All Schools")
    st.markdown("*Comprehensive financial health analysis across all NOLA charter schools*")
    
    # Filter controls
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_quarter = st.selectbox(
            "📅 Select Quarter for Analysis:",
            fiscal_options,
            index=len(fiscal_options)-1  # Latest quarter
        )
    with col2:
        min_score = st.slider(
            "🎯 Filter by Min Health Score:",
            0, 100, 0,
            help="Show only schools with health score above this threshold"
        )
    
    # Get data for selected quarter
    quarter_data = df[df["Fiscal Year"] == selected_quarter].copy()
    
    if quarter_data.empty:
        st.warning("No data available for selected quarter")
        st.stop()
    
    # Calculate health scores for all schools
    analyzer = FinancialHealthAnalyzer()
    school_scores = []
    
    for _, row in quarter_data.iterrows():
        school_name = row["Schools"]
        health_score = analyzer.calculate_health_score(row)
        health_rating, health_class = analyzer.get_health_rating(health_score)
        
        school_scores.append({
            'School': school_name,
            'Health Score': health_score,
            'Rating': health_rating,
            'Class': health_class,
            'FB Ratio': row.get('FB Ratio', 0),
            'Current Ratio': row.get('Current Ratio', 0),
            'Liabilities to Assets': row.get('Liabilities to Assets', 0),
            'Days COH': row.get('Unrestricted Days COH', 0),
            'Total Revenue': row.get('Total Revenue', 0),
            'Total Expenses': row.get('Total Expenses', 0),
            'Total Assets': row.get('Total Assets', 0)
        })
    
    # Create DataFrame and sort by health score
    schools_df = pd.DataFrame(school_scores)
    schools_df = schools_df[schools_df['Health Score'] >= min_score].sort_values('Health Score', ascending=False)
    
    # Summary Statistics
    st.markdown("### 📈 Portfolio Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Schools",
            len(schools_df),
            delta=f"{len(schools_df[schools_df['Rating']=='Excellent'])} Excellent"
        )
    
    with col2:
        avg_score = schools_df['Health Score'].mean()
        st.metric(
            "Average Health Score",
            f"{avg_score:.0f}/100",
            delta=f"{schools_df['Rating'].value_counts().get('Excellent', 0)} at top tier"
        )
    
    with col3:
        at_risk = len(schools_df[schools_df['Rating'] == 'At Risk'])
        st.metric(
            "Schools At Risk",
            at_risk,
            delta="Needs Attention" if at_risk > 0 else "All Healthy",
            delta_color="inverse"
        )
    
    with col4:
        total_assets = schools_df['Total Assets'].sum()
        st.metric(
            "Total Assets",
            f"${total_assets/1e6:.1f}M",
            delta=f"{len(schools_df)} schools"
        )
    
    st.divider()
    
    # School Cards Grid
    st.markdown("### 🏫 School Financial Health Cards")
    
    # Create 3-column layout
    num_cols = 3
    cols = st.columns(num_cols)
    
    for idx, (_, school) in enumerate(schools_df.iterrows()):
        col_idx = idx % num_cols
        
        with cols[col_idx]:
            # Health score card
            card_class = f"exec-card-{school['Class']}"
            score_class = f"health-score-{school['Class']}"
            
            st.markdown(f"""
            <div class="exec-card {card_class}">
                <div class="school-name">{school['School']}</div>
                <div class="{score_class} health-score">{school['Health Score']:.0f}</div>
                <div class="health-label">{school['Rating']}</div>
                <hr style="margin: 12px 0; border: none; border-top: 1px solid #eee;">
                <div style="font-size: 12px; color: #666;">
                    <div style="margin: 4px 0;">
                        <span class="metric-badge badge-{('excellent' if school['FB Ratio'] >= 0.15 else 'warning' if school['FB Ratio'] >= 0.10 else 'danger')}">
                            FB: {school['FB Ratio']:.1%}
                        </span>
                        <span class="metric-badge badge-{('excellent' if school['Current Ratio'] >= 2.0 else 'warning' if school['Current Ratio'] >= 1.5 else 'danger')}">
                            CR: {school['Current Ratio']:.2f}
                        </span>
                    </div>
                    <div style="margin: 4px 0;">
                        <span class="metric-badge badge-{('excellent' if school['Liabilities to Assets'] <= 0.7 else 'warning' if school['Liabilities to Assets'] <= 0.9 else 'danger')}">
                            L/A: {school['Liabilities to Assets']:.1%}
                        </span>
                        <span class="metric-badge badge-{('excellent' if school['Days COH'] >= 90 else 'warning' if school['Days COH'] >= 60 else 'danger')}">
                            COH: {school['Days COH']:.0f}d
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Comparative Analysis Charts
    st.markdown("### 📊 Comparative Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Health Scores", "CSAF Metrics", "Financial Overview"])
    
    with tab1:
        # Health Score Distribution
        fig = go.Figure()
        
        colors = schools_df['Class'].map({
            'excellent': EXEC_EXCELLENT,
            'good': EXEC_GOOD,
            'warning': EXEC_WARNING,
            'danger': EXEC_DANGER
        })
        
        fig.add_trace(go.Bar(
            x=schools_df['School'],
            y=schools_df['Health Score'],
            marker_color=colors,
            text=schools_df['Health Score'].apply(lambda x: f"{x:.0f}"),
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Financial Health Scores - All Schools",
            xaxis_title="",
            yaxis_title="Health Score (0-100)",
            height=500,
            showlegend=False,
            paper_bgcolor=PLOT_BG,
            plot_bgcolor=PLOT_BG
        )
        
        fig.update_xaxes(tickangle=-45)
        fig.add_hline(y=85, line_dash="dash", line_color="green", annotation_text="Excellent (85+)")
        fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Good (70+)")
        fig.add_hline(y=55, line_dash="dash", line_color="red", annotation_text="At Risk (<55)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # CSAF Metrics Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=schools_df['School'],
                y=schools_df['FB Ratio'],
                name='FB Ratio',
                marker_color='#2196F3'
            ))
            fig1.add_hline(y=0.10, line_dash="dash", line_color="red", annotation_text="10% Threshold")
            fig1.update_layout(
                title="Fund Balance Ratio",
                xaxis_title="",
                yaxis_title="FB Ratio",
                height=400,
                paper_bgcolor=PLOT_BG,
                plot_bgcolor=PLOT_BG
            )
            fig1.update_xaxes(tickangle=-45)
            fig1.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=schools_df['School'],
                y=schools_df['Current Ratio'],
                name='Current Ratio',
                marker_color='#00C853'
            ))
            fig2.add_hline(y=1.50, line_dash="dash", line_color="red", annotation_text="1.50 Threshold")
            fig2.update_layout(
                title="Current Ratio",
                xaxis_title="",
                yaxis_title="Current Ratio",
                height=400,
                paper_bgcolor=PLOT_BG,
                plot_bgcolor=PLOT_BG
            )
            fig2.update_xaxes(tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        # Financial Overview
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=schools_df['School'],
                y=schools_df['Total Revenue'],
                name='Revenue',
                marker_color='#4CAF50'
            ))
            fig3.add_trace(go.Bar(
                x=schools_df['School'],
                y=schools_df['Total Expenses'],
                name='Expenses',
                marker_color='#F44336'
            ))
            fig3.update_layout(
                title="Revenue vs Expenses",
                xaxis_title="",
                yaxis_title="Amount ($)",
                barmode='group',
                height=400,
                paper_bgcolor=PLOT_BG,
                plot_bgcolor=PLOT_BG
            )
            fig3.update_xaxes(tickangle=-45)
            fig3.update_yaxes(tickprefix="$", separatethousands=True)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(
                x=schools_df['School'],
                y=schools_df['Days COH'],
                marker_color='#FF9800'
            ))
            fig4.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="60 Days Threshold")
            fig4.update_layout(
                title="Days Cash on Hand",
                xaxis_title="",
                yaxis_title="Days",
                height=400,
                paper_bgcolor=PLOT_BG,
                plot_bgcolor=PLOT_BG
            )
            fig4.update_xaxes(tickangle=-45)
            st.plotly_chart(fig4, use_container_width=True)
    
    st.divider()
    
    # Detailed School Selector for Insights
    st.markdown("### 🔍 Detailed School Analysis")
    selected_school_detail = st.selectbox(
        "Select a school for detailed insights:",
        schools_df['School'].tolist()
    )
    
    # Get selected school data
    school_row = quarter_data[quarter_data["Schools"] == selected_school_detail].iloc[0]
    insights = analyzer.generate_insights(school_row, selected_school_detail)
    
    # Display insights
    if insights:
        st.markdown(f"#### 💡 Financial Insights: {selected_school_detail}")
        for insight in insights:
            icon = insight['icon']
            title = insight['title']
            message = insight['message']
            insight_class = f"insight-{insight['type']}"
            
            st.markdown(f"""
            <div class="insight-box {insight_class}">
                <strong>{icon} {title}</strong><br>
                {message}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No specific insights available for this school")
    
    # Download option
    st.divider()
    st.markdown("### 📥 Export Data")
    
    csv = schools_df.to_csv(index=False)
    st.download_button(
        label="📊 Download Full Report (CSV)",
        data=csv,
        file_name=f"nola_executive_summary_{selected_quarter}.csv",
        mime="text/csv"
    )

# Continue with all existing dashboards...
# [REST OF THE CODE REMAINS THE SAME - CSAF Metrics, CSAF Predicted, Other Metrics, Budget/Enrollment, etc.]
