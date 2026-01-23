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
    """
    For labels like: 'FY25 Q1', 'FY26 Q2' ...
    Returns (year, quarter).
    """
    try:
        s = str(x).strip().upper().replace("\u00A0", " ")
        m = re.search(r"FY\s*(\d{2,4})", s)
        y = int(m.group(1)) % 100 if m else 999
        qm = re.search(r"Q\s*(\d)", s)
        q = int(qm.group(1)) if qm else 9
        return (y, q)
    except:
        return (999, 9)

def sort_fy_only(x):
    """
    Robust FY extractor for labels like:
    'FY22', 'FY 22', 'FY2022', 'FY22 Q1', 'FY22 (Budget)'
    Returns 22, 23, ... ; returns 999 if not found.
    """
    try:
        s = str(x)
        s = s.replace("\u00A0", " ").replace("\n", " ").replace("\r", " ")
        s = s.strip().upper()
        m = re.search(r"FY\s*(\d{2,4})", s)
        if not m:
            return 999
        n = int(m.group(1))
        return n % 100
    except:
        return 999

def standardize_fy_label(x):
    """
    Forces FY label to the format FY22, FY23, ...
    """
    y = sort_fy_only(x)
    if y == 999:
        return str(x).strip()
    return f"FY{y:02d}"

def normalize_col(c):
    return re.sub(r"\s+", " ", str(c).strip()).lower()

def clean_series(y):
    return pd.to_numeric(pd.Series(y), errors="coerce").values.astype(float)

# =========================
# LOAD FY25 (CSAF + OTHER)
# =========================
fy25_path = "FY25.xlsx"
try:
    df = pd.read_excel(fy25_path, sheet_name="FY25", header=0, engine="openpyxl")
except Exception as e:
    st.error(f"❌ Could not load {fy25_path}: {e}")
    st.write("📁 Files in app folder:", sorted(os.listdir(".")))
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
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}. Put it in the repo root (same folder as this app)."
        )

    xl = pd.ExcelFile(file_path, engine="openpyxl")
    sheet_names = xl.sheet_names

    # Pick sheet (exact or best match)
    if preferred_sheet in sheet_names:
        sheet = preferred_sheet
    else:
        candidates = []
        for s in sheet_names:
            sn = s.lower()
            score = 0
            if "enroll" in sn: score += 2
            if "student" in sn: score += 2
            if "fy26" in sn or "26" in sn: score += 1
            candidates.append((score, s))
        candidates.sort(reverse=True)
        sheet = candidates[0][1] if candidates else sheet_names[0]

    # Try headers
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
        # header=None fallback then detect row
        tmp = pd.read_excel(file_path, sheet_name=sheet, header=None, engine="openpyxl")
        best_row = None
        for i in range(min(40, len(tmp))):
            row = tmp.iloc[i].astype(str).str.lower().tolist()
            if any("school" in v for v in row) and any(("fiscal" in v and "year" in v) or v.strip() in {"fy","fiscal year"} for v in row):
                best_row = i
                break
        if best_row is None:
            raise ValueError(
                f"Could not detect header row in sheet '{sheet}'. Sheets found: {sheet_names}"
            )
        df_raw = pd.read_excel(file_path, sheet_name=sheet, header=best_row, engine="openpyxl")
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        used_header = best_row

    # Rename flexibly
    rename_map = {}
    for c in df_raw.columns:
        cn = normalize_col(c)

        if cn in {"school", "schools", "campus", "site"} or "school" in cn:
            rename_map[c] = "Schools"

        if cn in {"fy", "fiscal year", "fiscal_year"} or ("fiscal" in cn and "year" in cn):
            rename_map[c] = "Fiscal Year"

        # Budgeted (avoid matching ratio)
        if ("budget" in cn) and ("ratio" not in cn) and ("%" not in cn) and ("enroll" not in cn):
            rename_map[c] = "Budgetted"

        if ("oct" in cn) and (("count" in cn) or ("enroll" in cn) or ("enrollment" in cn)):
            rename_map[c] = "October 1 Count"

        if ("feb" in cn or "february" in cn) and (("count" in cn) or ("enroll" in cn) or ("enrollment" in cn)):
            rename_map[c] = "February 1 Count"

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
        raise ValueError(
            f"Loaded sheet '{sheet}' (header={used_header}) but missing required columns: {missing}. "
            f"Columns detected: {list(df_raw.columns)}"
        )

    df_raw = df_raw.dropna(subset=["Schools", "Fiscal Year"]).copy()
    df_raw["Schools"] = df_raw["Schools"].astype(str).str.strip()
    df_raw["FY"] = df_raw["Fiscal Year"].apply(standardize_fy_label)

    metric_cols = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    for c in metric_cols:
        if c in df_raw.columns:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

    # Avoid ugly vertical drops if file uses 0 for missing
    for c in metric_cols:
        if c in df_raw.columns:
            df_raw.loc[df_raw[c] <= 0, c] = np.nan

    use_cols = [c for c in metric_cols if c in df_raw.columns]
    df_long_budget = df_raw.melt(
        id_vars=["Schools", "FY"],
        value_vars=use_cols,
        var_name="Metric",
        value_name="Value"
    ).dropna(subset=["Value"]).copy()

    fiscal_opts = sorted(df_long_budget["FY"].dropna().unique(), key=sort_fy_only)
    school_opts = sorted(df_long_budget["Schools"].dropna().unique())

    debug = {
        "file": file_path,
        "sheet_used": sheet,
        "header_used": used_header,
        "columns_after_rename": list(df_raw.columns),
        "rows_long": int(len(df_long_budget)),
        "fiscal_options_budget": fiscal_opts
    }

    return df_long_budget, fiscal_opts, school_opts, debug

# ---- Load Enrollment FY26
fy26_path = "Enrollment FY26.xlsx"
df_budget_long = pd.DataFrame()
fiscal_options_budget, school_options_budget = [], []
budget_debug = None

try:
    df_budget_long, fiscal_options_budget, school_options_budget, budget_debug = load_budget_enrollment_excel(
        fy26_path,
        preferred_sheet="FY26 Student enrollment"
    )
except Exception as e:
    st.error(f"❌ Enrollment FY26 failed to load: {e}")
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

with st.expander("✅ Enrollment FY26 Loader Debug (open if something looks wrong)", expanded=False):
    st.json(budget_debug)

st.sidebar.header("🔎 Filters")

modes = ["CSAF Metrics", "CSAF Predicted", "Other Metrics"]
if not df_budget_long.empty:
    modes += ["Budget to Enrollment", "Budget to Enrollment Predicted"]

metric_group = st.sidebar.radio("Choose Dashboard:", modes)
viz_type = st.sidebar.selectbox("📈 Visualization Type:", ["Bar Chart", "Line Chart"])

# budget line style
smooth_lines = st.sidebar.checkbox("✨ Smooth lines (Budget charts)", value=True)
show_line_labels = st.sidebar.checkbox("🔤 Show point labels on Line Chart", value=False)

# =========================
# CSAF PREDICTED
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
    viz_type_local = st.sidebar.selectbox("📈 CSAF Prediction Chart Type:", ["Line Chart", "Bar Chart"])
    run_pred = st.sidebar.button("▶ Run Prediction")

    fyq_re = re.compile(r"FY\s*(\d{2})\s*Q\s*(\d)")

    def parse_fyq(label: str):
        m = fyq_re.search(str(label))
        if not m:
            return None, None
        return int(m.group(1)), int(m.group(2))

    def make_future_labels(last_label: str, n: int, quarters_per_year=3):
        fy, q = parse_fyq(last_label)
        if fy is None:
            fy, q = 26, 0
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
    q_hist = quarter_index(fy_labels_hist)

    valid_mask = ~np.isnan(y_hist) & ~np.isnan(q_hist)
    y_hist = y_hist[valid_mask]
    fy_labels_hist = [fy_labels_hist[i] for i, m in enumerate(valid_mask) if m]
    q_hist = q_hist[valid_mask]

    if len(y_hist) < 4:
        st.warning("⚠️ Not enough historical points to produce a reliable forecast (need ≥ 4).")
        st.stop()

    def forecast_seasonal_naive(y, q, n_future):
        result = []
        for i in range(n_future):
            idx = len(y) - 3 + i - (i // 3) * 3
            if idx < 0:
                result.append(y[-1])
            else:
                result.append(y[max(0, idx)])
        return np.array(result, dtype=float)

    def forecast_robust_seasonal(y, q, n_future):
        t = np.arange(len(y)).reshape(-1, 1)
        Qd = seasonal_groups(q)
        X = np.hstack([t, Qd])

        y_log = np.log1p(np.clip(y, 0, None))
        model = HuberRegressor().fit(X, y_log)

        t_future = np.arange(len(y), len(y) + n_future).reshape(-1, 1)
        q_future = ((q[-1] + np.arange(1, n_future + 1) - 1) % 3) + 1
        Qd_future = seasonal_groups(q_future)
        Xf = np.hstack([t_future, Qd_future])

        y_pred_log = model.predict(Xf)
        y_pred = np.expm1(y_pred_log)
        return np.clip(y_pred, 0, None)

    def forecast_trend_times_seasonal(y, q, n_future):
        season_means = {s: np.nanmean(y[q == s]) for s in [1, 2, 3]}
        global_mean = np.nanmean(y)

        for s in [1, 2, 3]:
            if not np.isfinite(season_means.get(s, np.nan)):
                season_means[s] = global_mean

        core = y / np.array([season_means[int(s)] for s in q])
        core = np.where(np.isfinite(core), core, global_mean)

        t = np.arange(len(core)).reshape(-1, 1)
        model = LinearRegression().fit(t, np.log1p(np.clip(core, 0, None)))

        t_future = np.arange(len(core), len(core) + n_future).reshape(-1, 1)
        q_future = ((q[-1] + np.arange(1, n_future + 1) - 1) % 3) + 1
        core_pred = np.expm1(model.predict(t_future))

        y_pred = core_pred * np.array([season_means[int(s)] for s in q_future])
        return np.clip(y_pred, 0, None)

    def forecast_ensemble(y, q, n_future):
        a = forecast_seasonal_naive(y, q, n_future)
        b = forecast_robust_seasonal(y, q, n_future)
        c = forecast_trend_times_seasonal(y, q, n_future)
        return (0.2 * a) + (0.4 * b) + (0.4 * c)

    if forecast_method.startswith("Ensemble"):
        y_future = forecast_ensemble(y_hist, q_hist, n_future)
    elif forecast_method.startswith("Seasonal Naive"):
        y_future = forecast_seasonal_naive(y_hist, q_hist, n_future)
    elif forecast_method.startswith("Robust Seasonal"):
        y_future = forecast_robust_seasonal(y_hist, q_hist, n_future)
    else:
        y_future = forecast_trend_times_seasonal(y_hist, q_hist, n_future)

    future_labels = make_future_labels(train_through, n_future, quarters_per_year=3)

    actual_now = df[df["Schools"] == selected_school].copy()
    actual_now["sort_key"] = actual_now["Fiscal Year"].apply(sort_fy)
    actual_now = actual_now.sort_values("sort_key")

    actual_series = pd.DataFrame({
        "Quarter": actual_now["Fiscal Year"].astype(str),
        "Value": pd.to_numeric(actual_now[selected_metric], errors="coerce"),
        "Type": "Actual"
    }).dropna(subset=["Value"])

    forecast_key = f"CSAF__{selected_school}__{selected_metric}__{train_through}__{forecast_method}__{n_future}"
    st.session_state.setdefault("forecast_store", {})

    if forecast_key not in st.session_state["forecast_store"]:
        frozen_pred = pd.DataFrame({"Quarter": future_labels, "Value": y_future, "Type": "Forecast (Frozen)"})
        st.session_state["forecast_store"][forecast_key] = frozen_pred.copy()
    else:
        frozen_pred = st.session_state["forecast_store"][forecast_key].copy()

    combined = pd.concat([actual_series[["Quarter", "Value", "Type"]], frozen_pred[["Quarter", "Value", "Type"]]], ignore_index=True)

    metric_label, formula_txt, threshold, best_label = csaf_formulas[selected_metric]

    if viz_type_local == "Line Chart":
        fig = px.line(
            combined, x="Quarter", y="Value", color="Type",
            color_discrete_map={"Actual": "blue", "Forecast (Frozen)": "red"},
            markers=True,
            title=f"{selected_school} — {selected_metric} (Actual vs Frozen Forecast)"
        )
        fig.update_traces(text=None, connectgaps=False)
    else:
        fig = px.bar(
            combined, x="Quarter", y="Value", color="Type",
            color_discrete_map={"Actual": "blue", "Forecast (Frozen)": "red"},
            barmode="group", text="Value",
            title=f"{selected_school} — {selected_metric} (Actual vs Frozen Forecast)"
        )
        if selected_metric == "FB Ratio":
            fig.update_traces(texttemplate="%{y:.1%}", textposition="outside")
        elif selected_metric in ("Liabilities to Assets", "Current Ratio"):
            fig.update_traces(texttemplate="%{y:.2f}", textposition="outside")
        else:
            fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside")

    if "≥" in best_label:
        fig.add_hline(y=threshold, line_dash="dot", line_color="green",
                      annotation_text=f"Best Practice {best_label}", annotation_position="top left")
    else:
        fig.add_hline(y=threshold, line_dash="dot", line_color="red",
                      annotation_text=f"Best Practice {best_label}", annotation_position="top left")

    fig.update_xaxes(tickangle=45)

    if selected_metric == "FB Ratio":
        fig.update_layout(yaxis_tickformat=".1%")
    elif selected_metric in ("Liabilities to Assets", "Current Ratio"):
        fig.update_layout(yaxis_tickformat=",.2f")
    else:
        fig.update_layout(yaxis_tickformat=",.0f")

    try:
        x0 = frozen_pred["Quarter"].iloc[0]
        x1 = frozen_pred["Quarter"].iloc[-1]
        fig.add_vrect(x0=x0, x1=x1, fillcolor="orange", opacity=0.08, line_width=0)
    except:
        pass

    fig.update_layout(height=560, legend_title="Series")
    st.plotly_chart(fig, use_container_width=True)

    def fmt(v):
        if selected_metric == "FB Ratio":
            return f"{v:.1%}"
        elif selected_metric in ("Liabilities to Assets", "Current Ratio"):
            return f"{v:.2f}"
        else:
            return f"{v:,.0f}"

    pred_table = frozen_pred.copy()
    pred_table["Forecast (Frozen)"] = pred_table["Value"].map(fmt)
    pred_table = pred_table[["Quarter", "Forecast (Frozen)"]]

    st.markdown("### 📋 Forecast (Frozen) Values")
    st.dataframe(pred_table, use_container_width=True)
    st.caption(f"**Metric Formula:** {formula_txt}")
    st.caption(f"**Method:** {forecast_method}")
    st.caption(f"**Forecast Origin (Frozen at):** {train_through}")

# =========================
# BUDGET TO ENROLLMENT (COMPARISON)
# =========================
elif metric_group == "Budget to Enrollment":
    st.markdown("## 📈 Budget to Enrollment (FY)")

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
            df_f, x="FY", y="Value",
            color="Metric",
            color_discrete_map=budget_metric_color_map,
            markers=True,
            facet_col="Schools", facet_col_wrap=2,
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
            df_f, x="FY", y="Value",
            color="Metric",
            color_discrete_map=budget_metric_color_map,
            barmode="group", text="Value",
            facet_col="Schools", facet_col_wrap=2,
            title=title
        )

        for tr in fig.data:
            name = tr.name
            if name == "Budget to Enrollment Ratio":
                subset = pd.to_numeric(df_f[df_f["Metric"] == name]["Value"], errors="coerce")
                if not subset.empty and subset.max() <= 1.2:
                    tr.texttemplate = "%{text:.0%}"
                else:
                    tr.texttemplate = "%{text:,.2f}"
            elif name in {"Budgetted", "October 1 Count", "February 1 Count"}:
                tr.texttemplate = "%{text:,.0f}"
            else:
                tr.texttemplate = "%{text}"
        fig.update_traces(textposition="outside")

    fig.update_xaxes(categoryorder="array", categoryarray=fy_order, tickangle=45)
    fig.update_layout(height=700, legend_title="Metric", title_x=0.5, bargap=0.15, bargroupgap=0.05)
    st.plotly_chart(fig, use_container_width=True)

    def fmt_budget(row):
        m, v = row["Metric"], row["Value"]
        try:
            v = float(v)
            if m == "Budget to Enrollment Ratio":
                return f"{v:.0%}" if v <= 1.2 else f"{v:,.2f}"
            if m in {"Budgetted", "October 1 Count", "February 1 Count"}:
                return f"{v:,.0f}"
            return v
        except:
            return v

    df_show = df_f.copy()
    df_show["Formatted Value"] = df_show.apply(fmt_budget, axis=1)
    df_show = df_show[["Schools", "FY", "Metric", "Formatted Value"]]
    st.markdown("### 📋 Budget to Enrollment Data (By School)")
    st.dataframe(df_show, use_container_width=True)

# =========================
# BUDGET TO ENROLLMENT PREDICTED (MULTI-METRIC + FROZEN)
# =========================
elif metric_group == "Budget to Enrollment Predicted":
    st.markdown("## 🔮 Budget to Enrollment Predicted (Frozen Forecast)")

    schools_b = sorted(df_budget_long["Schools"].dropna().unique())
    selected_school_b = st.sidebar.selectbox("🏫 Select School:", schools_b, index=0 if schools_b else None)

    metrics_b_all = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    metrics_b_all = [m for m in metrics_b_all if m in df_budget_long["Metric"].unique()]

    selected_metrics_b = st.sidebar.multiselect("📊 Choose Metric(s):", metrics_b_all, default=metrics_b_all)

    fiscal_years_b = sorted(df_budget_long["FY"].dropna().astype(str).unique(), key=sort_fy_only)
    selected_fy_hist_b = st.sidebar.multiselect("📅 History Fiscal Years (training):", fiscal_years_b, default=fiscal_years_b)

    train_through_b = st.sidebar.selectbox("🧊 Forecast Origin (freeze at):", fiscal_years_b, index=max(0, len(fiscal_years_b) - 1))

    forecast_method_b = st.sidebar.selectbox(
        "🧠 Forecast Method",
        [
            "Ensemble (Robust Trend + Linear Trend)",
            "Robust Trend (Huber log1p)",
            "Linear Trend (log1p)",
        ],
        index=0,
    )

    n_future_b = st.sidebar.slider("🔮 Forecast horizon (years)", 1, 6, 3)
    viz_type_b = st.sidebar.selectbox("📈 Budget Prediction Chart Type:", ["Line Chart", "Bar Chart"])
    run_pred_b = st.sidebar.button("▶ Run Budget Prediction")

    if not selected_metrics_b:
        st.warning("⚠️ Select at least one metric.")
        st.stop()

    if not run_pred_b:
        st.info("Use the sidebar to pick School, Metrics, History Years, Forecast Origin, Method, then click **Run Budget Prediction**.")
        st.stop()

    def forecast_budget_metric(values, n_future):
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
            return np.clip(0.5 * pred_huber() + 0.5 * pred_linear(), 0, None)
        elif forecast_method_b.startswith("Robust"):
            return np.clip(pred_huber(), 0, None)
        else:
            return np.clip(pred_linear(), 0, None)

    origin_year = sort_fy_only(train_through_b)
    future_labels = [f"FY{origin_year + i:02d}" for i in range(1, n_future_b + 1)]

    # Actual series (keeps updating from file)
    actual_now = df_budget_long[
        (df_budget_long["Schools"] == selected_school_b) &
        (df_budget_long["Metric"].isin(selected_metrics_b))
    ].copy()
    actual_now["sort_key"] = actual_now["FY"].apply(sort_fy_only)
    actual_now = actual_now.sort_values("sort_key")
    actual_now = actual_now.rename(columns={"FY": "Period"})
    actual_now["Type"] = "Actual"

    st.session_state.setdefault("forecast_store_budget", {})
    frozen_frames = []

    for met in selected_metrics_b:
        dfh = df_budget_long[
            (df_budget_long["Schools"] == selected_school_b) &
            (df_budget_long["FY"].isin(selected_fy_hist_b)) &
            (df_budget_long["Metric"] == met)
        ].copy()

        dfh["sort_key"] = dfh["FY"].apply(sort_fy_only)
        dfh = dfh[dfh["sort_key"] <= origin_year].sort_values("sort_key")
        dfh = dfh.dropna(subset=["Value"])

        if dfh.empty or dfh["Value"].notna().sum() < 3:
            continue

        key = f"BUDGET__{selected_school_b}__{met}__{train_through_b}__{forecast_method_b}__{n_future_b}"

        if key not in st.session_state["forecast_store_budget"]:
            y_future = forecast_budget_metric(dfh["Value"].values, n_future_b)
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
        st.warning("⚠️ Not enough data to forecast the selected metric(s). Need at least 3 points per metric.")
        st.stop()

    frozen_all = pd.concat(frozen_frames, ignore_index=True)

    combined = pd.concat(
        [
            actual_now[["Period", "Value", "Metric", "Type"]],
            frozen_all[["Period", "Value", "Metric", "Type"]],
        ],
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

    # Shade forecast region
    try:
        x0 = frozen_all["Period"].iloc[0]
        x1 = frozen_all["Period"].iloc[-1]
        fig.add_vrect(x0=x0, x1=x1, fillcolor="orange", opacity=0.08, line_width=0)
    except:
        pass

    fig.update_xaxes(categoryorder="array", categoryarray=period_order, tickangle=45)
    fig.update_layout(height=560, legend_title="Metric")
    st.plotly_chart(fig, use_container_width=True)

    # Table
    def fmt_budget_pred(metric, v):
        try:
            v = float(v)
            if metric == "Budget to Enrollment Ratio":
                return f"{v:.0%}" if v <= 1.2 else f"{v:,.2f}"
            return f"{v:,.0f}"
        except:
            return v

    table = frozen_all.copy()
    table["Forecast (Frozen)"] = table.apply(lambda r: fmt_budget_pred(r["Metric"], r["Value"]), axis=1)
    table = table[["Metric", "Period", "Forecast (Frozen)"]]
    st.markdown("### 📋 Budget Forecast (Frozen) Values")
    st.dataframe(table, use_container_width=True)
    st.caption(f"**Method:** {forecast_method_b}")
    st.caption(f"**Forecast Origin (Frozen at):** {train_through_b}")

# =========================
# CSAF METRICS + OTHER METRICS (FY25 DATASET)
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
    ].copy()

    if filtered.empty:
        st.warning("⚠️ Welcome To Finance Accountability Real-Time Dashboard. Try Adjusting your Left filters.")
        st.stop()

    filtered["sort_key"] = filtered["Fiscal Year"].apply(sort_fy)
    filtered = filtered.sort_values("sort_key")
    filtered["FY Group"] = filtered["Fiscal Year"].astype(str).str.split().str[0]

    if len(selected_metrics) == 1 and selected_metrics[0] in csaf_descriptions:
        desc = csaf_descriptions[selected_metrics[0]]["desc"]
        st.markdown(f"**{selected_metrics[0]}**<br><span style='font-size:13px'>{desc}</span>", unsafe_allow_html=True)
        chart_title = ""
    else:
        chart_title = f"{', '.join(selected_metrics)} across Fiscal Years"

    if len(selected_schools) > 8:
        if viz_type == "Bar Chart":
            fig = px.bar(filtered, x="Fiscal Year", y="Value", color="Schools", barmode="group", text="Value", title=chart_title)
        else:
            fig = px.line(filtered, x="Fiscal Year", y="Value", color="Schools", markers=True, title=chart_title)
            fig.update_traces(text=None, connectgaps=False)
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
            fig.update_traces(text=None, connectgaps=False)

    fiscal_order = filtered["Fiscal Year"].unique().tolist()
    fig.update_xaxes(categoryorder="array", categoryarray=fiscal_order, tickangle=45)

    if viz_type == "Bar Chart":
        fig.update_traces(marker_line_width=0, width=0.6, textposition="outside")
        fig.update_layout(bargap=0.1, bargroupgap=0.05)

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

    def format_value(val, metric):
        try:
            val = float(val)
            if metric in dollar_metrics:
                return f"${val:,.0f}"
            if metric == "FB Ratio":
                return f"{val:.0%}"
            if metric in ["Liabilities to Assets", "Current Ratio"]:
                return f"{val:.2f}"
            return f"{val:,.0f}"
        except:
            return val

    df_display = filtered.copy()
    df_display["Value"] = df_display.apply(lambda row: format_value(row["Value"], row["Metric"]), axis=1)
    st.markdown("### 📑 Data Table")
    st.dataframe(df_display, use_container_width=True)
