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
                    style="
                        animation: spin 5s linear infinite;
                        border-radius: 50%;
                    ">
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
        parts = str(x).split()
        year = int(parts[0].replace("FY", "").strip()) if str(parts[0]).upper().startswith("FY") else 999
        q = int(parts[1].replace("Q", "").strip()) if len(parts) > 1 and str(parts[1]).upper().startswith("Q") else 9
        return (year, q)
    except:
        return (999, 9)

def sort_fy_only(x):
    """
    Robust FY extractor for labels like:
    'FY22', 'FY 22', 'FY2022', 'FY22 Q1', 'FY22 (Budget)'
    Returns 22, 23, ... ; returns 999 if not found.
    """
    s = str(x)
    s = s.replace("\u00A0", " ")
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.strip().upper()

    m = re.search(r"FY\s*(\d{2,4})", s)
    if not m:
        return 999
    n = int(m.group(1))
    return n % 100  # FY2022 -> 22

def standardize_fy_label(x):
    """
    Forces FY label to the format FY22, FY23, ...
    Fixes hidden spaces / FY variants.
    """
    y = sort_fy_only(x)
    if y == 999:
        return str(x).strip()
    return f"FY{y:02d}"

def normalize_col(c):
    return re.sub(r"\s+", " ", str(c).strip()).lower()

def clean_series(y):
    return pd.to_numeric(pd.Series(y), errors="coerce").values.astype(float)

def style_bar_fig(fig):
    """Make bars thicker and labels bigger across charts."""
    fig.update_traces(width=0.85, textposition="outside", textfont_size=14)
    fig.update_layout(bargap=0.05, bargroupgap=0.03, uniformtext_minsize=12, uniformtext_mode="show")
    return fig

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
# LOAD BUDGET TO ENROLLMENT (Enrollment FY26.xlsx)
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
        st.error(f"❌ Enrollment FY26 loaded, but missing columns: {missing}")
        st.write("Columns found:", list(df_budget_raw.columns))
        st.stop()

    expected_cols = [
        "Schools", "Fiscal Year",
        "Budgetted", "October 1 Count", "February 1 Count",
        "Budget to Enrollment Ratio"
    ]

    df_budget_raw = df_budget_raw.dropna(subset=["Schools", "Fiscal Year"]).copy()

    # Standardize FY label
    df_budget_raw["Fiscal Year"] = df_budget_raw["Fiscal Year"].apply(standardize_fy_label)

    # Convert numeric columns safely
    for c in ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]:
        if c in df_budget_raw.columns:
            df_budget_raw[c] = pd.to_numeric(df_budget_raw[c], errors="coerce")

    df_budget_long = df_budget_raw.melt(
        id_vars=["Schools", "Fiscal Year"],
        value_vars=[c for c in expected_cols if c in df_budget_raw.columns and c not in ["Schools", "Fiscal Year"]],
        var_name="Metric",
        value_name="Value"
    ).dropna(subset=["Value"]).copy()

    fiscal_options_budget = sorted(df_budget_long["Fiscal Year"].dropna().unique(), key=sort_fy_only)
    school_options_budget = sorted(df_budget_long["Schools"].dropna().unique())

except Exception as e:
    st.error(f"❌ Enrollment FY26 failed to load: {e}")
    st.markdown("### 📁 Files in app folder:")
    st.write(sorted(os.listdir(".")))
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
    modes += ["Budget to Enrollment"]

metric_group = st.sidebar.radio("Choose Dashboard:", modes)

# ✅ REMOVE LINE CHART OPTION (bars only)
viz_type = "Bar Chart"

# =========================
# CSAF PREDICTED (BARS ONLY)
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

    # Best practice line
    if "≥" in best_label:
        fig.add_hline(y=threshold, line_dash="dot", line_color="green",
                      annotation_text=f"Best Practice {best_label}", annotation_position="top left")
    else:
        fig.add_hline(y=threshold, line_dash="dot", line_color="red",
                      annotation_text=f"Best Practice {best_label}", annotation_position="top left")

    # Shade forecast region
    try:
        x0 = frozen_pred["Quarter"].iloc[0]
        x1 = frozen_pred["Quarter"].iloc[-1]
        fig.add_vrect(x0=x0, x1=x1, fillcolor="orange", opacity=0.08, line_width=0)
    except:
        pass

    fig.update_xaxes(tickangle=45)
    style_bar_fig(fig)
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
# BUDGET TO ENROLLMENT (COMPARISON) — BARS ONLY + MULTI METRICS
# =========================
elif metric_group == "Budget to Enrollment":
    if df_budget_long.empty:
        st.warning("⚠️ Enrollment FY26 dataset not loaded.")
        st.stop()

    selected_schools = st.sidebar.multiselect("Select School(s):", school_options_budget)
    selected_fy = st.sidebar.multiselect("Select Fiscal Year(s):", fiscal_options_budget)

    metrics_list = ["Budgetted", "October 1 Count", "February 1 Count", "Budget to Enrollment Ratio"]
    metrics_list = [m for m in metrics_list if m in df_budget_long["Metric"].unique()]
    selected_metrics = st.sidebar.multiselect("Select Metrics:", metrics_list, default=metrics_list)  # ✅ MULTI

    if not selected_schools or not selected_fy or not selected_metrics:
        st.info("Select at least 1 School, 1 Fiscal Year, and 1 Metric.")
        st.stop()

    df_f = df_budget_long[
        (df_budget_long["Schools"].isin(selected_schools)) &
        (df_budget_long["Fiscal Year"].isin(selected_fy)) &
        (df_budget_long["Metric"].isin(selected_metrics))
    ].copy()

    if df_f.empty:
        st.warning("⚠️ No Budget to Enrollment data matches your filters.")
        st.stop()

    df_f["Fiscal Year"] = df_f["Fiscal Year"].astype(str).str.strip()
    df_f["sort_key"] = df_f["Fiscal Year"].apply(sort_fy_only)
    df_f = df_f.sort_values("sort_key")
    fy_order = df_f["Fiscal Year"].unique().tolist()

    title = f"Budget to Enrollment Comparison — {', '.join(selected_metrics)}"

    fig = px.bar(
        df_f, x="Fiscal Year", y="Value",
        color="Metric",
        color_discrete_map=budget_metric_color_map,
        barmode="group",
        text="Value",
        facet_col="Schools", facet_col_wrap=2,
        title=title
    )

    # Label formats per metric
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

    fig.update_xaxes(categoryorder="array", categoryarray=fy_order, tickangle=45)
    style_bar_fig(fig)
    fig.update_layout(height=700, legend_title="Metric", title_x=0.5)
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
    df_show = df_show[["Schools", "Fiscal Year", "Metric", "Formatted Value"]]
    st.markdown("### 📋 Budget to Enrollment Data (By School)")
    st.dataframe(df_show, use_container_width=True)

# =========================
# FY25 (CSAF METRICS + OTHER METRICS) — BARS ONLY
# =========================
else:
    school_options = sorted(df_long["Schools"].dropna().unique())
    selected_schools = st.sidebar.multiselect("Select School(s):", school_options)

    selected_fy = st.sidebar.multiselect("Select Fiscal Year and Quarter:", fiscal_options)

    if metric_group == "CSAF Metrics":
        selected_metrics = st.sidebar.multiselect("Select CSAF Metric(s):", csaf_metrics)
    else:
        selected_metrics = st.sidebar.multiselect("Select Other Metric(s):", other_metrics)

    if not selected_schools or not selected_fy or not selected_metrics:
        st.info("Select at least 1 School, 1 Fiscal Year, and 1 Metric.")
        st.stop()

    filtered = df_long[
        (df_long["Schools"].isin(selected_schools)) &
        (df_long["Fiscal Year"].isin(selected_fy)) &
        (df_long["Metric"].isin(selected_metrics))
    ]

    if filtered.empty:
        st.warning("⚠️ Welcome To Finance Accountability Real-Time Dashboard. Try Adjusting your Left filters.")
        st.stop()

    filtered = filtered.copy()
    filtered["sort_key"] = filtered["Fiscal Year"].apply(sort_fy)
    filtered = filtered.sort_values("sort_key")
    filtered["FY Group"] = filtered["Fiscal Year"].str.split().str[0]

    if len(selected_metrics) == 1 and selected_metrics[0] in csaf_descriptions:
        desc = csaf_descriptions[selected_metrics[0]]["desc"]
        st.markdown(f"**{selected_metrics[0]}**<br><span style='font-size:13px'>{desc}</span>", unsafe_allow_html=True)
        chart_title = ""
    else:
        chart_title = f"{', '.join(selected_metrics)} across Fiscal Years"

    facet_args = {}
    if len(selected_schools) > 1 and len(selected_metrics) > 1:
        facet_args = {"facet_row": "Schools", "facet_col": "Metric", "facet_col_wrap": 2}
    elif len(selected_schools) > 1:
        facet_args = {"facet_col": "Schools", "facet_col_wrap": 2}
    elif len(selected_metrics) > 1:
        facet_args = {"facet_col": "Metric", "facet_col_wrap": 2}

    fig = px.bar(
        filtered, x="Fiscal Year", y="Value",
        color="FY Group", color_discrete_map=fy_color_map,
        barmode="group", text="Value", title=chart_title, **facet_args
    )

    fiscal_order = filtered["Fiscal Year"].unique().tolist()
    fig.update_xaxes(categoryorder="array", categoryarray=fiscal_order, tickangle=45)

    # Formatting
    if selected_metrics and all(m in dollar_metrics for m in selected_metrics):
        fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",")
        fig.update_traces(texttemplate="$%{text:,.0f}")
    elif selected_metrics == ["FB Ratio"]:
        fig.update_traces(texttemplate="%{text:.0%}")
        fig.update_layout(yaxis_tickformat=".0%")
        fig.add_hline(y=csaf_descriptions["FB Ratio"]["threshold"], line_dash="dot", line_color="blue")
    elif selected_metrics == ["Liabilities to Assets"]:
        fig.update_traces(texttemplate="%{text:.2f}")
        fig.add_hline(y=csaf_descriptions["Liabilities to Assets"]["threshold"], line_dash="dot", line_color="blue")
    elif selected_metrics == ["Current Ratio"]:
        fig.update_traces(texttemplate="%{text:.2f}")
        fig.add_hline(y=csaf_descriptions["Current Ratio"]["threshold"], line_dash="dot", line_color="blue")
    elif selected_metrics == ["Unrestricted Days COH"]:
        fig.update_traces(texttemplate="%{text:,.0f}")
        fig.add_hline(y=csaf_descriptions["Unrestricted Days COH"]["threshold"], line_dash="dot", line_color="blue")
    else:
        fig.update_traces(texttemplate="%{text:,.0f}")

    style_bar_fig(fig)
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
