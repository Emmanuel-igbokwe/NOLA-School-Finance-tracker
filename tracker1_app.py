import pandas as pd
import streamlit as st
import plotly.express as px
import os
import numpy as np
import re
from sklearn.linear_model import HuberRegressor, LinearRegression
import base64  # needed for logo encoding

# =========================
# PAGE CONFIG + HEADER SECTION
# =========================
st.set_page_config(page_title="NOLA Financial Tracker", layout="wide")

logo_path = "nola_parish_logo.png"
col1, col2 = st.columns([1, 5])

# --- LEFT COLUMN: Rotating NOLA Logo ---
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

# --- RIGHT COLUMN: Animated Welcome Text ---
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
# Helper functions
# =========================
def sort_fy(x):
    """
    Expected label formats like: 'FY25 Q1', 'FY26 Q2', etc.
    Returns (year, quarter) for sorting.
    """
    try:
        parts = str(x).split()
        year = int(parts[0][2:]) if parts[0].startswith("FY") else 999
        q = int(parts[1][1:]) if len(parts) > 1 and parts[1].startswith("Q") else 9
        return (year, q)
    except:
        return (999, 9)

def load_excel_safe(path_candidates, **read_kwargs):
    for p in path_candidates:
        try:
            if os.path.exists(p):
                return pd.read_excel(p, engine="openpyxl", **read_kwargs)
        except Exception:
            pass
    return None

# =========================
# Load FY25 Dataset
# =========================
fy25_path = "FY25.xlsx"  # File in repo root
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

# ===== Metric groups & descriptions (FY25)
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

# FY color map (FY26 reserved)
fy_color_map = {"FY22": "purple", "FY23": "red", "FY24": "blue", "FY25": "green", "FY26": "orange"}

# =========================
# Load FY26 Budget-to-Enrollment
# =========================
fy26_path = "Enrollment FY26.xlsx"  # File in repo root
try:
    df_budget_raw = pd.read_excel(fy26_path, sheet_name="FY26 Student enrollment", header=1)
    df_budget_raw.columns = df_budget_raw.columns.str.strip()
    if "CMO" in df_budget_raw.columns:
        df_budget_raw.drop(columns=["CMO"], inplace=True)

    expected_cols = ["Schools", "Fiscal Year", "Budgetted", "October 1 Count",
                     "Variance", "%Variance", "Budget to Enrollment Ratio"]
    df_budget_raw = df_budget_raw.dropna(subset=["Schools", "Fiscal Year"])

    df_budget_long = df_budget_raw.melt(
        id_vars=["Schools", "Fiscal Year"],
        value_vars=[c for c in expected_cols if c in df_budget_raw.columns],
        var_name="Metric",
        value_name="Value"
    )

    fiscal_options_budget = sorted(df_budget_long["Fiscal Year"].dropna().unique(), key=sort_fy)
    school_options_budget = sorted(df_budget_long["Schools"].dropna().unique())

except Exception as e:
    st.warning(f"⚠️ Could not load {fy26_path} or sheet 'FY26 Student enrollment': {e}")
    df_budget_long = pd.DataFrame()
    fiscal_options_budget, school_options_budget = [], []

# === Metric color map for Budget-to-Enrollment
budget_metric_color_map = {
    "Budgetted": "#1f77b4",
    "October 1 Count": "#2ca02c",
    "Variance": "#d62728",
    "%Variance": "#9467bd",
    "Budget to Enrollment Ratio": "#ff7f0e",
}
percent_metrics_budget = {"%Variance", "Budget to Enrollment Ratio"}

# =========================
# UI
# =========================
st.title("📊 NOLA Schools Financial Tracker")
st.markdown("<p style='font-size:14px;color:gray;'>Built by Emmanuel Igbokwe</p>", unsafe_allow_html=True)
st.sidebar.header("🔎 Filters")

modes = ["CSAF Metrics", "CSAF Predicted", "Other Metrics"]
if not df_budget_long.empty:
    modes.append("Budget to Enrollment")
metric_group = st.sidebar.radio("Choose Dashboard:", modes)
viz_type = st.sidebar.selectbox("📈 Visualization Type:", ["Bar Chart", "Line Chart"])

# =========================
# CSAF PREDICTED — Frozen Forecast that remains with Actual
# =========================
if metric_group == "CSAF Predicted":
    st.markdown("## 🔮 CSAF Predicted Metrics (FY22–FY28)")

    csaf_formulas = {
        "FB Ratio": ("Fund Balance Ratio", "Unrestricted Fund Balance ÷ Total Expenses", 0.10, "≥ 10%"),
        "Liabilities to Assets": ("Liabilities to Assets Ratio", "Total Liabilities ÷ Total Assets", 0.90, "≤ 0.90"),
        "Current Ratio": ("Current Ratio", "Current Assets ÷ Current Liabilities", 1.50, "≥ 1.5"),
        "Unrestricted Days COH": ("Unrestricted Days Cash on Hand", "Unrestricted Cash ÷ ((Total Exp. - Depreciation) ÷ 365)", 60, "≥ 60 days"),
    }

    # ---------- Sidebar controls ----------
    schools = sorted(df["Schools"].unique())
    selected_school = st.sidebar.selectbox("🏫 Select School:", schools, index=0 if schools else None)
    selected_metric = st.sidebar.selectbox("📊 Choose Metric:", list(csaf_formulas.keys()))

    fiscal_years = sorted(df["Fiscal Year"].dropna().unique(), key=sort_fy)
    selected_fy_hist = st.sidebar.multiselect(
        "📅 History Fiscal Years (training):",
        fiscal_years,
        default=fiscal_years
    )

    # ✅ Forecast origin (freeze quarter). Forecast stays on chart even after actual updates.
    all_quarters = sorted(df["Fiscal Year"].dropna().astype(str).unique(), key=sort_fy)
    train_through = st.sidebar.selectbox(
        "🧊 Forecast Origin (freeze at):",
        all_quarters,
        index=max(0, len(all_quarters) - 1)
    )

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
    viz_type_local = st.sidebar.selectbox("📈 Visualization Type:", ["Line Chart", "Bar Chart"])
    run_pred = st.sidebar.button("▶ Run Prediction")

    # ---------- Helpers ----------
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

    def clean_series(y):
        return pd.to_numeric(pd.Series(y), errors="coerce").values.astype(float)

    if not run_pred:
        st.info("Use the sidebar to pick School, Metric, History Years, Forecast Origin, Method, then click **Run Prediction**.")
        st.stop()

    # --- Build training set (frozen up to train_through) ---
    hist_df = df[(df["Schools"] == selected_school) & (df["Fiscal Year"].isin(selected_fy_hist))].copy()
    if hist_df.empty:
        st.warning("⚠️ No rows for the selected school and history years.")
        st.stop()
    if selected_metric not in hist_df.columns:
        st.warning(f"⚠️ {selected_metric} not found for {selected_school}.")
        st.stop()

    hist_df["sort_key"] = hist_df["Fiscal Year"].apply(sort_fy)
    cut_key = sort_fy(train_through)

    # ✅ Freeze training data at the chosen origin
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

    # ---------- Forecast methods ----------
    def forecast_seasonal_naive(y, q, n_future):
        # repeats last year's same quarter (3-quarter seasonality)
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

        # log1p for stability; clip to avoid negatives in log transform
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

    # ✅ Future labels start AFTER the frozen origin, not after newest actual
    future_labels = make_future_labels(train_through, n_future, quarters_per_year=3)

    # --- Latest ACTUAL from file (updates whenever Excel updates) ---
    actual_now = df[df["Schools"] == selected_school].copy()
    actual_now["sort_key"] = actual_now["Fiscal Year"].apply(sort_fy)
    actual_now = actual_now.sort_values("sort_key")

    actual_series = pd.DataFrame({
        "Quarter": actual_now["Fiscal Year"].astype(str),
        "Value": pd.to_numeric(actual_now[selected_metric], errors="coerce"),
        "Type": "Actual"
    }).dropna(subset=["Value"])

    # --- Frozen forecast stored in session so it remains with actual ---
    forecast_key = f"{selected_school}__{selected_metric}__{train_through}__{forecast_method}__{n_future}"
    st.session_state.setdefault("forecast_store", {})

    if forecast_key not in st.session_state["forecast_store"]:
        frozen_pred = pd.DataFrame({"Quarter": future_labels, "Value": y_future, "Type": "Forecast (Frozen)"})
        st.session_state["forecast_store"][forecast_key] = frozen_pred.copy()
    else:
        frozen_pred = st.session_state["forecast_store"][forecast_key].copy()

    combined = pd.concat(
        [
            actual_series[["Quarter", "Value", "Type"]],
            frozen_pred[["Quarter", "Value", "Type"]],
        ],
        ignore_index=True
    )

    metric_label, formula_txt, threshold, best_label = csaf_formulas[selected_metric]

    # --- Plot ---
    if viz_type_local == "Line Chart":
        fig = px.line(
            combined,
            x="Quarter",
            y="Value",
            color="Type",
            color_discrete_map={"Actual": "blue", "Forecast (Frozen)": "red"},
            markers=True,
            title=f"{selected_school} — {selected_metric} (Actual vs Frozen Forecast)"
        )
    else:
        fig = px.bar(
            combined,
            x="Quarter",
            y="Value",
            color="Type",
            color_discrete_map={"Actual": "blue", "Forecast (Frozen)": "red"},
            barmode="group",
            text="Value",
            title=f"{selected_school} — {selected_metric} (Actual vs Frozen Forecast)"
        )
        if selected_metric == "FB Ratio":
            fig.update_traces(texttemplate="%{y:.1%}", textposition="outside")
        elif selected_metric in ("Liabilities to Assets", "Current Ratio"):
            fig.update_traces(texttemplate="%{y:.2f}", textposition="outside")
        else:
            fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside")

    # --- Best practice line ---
    if "≥" in best_label:
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Best Practice {best_label}",
            annotation_position="top left"
        )
    else:
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color="red",
            annotation_text=f"Best Practice {best_label}",
            annotation_position="top left"
        )

    fig.update_xaxes(tickangle=45)

    if selected_metric == "FB Ratio":
        fig.update_layout(yaxis_tickformat=".1%")
    elif selected_metric in ("Liabilities to Assets", "Current Ratio"):
        fig.update_layout(yaxis_tickformat=",.2f")
    else:
        fig.update_layout(yaxis_tickformat=",.0f")

    # --- Shade forecast region (frozen) ---
    try:
        x0 = frozen_pred["Quarter"].iloc[0]
        x1 = frozen_pred["Quarter"].iloc[-1]
        fig.add_vrect(x0=x0, x1=x1, fillcolor="orange", opacity=0.08, line_width=0)
    except Exception:
        pass

    fig.update_layout(height=560, legend_title="Series")
    st.plotly_chart(fig, use_container_width=True)

    # --- Forecast table (frozen) ---
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
# BUDGET TO ENROLLMENT (SCHOOL-SPECIFIC COMPARISON)
# =========================
elif metric_group == "Budget to Enrollment":
    selected_schools = st.sidebar.multiselect("Select School(s):", school_options_budget)
    if st.sidebar.checkbox("Select All Budget Schools"):
        selected_schools = school_options_budget

    selected_fy = st.sidebar.multiselect("Select Fiscal Year(s):", fiscal_options_budget)
    if st.sidebar.checkbox("Select All Budget Fiscal Years"):
        selected_fy = fiscal_options_budget

    metrics_list = ["Budgetted", "October 1 Count", "Variance", "%Variance", "Budget to Enrollment Ratio"]
    metrics_list = [m for m in metrics_list if m in df_budget_long["Metric"].unique()]
    selected_metrics = st.sidebar.multiselect("Select Metrics:", metrics_list)

    df_f = df_budget_long[
        (df_budget_long["Schools"].isin(selected_schools)) &
        (df_budget_long["Fiscal Year"].isin(selected_fy)) &
        (df_budget_long["Metric"].isin(selected_metrics))
    ]

    if not df_f.empty:
        df_f = df_f.copy()
        df_f["sort_key"] = df_f["Fiscal Year"].apply(sort_fy)
        df_f = df_f.sort_values("sort_key")

        title = f"Budget to Enrollment Comparison — {', '.join(selected_metrics)}"

        if viz_type == "Line Chart":
            fig = px.line(
                df_f, x="Fiscal Year", y="Value",
                color="Metric",
                color_discrete_map=budget_metric_color_map,
                markers=True,
                facet_col="Schools",
                facet_col_wrap=2,
                title=title
            )
        else:
            fig = px.bar(
                df_f, x="Fiscal Year", y="Value",
                color="Metric",
                color_discrete_map=budget_metric_color_map,
                barmode="group",
                text="Value",
                facet_col="Schools",
                facet_col_wrap=2,
                title=title
            )

            for tr in fig.data:
                name = tr.name
                if name in percent_metrics_budget:
                    subset = df_f[df_f["Metric"] == name]["Value"]
                    if subset.max() <= 1.2:
                        tr.texttemplate = "%{text:.0%}"
                    else:
                        tr.texttemplate = "%{text:,.2f}%"
                elif name in {"Budgetted", "October 1 Count", "Variance"}:
                    tr.texttemplate = "%{text:,.0f}"
                else:
                    tr.texttemplate = "%{text}"
            fig.update_traces(textposition="outside")

        fig.update_xaxes(tickangle=45)
        fig.update_layout(
            height=700,
            legend_title="Metric",
            title_x=0.5,
            bargap=0.15,
            bargroupgap=0.05
        )
        st.plotly_chart(fig, use_container_width=True)

        def fmt_budget(row):
            m, v = row["Metric"], row["Value"]
            try:
                if m in percent_metrics_budget:
                    if df_f[df_f["Metric"] == m]["Value"].max() <= 1.2:
                        return f"{v:.0%}"
                    else:
                        return f"{v:,.2f}%"
                else:
                    return f"{v:,.0f}"
            except:
                return v

        df_show = df_f.copy()
        df_show["Formatted Value"] = df_show.apply(fmt_budget, axis=1)
        df_show = df_show[["Schools", "Fiscal Year", "Metric", "Formatted Value"]]
        st.markdown("### 📋 Budget to Enrollment Data (By School)")
        st.dataframe(df_show, use_container_width=True)
    else:
        st.warning("⚠️ No Budget to Enrollment data matches your filters.")

# =========================
# FY25 (CSAF + Other) — unchanged visuals
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
    ]

    if not filtered.empty:
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

        if len(selected_schools) > 8:
            if viz_type == "Bar Chart":
                fig = px.bar(
                    filtered, x="Fiscal Year", y="Value",
                    color="Schools", barmode="group", text="Value", title=chart_title
                )
            else:
                fig = px.line(
                    filtered, x="Fiscal Year", y="Value",
                    color="Schools", markers=True, title=chart_title
                )
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

        fiscal_order = filtered["Fiscal Year"].unique().tolist()
        fig.update_xaxes(categoryorder="array", categoryarray=fiscal_order)

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
                if metric in dollar_metrics:
                    return f"${val:,.0f}"
                elif metric == "FB Ratio":
                    return f"{val:.0%}"
                elif metric in ["Liabilities to Assets", "Current Ratio"]:
                    return f"{val:.2f}"
                else:
                    return f"{val:,.0f}"
            except:
                return val

        df_display = filtered.copy()
        df_display["Value"] = df_display.apply(lambda row: format_value(row["Value"], row["Metric"]), axis=1)
        st.markdown("### 📑 Data Table")
        st.dataframe(df_display, use_container_width=True)
    else:
        st.warning("⚠️ Welcome To Finance Accountability Real-Time Dashboard. Try Adjusting your Left filters.")
