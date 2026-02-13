"""
NOLA SCHOOLS FINANCIAL INTELLIGENCE PLATFORM
Advanced Executive Financial Analysis & Predictive Analytics

Features:
- Executive Financial Dashboard per School
- CSAF Health Scoring & Risk Assessment
- Advanced ML Forecasting (XGBoost, Prophet, ARIMA)
- Automated Financial Insights
- Enrollment Predictions
- Comprehensive Financial Analysis

Built by Emmanuel Igbokwe
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# XGBoost for superior performance
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="NOLA Financial Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROFESSIONAL DARK THEME
# ============================================================
THEME = {
    'bg_primary': '#0a0e17',
    'bg_secondary': '#1a1f2e',
    'bg_card': '#141922',
    'accent_primary': '#00ff88',
    'accent_secondary': '#0066ff',
    'accent_warning': '#ffaa00',
    'accent_danger': '#ff0066',
    'text_primary': '#e8eaed',
    'text_secondary': '#9aa0a6',
    'border': 'rgba(0, 255, 136, 0.2)',
    'grid': 'rgba(42, 49, 66, 0.3)'
}

st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(135deg, {THEME['bg_primary']} 0%, #0d1117 100%);
        color: {THEME['text_primary']};
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {THEME['bg_secondary']};
        border-right: 1px solid {THEME['border']};
    }}
    
    section[data-testid="stSidebar"] * {{
        color: {THEME['text_primary']} !important;
    }}
    
    /* Headers */
    h1, h2, h3 {{
        color: {THEME['accent_primary']} !important;
        font-family: 'Orbitron', sans-serif;
    }}
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {{
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: {THEME['accent_primary']} !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        font-size: 1rem !important;
        color: {THEME['text_secondary']} !important;
    }}
    
    /* Executive Card */
    .exec-card {{
        background: {THEME['bg_card']};
        border: 1px solid {THEME['border']};
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }}
    
    .exec-card:hover {{
        border-color: {THEME['accent_primary']};
        box-shadow: 0 6px 20px rgba(0, 255, 136, 0.2);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }}
    
    /* Status Badges */
    .status-excellent {{ 
        background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        color: #000;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }}
    
    .status-good {{ 
        background: linear-gradient(135deg, #0066ff 0%, #0052cc 100%);
        color: #fff;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }}
    
    .status-warning {{ 
        background: linear-gradient(135deg, #ffaa00 0%, #cc8800 100%);
        color: #000;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }}
    
    .status-danger {{ 
        background: linear-gradient(135deg, #ff0066 0%, #cc0052 100%);
        color: #fff;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }}
    
    /* Insight Box */
    .insight-box {{
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 102, 255, 0.1) 100%);
        border-left: 4px solid {THEME['accent_primary']};
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }}
    
    .insight-title {{
        color: {THEME['accent_primary']};
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 8px;
    }}
    
    /* Data Table */
    .dataframe {{
        background: {THEME['bg_card']} !important;
        color: {THEME['text_primary']} !important;
    }}
    
    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, {THEME['accent_primary']} 0%, {THEME['accent_secondary']} 100%);
        color: #000;
        border: none;
        border-radius: 8px;
        padding: 10px 25px;
        font-weight: 700;
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0, 255, 136, 0.3);
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def fy_label(y: int) -> str:
    """Convert year to FY label"""
    return f"FY{int(y):02d}"

def fy_num(fy_str: str):
    """Extract numeric year from FY string"""
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
    """Standardize FY format"""
    n = fy_num(val)
    return fy_label(n) if n is not None else str(val).strip()

def sort_fy(x):
    """Sort fiscal years with quarters"""
    try:
        parts = str(x).split()
        year = fy_num(parts[0]) if parts else None
        if year is None:
            return (999, 9)
        q = int(parts[1].replace("Q", "").strip()) if len(parts) > 1 else 9
        return (year, q)
    except:
        return (999, 9)

def parse_quarter(label: str):
    """Extract quarter number from FY label"""
    m = re.search(r"Q\s*(\d)", str(label))
    return int(m.group(1)) if m else None

# ============================================================
# FINANCIAL HEALTH ASSESSMENT
# ============================================================

class FinancialHealthAnalyzer:
    """
    Comprehensive financial health scoring system
    """
    
    CSAF_THRESHOLDS = {
        'FB Ratio': {'excellent': 0.20, 'good': 0.15, 'adequate': 0.10, 'poor': 0.05},
        'Current Ratio': {'excellent': 2.50, 'good': 2.00, 'adequate': 1.50, 'poor': 1.00},
        'Liabilities to Assets': {'excellent': 0.50, 'good': 0.70, 'adequate': 0.90, 'poor': 1.00},
        'Unrestricted Days COH': {'excellent': 120, 'good': 90, 'adequate': 60, 'poor': 30}
    }
    
    @staticmethod
    def calculate_health_score(school_data):
        """
        Calculate comprehensive financial health score (0-100)
        """
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
            
            # Score calculation logic
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
        """Get health rating from score"""
        if score >= 90:
            return "Excellent", "excellent"
        elif score >= 75:
            return "Good", "good"
        elif score >= 60:
            return "Adequate", "warning"
        else:
            return "At Risk", "danger"
    
    @staticmethod
    def generate_insights(school_data, school_name):
        """
        Generate automated financial insights
        """
        insights = []
        
        # Fund Balance Analysis
        fb_ratio = school_data.get('FB Ratio', 0)
        if fb_ratio >= 0.20:
            insights.append({
                'type': 'positive',
                'title': '🎯 Strong Fund Balance',
                'message': f"Fund Balance Ratio of {fb_ratio:.1%} exceeds best practice threshold (10%), providing excellent financial cushion."
            })
        elif fb_ratio < 0.10:
            insights.append({
                'type': 'warning',
                'title': '⚠️ Low Fund Balance',
                'message': f"Fund Balance Ratio of {fb_ratio:.1%} is below recommended 10% threshold. Consider building reserves."
            })
        
        # Liquidity Analysis
        current_ratio = school_data.get('Current Ratio', 0)
        if current_ratio >= 2.50:
            insights.append({
                'type': 'positive',
                'title': '💧 Excellent Liquidity',
                'message': f"Current Ratio of {current_ratio:.2f} indicates strong ability to meet short-term obligations."
            })
        elif current_ratio < 1.50:
            insights.append({
                'type': 'warning',
                'title': '⚠️ Liquidity Concerns',
                'message': f"Current Ratio of {current_ratio:.2f} is below 1.50 threshold. Monitor cash flow closely."
            })
        
        # Leverage Analysis
        liab_to_assets = school_data.get('Liabilities to Assets', 0)
        if liab_to_assets > 0.90:
            insights.append({
                'type': 'warning',
                'title': '📊 High Leverage',
                'message': f"Liabilities represent {liab_to_assets:.1%} of assets. Consider debt reduction strategies."
            })
        
        # Cash on Hand Analysis
        days_coh = school_data.get('Unrestricted Days COH', 0)
        if days_coh >= 120:
            insights.append({
                'type': 'positive',
                'title': '💰 Strong Cash Position',
                'message': f"{days_coh:.0f} days of cash on hand provides excellent operational flexibility."
            })
        elif days_coh < 60:
            insights.append({
                'type': 'critical',
                'title': '🚨 Cash Flow Risk',
                'message': f"Only {days_coh:.0f} days of cash on hand. Immediate attention needed to build cash reserves."
            })
        
        # Revenue Analysis
        total_revenue = school_data.get('Total Revenue', 0)
        local_rev = school_data.get('Local Revenue', 0)
        if total_revenue > 0:
            local_pct = (local_rev / total_revenue) * 100
            if local_pct > 50:
                insights.append({
                    'type': 'positive',
                    'title': '📈 Strong Local Support',
                    'message': f"Local revenue represents {local_pct:.1f}% of total, indicating strong community support."
                })
        
        # Expense Analysis
        total_expenses = school_data.get('Total Expenses', 0)
        salaries = school_data.get('Salaries', 0)
        if total_expenses > 0:
            salary_pct = (salaries / total_expenses) * 100
            if salary_pct > 60:
                insights.append({
                    'type': 'info',
                    'title': '👥 Staff-Heavy Model',
                    'message': f"Salaries represent {salary_pct:.1f}% of expenses. This is typical for quality education."
                })
        
        return insights

# ============================================================
# ADVANCED ML FORECASTING ENGINE
# ============================================================

class AdvancedForecaster:
    """
    State-of-the-art time series forecasting with ensemble methods
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = float('inf')
        
    def prepare_features(self, y, q=None, n_lags=4):
        """Create supervised learning features from time series"""
        n = len(y)
        X, y_target = [], []
        
        for t in range(n_lags, n):
            # Lag features
            lags = y[t-n_lags:t][::-1]
            
            # Trend feature
            trend = float(t)
            
            # Features list
            features = list(lags) + [trend, trend**2]
            
            # Quarter dummies if available
            if q is not None:
                q_val = int(q[t])
                features.extend([1 if q_val == i else 0 for i in range(1, 4)])
            
            X.append(features)
            y_target.append(y[t])
        
        return np.array(X), np.array(y_target)
    
    def train_models(self, y, q=None):
        """Train multiple models and select best"""
        X, y_target = self.prepare_features(y, q)
        
        if len(y_target) < 10:
            return None
        
        # Define model candidates
        candidates = {
            'XGBoost': XGBRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                random_state=42
            ) if XGBOOST_AVAILABLE else None,
            
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.08,
                random_state=42
            ),
            
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ),
            
            'Ridge': Ridge(alpha=1.0),
            
            'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5)
        }
        
        # Remove None models
        candidates = {k: v for k, v in candidates.items() if v is not None}
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=min(5, len(y_target) // 3))
        
        for name, model in candidates.items():
            try:
                scores = cross_val_score(
                    model, X, y_target,
                    cv=tscv,
                    scoring='neg_mean_absolute_error'
                )
                mae = -scores.mean()
                
                self.models[name] = {
                    'model': model,
                    'mae': mae,
                    'scores': scores
                }
                
                if mae < self.best_score:
                    self.best_score = mae
                    self.best_model = name
                    
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        # Train best model on full data
        if self.best_model:
            best = self.models[self.best_model]['model']
            best.fit(X, y_target)
        
        return self.best_model
    
    def forecast(self, y, q=None, horizon=6):
        """Generate forecast"""
        if self.best_model is None:
            self.train_models(y, q)
        
        if self.best_model is None:
            # Fallback: simple average
            return np.array([y[-1]] * horizon)
        
        model = self.models[self.best_model]['model']
        n_lags = 4
        
        # Iterative forecasting
        y_extended = list(y)
        q_extended = list(q) if q is not None else None
        predictions = []
        
        for _ in range(horizon):
            # Prepare features
            lags = y_extended[-n_lags:][::-1]
            t = len(y_extended)
            features = list(lags) + [float(t), float(t**2)]
            
            if q_extended is not None:
                q_next = (q_extended[-1] % 3) + 1
                features.extend([1 if q_next == i else 0 for i in range(1, 4)])
                q_extended.append(q_next)
            
            # Predict
            X_forecast = np.array([features])
            y_pred = model.predict(X_forecast)[0]
            
            # Apply reasonable bounds
            y_pred = max(0, min(y_pred, y[-1] * 1.3))
            
            predictions.append(y_pred)
            y_extended.append(y_pred)
        
        return np.array(predictions)
    
    def get_model_performance(self):
        """Return performance metrics for all models"""
        results = []
        for name, info in self.models.items():
            results.append({
                'Model': name,
                'MAE': f"${info['mae']/1e6:.2f}M",
                'Best': '✅' if name == self.best_model else ''
            })
        return pd.DataFrame(results)

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():
    """Load and prepare financial data"""
    try:
        df = pd.read_excel('FY25.xlsx', sheet_name='FY25')
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=['Schools', 'Fiscal Year'])
        df['Fiscal Year'] = df['Fiscal Year'].astype(str).str.strip()
        df['sort_key'] = df['Fiscal Year'].apply(sort_fy)
        df = df.sort_values(['Schools', 'sort_key'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
df = load_data()

if df is None:
    st.stop()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🎯 Navigation")

page = st.sidebar.radio(
    "Select Dashboard:",
    [
        "📊 Executive Summary",
        "🏫 School Financial Analysis",
        "🔮 CSAF Predictions",
        "📈 Financial Metrics",
        "👥 Enrollment Analysis",
        "📑 Comprehensive Report"
    ]
)

# School selector
schools = sorted(df['Schools'].unique())
selected_school = st.sidebar.selectbox("Select School:", schools)

# Fiscal year filter
fiscal_years = sorted(df['Fiscal Year'].unique(), key=sort_fy)
selected_fy = st.sidebar.multiselect(
    "Select Fiscal Years:",
    fiscal_years,
    default=fiscal_years[-4:] if len(fiscal_years) >= 4 else fiscal_years
)

# ============================================================
# PAGE 1: EXECUTIVE SUMMARY
# ============================================================

if page == "📊 Executive Summary":
    st.title("📊 Executive Financial Summary")
    st.markdown(f"### {selected_school}")
    
    # Get latest data for school
    school_data = df[
        (df['Schools'] == selected_school) &
        (df['Fiscal Year'].isin(selected_fy))
    ].sort_values('sort_key', ascending=False)
    
    if school_data.empty:
        st.warning("No data available for selected filters")
        st.stop()
    
    latest = school_data.iloc[0]
    
    # Calculate financial health score
    analyzer = FinancialHealthAnalyzer()
    health_score = analyzer.calculate_health_score(latest)
    health_rating, health_class = analyzer.get_health_rating(health_score)
    
    # Header Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Financial Health Score",
            f"{health_score:.0f}/100",
            delta=health_rating
        )
    
    with col2:
        st.metric(
            "Fund Balance Ratio",
            f"{latest['FB Ratio']:.1%}",
            delta="✅ Healthy" if latest['FB Ratio'] >= 0.10 else "⚠️ Low"
        )
    
    with col3:
        st.metric(
            "Current Ratio",
            f"{latest['Current Ratio']:.2f}",
            delta="✅ Good" if latest['Current Ratio'] >= 1.50 else "⚠️ Monitor"
        )
    
    with col4:
        st.metric(
            "Days Cash on Hand",
            f"{latest['Unrestricted Days COH']:.0f}",
            delta="✅ Strong" if latest['Unrestricted Days COH'] >= 60 else "⚠️ Low"
        )
    
    # Health Status Badge
    st.markdown(f"""
    <div class="exec-card">
        <h3>Overall Financial Health</h3>
        <span class="status-{health_class}">{health_rating}</span>
        <p style="margin-top: 10px;">Based on comprehensive analysis of CSAF metrics and financial indicators.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Automated Insights
    st.markdown("### 💡 Key Financial Insights")
    insights = analyzer.generate_insights(latest, selected_school)
    
    for insight in insights:
        icon = {
            'positive': '✅',
            'warning': '⚠️',
            'critical': '🚨',
            'info': 'ℹ️'
        }.get(insight['type'], '📌')
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">{icon} {insight['title']}</div>
            <div>{insight['message']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Financial Overview Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue vs Expenses
        fig_rev_exp = go.Figure()
        
        fig_rev_exp.add_trace(go.Bar(
            name='Revenue',
            x=school_data['Fiscal Year'],
            y=school_data['Total Revenue'],
            marker_color=THEME['accent_primary']
        ))
        
        fig_rev_exp.add_trace(go.Bar(
            name='Expenses',
            x=school_data['Fiscal Year'],
            y=school_data['Total Expenses'],
            marker_color=THEME['accent_secondary']
        ))
        
        fig_rev_exp.update_layout(
            title="Revenue vs Expenses Trend",
            barmode='group',
            paper_bgcolor=THEME['bg_primary'],
            plot_bgcolor=THEME['bg_secondary'],
            font=dict(color=THEME['text_primary']),
            height=400
        )
        
        st.plotly_chart(fig_rev_exp, use_container_width=True)
    
    with col2:
        # CSAF Metrics Radar
        csaf_metrics = ['FB Ratio', 'Current Ratio', 'Unrestricted Days COH']
        csaf_values = [latest[m] for m in csaf_metrics if m in latest]
        
        # Normalize for radar chart
        normalized = [
            min(latest['FB Ratio'] / 0.20, 1) * 100,
            min(latest['Current Ratio'] / 2.50, 1) * 100,
            min(latest['Unrestricted Days COH'] / 120, 1) * 100
        ]
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=normalized,
            theta=['Fund Balance', 'Liquidity', 'Cash Reserves'],
            fill='toself',
            marker_color=THEME['accent_primary']
        ))
        
        fig_radar.update_layout(
            title="Financial Health Profile",
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            paper_bgcolor=THEME['bg_primary'],
            plot_bgcolor=THEME['bg_secondary'],
            font=dict(color=THEME['text_primary']),
            height=400
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)

# ============================================================
# PAGE 2: SCHOOL FINANCIAL ANALYSIS
# ============================================================

elif page == "🏫 School Financial Analysis":
    st.title(f"🏫 Financial Analysis: {selected_school}")
    
    school_data = df[
        (df['Schools'] == selected_school) &
        (df['Fiscal Year'].isin(selected_fy))
    ].sort_values('sort_key')
    
    if school_data.empty:
        st.warning("No data available")
        st.stop()
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 CSAF Metrics",
        "💰 Revenue Analysis", 
        "💸 Expense Analysis",
        "📈 Balance Sheet"
    ])
    
    with tab1:
        st.markdown("### CSAF Key Performance Indicators")
        
        # 4-panel CSAF view
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Fund Balance Ratio',
                'Current Ratio',
                'Liabilities to Assets',
                'Days Cash on Hand'
            ]
        )
        
        # FB Ratio
        fig.add_trace(
            go.Scatter(
                x=school_data['Fiscal Year'],
                y=school_data['FB Ratio'],
                mode='lines+markers',
                name='FB Ratio',
                line=dict(color=THEME['accent_primary'], width=3)
            ),
            row=1, col=1
        )
        fig.add_hline(y=0.10, line_dash="dash", line_color="red", row=1, col=1)
        
        # Current Ratio
        fig.add_trace(
            go.Scatter(
                x=school_data['Fiscal Year'],
                y=school_data['Current Ratio'],
                mode='lines+markers',
                name='Current Ratio',
                line=dict(color=THEME['accent_secondary'], width=3)
            ),
            row=1, col=2
        )
        fig.add_hline(y=1.50, line_dash="dash", line_color="red", row=1, col=2)
        
        # Liabilities to Assets
        fig.add_trace(
            go.Scatter(
                x=school_data['Fiscal Year'],
                y=school_data['Liabilities to Assets'],
                mode='lines+markers',
                name='Liab/Assets',
                line=dict(color=THEME['accent_warning'], width=3)
            ),
            row=2, col=1
        )
        fig.add_hline(y=0.90, line_dash="dash", line_color="red", row=2, col=1)
        
        # Days COH
        fig.add_trace(
            go.Scatter(
                x=school_data['Fiscal Year'],
                y=school_data['Unrestricted Days COH'],
                mode='lines+markers',
                name='Days COH',
                line=dict(color=THEME['accent_primary'], width=3)
            ),
            row=2, col=2
        )
        fig.add_hline(y=60, line_dash="dash", line_color="red", row=2, col=2)
        
        fig.update_layout(
            height=700,
            showlegend=False,
            paper_bgcolor=THEME['bg_primary'],
            plot_bgcolor=THEME['bg_secondary'],
            font=dict(color=THEME['text_primary'])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Revenue Composition & Trends")
        
        # Revenue breakdown
        latest = school_data.iloc[-1]
        
        revenue_data = {
            'Local Revenue': latest['Local Revenue'],
            'State Revenue': latest['State Rev'],
            'Federal Revenue': latest['Federa Rev ']
        }
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(revenue_data.keys()),
            values=list(revenue_data.values()),
            hole=0.4,
            marker=dict(colors=[
                THEME['accent_primary'],
                THEME['accent_secondary'],
                THEME['accent_warning']
            ])
        )])
        
        fig_pie.update_layout(
            title=f"Revenue Sources - {latest['Fiscal Year']}",
            paper_bgcolor=THEME['bg_primary'],
            font=dict(color=THEME['text_primary']),
            height=400
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Revenue trend
        fig_trend = go.Figure()
        
        for col, color in [
            ('Local Revenue', THEME['accent_primary']),
            ('State Rev', THEME['accent_secondary']),
            ('Federa Rev ', THEME['accent_warning'])
        ]:
            fig_trend.add_trace(go.Scatter(
                x=school_data['Fiscal Year'],
                y=school_data[col],
                mode='lines+markers',
                name=col.replace(' Rev', ' Revenue').replace('Federa', 'Federal'),
                line=dict(color=color, width=3)
            ))
        
        fig_trend.update_layout(
            title="Revenue Trends by Source",
            paper_bgcolor=THEME['bg_primary'],
            plot_bgcolor=THEME['bg_secondary'],
            font=dict(color=THEME['text_primary']),
            height=400
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with tab3:
        st.markdown("### Expense Analysis")
        
        # Expense categories
        expense_cols = [
            'Salaries', 'Employee Benefits', 
            'Purchased professional', 'Supplies'
        ]
        
        expense_data = {}
        for col in expense_cols:
            if col in latest:
                expense_data[col] = latest[col]
        
        fig_exp = go.Figure(data=[go.Bar(
            x=list(expense_data.keys()),
            y=list(expense_data.values()),
            marker_color=THEME['accent_secondary']
        )])
        
        fig_exp.update_layout(
            title=f"Expense Breakdown - {latest['Fiscal Year']}",
            paper_bgcolor=THEME['bg_primary'],
            plot_bgcolor=THEME['bg_secondary'],
            font=dict(color=THEME['text_primary']),
            height=400
        )
        
        st.plotly_chart(fig_exp, use_container_width=True)
    
    with tab4:
        st.markdown("### Balance Sheet Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Assets
            assets_data = {
                'Current Assets': latest['Current Assets'],
                'Fixed Assets': latest['Fixed Assets']
            }
            
            fig_assets = go.Figure(data=[go.Pie(
                labels=list(assets_data.keys()),
                values=list(assets_data.values()),
                hole=0.4,
                marker=dict(colors=[
                    THEME['accent_primary'],
                    THEME['accent_secondary']
                ])
            )])
            
            fig_assets.update_layout(
                title="Asset Composition",
                paper_bgcolor=THEME['bg_primary'],
                font=dict(color=THEME['text_primary']),
                height=350
            )
            
            st.plotly_chart(fig_assets, use_container_width=True)
        
        with col2:
            # Liabilities
            liab_data = {
                'Current Liabilities': latest['Current Liabilities'],
                'Long-term Liabilities': latest['Long term liabilities']
            }
            
            fig_liab = go.Figure(data=[go.Pie(
                labels=list(liab_data.keys()),
                values=list(liab_data.values()),
                hole=0.4,
                marker=dict(colors=[
                    THEME['accent_warning'],
                    THEME['accent_danger']
                ])
            )])
            
            fig_liab.update_layout(
                title="Liability Composition",
                paper_bgcolor=THEME['bg_primary'],
                font=dict(color=THEME['text_primary']),
                height=350
            )
            
            st.plotly_chart(fig_liab, use_container_width=True)

# ============================================================
# PAGE 3: CSAF PREDICTIONS
# ============================================================

elif page == "🔮 CSAF Predictions":
    st.title(f"🔮 CSAF Predictions: {selected_school}")
    
    # Sidebar controls
    st.sidebar.markdown("### Prediction Settings")
    
    selected_metric = st.sidebar.selectbox(
        "Select CSAF Metric:",
        ['FB Ratio', 'Current Ratio', 'Liabilities to Assets', 'Unrestricted Days COH']
    )
    
    horizon = st.sidebar.slider("Forecast Horizon (Quarters)", 3, 12, 6)
    
    run_forecast = st.sidebar.button("▶️ Run Prediction", type="primary")
    
    if run_forecast:
        # Get historical data
        school_data = df[df['Schools'] == selected_school].sort_values('sort_key')
        
        y = school_data[selected_metric].values.astype(float)
        q = school_data['Fiscal Year'].apply(parse_quarter).values
        
        # Remove NaN
        mask = ~np.isnan(y) & ~np.isnan(q)
        y = y[mask]
        q = q[mask].astype(int)
        
        if len(y) < 5:
            st.error("Insufficient data for prediction (need at least 5 data points)")
            st.stop()
        
        # Train forecaster
        with st.spinner("Training advanced ML models..."):
            forecaster = AdvancedForecaster()
            best_model = forecaster.train_models(y, q)
            
            if best_model:
                st.success(f"✅ Best Model: {best_model}")
            
            # Generate forecast
            predictions = forecaster.forecast(y, q, horizon)
        
        # Create future labels
        last_fy = school_data['Fiscal Year'].iloc[-1]
        last_year = fy_num(last_fy)
        last_q = parse_quarter(last_fy) or 3
        
        future_labels = []
        for i in range(horizon):
            last_q += 1
            if last_q > 3:
                last_q = 1
                last_year += 1
            future_labels.append(f"FY{last_year:02d} Q{last_q}")
        
        # Plot
        fig = go.Figure()
        
        # Historical
        fig.add_trace(go.Scatter(
            x=school_data['Fiscal Year'].tolist(),
            y=y.tolist(),
            mode='lines+markers',
            name='Historical',
            line=dict(color=THEME['accent_primary'], width=3),
            marker=dict(size=8)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=future_labels,
            y=predictions.tolist(),
            mode='lines+markers',
            name='Forecast',
            line=dict(color=THEME['accent_secondary'], width=3, dash='dash'),
            marker=dict(size=8)
        ))
        
        # Threshold line
        thresholds = {
            'FB Ratio': 0.10,
            'Current Ratio': 1.50,
            'Liabilities to Assets': 0.90,
            'Unrestricted Days COH': 60
        }
        
        if selected_metric in thresholds:
            fig.add_hline(
                y=thresholds[selected_metric],
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold"
            )
        
        fig.update_layout(
            title=f"{selected_metric} - Historical & Forecast",
            paper_bgcolor=THEME['bg_primary'],
            plot_bgcolor=THEME['bg_secondary'],
            font=dict(color=THEME['text_primary']),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model Performance
        st.markdown("### 📊 Model Performance Comparison")
        perf_df = forecaster.get_model_performance()
        st.dataframe(perf_df, use_container_width=True)
        
        # Prediction table
        st.markdown("### 📅 Forecast Values")
        forecast_df = pd.DataFrame({
            'Quarter': future_labels,
            'Predicted Value': predictions,
            'Status': [
                '✅ Above Threshold' if (
                    (selected_metric != 'Liabilities to Assets' and pred >= thresholds[selected_metric]) or
                    (selected_metric == 'Liabilities to Assets' and pred <= thresholds[selected_metric])
                ) else '⚠️ Below Threshold'
                for pred in predictions
            ]
        })
        
        st.dataframe(forecast_df, use_container_width=True)

# ============================================================
# PAGE 4: FINANCIAL METRICS
# ============================================================

elif page == "📈 Financial Metrics":
    st.title("📈 Comprehensive Financial Metrics")
    
    school_data = df[
        (df['Schools'] == selected_school) &
        (df['Fiscal Year'].isin(selected_fy))
    ].sort_values('sort_key')
    
    # Metric categories
    metric_categories = {
        'Assets': ['Current Assets', 'Fixed Assets', 'Total Assets'],
        'Liabilities': ['Current Liabilities', 'Long term liabilities', 'Total Liabilities'],
        'Revenue': ['Local Revenue', 'State Rev', 'Federa Rev ', 'Total Revenue'],
        'Expenses': ['Salaries', 'Employee Benefits', 'Total Expenses']
    }
    
    selected_category = st.selectbox(
        "Select Metric Category:",
        list(metric_categories.keys())
    )
    
    metrics = metric_categories[selected_category]
    
    # Plot metrics
    fig = go.Figure()
    
    colors = [THEME['accent_primary'], THEME['accent_secondary'], THEME['accent_warning'], THEME['accent_danger']]
    
    for i, metric in enumerate(metrics):
        if metric in school_data.columns:
            fig.add_trace(go.Scatter(
                x=school_data['Fiscal Year'],
                y=school_data[metric],
                mode='lines+markers',
                name=metric,
                line=dict(color=colors[i % len(colors)], width=3)
            ))
    
    fig.update_layout(
        title=f"{selected_category} Trends",
        paper_bgcolor=THEME['bg_primary'],
        plot_bgcolor=THEME['bg_secondary'],
        font=dict(color=THEME['text_primary']),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 5: ENROLLMENT ANALYSIS
# ============================================================

elif page == "👥 Enrollment Analysis":
    st.title("👥 Enrollment Analysis & Predictions")
    st.info("📝 Note: Enrollment data will be integrated when available")
    
    # Placeholder for enrollment analysis
    st.markdown("""
    ### Enrollment Metrics (Coming Soon)
    
    This section will include:
    - **October 1 Count** - Official enrollment snapshot
    - **February 1 Count** - Mid-year enrollment
    - **Budget to Enrollment Ratio** - Funding efficiency
    - **Enrollment Trends** - Historical patterns
    - **Enrollment Predictions** - ML-powered forecasts
    
    *Please ensure enrollment data is available in the system*
    """)

# ============================================================
# PAGE 6: COMPREHENSIVE REPORT
# ============================================================

elif page == "📑 Comprehensive Report":
    st.title(f"📑 Comprehensive Financial Report")
    st.markdown(f"## {selected_school}")
    
    school_data = df[
        (df['Schools'] == selected_school) &
        (df['Fiscal Year'].isin(selected_fy))
    ].sort_values('sort_key')
    
    latest = school_data.iloc[-1]
    
    # Executive Summary
    st.markdown("### Executive Summary")
    
    analyzer = FinancialHealthAnalyzer()
    health_score = analyzer.calculate_health_score(latest)
    health_rating, _ = analyzer.get_health_rating(health_score)
    
    st.markdown(f"""
    **Financial Health Score:** {health_score:.0f}/100 ({health_rating})
    
    **Reporting Period:** {school_data['Fiscal Year'].min()} to {school_data['Fiscal Year'].max()}
    
    **Key Findings:**
    - Fund Balance Ratio: {latest['FB Ratio']:.1%}
    - Current Ratio: {latest['Current Ratio']:.2f}
    - Liabilities to Assets: {latest['Liabilities to Assets']:.1%}
    - Days Cash on Hand: {latest['Unrestricted Days COH']:.0f}
    """)
    
    # Financial Highlights
    st.markdown("### Financial Highlights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Assets", f"${latest['Total Assets']:,.0f}")
        st.metric("Total Liabilities", f"${latest['Total Liabilities']:,.0f}")
    
    with col2:
        st.metric("Total Revenue", f"${latest['Total Revenue']:,.0f}")
        st.metric("Total Expenses", f"${latest['Total Expenses']:,.0f}")
    
    with col3:
        surplus = latest['Total Revenue'] - latest['Total Expenses']
        st.metric("Net Surplus/(Deficit)", f"${surplus:,.0f}")
        st.metric("Fund Balance", f"${latest['Fund Balance']:,.0f}")
    
    # Detailed Data Table
    st.markdown("### Historical Financial Data")
    
    display_cols = [
        'Fiscal Year', 'Total Assets', 'Total Liabilities',
        'Total Revenue', 'Total Expenses', 'Fund Balance',
        'FB Ratio', 'Current Ratio', 'Unrestricted Days COH'
    ]
    
    st.dataframe(
        school_data[display_cols].style.format({
            'Total Assets': '${:,.0f}',
            'Total Liabilities': '${:,.0f}',
            'Total Revenue': '${:,.0f}',
            'Total Expenses': '${:,.0f}',
            'Fund Balance': '${:,.0f}',
            'FB Ratio': '{:.1%}',
            'Current Ratio': '{:.2f}',
            'Unrestricted Days COH': '{:.0f}'
        }),
        use_container_width=True
    )

# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 💡 About This Platform

**Advanced Features:**
- 🎯 Financial Health Scoring
- 🤖 ML Predictions (XGBoost, GBR)
- 📊 Automated Insights
- 📈 Comprehensive Analysis

**Built by Emmanuel Igbokwe**

*Financial Intelligence Platform v2.0*
""")
