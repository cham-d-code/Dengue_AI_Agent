import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# ==============================================================================
# 1. CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(
    page_title="DengueGuard AI | National Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- MODERN CSS ---------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* GLOBAL RESET */
* { font-family: 'Plus Jakarta Sans', sans-serif !important; }
html, body, .stApp { background-color: #F8FAFC !important; color: #0F172A !important; }

/* HIDE DEFAULT FOOTER ONLY - Keep header for sidebar toggle */
header[data-testid="stHeader"] { 
    background: transparent !important; 
    visibility: visible !important;
}
header[data-testid="stHeader"] > div:first-child {
    background: transparent !important;
}
/* Keep sidebar toggle button visible - HIGH CONTRAST */
button[data-testid="stSidebarCollapseButton"],
button[data-testid="collapsedControl"],
button[data-testid="baseButton-headerNoPadding"] {
    visibility: visible !important;
    display: flex !important;
    background-color: #1E40AF !important; /* Bright blue for contrast */
    color: #FFFFFF !important;
    border: 2px solid #3B82F6 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    padding: 8px !important;
    margin: 8px !important;
}
button[data-testid="stSidebarCollapseButton"]:hover,
button[data-testid="collapsedControl"]:hover,
button[data-testid="baseButton-headerNoPadding"]:hover {
    background-color: #2563EB !important;
}
button[data-testid="stSidebarCollapseButton"] svg,
button[data-testid="collapsedControl"] svg,
button[data-testid="baseButton-headerNoPadding"] svg {
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
}
footer { display: none; }


/* SIDEBAR STYLING - HIGH CONTRAST */
section[data-testid="stSidebar"] {
    background-color: #0F172A !important; /* Dark Navy */
    border-right: 1px solid #1E293B;
}

section[data-testid="stSidebar"] * {
    color: #F8FAFC !important; /* White text */
}

/* Hide broken Material Icons text (keyboard_double_) but keep button functional */
button[data-testid="stSidebarCollapseButton"] span {
    font-size: 0 !important;
    color: transparent !important;
}

/* Style the sidebar collapse button - show as arrow */
button[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    background-color: #1E40AF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px !important;
    margin: 8px !important;
    cursor: pointer !important;
}

/* Add arrow icon using ::before pseudo-element */
button[data-testid="stSidebarCollapseButton"]::before {
    content: "←" !important;
    font-size: 18px !important;
    color: #FFFFFF !important;
    font-weight: bold !important;
}


/* Sidebar selectbox override */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #1E293B !important;
    border-color: #334155 !important;
}

/* CARDS ("Glass" Effect normalized for Light Mode) */
.stat-card {
    background: #FFFFFF;
    border: 1px solid #CBD5E1; /* Darker border for contrast */
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); /* Stronger shadow */
    height: 100%;
    min-height: 180px; /* Fixed height for uniformity */
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.2s ease;
}
.stat-card:hover {
    border-color: #94A3B8;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    transform: translateY(-2px);
}

/* TYPOGRAPHY - HIGH CONTRAST */
.big-stat { 
    font-size: 2.25rem; 
    font-weight: 800; 
    color: #0F172A; /* Almost black */
    letter-spacing: -0.02em; 
    margin-bottom: 4px;
}
.label { 
    font-size: 0.8rem; 
    font-weight: 700; 
    color: #475569; /* Darker slate */
    text-transform: uppercase; 
    letter-spacing: 0.05em; 
    margin-bottom: 8px;
}
.sub-stat { 
    font-size: 0.875rem; 
    font-weight: 500;
    color: #64748B; 
}

/* UNIFIED CARD STYLES */
/* Use a single solid style for all metrics to ensure consistency */
.metric-box {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px;
}

/* BADGES - HIGH VISIBILITY */
.risk-badge {
    padding: 6px 12px;
    border-radius: 6px; /* Squarer, more button-like */
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
    text-transform: uppercase;
}
.risk-critical { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
.risk-high { background: #ffedd5; color: #9a3412; border: 1px solid #fed7aa; }
.risk-mod { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
.risk-low { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }


/* CHAT INTERFACE - FIX OVERLAP */
.chat-container {
    background: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    padding: 24px;
    margin-top: 40px; /* Increased spacing */
}

/* Fix for overlapping avatar/text in Streamlit chat messages */
div[data-testid="stChatMessage"] {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    gap: 1rem; /* Ensure space between avatar and text */
}

div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
    padding-top: 2px; /* align text with avatar center */
}

/* Ensure avatar doesn't shrink */
div[data-testid="stChatMessage"] img, 
div[data-testid="stChatMessage"] .st-emotion-cache-1p1m4ay {
    min-width: 40px;
    width: 40px;
    height: 40px;
}

</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA ENGINE
# ==============================================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    paths = {
        "current": os.path.join(ROOT, "data_processed", "forecast_2024_with_capacity.csv"),
        "future": os.path.join(ROOT, "data_processed", "future_forecast_next_2_weeks_weather_based.csv"),
        "weather_forecast": os.path.join(ROOT, "data_raw", "weather_forecast_openmeteo_7day.csv"),
        "weather_past": os.path.join(ROOT, "data_raw", "weather_past_14_days.csv")  # Historical weather for t-2 lag
    }
    return {k: pd.read_csv(v) if os.path.exists(v) else pd.DataFrame() for k, v in paths.items()}

data = load_data()
df_curr = data['current']
df_fut = data['future']
df_wx = data['weather_forecast']
df_wx_past = data['weather_past']

# Rename weather columns for consistency
if not df_wx.empty:
    col_map = {
        'rain_daily_total': 'rainfall_mm',
        'rain_sum_7d': 'rainfall_mm',
        'temp_daily_avg': 'temp_avg',
        'humidity_daily_avg': 'humidity_pct'
    }
    df_wx = df_wx.rename(columns={k: v for k, v in col_map.items() if k in df_wx.columns})

# Rename past weather columns for consistency
if not df_wx_past.empty:
    col_map_past = {
        'rain_sum': 'rainfall_mm',
        'temp_max': 'temp_max',
        'temp_min': 'temp_min'
    }
    df_wx_past = df_wx_past.rename(columns={k: v for k, v in col_map_past.items() if k in df_wx_past.columns})
    # Calculate average temp
    if 'temp_max' in df_wx_past.columns and 'temp_min' in df_wx_past.columns:
        df_wx_past['temp_avg'] = (df_wx_past['temp_max'] + df_wx_past['temp_min']) / 2

if df_curr.empty:
    st.error("⚠️ System Offline: No forecast data available.")
    st.stop()


# ==============================================================================
# 3. CHATBOT LOGIC - Senior Epidemiological Data Analyst
# ==============================================================================
def agent_response(prompt):
    """
    Role: Senior Epidemiological Data Analyst for NDCU Sri Lanka.
    Provides deep-dive analysis with historical comparison, weather audit,
    biological justification, and actionable resource recommendations.
    """
    from datetime import datetime, timedelta
    import random
    
    prompt = prompt.lower()
    
    # Get the currently selected district from sidebar
    sel_dist = st.session_state.get('selected_district', 'Colombo')
    
    # Get Context Data for selected district
    dist_data = df_curr[df_curr['district'] == sel_dist]
    if dist_data.empty:
        return f"No data available for {sel_dist}. Please select another district."
    
    d_row = dist_data.iloc[0]
    
    # Extract key metrics
    pred = int(d_row['predicted_cases'])
    obs = int(d_row['cases'])
    risk = d_row['risk']
    beds = int(d_row['beds_needed'])
    blood_units = max(1, beds // 2)
    
    # Calculate date ranges
    today = datetime.now()
    week_num = today.isocalendar()[1]
    next_week_start = today + timedelta(days=(7 - today.weekday()))
    next_week_end = next_week_start + timedelta(days=6)
    
    # ===========================================================================
    # REAL HISTORICAL DATA (2021-2024) - Replaces simulation
    # ===========================================================================
    from historical_analysis import (
        get_historical_cases, 
        get_historical_with_weather,
        get_weather_lagged,
        find_similar_weather_periods,
        calculate_weather_case_correlation,
        get_trend_analysis
    )
    
    # Get real historical cases for same week across years
    historical_data = get_historical_with_weather(sel_dist, week_num, years=[2024, 2023, 2022, 2021])
    
    # Extract values for backwards compatibility
    cases_by_year = {h['year']: h['cases'] for h in historical_data if h['cases'] is not None}
    historical_same_week_last_year = cases_by_year.get(2024, cases_by_year.get(2023, pred))
    
    # Get trend from 2024 data
    trend_info = get_trend_analysis(sel_dist, week_num)
    velocity_pct = trend_info['change_pct']
    
    # For display - get cases from weeks ago in 2024 data
    cases_4_weeks_ago = get_historical_cases(sel_dist, max(1, week_num - 4), [2024]).get(2024, obs)
    cases_2_weeks_ago = get_historical_cases(sel_dist, max(1, week_num - 2), [2024]).get(2024, obs)
    
    # Get current weather (t-2 lag from 2024)
    current_weather = get_weather_lagged(sel_dist, 2024, week_num, lag=2)
    rainfall_current = current_weather['rainfall_mm']
    temp_current = current_weather['temp_avg_c']
    humidity_current = current_weather['humidity_pct']
    
    # Get last year's weather for same period
    last_year_weather = get_weather_lagged(sel_dist, 2023, week_num, lag=2)
    rainfall_last_year = last_year_weather['rainfall_mm']
    temp_last_year = last_year_weather['temp_avg_c']
    humidity_last_year = last_year_weather['humidity_pct']
    
    # Calculate year-over-year change
    if historical_same_week_last_year and historical_same_week_last_year > 0:
        yoy_change = ((pred - historical_same_week_last_year) / historical_same_week_last_year) * 100
    else:
        yoy_change = 0
    
    # Rainfall impact calculation
    if rainfall_last_year > 0:
        rainfall_change = ((rainfall_current - rainfall_last_year) / rainfall_last_year) * 100
    else:
        rainfall_change = 100 if rainfall_current > 0 else 0
    
    # Hydro-Humid Interaction check
    hydro_humid_active = humidity_current > 75 and rainfall_current > 30
    
    # Biological limits check
    temp_optimal = 24 <= temp_current <= 30
    
    # Get weather-case correlations
    correlations = calculate_weather_case_correlation(sel_dist)
    
    # =========================================================================
    # BUILD DETAILED RESPONSE
    # =========================================================================
    
    def generate_detailed_forecast():
        # Build historical comparison table
        hist_table = "| Year | Week " + str(week_num) + " Cases | Weather (t-2 lag) |\n|------|---------------|-------------------|\n"
        for h in historical_data:
            cases_str = str(h['cases']) if h['cases'] is not None else "No data"
            weather_str = f"{h['rainfall_mm']:.0f}mm, {h['temp_avg_c']:.1f}°C"
            hist_table += f"| {h['year']} | {cases_str} | {weather_str} |\n"
        
        response = f"""## Detailed Forecast: {sel_dist} District (Week {week_num + 1})

---

### 1. Historical Data Analysis (2021-2024)

**Prediction Period:** {next_week_start.strftime('%b %d')} – {next_week_end.strftime('%b %d, %Y')}

**Same Week Across Years (Real Data):**

{hist_table}

**Current Trend:** {trend_info['trend'].capitalize()} ({velocity_pct:+.0f}% over recent weeks)

---

### 2. Weather-Case Correlation (t-2 Lag)

Weather from 2 weeks ago determines next week's cases. Correlation analysis for {sel_dist}:

| Weather Factor | Correlation with Cases |
|---------------|------------------------|
| Rainfall | {correlations['rainfall']:+.2f} {'📈 Strong' if abs(correlations['rainfall']) > 0.3 else '➖ Weak'} |
| Temperature | {correlations['temperature']:+.2f} {'📈 Strong' if abs(correlations['temperature']) > 0.3 else '➖ Weak'} |
| Humidity | {correlations['humidity']:+.2f} {'📈 Strong' if abs(correlations['humidity']) > 0.3 else '➖ Weak'} |

**Current Weather (2 weeks ago):** {rainfall_current:.0f}mm rain, {temp_current:.1f}°C, {humidity_current:.0f}% humidity
**Same period 2023:** {rainfall_last_year:.0f}mm rain, {temp_last_year:.1f}°C, {humidity_last_year:.0f}% humidity

---

### 3. Prediction Basis

The model predicts **{pred} cases** based on:

"""
        # Add reasoning based on conditions
        if hydro_humid_active:
            response += f"""**🔬 Hydro-Humid Interaction: ACTIVE**
- High rainfall ({rainfall_current:.0f}mm) + humidity ({humidity_current:.0f}%) = more breeding sites

"""
        
        if temp_optimal:
            response += f"""**🌡️ Temperature: OPTIMAL ({temp_current:.1f}°C)**
- Within 24-30°C range for mosquito activity

"""
        else:
            response += f"""**🌡️ Biological Hard Limits:**

Temperature ({temp_current:.1f}°C) is {'above' if temp_current > 30 else 'below'} the optimal 24-30°C range, which {'may stress mosquito populations' if temp_current > 34 else 'slows mosquito development'}.

"""
        
        if velocity_pct > 20:
            response += f"""**📈 Momentum Analysis:**

The {velocity_pct:.0f}% surge over 14 days indicates active community transmission. Without intervention, exponential growth is likely.

"""
        
        response += f"""---

### 4. Actionable Preparation

| Metric | Recommendation |
|--------|----------------|
| **Risk Level** | {risk.upper()} {'(Escalating from previous week)' if velocity_pct > 10 else ''} |
| **Hospital Readiness** | Prepare **{beds} additional beds** and **{blood_units} blood units** for {sel_dist} |
| **Vector Control** | {'🚨 PRIORITY fogging needed in urban areas where rain accumulation was highest' if rainfall_current > 40 else 'Standard surveillance recommended'} |
| **Lab Capacity** | Ensure **{max(10, pred * 2)} NS1/IgM rapid test kits** are stocked |

---

*Analysis generated by NDCU Epidemiological Intelligence Unit*
*Data sources: National Disease Surveillance, Meteorological Department, Historical Case Registry*
"""
        return response
    
    # --- INTENT HANDLING ---
    
    # 1. Detailed/Prediction Queries
    if any(k in prompt for k in ['predict', 'forecast', 'detail', 'analysis', 'next week', 'cases', 'how many', 'report']):
        return generate_detailed_forecast()
        
    # 2. Weather Queries
    elif any(k in prompt for k in ['weather', 'rain', 'temp', 'climate', 'humid']):
        return f"""## Weather Analysis: {sel_dist}

**Current Conditions (2-Week Lag Window):**
- Rainfall: **{rainfall_current:.1f}mm** (Last year: {rainfall_last_year:.1f}mm)
- Temperature: **{temp_current:.1f}°C** (Last year: {temp_last_year:.1f}°C)  
- Humidity: **{humidity_current:.0f}%** (Last year: {humidity_last_year:.0f}%)

**Epidemiological Impact:**
{'⚠️ **Hydro-Humid Interaction Active:** High rainfall + humidity > 75% = elevated breeding success.' if hydro_humid_active else '✅ Conditions are within normal parameters.'}

{'🌡️ Temperature is in optimal range (24-30°C) for Aedes aegypti activity.' if temp_optimal else '🌡️ Temperature outside optimal range may limit vector activity.'}
"""
        
    # 3. National Queries
    elif any(k in prompt for k in ['national', 'sri lanka', 'all district', 'country', 'total']):
        total = int(df_fut['predicted_cases'].sum()) if not df_fut.empty else 0
        crit_districts = df_curr[df_curr['risk'] == 'critical']['district'].tolist()
        high_districts = df_curr[df_curr['risk'] == 'high']['district'].tolist()
        return f"""## National Dengue Situation Report

**Island-Wide Forecast:** {total:,} cases expected next week

**Critical Hotspots ({len(crit_districts)}):** {', '.join(crit_districts[:5]) if crit_districts else 'None currently'}

**High Risk Districts ({len(high_districts)}):** {', '.join(high_districts[:5]) if high_districts else 'None currently'}

**Resource Allocation Priority:**
1. Surge medical supplies to critical districts
2. Deploy rapid response teams
3. Activate community vector control programs
"""
        
    # 4. Action/Hospital Queries
    elif any(k in prompt for k in ['do', 'action', 'recommend', 'prepare', 'hospital', 'resource']):
        return f"""## Action Protocol: {sel_dist}

**Immediate Actions Required:**

{d_row['recommended_actions']}

**Resource Requirements:**
- Hospital Beds: **{beds}**
- Blood Units: **{blood_units}**
- Rapid Test Kits: **{max(10, pred * 2)}**
- Fogging Priority: **{'HIGH' if rainfall_current > 40 else 'STANDARD'}**

**Timeline:** Actions should be initiated within 48 hours to prepare for predicted case surge.
"""
    
    # 5. Why/Reason Queries
    elif any(k in prompt for k in ['why', 'reason', 'explain', 'how', 'logic']):
        return generate_detailed_forecast()
        
    # Default - show brief status with option for detailed analysis
    return f"""## {sel_dist} Quick Status

**Forecast:** {pred} cases next week ({risk.upper()} risk)
**Trend:** {velocity_pct:+.0f}% over 14 days
**Beds Needed:** {beds}

---

For detailed analysis with historical comparison and weather audit, ask:
- "Give me a detailed forecast"
- "Why this prediction?"
- "What's the weather impact?"
"""


# ==============================================================================
# 4. UI LAYOUT
# ==============================================================================

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2382/2382461.png", width=50)
    st.title("DengueGuard")
    st.caption("v2.0 National Framework")
    st.markdown("---")
    
    st.markdown("### Location")
    districts = sorted(df_curr['district'].unique())
    # Initialize session state for district if not set
    if 'selected_district' not in st.session_state:
        st.session_state.selected_district = "Colombo"
        
    selected_district = st.selectbox("Select District", districts, key='selected_district')
    
    st.markdown("---")
    st.markdown("### Agent Status")
    st.success("Model Online")
    st.caption("Updated: " + datetime.now().strftime("%H:%M"))
    
    # CHAT TOGGLE BUTTON (Hidden real button, we use CSS for the FAB look)
    if 'chat_open' not in st.session_state:
        st.session_state.chat_open = False
        
    def toggle_chat():
        st.session_state.chat_open = not st.session_state.chat_open
    
    # We will use a regular button in sidebar to toggle, 
    # but for the requested "Icon Button Bottom Left", we need a floating UI in the main area.
    # See the bottom of the script for the FAB implementation.

# --- MAIN PAGE: DASHBOARD ---

# Header
st.markdown(f"# National Defense <span style='color:#94A3B8; font-weight:300'>| {selected_district} Sector</span>", unsafe_allow_html=True)
st.markdown(" ")

# 1. NATIONAL OVERVIEW (Top Row)
st.markdown("### National Situation")
c1, c2, c3, c4 = st.columns(4)

total_cases = int(df_fut['predicted_cases'].sum()) if not df_fut.empty else 0
# Count UNIQUE districts with critical/high risk (not row count)
crit_count = df_curr[df_curr['risk'] == 'critical']['district'].nunique()
high_count = df_curr[df_curr['risk'] == 'high']['district'].nunique()


with c1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="label">National Forecast (2W)</div>
        <div class="big-stat" style="color:#2563EB">{total_cases:,}</div>
        <div class="sub-stat">Total expected patients</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="label">Critical Hotspots</div>
        <div class="big-stat" style="color:#DC2626">{crit_count}</div>
        <div class="sub-stat">Districts needing immediate surge</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="label">High Risk Zones</div>
        <div class="big-stat" style="color:#D97706">{high_count}</div>
        <div class="sub-stat">Districts on alert</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="label">Readiness</div>
        <div class="big-stat" style="color:#16A34A">AA-25</div>
        <div class="sub-stat">Anticipatory Action Active</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# 2. DISTRICT DEEP DIVE
d_row = df_curr[df_curr['district'] == selected_district].iloc[0]
risk_lvl = d_row['risk']
risk_cls = f"risk-{risk_lvl}" if risk_lvl in ['low', 'moderate', 'high', 'critical'] else "risk-mod"

st.markdown(f"### District Focus: {selected_district} <span class='risk-badge {risk_cls}'>{risk_lvl.upper()}</span>", unsafe_allow_html=True)

# Detailed Metrics (3 columns, full width)
dc1, dc2, dc3 = st.columns(3)

pred = int(d_row['predicted_cases'])
beds = int(d_row['beds_needed'])

with dc1:
    st.markdown(f"""
    <div class="stat-card" style="border-left:4px solid #3B82F6;">
        <div class="label">Predicted Load</div>
        <div class="big-stat">{pred}</div>
        <div class="sub-stat">Cases next week</div>
    </div>
    """, unsafe_allow_html=True)
    
with dc2:
    st.markdown(f"""
    <div class="stat-card" style="border-left:4px solid #8B5CF6;">
        <div class="label">Logistics</div>
        <div class="big-stat">{beds}</div>
        <div class="sub-stat">Beds required</div>
    </div>
    """, unsafe_allow_html=True)
    
with dc3:
    w_text = "Stable"
    rain_mm = 0
    # Use PAST weather data (t-2 lag) for rainfall
    if not df_wx_past.empty:
        dw_past = df_wx_past[df_wx_past['district'] == selected_district]
        if not dw_past.empty:
            # Get most recent rainfall data
            dw_sorted = dw_past.sort_values('date', ascending=False)
            rain_mm = dw_sorted.iloc[0].get('rainfall_mm', 0)
            if rain_mm > 0:
                w_text = f"{rain_mm:.0f}mm Rain"
            else:
                w_text = "Dry"
            
    st.markdown(f"""
    <div class="stat-card" style="border-left:4px solid #10B981;">
        <div class="label">Environment</div>
        <div class="big-stat" style="font-size:1.5rem; line-height:1.5;">{w_text}</div>
        <div class="sub-stat">Vector opportunity</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")


# 3. STATIC FLOATING CHATBOT
# ==========================================================

# Initialize Chat History and track district changes
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_district = selected_district

# Update greeting when district changes
if st.session_state.get('chat_district') != selected_district:
    st.session_state.chat_district = selected_district
    # Clear chat and add new greeting for new district
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hello! I am monitoring **{selected_district}**. Ask me for predictions with reasoning."}
    ]

# Add initial greeting if chat is empty
if not st.session_state.messages:
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hello! I am monitoring **{selected_district}**. Ask me for predictions with reasoning."}
    ]


# Static FAB Button (Pure HTML/CSS - Fixed Position at Bottom Left)
st.markdown("""
<style>
/* Static Floating Action Button */
.static-fab {
    position: fixed;
    bottom: 30px;
    left: 30px;
    width: 56px;
    height: 56px;
    background-color: #0F172A;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    z-index: 9999;
    text-decoration: none;
    transition: all 0.2s ease;
}
.static-fab:hover {
    background-color: #1E293B;
    transform: scale(1.05);
}
.static-fab span {
    font-size: 24px;
}
</style>
""", unsafe_allow_html=True)

# Gemini-Style Chat Interface at Bottom
st.markdown("""
<style>
.chat-greeting {
    text-align: center;
    padding: 40px 20px 20px 20px;
}
.chat-greeting-icon {
    font-size: 1.5rem;
    background: linear-gradient(135deg, #4285F4, #EA4335, #FBBC05, #34A853);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
}
.chat-greeting-text {
    font-size: 1.1rem;
    color: #5F6368;
    font-weight: 500;
    margin-left: 8px;
}
.chat-question {
    font-size: 2rem;
    font-weight: 400;
    color: #202124;
    text-align: center;
    margin-bottom: 30px;
}
.chat-input-container {
    max-width: 800px;
    margin: 0 auto;
    background: #FFFFFF;
    border-radius: 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08);
    padding: 16px 24px;
    border: 1px solid #E8EAED;
}
.chat-input-box {
    display: flex;
    align-items: center;
    gap: 12px;
}
.chat-toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #F1F3F4;
}
.toolbar-left {
    display: flex;
    gap: 16px;
    align-items: center;
}
.toolbar-btn {
    background: none;
    border: none;
    color: #5F6368;
    font-size: 0.9rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
}
.toolbar-right {
    display: flex;
    gap: 12px;
    align-items: center;
    color: #5F6368;
    font-size: 0.9rem;
}

/* Hide chat message avatars */
div[data-testid="stChatMessage"] > div:first-child {
    display: none !important;
}

/* Make chat text black */
div[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 8px 0 !important;
}

div[data-testid="stChatMessage"] p {
    color: #202124 !important;
}

/* White send button with dark arrow icon */
div[data-testid="column"]:last-child button {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    min-width: 50px !important;
}
div[data-testid="column"]:last-child button p,
div[data-testid="column"]:last-child button span,
div[data-testid="column"]:last-child button div {
    color: #0F172A !important;
}
div[data-testid="column"]:last-child button:hover,
div[data-testid="column"]:last-child button:focus,
div[data-testid="column"]:last-child button:active {
    background-color: #F1F5F9 !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
}
/* Make text input white */
div[data-testid="stTextInput"] input {
    background-color: #FFFFFF !important;
    color: #202124 !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
}
div[data-testid="stTextInput"] input::placeholder {
    color: #94A3B8 !important;
}
</style>


""", unsafe_allow_html=True)

# Initialize chat (empty - no default message)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize input key for clearing
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# Show conversation if there are messages
if st.session_state.messages:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<p style="color: #202124; font-weight: 500;"><strong>You:</strong> {message["content"]}</p>', unsafe_allow_html=True)
        else:
            st.markdown(message["content"], unsafe_allow_html=True)

# Simple text input (white/light style)
col1, col2 = st.columns([10, 1])
with col1:
    user_input = st.text_input("", placeholder="Ask about dengue predictions...", label_visibility="collapsed", key=f"chat_input_{st.session_state.input_key}")
with col2:
    send_btn = st.button("➤", key="send_btn", help="Send message")

if send_btn and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    response_text = agent_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    # Increment key to clear input
    st.session_state.input_key += 1
    st.rerun()
