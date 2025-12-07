import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import plotly.express as px

# ---------------------- Paths ---------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FCAST_CURRENT = os.path.join(ROOT, "data_processed", "forecast_2024_with_capacity.csv")
FCAST_FUTURE = os.path.join(ROOT, "data_processed", "future_forecast_next_2_weeks_weather_based.csv")
WEATHER_ALERTS = os.path.join(ROOT, "data_processed", "weather_risk_alerts_next_7_days.csv")

# Approximate centroid coordinates for Sri Lankan districts
DISTRICT_COORDS = {
    "Ampara": (7.283, 81.683),
    "Anuradhapura": (8.350, 80.383),
    "Badulla": (6.985, 81.055),
    "Batticaloa": (7.717, 81.700),
    "Colombo": (6.927, 79.862),
    "Galle": (6.053, 80.221),
    "Gampaha": (7.089, 79.994),
    "Hambantota": (6.124, 81.118),
    "Jaffna": (9.668, 80.020),
    "Kalutara": (6.585, 79.960),
    "Kandy": (7.291, 80.635),
    "Kegalle": (7.251, 80.345),
    "Kilinochchi": (9.389, 80.385),
    "Kurunegala": (7.480, 80.366),
    "Mannar": (8.980, 79.914),
    "Matale": (7.467, 80.623),
    "Matara": (5.949, 80.546),
    "Monaragala": (6.872, 81.350),
    "Mullaitivu": (9.267, 80.815),
    "Nuwara Eliya": (6.972, 80.782),
    "Polonnaruwa": (7.950, 81.000),
    "Puttalam": (8.040, 79.840),
    "Ratnapura": (6.705, 80.384),
    "Trincomalee": (8.567, 81.233),
    "Vavuniya": (8.754, 80.497),
}

# ------------------ Streamlit UI Config ------------------ #
st.set_page_config(
    page_title="Sri Lanka Dengue Early Warning Dashboard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------- Strong CSS: Force light mode everywhere ------------- #
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');

/* GLOBAL: background + font */
html, body, .stApp {
    background-color: #F5F7FB !important;
    color: #111111 !important;
    font-family: 'Montserrat', sans-serif !important;
}

/* Main view container */
[data-testid="stAppViewContainer"] {
    background-color: #F5F7FB !important;
}

/* Header */
[data-testid="stHeader"] {
    background-color: #F5F7FB !important;
    color: #111111 !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E0E0E0 !important;
}
section[data-testid="stSidebar"] * {
    color: #111111 !important;
    font-family: 'Montserrat', sans-serif !important;
}

/* Sidebar selectbox input */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #111111 !important;
    border: 1px solid #CCCCCC !important;
}

/* Global selectbox (text + arrow) */
div[data-baseweb="select"] span {
    color: #111111 !important;
}
div[data-baseweb="select"] svg {
    fill: #111111 !important;
}

/* Dropdown menu background */
div[role="listbox"] {
    background-color: #FFFFFF !important;
    border: 1px solid #CCCCCC !important;
}

/* Dropdown options */
div[role="option"] {
    background-color: #FFFFFF !important;
    color: #111111 !important;
}
div[role="option"]:hover {
    background-color: #F2F2F2 !important;
    color: #000000 !important;
}
div[role="option"][aria-selected="true"] {
    background-color: #E8E8E8 !important;
    color: #000000 !important;
}

/* CARDS */
.metric-card {
    padding: 20px;
    border-radius: 14px;
    background: #FFFFFF;
    box-shadow: 0 4px 14px rgba(0,0,0,0.07);
    border: 1px solid #E0E0E0;
}

.section-card {
    padding: 24px;
    border-radius: 16px;
    background: #FFFFFF;
    box-shadow: 0 4px 14px rgba(0,0,0,0.07);
    border: 1px solid #E0E0E0;
    margin-top: 24px;
    margin-bottom: 20px;
}

/* DOWNLOAD BUTTONS */
.stDownloadButton button {
    background-color: #FFFFFF !important;
    color: #111111 !important;
    border: 1px solid #CCCCCC !important;
    border-radius: 10px !important;
    padding: 8px 18px !important;
    font-weight: 600 !important;
}
.stDownloadButton button:hover {
    background-color: #F0F0F0 !important;
    color: #000000 !important;
    border-color: #AAAAAA !important;
}

/* HEADINGS */
h1, h2, h3, h4 {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    color: #111111 !important;
}

/* Body text */
p, span, div {
    font-size: 15px;
}

/* Tables */
table td, table th {
    color: #222222 !important;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ------------------- Load Data ------------------- #
def load_data():
    df_current = pd.read_csv(FCAST_CURRENT) if os.path.exists(FCAST_CURRENT) else None
    df_future = pd.read_csv(FCAST_FUTURE) if os.path.exists(FCAST_FUTURE) else None
    df_weather = pd.read_csv(WEATHER_ALERTS) if os.path.exists(WEATHER_ALERTS) else None
    return df_current, df_future, df_weather

df_current, df_future, df_weather = load_data()

st.title("Sri Lanka Dengue Early Warning System")
st.write("Real-time epidemic forecasts, weather-driven alerts, and hospital planning support.")

st.markdown("---")

if df_current is None or df_current.empty:
    st.error("Current forecast data not found. Make sure forecast_2024_with_capacity.csv exists in data_processed.")
    st.stop()

# ------------------- Helper: last year's observed cases ------------------- #
def get_last_year_cases(df, district, cur_year, cur_week, fallback_cases):
    prev_year = cur_year - 1
    mask = (
        (df["district"] == district) &
        (df["year"] == prev_year) &
        (df["week"] == cur_week)
    )
    subset = df.loc[mask]
    if not subset.empty:
        return int(round(subset["cases"].iloc[0]))
    return int(round(fallback_cases))

# ------------------- NATIONAL OVERVIEW ------------------- #
st.markdown("## National Overview")

col_nat1, col_nat2, col_nat3 = st.columns(3)

total_expected_2w = None
high_crit_districts = None
critical_districts = None

if df_future is not None and not df_future.empty:
    total_expected_2w = int(round(df_future["predicted_cases"].sum()))

    best_risk = (
        df_future
        .sort_values(["district", "risk"])
        .groupby("district")
        .tail(1)
    )
    high_crit_districts = int(((best_risk["risk"] == "high") | (best_risk["risk"] == "critical")).sum())
    critical_districts = int((best_risk["risk"] == "critical").sum())

with col_nat1:
    val = total_expected_2w if total_expected_2w is not None else 0
    st.markdown(f"""
    <div class="metric-card">
        <h3>Expected Patients (Next 2 Weeks)</h3>
        <h2>{val}</h2>
        <p style="color:#555555;">All districts combined</p>
    </div>
    """, unsafe_allow_html=True)

with col_nat2:
    val = high_crit_districts if high_crit_districts is not None else 0
    st.markdown(f"""
    <div class="metric-card">
        <h3>High / Critical Risk Districts</h3>
        <h2>{val}</h2>
        <p style="color:#555555;">Based on short-term forecast</p>
    </div>
    """, unsafe_allow_html=True)

with col_nat3:
    val = critical_districts if critical_districts is not None else 0
    st.markdown(f"""
    <div class="metric-card">
        <h3>Critical Risk Districts</h3>
        <h2>{val}</h2>
        <p style="color:#555555;">Require urgent attention</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------- NATIONAL RISK MAP ------------------- #
st.markdown("## National District Risk Map")

latest_by_district = (
    df_current.sort_values(["district", "year", "week"])
    .groupby("district")
    .tail(1)
)

map_rows = []
for _, row in latest_by_district.iterrows():
    d = row["district"]
    coords = DISTRICT_COORDS.get(d)
    if not coords:
        continue
    lat, lon = coords
    pred_cases_int = int(round(row["predicted_cases"]))
    map_rows.append({
        "district": d,
        "lat": lat,
        "lon": lon,
        "predicted_cases": pred_cases_int,
        "risk": str(row["risk"]).lower()
    })

if map_rows:
    map_df = pd.DataFrame(map_rows)
    color_map = {
        "low": "#4CAF50",
        "moderate": "#FFB300",
        "high": "#E53935",
        "critical": "#7B1FA2"
    }
    fig = px.scatter_geo(
        map_df,
        lat="lat",
        lon="lon",
        hover_name="district",
        hover_data={"predicted_cases": True, "lat": False, "lon": False},
        size="predicted_cases",
        size_max=30,
        color="risk",
        color_discrete_map=color_map,
        projection="natural earth",
        scope="asia",
        center={"lat": 7.5, "lon": 81.0},
    )
    fig.update_layout(
        margin=dict(r=0, l=0, b=0, t=0),
        height=450,
        legend_title_text="Risk Level",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No geographic data available to render the map.")

st.markdown("---")

# ------------------- Sidebar selection ------------------- #
st.sidebar.header("District Selection")
districts = sorted(df_current["district"].unique())
selected_district = st.sidebar.selectbox("Choose District", districts)

today = datetime.now().strftime("%Y-%m-%d")
st.sidebar.write(f"Today: **{today}**")

# ------------------- District-specific data ------------------- #
current_row = (
    df_current[df_current["district"] == selected_district]
    .sort_values(["year", "week"])
    .tail(1)
    .iloc[0]
)

current_year = int(current_row["year"])
current_week = int(current_row["week"])

observed_display = get_last_year_cases(
    df_current, selected_district, current_year, current_week, current_row["cases"]
)
predicted_display = int(round(current_row["predicted_cases"]))

# ------------------- DISTRICT: Current Status ------------------- #
st.markdown(f"## Current Epidemiological Status — **{selected_district}**")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Observed Cases (Last Year Same Week)</h3>
        <h2>{observed_display}</h2>
        <p style="color:#555555;">Baseline from previous year</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Predicted Cases (This Week)</h3>
        <h2>{predicted_display}</h2>
        <p style="color:#555555;">Model estimate for current week</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    risk_color = {
        "low": "#4CAF50",
        "moderate": "#FF9800",
        "high": "#E53935",
        "critical": "#7B1FA2"
    }.get(str(current_row["risk"]).lower(), "#333333")

    st.markdown(f"""
    <div class="metric-card" style="border-left: 8px solid {risk_color};">
        <h3>Risk Level</h3>
        <h2 style="color:{risk_color};">{str(current_row['risk']).upper()}</h2>
        <p style="color:#555555;">Based on predicted cases and thresholds</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------- Recommended Actions ------------------- #
st.markdown(f"""
<div class="section-card">
    <h3>Recommended Actions</h3>
    <p>{current_row['recommended_actions']}</p>
</div>
""", unsafe_allow_html=True)

# ------------------- DISTRICT: Future 2-Week Forecast ------------------- #
st.markdown("## Forecast for Selected District: Next 2 Weeks")

if df_future is None or df_future.empty:
    st.warning("No future forecasts available. Run forecast_future_real.py.")
else:
    df_future_disp = df_future.copy()
    if "horizon_weeks_ahead" in df_future_disp.columns:
        df_future_disp = df_future_disp.rename(columns={"horizon_weeks_ahead": "Weeks Ahead"})
    for col in ["predicted_cases", "beds_needed", "blood_units_needed"]:
        if col in df_future_disp.columns:
            df_future_disp[col] = df_future_disp[col].round().astype("Int64")

    future_d = df_future_disp[df_future_disp["district"] == selected_district]

    if future_d.empty:
        st.info("No future forecast records for this district.")
    else:
        st.dataframe(
            future_d,
            hide_index=True,
            use_container_width=True
        )
        total_expected = int(future_d["predicted_cases"].sum())
        st.markdown(f"""
        <div class="metric-card">
            <h3>Expected Patients (Next 2 Weeks)</h3>
            <h2>{total_expected}</h2>
            <p style="color:#555555;">Sum of predictions for Weeks Ahead = 1 and 2</p>
        </div>
        """, unsafe_allow_html=True)

# ------------------- DISTRICT: Weather Risk Alerts ------------------- #
st.markdown("## Weather-Triggered Dengue Alerts (Next 7 Days)")

if df_weather is None or df_weather.empty:
    st.info("Weather risk alerts not generated yet. Run weather_trigger_alerts.py.")
else:
    weather_d = df_weather[df_weather["district"] == selected_district]

    if weather_d.empty:
        st.success("No weather-triggered risks detected for this district in the next 7 days.")
    else:
        st.dataframe(weather_d, hide_index=True, use_container_width=True)

        if "rainfall_mm" in weather_d.columns:
            worst = weather_d.sort_values("rainfall_mm", ascending=False).iloc[0]
        else:
            worst = weather_d.iloc[0]

        st.markdown(f"""
        <div class="metric-card" style="border-left: 8px solid #1976D2;">
            <h3>Highest Weather Risk</h3>
            <h2>{worst['env_risk'].replace('_',' ').title()}</h2>
            <p style="color:#555555;">Driven by forecasted rainfall, temperature and humidity</p>
        </div>
        """, unsafe_allow_html=True)

# ------------------- Downloads ------------------- #
st.markdown("## Download Data")

colA, colB, colC = st.columns(3)

with colA:
    st.download_button(
        "Current Forecast CSV",
        df_current.to_csv(index=False),
        file_name="current_forecast.csv"
    )

with colB:
    if df_future is not None:
        st.download_button(
            "Future Forecast CSV",
            df_future.to_csv(index=False),
            file_name="future_forecast_next_2_weeks.csv"
        )

with colC:
    if df_weather is not None:
        st.download_button(
            "Weather Alerts CSV",
            df_weather.to_csv(index=False),
            file_name="weather_risk_alerts.csv"
        )
