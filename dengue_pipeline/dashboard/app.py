import os
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Sri Lanka Dengue Early Warning",
    page_icon="🦟",
    layout="wide",
)

# ------------------------------------------
# Paths
# ------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FCAST_FILE = os.path.join(ROOT, "data_processed", "forecast_2024_with_capacity.csv")
MULTI_FILE = os.path.join(ROOT, "data_processed", "multiweek_forecast_next_2_weeks.csv")
WEATHER_RISK = os.path.join(ROOT, "data_processed", "weather_risk_alerts_next_7_days.csv")


RISK_STYLES = {
    "low": ("Low", "#228B22"),
    "moderate": ("Moderate", "#ffb300"),
    "high": ("High", "#fb8c00"),
    "critical": ("Critical", "#c62828"),
}


def format_capacity_value(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{int(round(value)):,}"


def risk_chip(risk_level: str) -> str:
    label, color = RISK_STYLES.get(
        str(risk_level).lower(), (str(risk_level).title(), "#6c757d")
    )
    return (
        f"<span style='background:{color};padding:4px 12px;border-radius:999px;"
        f"color:white;font-weight:600;font-size:0.9rem'>{label}</span>"
    )


def action_sentences(text: str) -> list[str]:
    sentences = [seg.strip(" .") for seg in str(text).split(". ") if seg.strip()]
    return [s for s in sentences if s]


# ------------------------------------------
# Load Data
# ------------------------------------------
@st.cache_data
def load_data():
    df_main = pd.read_csv(FCAST_FILE) if os.path.exists(FCAST_FILE) else None
    df_multi = pd.read_csv(MULTI_FILE) if os.path.exists(MULTI_FILE) else None
    df_weather = pd.read_csv(WEATHER_RISK) if os.path.exists(WEATHER_RISK) else None
    return df_main, df_multi, df_weather


df_main, df_multi, df_weather = load_data()

if df_main is None or df_main.empty:
    st.error(
        "No forecast data found. Please run `python scripts/dengue_forecast_model.py` "
        "to generate the latest weekly forecast before opening the dashboard."
    )
    st.stop()

if df_multi is None:
    df_multi = pd.DataFrame()
if df_weather is None:
    df_weather = pd.DataFrame()

st.title("🦟 Sri Lanka Dengue Early Warning & Forecast Dashboard")
st.caption(
    "Live dengue surveillance, resource planning, and weather-triggered alerts "
    "for every district."
)

st.divider()

# ------------------------------------------
# Sidebar
# ------------------------------------------
st.sidebar.header("Select District")
st.sidebar.caption("Choose a district to view the latest situation and actions.")
all_districts = sorted(df_main["district"].unique())
district = st.sidebar.selectbox("District", all_districts)
# Show quick pipeline reminder for operators.
st.sidebar.info(
    "Update data by running the automation scripts in `dengue_pipeline/scripts`."
)

# Latest week for main forecast
latest_week_row = (
    df_main[df_main["district"] == district]
    .sort_values(["year", "week"])
    .tail(1)
    .iloc[0]
)

# ------------------------------------------
# Display current status
# ------------------------------------------
st.header(f"📍 Current Situation — {district}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Observed Cases (latest week)", f"{latest_week_row['cases']:.0f}")
with col2:
    st.metric("Predicted Cases", f"{latest_week_row['predicted_cases']:.1f}")
with col3:
    st.markdown("**Risk Level**")
    st.markdown(risk_chip(latest_week_row["risk"]), unsafe_allow_html=True)
with col4:
    st.metric(
        "Latest Week",
        f"W{int(latest_week_row['week'])} • {int(latest_week_row['year'])}",
    )

col5, col6 = st.columns(2)
with col5:
    st.metric("Hospital beds needed (est.)", format_capacity_value(latest_week_row.get("beds_needed")))
with col6:
    st.metric("Blood units needed (est.)", format_capacity_value(latest_week_row.get("blood_units_needed")))

st.subheader("Recommended MoH Actions")
actions = action_sentences(latest_week_row.get("recommended_actions", ""))
if actions:
    st.markdown("\n".join(f"- {step}" for step in actions))
else:
    st.write("No specific actions available for this district/week.")

st.divider()

# ------------------------------------------
# 2-week forecast
# ------------------------------------------
st.header("🔮 Next 2 Weeks Dengue Forecast")

if df_multi.empty:
    st.warning(
        "Two-week forecasts are not available yet. Run "
        "`python scripts/forecast_next_two_weeks.py` to generate them."
    )
else:
    district_future = df_multi[df_multi["district"] == district]

    if district_future.empty:
        st.warning("No multi-week forecast available for this district yet.")
    else:
        display_cols = [
            "target_year",
            "target_week",
            "horizon_weeks_ahead",
            "predicted_cases",
            "historical_avg_cases",
            "beds_needed",
            "blood_units_needed",
            "risk",
        ]
        future_table = district_future[display_cols].copy()
        future_table["predicted_cases"] = future_table["predicted_cases"].round(1)
        future_table["historical_avg_cases"] = future_table["historical_avg_cases"].round(1)
        future_table["beds_needed"] = (
            future_table["beds_needed"].fillna(0).round().astype(int)
        )
        future_table["blood_units_needed"] = (
            future_table["blood_units_needed"].fillna(0).round().astype(int)
        )

        st.dataframe(
            future_table.rename(
                columns={
                    "target_year": "Year",
                    "target_week": "Week",
                    "horizon_weeks_ahead": "Weeks Ahead",
                    "predicted_cases": "Predicted Cases",
                    "historical_avg_cases": "Historical Avg",
                    "beds_needed": "Beds Needed",
                    "blood_units_needed": "Blood Units Needed",
                    "risk": "Risk",
                }
            ),
            use_container_width=True,
        )

        high_risk_rows = district_future[district_future["risk"].isin(["high", "critical"])]
        if not high_risk_rows.empty:
            st.subheader("⚠️ Expected Patient Surge")
            total_expected = high_risk_rows["predicted_cases"].sum()
            soonest = high_risk_rows.sort_values("horizon_weeks_ahead").iloc[0]
            st.metric(
                "Estimated total cases (next 2 weeks)",
                f"{total_expected:.0f}",
                help="Sum of predicted cases for horizon 1 and 2 weeks.",
            )
            st.info(
                f"The earliest high-risk week is Week {int(soonest['target_week'])} "
                f"({int(soonest['horizon_weeks_ahead'])} week(s) ahead)."
            )
        else:
            st.success("No high or critical risks detected in the next two weeks.")

st.divider()

# ------------------------------------------
# Weather-Based Risk
# ------------------------------------------
st.header("🌧 Weather Risk (Next 7 Days)")

if df_weather.empty:
    st.info(
        "Weather-trigger alerts have not been generated. "
        "Run `python scripts/weather_trigger_alerts.py` after fetching weather forecasts."
    )
else:
    weather_dist = df_weather[df_weather["district"] == district]

    if weather_dist.empty:
        st.info("No weather risk detected for this district yet.")
    else:
        risk_counts = weather_dist["env_risk"].value_counts()
        high_days = risk_counts.get("environment_high", 0)
        moderate_days = risk_counts.get("environment_moderate", 0)
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.metric("High-risk weather days", f"{high_days}")
        with col_w2:
            st.metric("Moderate-risk weather days", f"{moderate_days}")

        st.dataframe(
            weather_dist.rename(
                columns={
                    "date": "Date",
                    "rain_daily_total": "Rain (mm)",
                    "temp_daily_avg": "Temp °C",
                    "humidity_daily_avg": "Humidity %",
                    "env_risk": "Risk",
                }
            ),
            use_container_width=True,
        )

        highest = weather_dist.sort_values("env_risk", ascending=False).iloc[0]
        st.markdown(
            f"**Highest weather risk:** {highest['env_risk'].replace('_', ' ').title()} "
            f"on {highest['date']} with {highest['rain_daily_total']:.1f} mm rain."
        )

st.divider()

# ------------------------------------------
# Download Buttons
# ------------------------------------------
st.header("📥 Download Data")

st.download_button(
    "Download Current Forecast (CSV)",
    data=df_main.to_csv(index=False),
    file_name="forecast_current.csv",
)

if not df_multi.empty:
    st.download_button(
        "Download 2-Week Forecast (CSV)",
        data=df_multi.to_csv(index=False),
        file_name="forecast_next_2_weeks.csv",
    )

if not df_weather.empty:
    st.download_button(
        "Download Weather Risk Alerts (CSV)",
        data=df_weather.to_csv(index=False),
        file_name="weather_risk_next_7_days.csv",
    )
