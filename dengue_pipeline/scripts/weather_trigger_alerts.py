import os
import sys
import pandas as pd

# ---------- Path Setup ---------- #

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FORECAST_FILE = os.path.join(PROJECT_ROOT, "data_raw", "weather_forecast_openmeteo_7day.csv")
OUT_ALERTS = os.path.join(PROJECT_ROOT, "data_processed", "weather_risk_alerts_next_7_days.csv")

os.makedirs(os.path.dirname(OUT_ALERTS), exist_ok=True)

# ---------- Dengue Weather Thresholds ---------- #

RAIN_THRESHOLD_HIGH = 60.0      # mm over next week
RAIN_THRESHOLD_MODERATE = 30.0  # mm
TEMP_MIN = 24.0
TEMP_MAX = 32.0
HUMIDITY_MIN = 70.0


def classify_env_risk(row):
    """Determine dengue-favourable weather level."""
    rain = row["rain_daily_total"]
    temp = row["temp_daily_avg"]
    hum = row["humidity_daily_avg"]

    # High risk
    if (
        rain >= RAIN_THRESHOLD_HIGH
        and TEMP_MIN <= temp <= TEMP_MAX
        and hum >= HUMIDITY_MIN
    ):
        return "environment_high"

    # Moderate risk
    if rain >= RAIN_THRESHOLD_MODERATE:
        return "environment_moderate"

    return "environment_low"


def main():
    print("Loading forecast:", FORECAST_FILE)

    if not os.path.exists(FORECAST_FILE):
        raise FileNotFoundError(
            f"❌ Forecast file missing.\nRun fetch_openmeteo_forecast.py first."
        )

    df = pd.read_csv(FORECAST_FILE)

    if df.empty:
        print("❌ Forecast file is empty.")
        return

    # Because this forecast is per-day for 7 days, we directly use daily metrics.
    df["env_risk"] = df.apply(classify_env_risk, axis=1)

    # Only keep districts with moderate/high
    alerts = df[df["env_risk"].isin(["environment_moderate", "environment_high"])]

    if alerts.empty:
        print("✅ No dengue-favourable weather alerts for the next 7 days.")
        return

    alerts.to_csv(OUT_ALERTS, index=False)

    print("\n🎉 Weather dengue risk alerts saved to:")
    print("   ", OUT_ALERTS)

    print("\n=== WEATHER-BASED DENGUE RISK ALERTS (Next 7 days) ===\n")

    # Print MoH-style summary
    for _, row in alerts.sort_values(["env_risk", "rain_daily_total"], ascending=[False, False]).iterrows():
        print("-----------------------------")
        print(f"District     : {row['district']}")
        print(f"Date         : {row['date']}")
        print(f"Rain (mm)    : {row['rain_daily_total']:.1f}")
        print(f"Temp (avg)   : {row['temp_daily_avg']:.1f} °C")
        print(f"Humidity     : {row['humidity_daily_avg']:.1f}%")
        print(f"Risk Level   : {row['env_risk']}")

        if row["env_risk"] == "environment_high":
            print("⚠️ ACTION: High dengue weather risk – activate vector control immediately.")
        elif row["env_risk"] == "environment_moderate":
            print("⚠️ ACTION: Moderate risk – start pre-emptive clean-up and awareness.")

    print("\n======================================================\n")


if __name__ == "__main__":
    main()
