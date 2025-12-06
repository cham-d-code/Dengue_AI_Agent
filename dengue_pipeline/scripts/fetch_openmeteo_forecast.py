import os
import sys
import requests
import pandas as pd
from urllib.parse import urlencode

# ---------- PATH SETUP ---------- #

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.district_config import DISTRICTS  # You already have this

OUT_FILE = os.path.join(PROJECT_ROOT, "data_raw", "weather_forecast_openmeteo_7day.csv")
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ---------- API CONFIG ---------- #

BASE_URL = "https://api.open-meteo.com/v1/forecast"

COMMON_PARAMS = {
    "daily": (
        "rain_sum,precipitation_sum,daylight_duration,uv_index_max,"
        "temperature_2m_max,temperature_2m_min"
    ),
    "hourly": (
        "temperature_2m,relative_humidity_2m,rain,"
        "precipitation_probability,dew_point_2m"
    ),
    "current": "temperature_2m,relative_humidity_2m,precipitation",
    "forecast_days": 7,
    "timezone": "Asia/Colombo",
}


def build_url(lat, lon):
    params = {"latitude": lat, "longitude": lon, **COMMON_PARAMS}
    return f"{BASE_URL}?{urlencode(params)}"


def fetch_forecast(district_name, lat, lon):
    url = build_url(lat, lon)
    print(f"Fetching weather for {district_name}...")
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to fetch {district_name}: {e}")
        return None

    data = r.json()
    if "hourly" not in data or "daily" not in data:
        print(f"⚠️ Unexpected response format for {district_name}")
        return None

    # ---------- Process HOURLY data ---------- #
    h = data["hourly"]
    hourly_df = pd.DataFrame({
        "datetime": h["time"],
        "temperature_2m": h["temperature_2m"],
        "humidity_2m": h["relative_humidity_2m"],
        "rain_mm": h["rain"],
        "precipitation_probability": h.get("precipitation_probability", [None] * len(h["time"])),
        "dew_point_2m": h.get("dew_point_2m", [None] * len(h["time"])),
    })
    hourly_df["district"] = district_name
    hourly_df["date"] = pd.to_datetime(hourly_df["datetime"]).dt.date

    # ---------- Aggregate HOURLY → DAILY ---------- #
    daily_from_hourly = (
        hourly_df.groupby(["district", "date"], as_index=False)
        .agg({
            "temperature_2m": "mean",
            "humidity_2m": "mean",
            "rain_mm": "sum",
            "precipitation_probability": "mean",
            "dew_point_2m": "mean"
        })
        .rename(columns={
            "temperature_2m": "temp_daily_avg",
            "humidity_2m": "humidity_daily_avg",
            "rain_mm": "rain_daily_total",
            "precipitation_probability": "precip_prob_daily_avg",
            "dew_point_2m": "dew_point_avg",
        })
    )

        # ---------- Process DAILY data ---------- #
    d = data["daily"]
    daily_dates = pd.to_datetime(d["time"])  # DatetimeIndex

    daily_df = pd.DataFrame({
        "district": district_name,
        # DatetimeIndex has .date attribute returning array of python dates
        "date": daily_dates.date,
        "rain_sum": d["rain_sum"],
        "precipitation_sum": d["precipitation_sum"],
        "daylight_duration": d["daylight_duration"],
        "uv_index_max": d.get("uv_index_max", [None] * len(d["time"])),
        "temp_max": d["temperature_2m_max"],
        "temp_min": d["temperature_2m_min"],
    })


    # ---------- Merge daily forecast ---------- #
    merged = pd.merge(daily_df, daily_from_hourly,
                      on=["district", "date"],
                      how="left")

    return merged


def main():
    all_data = []

    for district_name, coords in DISTRICTS.items():
        df = fetch_forecast(
            district_name,
            coords["lat"],
            coords["lon"]
        )
        if df is not None:
            all_data.append(df)

    if not all_data:
        print("❌ No weather data fetched. Check API or network.")
        return

    result = pd.concat(all_data, ignore_index=True)

    # Save
    result.to_csv(OUT_FILE, index=False)

    print("\n🎉 Weather forecast (Open-Meteo, 7-day) saved to:")
    print("   ", OUT_FILE)
    print("\nPreview:")
    print(result.head())


if __name__ == "__main__":
    main()
