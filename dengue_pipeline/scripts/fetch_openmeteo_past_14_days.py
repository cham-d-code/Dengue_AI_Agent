import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.district_config import DISTRICTS

OUT_FILE = os.path.join(PROJECT_ROOT, "data_raw", "weather_past_14_days.csv")
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

def fetch_past_weather(lat, lon, start_date, end_date):
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m,rain,dew_point_2m"
        "&daily=rain_sum,temperature_2m_max,temperature_2m_min"
        f"&start_date={start_date}&end_date={end_date}"
        "&timezone=Asia%2FColombo"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    today = datetime.now().date()
    start = today - timedelta(days=14)
    end = today - timedelta(days=1)

    all_rows = []

    for district, coords in DISTRICTS.items():
        print("Fetching past weather for:", district)
        data = fetch_past_weather(coords["lat"], coords["lon"], start, end)

        if "daily" not in data:
            continue

        df = pd.DataFrame({
            "district": district,
            "date": data["daily"]["time"],
            "rain_sum": data["daily"]["rain_sum"],
            "temp_max": data["daily"]["temperature_2m_max"],
            "temp_min": data["daily"]["temperature_2m_min"],
        })
        all_rows.append(df)

    if all_rows:
        result = pd.concat(all_rows)
        result.to_csv(OUT_FILE, index=False)
        print("Saved:", OUT_FILE)
    else:
        print("❌ No data fetched")

if __name__ == "__main__":
    main()
