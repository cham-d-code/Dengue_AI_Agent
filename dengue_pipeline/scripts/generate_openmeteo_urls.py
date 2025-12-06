import os
import sys
from urllib.parse import urlencode

# ---------- Path + imports ---------- #

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.district_config import DISTRICTS  # you already have this

BASE_URL = "https://api.open-meteo.com/v1/forecast"

COMMON_PARAMS = {
    "daily": (
        "rain_sum,precipitation_sum,daylight_duration,"
        "uv_index_max,temperature_2m_max,temperature_2m_min"
    ),
    "hourly": (
        "temperature_2m,relative_humidity_2m,rain,"
        "precipitation_probability,dew_point_2m"
    ),
    "current": "temperature_2m,relative_humidity_2m,precipitation",
    "forecast_days": 7,
    "timezone": "Asia/Colombo",
}


def make_url(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        **COMMON_PARAMS,
    }
    return f"{BASE_URL}?{urlencode(params)}"


def main():
    print("Open-Meteo URLs for all districts:\n")
    for name, coords in DISTRICTS.items():
        lat = coords["lat"]
        lon = coords["lon"]
        url = make_url(lat, lon)
        print(f"{name}:")
        print(f"  {url}\n")


if __name__ == "__main__":
    main()
