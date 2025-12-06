# scripts/download_nasa_power.py

import os
import sys
import time
import requests
import pandas as pd
from tqdm import tqdm

# ----------------- Make project root importable ----------------- #

# This file is in: <project_root>/scripts/download_nasa_power.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)  # one level up

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.district_config import DISTRICTS  # now this should work

# ----------------- NASA POWER config ----------------- #

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

START = "2010-01-01"
END   = "2024-12-31"

PARAMETERS = ["PRECTOTCORR", "T2M", "RH2M"]

OUT_DIR = os.path.join(PROJECT_ROOT, "data_raw", "weather_nasa")
os.makedirs(OUT_DIR, exist_ok=True)


def fetch_nasa_power(lat, lon, start, end):
    params = {
        "start": start.replace("-", ""),
        "end": end.replace("-", ""),
        "latitude": lat,
        "longitude": lon,
        "parameters": ",".join(PARAMETERS),
        "format": "JSON",
        "community": "AG",
    }

    r = requests.get(BASE_URL, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]

    dates = sorted(list(data[PARAMETERS[0]].keys()))
    rows = []

    for d in dates:
        row = {"date": pd.to_datetime(d, format="%Y%m%d")}
        for p in PARAMETERS:
            row[p] = data[p].get(d)
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("Starting NASA POWER weather download...")

    for district, coords in tqdm(DISTRICTS.items()):
        lat, lon = coords["lat"], coords["lon"]
        out_path = os.path.join(
            OUT_DIR,
            f"nasa_daily_{district.replace(' ', '_')}_2010_2024.csv"
        )

        if os.path.exists(out_path):
            print(f"✔ {district} already downloaded")
            continue

        try:
            df = fetch_nasa_power(lat, lon, START, END)
            df.insert(0, "district", district)
            df.to_csv(out_path, index=False)
        except Exception as e:
            print(f"❌ Failed for {district}: {e}")
            continue

        time.sleep(1)

    print("🎉 NASA POWER data download complete!")


if __name__ == "__main__":
    main()
