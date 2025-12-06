# scripts/combine_weather_nasa.py

import os
import glob
import pandas as pd

IN_DIR = "data_raw/weather_nasa"
OUT_DAILY = "data_interim/weather_daily_all_districts_2010_2024.csv"
OUT_WEEKLY = "data_processed/weather_weekly_all_districts_2010_2024.csv"

os.makedirs("data_interim", exist_ok=True)
os.makedirs("data_processed", exist_ok=True)


def main():
    files = glob.glob(f"{IN_DIR}/nasa_daily_*.csv")
    if not files:
        raise RuntimeError("❌ No NASA files found. Run download_nasa_power.py first.")

    dfs = [pd.read_csv(f, parse_dates=["date"]) for f in files]
    daily = pd.concat(dfs, ignore_index=True)
    daily.sort_values(["district", "date"], inplace=True)
    daily.to_csv(OUT_DAILY, index=False)

    daily["year"] = daily["date"].dt.isocalendar().year.astype(int)
    daily["week"] = daily["date"].dt.isocalendar().week.astype(int)

    weekly = (
        daily.groupby(["district", "year", "week"], as_index=False)
        .agg({
            "PRECTOTCORR": "sum",
            "T2M": "mean",
            "RH2M": "mean"
        })
        .rename(columns={
            "PRECTOTCORR": "rainfall_mm",
            "T2M": "temp_avg_c",
            "RH2M": "humidity_pct",
        })
    )

    weekly.to_csv(OUT_WEEKLY, index=False)
    print(f"🎉 Saved weekly weather → {OUT_WEEKLY}")


if __name__ == "__main__":
    main()
