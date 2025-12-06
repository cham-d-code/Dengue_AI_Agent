import os
import pandas as pd

DENGUE_FILE = "data_processed/dengue_weekly_district_2021_2024.csv"
WEATHER_FILE = "data_processed/weather_weekly_all_districts_2010_2024.csv"
OUT_FILE = "data_processed/weekly_district_dataset_2021_2024.csv"


def main():
    if not os.path.exists(DENGUE_FILE):
        raise FileNotFoundError(f"Missing dengue file: {DENGUE_FILE}")
    if not os.path.exists(WEATHER_FILE):
        raise FileNotFoundError(f"Missing weather file: {WEATHER_FILE}")

    dengue = pd.read_csv(DENGUE_FILE)
    weather = pd.read_csv(WEATHER_FILE)

    dengue["district"] = dengue["district"].str.strip()
    weather["district"] = weather["district"].str.strip()

    # Merge on district + year + week
    df = dengue.merge(
        weather,
        on=["district", "year", "week"],
        how="left"
    )

    # Optionally drop rows without weather
    df = df.dropna(subset=["rainfall_mm", "temp_avg_c"])

    df.to_csv(OUT_FILE, index=False)
    print(f"🎉 Saved combined dataset → {OUT_FILE}")
    print(df.head())


if __name__ == "__main__":
    main()
