# scripts/combine_dengue_weather_weekly.py

import os
import pandas as pd

DENGUE = "data_processed/dengue_weekly_district_2010_2024.csv"
WEATHER = "data_processed/weather_weekly_all_districts_2010_2024.csv"
OUT = "data_processed/weekly_district_dataset_2010_2024.csv"


def main():
    dengue = pd.read_csv(DENGUE)
    weather = pd.read_csv(WEATHER)

    dengue["district"] = dengue["district"].str.strip()
    weather["district"] = weather["district"].str.strip()

    df = dengue.merge(weather, on=["district", "year", "week"], how="left")

    df.to_csv(OUT, index=False)
    print(f"🎉 Final combined dataset saved → {OUT}")
    print(df.head())


if __name__ == "__main__":
    main()
