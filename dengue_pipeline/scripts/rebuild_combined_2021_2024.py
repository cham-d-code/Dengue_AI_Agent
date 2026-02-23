import os
import sys
import re
import pandas as pd

# ---------- Paths ---------- #

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

DENGUE_RAW = os.path.join(PROJECT_ROOT, "data_processed", "dengue_weekly_district_2021_2024.csv")
WEATHER = os.path.join(PROJECT_ROOT, "data_processed", "weather_weekly_all_districts_2010_2024.csv")

DENGUE_CLEAN_OUT = os.path.join(PROJECT_ROOT, "data_processed", "dengue_weekly_district_clean_2021_2024.csv")
COMBINED_OUT = os.path.join(PROJECT_ROOT, "data_processed", "weekly_district_dataset_2021_2024.csv")

# ---------- Known Sri Lankan districts ---------- #

DISTRICT_NAMES = [
    "Colombo","Gampaha","Kalutara",
    "Kandy","Matale","Nuwara Eliya",
    "Galle","Matara","Hambantota",
    "Jaffna","Kilinochchi","Mannar",
    "Vavuniya","Mullaitivu",
    "Batticaloa","Ampara","Trincomalee",
    "Kurunegala","Puttalam",
    "Anuradhapura","Polonnaruwa",
    "Badulla","Monaragala",
    "Ratnapura","Kegalle",
]


def clean_text(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def extract_district(raw):
    """Find which known district name appears inside the messy string."""
    s = clean_text(raw)
    for d in DISTRICT_NAMES:
        if d.lower() in s.lower():
            return d
    return None


def main():
    if not os.path.exists(DENGUE_RAW):
        raise FileNotFoundError(f"Missing dengue file: {DENGUE_RAW}")
    if not os.path.exists(WEATHER):
        raise FileNotFoundError(f"Missing weather file: {WEATHER}")

    print(f"Loading dengue data from {DENGUE_RAW} ...")
    df = pd.read_csv(DENGUE_RAW)

    # Extract clean district name
    df["clean_district"] = df["district"].apply(extract_district)

    # Keep only rows with recognised districts
    df_clean = df[df["clean_district"].notna()].copy()

    # Group by district + year + week (in case multiple rows per district/week)
    dengue_agg = (
        df_clean
        .groupby(["clean_district", "year", "week"], as_index=False)["cases"]
        .sum()
        .rename(columns={"clean_district": "district"})
    )

    print("Dengue cleaned shape:", dengue_agg.shape)
    dengue_agg.to_csv(DENGUE_CLEAN_OUT, index=False)
    print(f"✔ Saved cleaned dengue to {DENGUE_CLEAN_OUT}")

    # Load weather
    print(f"Loading weather data from {WEATHER} ...")
    weather = pd.read_csv(WEATHER)

    dengue_agg["district"] = dengue_agg["district"].str.strip()
    weather["district"] = weather["district"].str.strip()

    # Merge dengue + weather
    combined = dengue_agg.merge(
        weather,
        on=["district", "year", "week"],
        how="left"
    )

    print("Combined shape:", combined.shape)
    combined.to_csv(COMBINED_OUT, index=False)
    print(f"🎉 Saved rebuilt combined dataset to {COMBINED_OUT}")


if __name__ == "__main__":
    main()
