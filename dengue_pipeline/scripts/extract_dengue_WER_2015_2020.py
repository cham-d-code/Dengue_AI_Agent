import os
import re
import glob
import pdfplumber
import pandas as pd

IN_DIR = "data_raw/dengue_pdfs"
OUT_FILE = "data_processed/dengue_weekly_district_2015_2020.csv"

os.makedirs("data_processed", exist_ok=True)

# Same district list as in config/district_config.py
DISTRICT_NAMES = [
    "Colombo", "Gampaha", "Kalutara",
    "Kandy", "Matale", "Nuwara Eliya",
    "Galle", "Matara", "Hambantota",
    "Jaffna", "Kilinochchi", "Mannar",
    "Vavuniya", "Mullaitivu",
    "Batticaloa", "Ampara", "Trincomalee",
    "Kurunegala", "Puttalam",
    "Anuradhapura", "Polonnaruwa",
    "Badulla", "Monaragala",
    "Ratnapura", "Kegalle",
]


def clean_text(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def extract_year_week(pdf_path: str, first_page_text: str):
    """Detect year & week from first-page text or filename (vol_XX)."""
    text = first_page_text or ""
    fname = os.path.basename(pdf_path)

    year = None
    week = None

    # year from text, e.g. 2015, 2018...
    y = re.search(r"(20\d{2})", text)
    if y:
        year = int(y.group(1))

    # week from text, e.g. Week 15, WEEK: 32
    w = re.search(r"[Ww]eek[\s:]*([0-5]?\d)", text)
    if w:
        week = int(w.group(1))

    # fallback year from vol_XX in filename:  vol_42 → 2014, vol_47 → 2019, etc.
    if year is None:
        m = re.search(r"vol[_\- ]?(\d+)", fname, re.IGNORECASE)
        if m:
            vol_num = int(m.group(1))
            # mapping: 42→2014, 52→2024 (volume + 1972)
            year = vol_num + 1972

    return year, week


def normalise_header_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to find a row that looks like header (contains several district names)
    and set it as columns.
    """
    df = df.copy()
    df = df.dropna(how="all").reset_index(drop=True)

    header_row_idx = None
    for idx, row in df.iterrows():
        cells = [clean_text(c) for c in row.tolist()]
        if not any(cells):
            continue
        hits = sum(1 for c in cells if any(d.lower() in c.lower() for d in DISTRICT_NAMES))
        if hits >= 3:  # row that has multiple district names
            header_row_idx = idx
            break

    if header_row_idx is None:
        # fall back to first row as header
        header_row_idx = 0

    header = df.iloc[header_row_idx].fillna("").astype(str).tolist()
    df = df.iloc[header_row_idx + 1 :].reset_index(drop=True)
    df.columns = [clean_text(c) for c in header]
    return df


def parse_wer_style_table(df: pd.DataFrame, year: int, week: int) -> pd.DataFrame:
    """
    WER layout assumption:
      - rows: diseases (Dengue fever, Chickenpox, etc.)
      - columns: districts
      - first column: Disease name / 'Notifiable disease'
      - some header cells match known district names
    We find the row where disease contains 'dengue' and extract
    case counts per district.
    """
    df = df.copy()
    df = df.dropna(how="all")
    if df.empty:
        return pd.DataFrame()

    df = normalise_header_row(df)

    cols = list(df.columns)
    if not cols:
        return pd.DataFrame()

    first_col = cols[0]

    # Which columns are districts?
    district_cols = []
    for c in cols[1:]:
        cname = clean_text(c)
        if any(d.lower() in cname.lower() for d in DISTRICT_NAMES):
            district_cols.append(c)

    if not district_cols:
        # Table probably not WER disease-by-district
        return pd.DataFrame()

    # Find row where disease name contains 'dengue'
    dengue_row = None
    for _, row in df.iterrows():
        disease_name = clean_text(row[first_col])
        if "dengue" in disease_name.lower():
            dengue_row = row
            break

    if dengue_row is None:
        return pd.DataFrame()

    out_rows = []
    for c in district_cols:
        district_header = clean_text(c)
        # choose the actual district name from header
        district = None
        for d in DISTRICT_NAMES:
            if d.lower() in district_header.lower():
                district = d
                break
        if district is None:
            continue

        cell = clean_text(dengue_row[c])
        val = re.sub(r"[^\d]", "", cell)
        if not val.isdigit():
            continue

        out_rows.append({
            "district": district,
            "year": int(year),
            "week": int(week),
            "cases": int(val),
        })

    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows)


def main():
    pdf_files = glob.glob(os.path.join(IN_DIR, "*.pdf"))
    if not pdf_files:
        print(f"❌ No PDFs in {IN_DIR}")
        return

    all_rows = []

    for pdf_path in pdf_files:
        fname = os.path.basename(pdf_path)

        # We only care about 2015–2020 for this script:
        if "vol_" in fname.lower():
            m = re.search(r"vol[_\- ]?(\d+)", fname, re.IGNORECASE)
            if m:
                vol_num = int(m.group(1))
                year_guess = vol_num + 1972
                if year_guess < 2015 or year_guess > 2020:
                    # skip volumes outside 2015–2020
                    continue

        print(f"\n📄 Processing: {fname}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                first_text = pdf.pages[0].extract_text() or ""
        except Exception as e:
            print(f"  ❌ Failed to read first page: {e}")
            continue

        year, week = extract_year_week(pdf_path, first_text)
        print(f"  ➡ Detected year={year}, week={week}")

        if year is None or week is None:
            print("  ⚠️ Missing year or week, skipping (maybe not a weekly report).")
            continue

        if year < 2015 or year > 2020:
            print("  ⚠️ Year outside 2015–2020, skipping.")
            continue

        # Now scan pages for WER tables
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for p_i, page in enumerate(pdf.pages):
                    table = page.extract_table()
                    if not table:
                        continue

                    raw_df = pd.DataFrame(table)
                    parsed = parse_wer_style_table(raw_df, year, week)
                    if not parsed.empty:
                        print(f"  ✔ Found dengue table on page {p_i + 1} ({len(parsed)} rows)")
                        all_rows.append(parsed)
        except Exception as e:
            print(f"  ❌ Error parsing tables in {fname}: {e}")
            continue

    if not all_rows:
        print("\n❌ No dengue tables extracted from any 2015–2020 WER PDF.")
        return

    result = pd.concat(all_rows, ignore_index=True)
    result["district"] = result["district"].str.strip()

    result.to_csv(OUT_FILE, index=False)
    print("\n🎉 Extraction complete!")
    print(f"Saved: {OUT_FILE}")
    print(result.head())


if __name__ == "__main__":
    main()
