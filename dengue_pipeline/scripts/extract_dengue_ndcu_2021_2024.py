import os
import re
import glob
import pdfplumber
import pandas as pd

IN_DIR = "data_raw/dengue_pdfs"
OUT_FILE = "data_processed/dengue_weekly_district_2021_2025.csv"

os.makedirs("data_processed", exist_ok=True)


def clean_text(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def extract_year_week_from_text(text: str):
    """
    Extract year and week number from free text.
    Looks for patterns like:
      - 2023 week 15
      - Week 15 - 2023
      - WEEK: 32, 2022
    """
    year = None
    week = None

    # Year: first 20xx
    y = re.search(r"(20\d{2})", text)
    if y:
        year = int(y.group(1))

    # Week: 'Week 15', 'WEEK: 32'
    w = re.search(r"[Ww]eek[\s:]*([0-5]?\d)", text)
    if w:
        week = int(w.group(1))

    return year, week


def detect_year_week(pdf_path: str):
    """Read first page text and detect year/week."""
    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages[0]
        txt = first_page.extract_text() or ""

    year, week = extract_year_week_from_text(txt)
    return year, week


def parse_ndcu_table(df: pd.DataFrame, year: int, week: int) -> pd.DataFrame:
    """
    Try to parse a NDCU-style dengue table:
      - One row per district
      - One column 'District'
      - One main column with dengue cases (e.g. 'Cases', 'Reported cases', etc.)
    """
    df = df.copy()
    df = df.dropna(how="all")
    if df.empty:
        return pd.DataFrame()

    # Use first non-empty row as header
    header_row_idx = None
    for i, row in df.iterrows():
        if any([clean_text(c) for c in row]):
            header_row_idx = i
            break

    if header_row_idx is None:
        return pd.DataFrame()

    header = df.iloc[header_row_idx].fillna("").astype(str).tolist()
    df = df.iloc[header_row_idx + 1 :].reset_index(drop=True)
    df.columns = [clean_text(c) for c in header]

    # Try to find district column
    district_col = None
    for c in df.columns:
        if "district" in c.lower():
            district_col = c
            break
    if district_col is None:
        # maybe first column is the district
        district_col = df.columns[0]

    # Try to find dengue cases column
    dengue_col = None
    for c in df.columns:
        cl = c.lower()
        if "dengue" in cl or "cases" in cl or "no. of cases" in cl:
            # skip if it's obviously cumulative / other
            dengue_col = c
            # you can refine by checking "this week" vs "cumulative" if needed
            break

    if dengue_col is None:
        return pd.DataFrame()

    out_rows = []
    for _, row in df.iterrows():
        dist = clean_text(row.get(district_col, ""))
        if not dist:
            continue
        if dist.lower().startswith("total"):
            continue

        cell = clean_text(row.get(dengue_col, ""))
        val = re.sub(r"[^\d]", "", cell)
        if not val.isdigit():
            continue

        out_rows.append({
            "district": dist,
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
        print(f"❌ No PDFs found in {IN_DIR}")
        return

    all_rows = []

    for pdf_path in pdf_files:
        fname = os.path.basename(pdf_path)
        print(f"\n📄 Processing: {fname}")

        try:
            year, week = detect_year_week(pdf_path)
        except Exception as e:
            print(f"  ❌ Failed to read first page: {e}")
            continue

        print(f"  ➡ Detected year={year}, week={week}")
        if year is None or week is None:
            print("  ⚠️ Could not detect year/week, skipping.")
            continue

        if year < 2021 or year > 2025:
            print("  ⚠️ Year outside 2021–2025, skipping.")
            continue

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for p_i, page in enumerate(pdf.pages):
                    table = page.extract_table()
                    if not table:
                        continue
                    raw_df = pd.DataFrame(table)
                    parsed = parse_ndcu_table(raw_df, year, week)
                    if not parsed.empty:
                        print(f"  ✔ Parsed dengue table on page {p_i + 1} ({len(parsed)} rows)")
                        all_rows.append(parsed)
        except Exception as e:
            print(f"  ❌ Error reading tables in {fname}: {e}")
            continue

    if not all_rows:
        print("\n❌ No dengue tables extracted from any PDF.")
        return

    result = pd.concat(all_rows, ignore_index=True)
    result["district"] = result["district"].str.strip()

    result.to_csv(OUT_FILE, index=False)
    print("\n🎉 Extraction complete!")
    print(f"Saved: {OUT_FILE}")
    print(result.head())


if __name__ == "__main__":
    main()
