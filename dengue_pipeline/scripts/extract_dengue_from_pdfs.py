import os
import re
import glob
import pdfplumber
import pandas as pd

# Folder with all WER / flashback / dengue update PDFs
IN_DIR = "data_raw/dengue_pdfs"
OUT_FILE = "data_processed/dengue_weekly_district_2010_2024.csv"

os.makedirs("data_processed", exist_ok=True)


# ------------- Helpers ---------------- #

def clean_text(s: str) -> str:
    """Basic whitespace cleanup."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", s.strip())


def extract_year_week(pdf_path: str, first_page_text: str):
    """
    Extract YEAR and WEEK from:
      1) First page text
      2) 'vol_XX' pattern in filename as fallback for YEAR

    Rules:
      - Year from text: first 20xx we see
      - Week from text: patterns like 'Week 15', 'WEEK: 32'
      - If year missing, use vol mapping:
            year = volume_number + 1972
        e.g. vol_42 -> 2014, vol_52 -> 2024
    """
    text = first_page_text or ""
    fname = os.path.basename(pdf_path)

    year = None
    week = None

    # 1) Try to detect year from text
    y = re.search(r"(20\d{2})", text)
    if y:
        year = int(y.group(1))

    # 2) Detect week from text (Week 1–53)
    w = re.search(r"[Ww]eek[\s:]*([0-5]?\d)", text)
    if w:
        week = int(w.group(1))

    # 3) If year still missing, try from vol_XX pattern in filename
    if year is None:
        m = re.search(r"vol[_\- ]?(\d+)", fname, re.IGNORECASE)
        if m:
            vol_num = int(m.group(1))
            # Your mapping: vol 42 -> 2014, vol 52 -> 2024
            year = vol_num + 1972

    return year, week


def parse_dengue_table(df: pd.DataFrame, year: int, week: int) -> pd.DataFrame:
    """
    Convert a raw extracted table into:
      columns: district, year, week, cases
    Assumptions:
      - First row is header
      - First column is district name
      - One column has 'dengue' in its header (dengue cases)
    """
    df = df.copy()
    df = df.dropna(how="all")
    if df.empty:
        return pd.DataFrame()

    # Use first row as header
    header = df.iloc[0].fillna("").astype(str)
    df = df[1:]
    df.columns = header

    # Find dengue column
    dengue_col = None
    for c in df.columns:
        if "dengue" in str(c).lower():
            dengue_col = c
            break

    if dengue_col is None:
        return pd.DataFrame()

    # Assume first column is district
    district_col = df.columns[0]

    out_rows = []
    for _, row in df.iterrows():
        dist_raw = clean_text(str(row[district_col]))
        if not dist_raw:
            continue
        if dist_raw.lower().startswith("total"):
            continue

        # Extract digits from dengue cell
        cell = str(row[dengue_col])
        val = re.sub(r"[^\d]", "", cell)
        if not val.isdigit():
            continue

        out_rows.append({
            "district": dist_raw,
            "year": int(year),
            "week": int(week),
            "cases": int(val),
        })

    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows)


# ------------- Main pipeline ---------------- #

def main():
    pdf_files = glob.glob(os.path.join(IN_DIR, "*.pdf"))
    if not pdf_files:
        print(f"❌ No PDFs found in {IN_DIR}")
        return

    all_rows = []

    for pdf_path in pdf_files:
        fname = os.path.basename(pdf_path)
        print(f"\n📄 Processing: {fname}")

        # --- Read first page text to detect year & week ---
        try:
            with pdfplumber.open(pdf_path) as pdf:
                first_page = pdf.pages[0]
                first_text = first_page.extract_text() or ""
        except Exception as e:
            print(f"  ❌ Failed to read first page: {e}")
            continue

        year, week = extract_year_week(pdf_path, first_text)
        print(f"  ➡ Detected Year: {year}, Week: {week}")

        # Skip PDFs where we cannot confidently get both
        if year is None or week is None:
            print("  ⚠️ Missing year or week, skipping this PDF (probably annual summary).")
            continue

        # --- Extract tables from all pages ---
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    table = page.extract_table()
                    if not table:
                        continue

                    df_raw = pd.DataFrame(table)
                    parsed = parse_dengue_table(df_raw, year, week)
                    if not parsed.empty:
                        print(f"  ✔ Found dengue table on page {page_idx + 1} with {len(parsed)} rows")
                        all_rows.append(parsed)
        except Exception as e:
            print(f"  ❌ Error parsing tables: {e}")
            continue

    if not all_rows:
        print("\n❌ No dengue tables extracted from any PDF.")
        return

    result = pd.concat(all_rows, ignore_index=True)

    # Optional: clean whitespace in district names
    result["district"] = result["district"].str.strip()

    result.to_csv(OUT_FILE, index=False)

    print("\n🎉 Extraction complete!")
    print(f"Saved: {OUT_FILE}")
    print(result.head())


if __name__ == "__main__":
    main()
