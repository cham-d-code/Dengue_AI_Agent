import os
import time
from typing import Optional
import requests
from urllib.parse import urljoin, urlparse, parse_qs, unquote

from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Year → category URL on NDCU site
YEAR_URLS = {
    2021: "https://dengue.health.gov.lk/web/index.php/en/publication-and-resources/publications/category/21-2021",
    2022: "https://dengue.health.gov.lk/web/index.php/en/publication-and-resources/publications/category/22-2022",
    2023: "https://dengue.health.gov.lk/web/index.php/en/publication-and-resources/publications/category/23-2023",
    2024: "https://dengue.health.gov.lk/web/index.php/en/publication-and-resources/publications/category/26-2024",
    2025: "https://dengue.health.gov.lk/web/index.php/en/publication-and-resources/publications/category/27-2025",
}

OUT_DIR = "data_raw/dengue_pdfs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- HTTP session with retries ---------- #

def make_session() -> requests.Session:
    session = requests.Session()
    retry_cfg = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2.0,          # 1st retry after 2s, then 4s, 8s, ...
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET", "HEAD"]
    )
    adapter = HTTPAdapter(max_retries=retry_cfg)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

SESSION = make_session()
TIMEOUT = 120  # seconds


def get_soup(url: str) -> Optional[BeautifulSoup]:
    print(f"Fetching page: {url}")
    try:
        r = SESSION.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"  ❌ Failed to fetch page {url}: {e}")
        return None


def extract_pdf_links_from_page(soup: BeautifulSoup, base_url: str):
    """
    Weekly dengue update PDFs appear as links with ?download= in href.
    """
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "download=" in href:
            links.append(urljoin(base_url, href))

    # de-duplicate while keeping order
    return list(dict.fromkeys(links))


def make_filename_from_url(url: str, year_hint: int | None = None) -> str:
    """
    Generate a readable filename, e.g.
    URL ...?download=241%3A2023-week-25  →  2023-week-25.pdf
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    download_param = qs.get("download", ["file"])[0]
    decoded = unquote(download_param)

    # Often like "241:2023-week-25"
    name_part = decoded.split(":")[-1].strip()

    # Add year hint if we didn't get anything meaningful
    if not name_part or name_part.lower() == "file":
        base = os.path.basename(parsed.path) or "file"
        name_part = f"{year_hint}-{base}"

    if not name_part.lower().endswith(".pdf"):
        name_part += ".pdf"

    # Make it filesystem-safe
    name_part = name_part.replace("/", "_").replace("\\", "_")
    return name_part


def find_next_page_url(soup: BeautifulSoup, current_url: str) -> Optional[str]:
    """
    If there is pagination, find a link with text 'Next' or '>' and follow it.
    """
    for a in soup.find_all("a", href=True):
        txt = a.get_text(strip=True).lower()
        if txt in ("next", ">"):
            return urljoin(current_url, a["href"])
    return None


def download_all_for_year(year: int, base_url: str):
    print(f"\n===== YEAR {year} =====")
    url = base_url
    seen_urls = set()
    downloaded = 0

    while url:
        soup = get_soup(url)
        if soup is None:
            print(f"  ⚠️ Skipping remaining pages for {year} due to repeated errors.")
            break

        pdf_urls = extract_pdf_links_from_page(soup, url)
        print(f"  Found {len(pdf_urls)} PDF links on this page.")

        for pdf_url in pdf_urls:
            if pdf_url in seen_urls:
                continue
            seen_urls.add(pdf_url)

            fname = make_filename_from_url(pdf_url, year_hint=year)
            out_path = os.path.join(OUT_DIR, fname)

            if os.path.exists(out_path):
                print(f"  ✔ Already downloaded: {fname}")
                continue

            print(f"  ⬇ Downloading: {fname}")
            try:
                r = SESSION.get(pdf_url, timeout=TIMEOUT)
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    f.write(r.content)
                downloaded += 1
            except Exception as e:
                print(f"    ❌ Failed {pdf_url}: {e}")
                continue

            time.sleep(1)  # be polite to server

        next_url = find_next_page_url(soup, url)
        if next_url and next_url != url:
            url = next_url
        else:
            url = None

    print(f"  → Year {year}: downloaded {downloaded} new PDFs")


def main():
    for year, url in YEAR_URLS.items():
        download_all_for_year(year, url)
    print(f"\n🎉 Done. All PDFs saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
