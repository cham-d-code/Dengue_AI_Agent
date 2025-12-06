import os
import time
import requests
from urllib.parse import urljoin, urlparse, parse_qs, unquote

from bs4 import BeautifulSoup

BASE_LIST_URL = "https://dengue.health.gov.lk/web/index.php/en/publication-and-resources/publications/category/26-2024"
OUT_DIR = "data_raw/dengue_pdfs"

os.makedirs(OUT_DIR, exist_ok=True)


def get_soup(url):
    print(f"Fetching page: {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def extract_pdf_links_from_page(soup, base_url):
    """
    Find all links with ?download= in href.
    These are the weekly dengue update PDFs.
    """
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "download=" in href:
            full_url = urljoin(base_url, href)
            links.append(full_url)
    return list(dict.fromkeys(links))  # dedupe while preserving order


def make_filename_from_url(url):
    """
    Try to produce a nice filename like '2024-week-15.pdf'
    from URLs like:
      ...?download=241%3A2024-week-15
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    download_param = qs.get("download", ["pdf"])[0]
    # decode URL encoding (e.g. %3A -> :)
    decoded = unquote(download_param)

    # often looks like "241:2024-week-15"
    name_part = decoded.split(":")[-1].strip()
    if not name_part:
        # fallback: use last part of path
        name_part = os.path.basename(parsed.path) or "file"

    if not name_part.lower().endswith(".pdf"):
        name_part += ".pdf"

    # ensure filesystem-safe
    name_part = name_part.replace("/", "_").replace("\\", "_")
    return name_part


def find_next_page_url(soup, current_url):
    """
    Look for a 'Next' link in the pagination.
    If not found, return None.
    """
    # Many Joomla sites use <a> with text 'Next'
    for a in soup.find_all("a", href=True):
        if a.get_text(strip=True).lower() in ("next", ">"):
            return urljoin(current_url, a["href"])
    return None


def main():
    downloaded = 0
    seen_urls = set()

    url = BASE_LIST_URL

    while url:
        soup = get_soup(url)

        pdf_urls = extract_pdf_links_from_page(soup, url)
        print(f"  Found {len(pdf_urls)} PDF links on this page.")

        for pdf_url in pdf_urls:
            if pdf_url in seen_urls:
                continue
            seen_urls.add(pdf_url)

            filename = make_filename_from_url(pdf_url)
            out_path = os.path.join(OUT_DIR, filename)

            if os.path.exists(out_path):
                print(f"✔ Already downloaded: {filename}")
                continue

            print(f"⬇ Downloading: {filename}")
            try:
                r = requests.get(pdf_url, timeout=60)
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    f.write(r.content)
                downloaded += 1
            except Exception as e:
                print(f"  ❌ Failed to download {pdf_url}: {e}")
                continue

            time.sleep(1)  # be polite to the server

        # Try to move to the next page (page 2, etc.)
        next_url = find_next_page_url(soup, url)
        if next_url and next_url != url:
            url = next_url
        else:
            url = None

    print(f"\n🎉 Done. Downloaded {downloaded} new PDF files into {OUT_DIR}")


if __name__ == "__main__":
    main()
