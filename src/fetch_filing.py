import os
import requests

HEADERS = {"User-Agent": "Payal <payal61503@gmail.com>"}

def fetch_filing(cik: str, accession: str, save_dir: str = "../data/raw/") -> str:
    """
    Downloads a filing from SEC EDGAR and saves it to disk.
    Returns the local filepath.
    """
    accession_nodash = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{accession}.txt"

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{cik}_{accession}.txt")

    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch filing: {url}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(resp.text)

    return filepath
