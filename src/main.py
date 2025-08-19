import argparse
import os
import json

from fetch_filing import fetch_filing
from parse_filing import clean_text, split_sections
from schema import create_filing_json

def main(cik, accession, company, filing_type, filing_date):
    # Step 1: Fetch raw filing
    raw_path = fetch_filing(cik, accession)

    # Step 2: Read + clean
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    cleaned = clean_text(raw_text)

    # Step 3: Split into sections
    sections = split_sections(cleaned)

    # Step 4: Convert to structured JSON
    structured = create_filing_json(company, cik, filing_type, filing_date, sections)

    # Step 5: Save JSON
    os.makedirs("../data/structured/", exist_ok=True)
    out_path = os.path.join("../data/structured/", f"{cik}_{accession}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2)

    print(f"✅ Structured filing saved at {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEC Filing Parser")
    parser.add_argument("--cik", required=True, help="Company CIK number")
    parser.add_argument("--accession", required=True, help="Accession number (with dashes)")
    parser.add_argument("--company", required=True, help="Company name")
    parser.add_argument("--filing_type", required=True, help="Filing type, e.g., 10-K, 10-Q")
    parser.add_argument("--filing_date", required=True, help="Filing date, YYYY-MM-DD")

    args = parser.parse_args()
    main(args.cik, args.accession, args.company, args.filing_type, args.filing_date)
