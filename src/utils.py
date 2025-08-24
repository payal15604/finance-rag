# src/utils.py
from schema import Filing, CompanyFilings
from typing import Dict, List

def create_filing_json(company: str, cik: str, filing_type: str, filing_date: str, sections: Dict[str, str]) -> dict:
    """
    Convert sections into a single structured Filing JSON object.
    """
    filing_data = Filing(
        company=company,
        cik=cik,
        filing_type=filing_type,
        filing_date=filing_date,
        sections=sections
    )
    return filing_data.dict()


def create_company_filings(filings_list: List[dict]) -> dict:
    """
    Combine multiple filings into a CompanyFilings JSON.
    """
    filings_objects = [Filing(**f) if isinstance(f, dict) else f for f in filings_list]
    company_filings = CompanyFilings(filings=filings_objects)
    return company_filings.dict()
