# src/schema.py
from pydantic import BaseModel
from typing import Dict, List

class Filing(BaseModel):
    """
    Represents a single filing for a company.

    Attributes:
        company: Name of the company (e.g., Tesla Inc)
        cik: Central Index Key of the company (string)
        filing_type: Type of filing (e.g., 10-K, 10-Q)
        filing_date: Date of the filing (YYYY-MM-DD)
        sections: Dictionary mapping section names to content
                  Example: { "Item 10.": "Directors, Executive Officers...", ... }
    """
    company: str
    cik: str
    filing_type: str
    filing_date: str
    sections: Dict[str, str]

class CompanyFilings(BaseModel):
    """
    Represents all filings for a company.

    Attributes:
        filings: List of Filing objects
    """
    filings: List[Filing]
