def create_filing_json(company: str, cik: str, filing_type: str, filing_date: str, sections: dict) -> list:
    """
    Convert sections into structured JSON format (list of dicts).
    """
    structured_data = []
    for section, content in sections.items():
        structured_data.append({
            "company": company,
            "cik": cik,
            "filing_type": filing_type,
            "filing_date": filing_date,
            "section": section,
            "content": content
        })
    return structured_data
