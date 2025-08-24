import re
import json
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    """Remove HTML tags and normalize spaces."""
    soup = BeautifulSoup(text, "lxml")
    clean = soup.get_text(separator=" ")
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()

def split_sections(text: str) -> list:
    """
    Split filing into sections (Item 1, Item 1A, Item 7, etc.)
    Returns a list of dicts: [{"section": section_title, "text": section_text}, ...]
    """
    pattern = r"(Item\s+[0-9A-Za-z\.]+)"
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    sections = []
    for i in range(1, len(parts), 2):  # every odd index is a header
        header = parts[i].strip()
        content = parts[i+1].strip() if i+1 < len(parts) else ""
        sections.append({"section": header, "text": content})

    return sections
