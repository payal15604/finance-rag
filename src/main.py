import argparse
import os
import json
import logging
from transformers import AutoTokenizer, AutoModel
import torch

from fetch_filing import fetch_filing
from parse_filing import clean_text, split_sections
from utils import create_filing_json

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        if i + chunk_size >= len(words):
            break
        i += chunk_size - overlap
    return chunks

def generate_embedding(text, tokenizer, model):
    """Generate a sentence embedding using a HuggingFace transformer model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token representation as the embedding
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding.tolist()

def main(cik, accession, company, filing_type, filing_date):
    # Step 1: Fetch raw filing
    try:
        raw_path = fetch_filing(cik, accession)
    except Exception as e:
        logging.error(f"Failed to fetch filing: {e}")
        return

    # Step 2: Read + clean
    try:
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except Exception as e:
        logging.error(f"Failed to read raw filing: {e}")
        return
    cleaned = clean_text(raw_text)

    # Step 3: Split into sections
    sections = split_sections(cleaned)

    # Step 4: Chunk sections and add metadata
    chunked_sections = []
    for section in sections:
        section_name = section.get("section", "Unknown")
        section_text = section.get("text", "")
        chunks = chunk_text(section_text)
        for idx, chunk in enumerate(chunks):
            chunked_sections.append({
                "company": company,
                "cik": cik,
                "filing_type": filing_type,
                "filing_date": filing_date,
                "section": section_name,
                "chunk_index": idx,
                "text": chunk
            })

    # Step 5: Convert to structured JSON
    structured = {"chunks": chunked_sections}

    # Step 6: Save JSON (debug: print output path and chunk count)
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "structured"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cik}_{accession}.json")
    print(f"[DEBUG] Output directory: {out_dir}")
    print(f"[DEBUG] Output file path: {out_path}")
    print(f"[DEBUG] Number of chunks: {len(chunked_sections)}")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(structured, f, indent=2)
        print(f"✅ Structured filing saved at {out_path}")
    except Exception as e:
        logging.error(f"Failed to save structured JSON: {e}")
        print(f"[ERROR] Failed to save structured JSON: {e}")

    # Step 7: Generate embeddings for each chunk
    print("[DEBUG] Loading HuggingFace model for embeddings...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    for chunk in chunked_sections:
        chunk['embedding'] = generate_embedding(chunk['text'], tokenizer, model)
    # Save or push to vector DB as needed

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="SEC Filing Parser")
    parser.add_argument("--cik", required=True, help="Company CIK number")
    parser.add_argument("--accession", required=True, help="Accession number (with dashes)")
    parser.add_argument("--company", required=True, help="Company name")
    parser.add_argument("--filing_type", required=True, help="Filing type, e.g., 10-K, 10-Q")
    parser.add_argument("--filing_date", required=True, help="Filing date, YYYY-MM-DD")

    args = parser.parse_args()
    main(args.cik, args.accession, args.company, args.filing_type, args.filing_date)
