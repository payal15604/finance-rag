# Enhanced Finance RAG System
# File: src/enhanced_main.py

import argparse
import os
import json
import logging
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from dataclasses import dataclass

from fetch_filing import fetch_filing
from parse_filing import clean_text, split_sections

@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata"""
    company: str
    cik: str
    filing_type: str
    filing_date: str
    section: str
    chunk_index: int
    text: str
    embedding: Optional[np.ndarray] = None
    doc_id: Optional[str] = None

class FinanceRAGSystem:
    """Enhanced Finance RAG system with vector storage and retrieval"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.chunks: List[DocumentChunk] = []
        self.faiss_index = None
        self.chunk_metadata = {}
        
    def process_filing(self, cik: str, accession: str, company: str, 
                      filing_type: str, filing_date: str, 
                      chunk_size: int = 1000, overlap: int = 200) -> List[DocumentChunk]:
        """Process a single SEC filing into chunks"""
        
        print(f"🔄 Processing {company} ({filing_type}) filing...")
        
        # Step 1: Fetch filing
        try:
            raw_path = fetch_filing(cik, accession)
            print(f"✅ Downloaded filing to {raw_path}")
        except Exception as e:
            logging.error(f"Failed to fetch filing: {e}")
            return []

        # Step 2: Clean and parse
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            cleaned = clean_text(raw_text)
            sections = split_sections(cleaned)
            print(f"✅ Parsed {len(sections)} sections")
        except Exception as e:
            logging.error(f"Failed to parse filing: {e}")
            return []

        # Step 3: Create chunks
        filing_chunks = []
        doc_id = f"{cik}_{accession}"
        
        for section in sections:
            section_name = section.get("section", "Unknown")
            section_text = section.get("text", "")
            
            if len(section_text.strip()) < 100:  # Skip very short sections
                continue
                
            chunks = self._chunk_text(section_text, chunk_size, overlap)
            
            for idx, chunk_text in enumerate(chunks):
                chunk = DocumentChunk(
                    company=company,
                    cik=cik,
                    filing_type=filing_type,
                    filing_date=filing_date,
                    section=section_name,
                    chunk_index=idx,
                    text=chunk_text,
                    doc_id=f"{doc_id}_s{len(filing_chunks)}"
                )
                filing_chunks.append(chunk)
        
        print(f"✅ Created {len(filing_chunks)} chunks")
        return filing_chunks
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i:i+chunk_size]
            chunks.append(' '.join(chunk_words))
            if i + chunk_size >= len(words):
                break
            i += chunk_size - overlap
        return chunks
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add chunks to the system and generate embeddings"""
        print(f"🔄 Generating embeddings for {len(chunks)} chunks...")
        
        # Generate embeddings in batches for efficiency
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, batch_size=32, show_progress_bar=True)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            
        self.chunks.extend(chunks)
        self._rebuild_faiss_index()
        print(f"✅ Added {len(chunks)} chunks to system")
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index with current chunks"""
        if not self.chunks:
            return
            
        embeddings = np.array([chunk.embedding for chunk in self.chunks])
        dimension = embeddings.shape[1]
        
        # Create FAISS index
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
        
        # Update metadata mapping
        self.chunk_metadata = {i: chunk for i, chunk in enumerate(self.chunks)}
        
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        if self.faiss_index is None:
            return []
            
        # Embed query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.faiss_index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunk_metadata[idx]
                results.append({
                    'chunk': chunk,
                    'score': float(score),
                    'text': chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
                    'metadata': {
                        'company': chunk.company,
                        'filing_type': chunk.filing_type,
                        'section': chunk.section,
                        'filing_date': chunk.filing_date
                    }
                })
        return results
    
    def save_system(self, filepath: str):
        """Save the entire RAG system to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_data = {
            'chunks': self.chunks,
            'embedding_model_name': self.embedding_model._modules['0'].transformer.config.name_or_path,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
            
        # Save FAISS index separately
        if self.faiss_index:
            faiss.write_index(self.faiss_index, filepath.replace('.pkl', '_faiss.index'))
        
        print(f"✅ System saved to {filepath}")
    
    def load_system(self, filepath: str):
        """Load RAG system from disk"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.chunks = save_data['chunks']
        
        # Load FAISS index
        faiss_path = filepath.replace('.pkl', '_faiss.index')
        if os.path.exists(faiss_path):
            self.faiss_index = faiss.read_index(faiss_path)
            self.chunk_metadata = {i: chunk for i, chunk in enumerate(self.chunks)}
        
        print(f"✅ System loaded from {filepath}")

# Enhanced Stock Data Integration
class StockDataProcessor:
    """Handles stock price data integration"""
    
    @staticmethod
    def fetch_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch stock data using yfinance"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except ImportError:
            print("❌ yfinance not installed. Run: pip install yfinance")
            return pd.DataFrame()
    
    @staticmethod
    def create_stock_summary(symbol: str, data: pd.DataFrame) -> str:
        """Create a text summary of stock performance"""
        if data.empty:
            return f"No data available for {symbol}"
        
        latest = data.iloc[-1]
        start = data.iloc[0]
        
        total_return = (latest['Close'] - start['Close']) / start['Close'] * 100
        volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized
        
        summary = f"""
        Stock Analysis for {symbol}:
        - Current Price: ${latest['Close']:.2f}
        - Total Return: {total_return:.1f}%
        - Annualized Volatility: {volatility:.1f}%
        - Average Volume: {data['Volume'].mean():.0f}
        - 52-Week High: ${data['High'].max():.2f}
        - 52-Week Low: ${data['Low'].min():.2f}
        """
        return summary.strip()

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Enhanced Finance RAG System")
    parser.add_argument("--mode", choices=['process', 'search', 'build'], required=True,
                       help="Mode: process new filing, search existing, or build system")
    parser.add_argument("--cik", help="Company CIK number")
    parser.add_argument("--accession", help="Accession number")
    parser.add_argument("--company", help="Company name")
    parser.add_argument("--filing_type", help="Filing type (10-K, 10-Q)")
    parser.add_argument("--filing_date", help="Filing date (YYYY-MM-DD)")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--save_path", default="../models/finance_rag_system.pkl", 
                       help="Path to save/load system")
    
    args = parser.parse_args()
    
    # Initialize system
    rag_system = FinanceRAGSystem()
    
    if args.mode == 'process':
        # Process new filing
        if not all([args.cik, args.accession, args.company, args.filing_type, args.filing_date]):
            print("❌ Missing required arguments for processing")
            return
            
        chunks = rag_system.process_filing(
            args.cik, args.accession, args.company, 
            args.filing_type, args.filing_date
        )
        
        if chunks:
            rag_system.add_chunks(chunks)
            rag_system.save_system(args.save_path)
            print(f"✅ Processed and saved {len(chunks)} chunks")
    
    elif args.mode == 'search':
        # Search existing system
        if not args.query:
            print("❌ Query required for search mode")
            return
            
        try:
            rag_system.load_system(args.save_path)
            results = rag_system.search(args.query)
            
            print(f"\n🔍 Search Results for: '{args.query}'\n")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['metadata']['company']} - {result['metadata']['section']}")
                print(f"   Filing: {result['metadata']['filing_type']} ({result['metadata']['filing_date']})")
                print(f"   Score: {result['score']:.3f}")
                print(f"   Text: {result['text']}\n")
                
        except FileNotFoundError:
            print(f"❌ System file not found: {args.save_path}")
    
    elif args.mode == 'build':
        # Build system from multiple filings (example)
        print("🏗️ Building comprehensive system...")
        
        # Example companies to process
        companies = [
            {"cik": "1318605", "company": "Tesla Inc", "symbol": "TSLA"},
            {"cik": "320193", "company": "Apple Inc", "symbol": "AAPL"},
            {"cik": "789019", "company": "Microsoft Corp", "symbol": "MSFT"}
        ]
        
        # Note: You'd need to find actual accession numbers for recent filings
        # This is just a framework
        print("⚠️ Build mode requires manual accession numbers")
        print("Use SEC EDGAR search to find recent 10-K/10-Q accession numbers")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()