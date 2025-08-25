# #!/usr/bin/env python3
# """
# Fixed Finance RAG System - Main Processing Script
# Addresses common issues and adds better error handling
# """

# import argparse
# import os
# import json
# import logging
# import pickle
# from datetime import datetime
# from typing import List, Dict, Any, Optional
# import numpy as np
# import pandas as pd
# from pathlib import Path

# # Try importing with fallbacks
# try:
#     from sentence_transformers import SentenceTransformer
#     HAS_SENTENCE_TRANSFORMERS = True
# except ImportError:
#     print("⚠️ sentence-transformers not found. Install with: pip install sentence-transformers")
#     HAS_SENTENCE_TRANSFORMERS = False

# try:
#     import faiss
#     HAS_FAISS = True
# except ImportError:
#     print("⚠️ faiss not found. Install with: pip install faiss-cpu")
#     HAS_FAISS = False

# from dataclasses import dataclass
# from fetch_filing import fetch_filing
# from parse_filing import clean_text, split_sections

# @dataclass
# class DocumentChunk:
#     """Represents a document chunk with metadata"""
#     company: str
#     cik: str
#     filing_type: str
#     filing_date: str
#     section: str
#     chunk_index: int
#     text: str
#     embedding: Optional[np.ndarray] = None
#     doc_id: Optional[str] = None

# class FinanceRAGSystem:
#     """Enhanced Finance RAG system with better error handling"""
    
#     def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
#         self.model_name = model_name
#         self.embedding_model = None
#         self.chunks: List[DocumentChunk] = []
#         self.faiss_index = None
#         self.chunk_metadata = {}
        
#         # Initialize embedding model with error handling
#         self._init_embedding_model()
        
#     def _init_embedding_model(self):
#         """Initialize embedding model with fallback"""
#         if not HAS_SENTENCE_TRANSFORMERS:
#             print("❌ Cannot initialize embedding model - sentence-transformers not available")
#             return
            
#         try:
#             self.embedding_model = SentenceTransformer(self.model_name)
#             print(f"✅ Loaded embedding model: {self.model_name}")
#         except Exception as e:
#             print(f"❌ Failed to load embedding model: {e}")
#             # Try fallback model
#             try:
#                 self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#                 print("✅ Loaded fallback embedding model: all-MiniLM-L6-v2")
#             except Exception as e2:
#                 print(f"❌ Failed to load fallback model: {e2}")
        
#     def process_filing(self, cik: str, accession: str, company: str, 
#                       filing_type: str, filing_date: str, 
#                       chunk_size: int = 1000, overlap: int = 200) -> List[DocumentChunk]:
#         """Process a single SEC filing into chunks with error handling"""
        
#         print(f"🔄 Processing {company} ({filing_type}) filing...")
        
#         # Step 1: Fetch filing
#         # try:
#         #     raw_path = fetch_filing(cik, accession)
#         #     print(f"✅ Downloaded filing to {raw_path}")
#         # except Exception as e:
#         #     logging.error(f"Failed to fetch filing: {e}")
#         #     print(f"❌ Failed to fetch filing: {e}")
#         #     return []
#         raw_path = "../data/raw/1318605_0001104659-25-042659.txt"  # For testing without fetching
#         # Step 2: Clean and parse
#         try:
#             with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:  # Added error handling
#                 raw_text = f.read()
#             cleaned = clean_text(raw_text)
#             sections = split_sections(cleaned)
#             print(f"✅ Parsed {len(sections)} sections")
#         except Exception as e:
#             logging.error(f"Failed to parse filing: {e}")
#             print(f"❌ Failed to parse filing: {e}")
#             return []

#         # Step 3: Create chunks
#         filing_chunks = []
#         doc_id = f"{cik}_{accession}"
        
#         for section in sections:
#             section_name = section.get("section", "Unknown")
#             section_text = section.get("text", "")
            
#             if len(section_text.strip()) < 100:  # Skip very short sections
#                 continue
                
#             chunks = self._chunk_text(section_text, chunk_size, overlap)
            
#             for idx, chunk_text in enumerate(chunks):
#                 chunk = DocumentChunk(
#                     company=company,
#                     cik=cik,
#                     filing_type=filing_type,
#                     filing_date=filing_date,
#                     section=section_name,
#                     chunk_index=idx,
#                     text=chunk_text,
#                     doc_id=f"{doc_id}_s{len(filing_chunks)}"
#                 )
#                 filing_chunks.append(chunk)
        
#         print(f"✅ Created {len(filing_chunks)} chunks")
#         return filing_chunks
    
#     def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
#         """Split text into overlapping chunks"""
#         if not text or not text.strip():
#             return []
            
#         words = text.split()
#         if len(words) <= chunk_size:
#             return [text]
            
#         chunks = []
#         i = 0
#         while i < len(words):
#             chunk_words = words[i:i+chunk_size]
#             chunks.append(' '.join(chunk_words))
#             if i + chunk_size >= len(words):
#                 break
#             i += chunk_size - overlap
#         return chunks
    
#     def add_chunks(self, chunks: List[DocumentChunk]):
#         """Add chunks to the system and generate embeddings"""
#         if not chunks:
#             print("⚠️ No chunks to add")
#             return
            
#         if not self.embedding_model:
#             print("❌ Cannot generate embeddings - model not loaded")
#             return
            
#         print(f"🔄 Generating embeddings for {len(chunks)} chunks...")
        
#         try:
#             # Generate embeddings in batches for efficiency
#             texts = [chunk.text for chunk in chunks]
#             embeddings = self.embedding_model.encode(texts, batch_size=32, show_progress_bar=True)
            
#             # Add embeddings to chunks
#             for chunk, embedding in zip(chunks, embeddings):
#                 chunk.embedding = embedding
                
#             self.chunks.extend(chunks)
#             self._rebuild_faiss_index()
#             print(f"✅ Added {len(chunks)} chunks to system")
#         except Exception as e:
#             print(f"❌ Failed to generate embeddings: {e}")
#             logging.error(f"Failed to generate embeddings: {e}")
    
#     def _rebuild_faiss_index(self):
#         """Rebuild FAISS index with current chunks"""
#         if not HAS_FAISS:
#             print("⚠️ FAISS not available - search will be disabled")
#             return
            
#         if not self.chunks:
#             return
            
#         try:
#             embeddings = np.array([chunk.embedding for chunk in self.chunks])
#             dimension = embeddings.shape[1]
            
#             # Create FAISS index
#             self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
            
#             # Normalize embeddings for cosine similarity
#             faiss.normalize_L2(embeddings)
#             self.faiss_index.add(embeddings)
            
#             # Update metadata mapping
#             self.chunk_metadata = {i: chunk for i, chunk in enumerate(self.chunks)}
#             print(f"✅ Built FAISS index with {len(self.chunks)} chunks")
            
#         except Exception as e:
#             print(f"❌ Failed to build FAISS index: {e}")
#             logging.error(f"Failed to build FAISS index: {e}")
            
#     def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
#         """Search for relevant chunks"""
#         if not self.faiss_index:
#             print("⚠️ No search index available")
#             return []
            
#         if not self.embedding_model:
#             print("❌ Cannot search - embedding model not loaded")
#             return []
            
#         try:
#             # Embed query
#             query_embedding = self.embedding_model.encode([query])
#             faiss.normalize_L2(query_embedding)
            
#             # Search
#             scores, indices = self.faiss_index.search(query_embedding, k)
            
#             results = []
#             for score, idx in zip(scores[0], indices[0]):
#                 if idx < len(self.chunks) and idx in self.chunk_metadata:
#                     chunk = self.chunk_metadata[idx]
#                     results.append({
#                         'chunk': chunk,
#                         'score': float(score),
#                         'text': chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
#                         'metadata': {
#                             'company': chunk.company,
#                             'filing_type': chunk.filing_type,
#                             'section': chunk.section,
#                             'filing_date': chunk.filing_date
#                         }
#                     })
#             return results
#         except Exception as e:
#             print(f"❌ Search failed: {e}")
#             logging.error(f"Search failed: {e}")
#             return []
    
#     def save_system(self, filepath: str):
#         """Save the entire RAG system to disk"""
#         try:
#             os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
#             save_data = {
#                 'chunks': self.chunks,
#                 'model_name': self.model_name,
#             }
            
#             with open(filepath, 'wb') as f:
#                 pickle.dump(save_data, f)
                
#             # Save FAISS index separately
#             if self.faiss_index:
#                 faiss_path = filepath.replace('.pkl', '_faiss.index')
#                 faiss.write_index(self.faiss_index, faiss_path)
            
#             print(f"✅ System saved to {filepath}")
#         except Exception as e:
#             print(f"❌ Failed to save system: {e}")
#             logging.error(f"Failed to save system: {e}")
            
#     @staticmethod
#     def load_system(filepath: str):
#         with open(filepath, 'rb') as f:
#             return pickle.load(f)

#     def save_system(self, filepath: str):
#         """Save RAG system to disk"""
#         try:
#             os.makedirs(os.path.dirname(filepath), exist_ok=True)

#             save_data = {
#                 'chunks': self.chunks,
#                 'model_name': self.model_name,
#             }

#             with open(filepath, 'wb') as f:
#                 pickle.dump(save_data, f)

#             # Save FAISS index separately
#             if self.faiss_index:
#                 faiss_path = filepath.replace('.pkl', '_faiss.index')
#                 faiss.write_index(self.faiss_index, faiss_path)

#             print(f"✅ System saved to {filepath}")
#         except Exception as e:
#             print(f"❌ Failed to save system: {e}")
#             logging.error(f"Failed to save system: {e}")

            
#     #         self.chunks = save_data['chunks']
            
#     #         # Load FAISS index
#     #         faiss_path = filepath.replace('.pkl', '_faiss.index')
#     #         if os.path.exists(faiss_path) and HAS_FAISS:
#     #             self.faiss_index = faiss.read_index(faiss_path)
#     #             self.chunk_metadata = {i: chunk for i, chunk in enumerate(self.chunks)}
            
#     #         print(f"✅ System loaded from {filepath} ({len(self.chunks)} chunks)")
#     #     except Exception as e:
#     #         print(f"❌ Failed to load system: {e}")
#     #         logging.error(f"Failed to load system: {e}")

# def main():
#     """Main execution function"""
#     parser = argparse.ArgumentParser(description="Finance RAG System (Fixed)")
#     parser.add_argument("--mode", choices=['process', 'search', 'test'], required=True,
#                        help="Mode: process new filing, search existing, or test system")
#     parser.add_argument("--cik", help="Company CIK number")
#     parser.add_argument("--accession", help="Accession number")
#     parser.add_argument("--company", help="Company name")
#     parser.add_argument("--filing_type", help="Filing type (10-K, 10-Q)")
#     parser.add_argument("--filing_date", help="Filing date (YYYY-MM-DD)")
#     parser.add_argument("--query", help="Search query")
#     parser.add_argument("--save_path", default="models/finance_rag_system.pkl", 
#                        help="Path to save/load system")
    
#     args = parser.parse_args()
    
#     # Setup logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler('logs/finance_rag.log'),
#             logging.StreamHandler()
#         ]
#     )
    
#     # Initialize system
#     #rag_system = FinanceRAGSystem()
#     rag_system = FinanceRAGSystem.load_system(args.save_path)

    
#     if args.mode == 'process':
#         # Process new filing
#         if not all([args.cik, args.accession, args.company, args.filing_type, args.filing_date]):
#             print("❌ Missing required arguments for processing")
#             print("Required: --cik, --accession, --company, --filing_type, --filing_date")
#             return
            
#         chunks = rag_system.process_filing(
#             args.cik, args.accession, args.company, 
#             args.filing_type, args.filing_date
#         )
        
#         if chunks:
#             rag_system.add_chunks(chunks)
#             rag_system.save_system(args.save_path)
#             print(f"✅ Processed and saved {len(chunks)} chunks")
    
#     elif args.mode == 'search':
#         # Search existing system
#         if not args.query:
#             print("❌ Query required for search mode")
#             return
            
#         try:
#             rag_system.load_system(args.save_path)
#             results = rag_system.search(args.query)
            
#             print(f"\n🔍 Search Results for: '{args.query}'\n")
#             for i, result in enumerate(results, 1):
#                 print(f"{i}. {result['metadata']['company']} - {result['metadata']['section']}")
#                 print(f"   Filing: {result['metadata']['filing_type']} ({result['metadata']['filing_date']})")
#                 print(f"   Score: {result['score']:.3f}")
#                 print(f"   Text: {result['text']}\n")
                
#         except FileNotFoundError:
#             print(f"❌ System file not found: {args.save_path}")
#             print("Run in 'process' mode first to create the system")
    
#     elif args.mode == 'test':
#         # Test system components
#         print("🧪 Testing system components...")
        
#         # Test 1: Check dependencies
#         print("\n1. Checking dependencies...")
#         deps = {
#             'sentence_transformers': HAS_SENTENCE_TRANSFORMERS,
#             'faiss': HAS_FAISS,
#         }
#         for dep, available in deps.items():
#             status = "✅" if available else "❌"
#             print(f"   {status} {dep}")
        
#         # Test 2: Test embedding model
#         print("\n2. Testing embedding model...")
#         if rag_system.embedding_model:
#             try:
#                 test_text = "This is a test sentence for embedding."
#                 embedding = rag_system.embedding_model.encode([test_text])
#                 print(f"   ✅ Generated embedding shape: {embedding.shape}")
#             except Exception as e:
#                 print(f"   ❌ Embedding test failed: {e}")
#         else:
#             print("   ❌ No embedding model available")
        
#         # Test 3: Test file structure
#         print("\n3. Checking file structure...")
#         required_dirs = ['data/raw', 'data/structured', 'models', 'logs']
#         for directory in required_dirs:
#             if os.path.exists(directory):
#                 print(f"   ✅ {directory}")
#             else:
#                 print(f"   ❌ {directory} (missing)")
# # inside main_2_with_vector_database.py

# import pandas as pd

# class StockDataProcessor:
#     """Dummy stock processor until implemented"""
#     def fetch_stock_data(self, symbol: str) -> pd.DataFrame:
#         print(f"⚠️ Stock data fetch not implemented for {symbol}")
#         return pd.DataFrame()

#     def create_stock_summary(self, symbol: str, data: pd.DataFrame) -> str:
#         return f"No stock data available for {symbol}"


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Fixed Finance RAG System - Main Processing Script
Addresses common issues and adds better error handling
"""

import argparse
import os
import json
import logging
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path

# Try importing with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    print("⚠️ sentence-transformers not found. Install with: pip install sentence-transformers")
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    print("⚠️ faiss not found. Install with: pip install faiss-cpu")
    HAS_FAISS = False

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
    """Enhanced Finance RAG system with better error handling"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model = None
        self.chunks: List[DocumentChunk] = []
        self.faiss_index = None
        self.chunk_metadata = {}
        self._init_embedding_model()
        
    def _init_embedding_model(self):
        """Initialize embedding model with fallback"""
        if not HAS_SENTENCE_TRANSFORMERS:
            print("❌ Cannot initialize embedding model - sentence-transformers not available")
            return
            
        try:
            self.embedding_model = SentenceTransformer(self.model_name)
            print(f"✅ Loaded embedding model: {self.model_name}")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                print("✅ Loaded fallback embedding model: all-MiniLM-L6-v2")
            except Exception as e2:
                print(f"❌ Failed to load fallback model: {e2}")
        
    def process_filing(self, cik: str, accession: str, company: str, 
                      filing_type: str, filing_date: str, 
                      chunk_size: int = 1000, overlap: int = 200) -> List[DocumentChunk]:
        """Process a single SEC filing into chunks with error handling"""
        print(f"🔄 Processing {company} ({filing_type}) filing...")
        
        # Step 1: Fetch filing (kept your hardcoded path)
        # raw_path = fetch_filing(cik, accession)
        raw_path = "../data/raw/1318605_0001104659-25-042659.txt"  # For testing without fetching
        
        # Step 2: Clean and parse
        try:
            with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
            cleaned = clean_text(raw_text)
            sections = split_sections(cleaned)
            print(f"✅ Parsed {len(sections)} sections")
        except Exception as e:
            logging.error(f"Failed to parse filing: {e}")
            print(f"❌ Failed to parse filing: {e}")
            return []

        # Step 3: Create chunks
        filing_chunks = []
        doc_id = f"{cik}_{accession}"
        
        for section in sections:
            section_name = section.get("section", "Unknown")
            section_text = section.get("text", "")
            if len(section_text.strip()) < 100:
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
        if not text or not text.strip():
            return []
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
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
        if not chunks:
            print("⚠️ No chunks to add")
            return
        if not self.embedding_model:
            print("❌ Cannot generate embeddings - model not loaded")
            return
            
        print(f"🔄 Generating embeddings for {len(chunks)} chunks...")
        try:
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_model.encode(texts, batch_size=32, show_progress_bar=True)
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            self.chunks.extend(chunks)
            self._rebuild_faiss_index()
            print(f"✅ Added {len(chunks)} chunks to system")
        except Exception as e:
            print(f"❌ Failed to generate embeddings: {e}")
            logging.error(f"Failed to generate embeddings: {e}")
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index with current chunks"""
        if not HAS_FAISS:
            print("⚠️ FAISS not available - search will be disabled")
            return
        if not self.chunks:
            return
        try:
            embeddings = np.array([chunk.embedding for chunk in self.chunks])
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
            faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings)
            self.chunk_metadata = {i: chunk for i, chunk in enumerate(self.chunks)}
            print(f"✅ Built FAISS index with {len(self.chunks)} chunks")
        except Exception as e:
            print(f"❌ Failed to build FAISS index: {e}")
            logging.error(f"Failed to build FAISS index: {e}")
            
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        if not self.faiss_index:
            print("⚠️ No search index available")
            return []
        if not self.embedding_model:
            print("❌ Cannot search - embedding model not loaded")
            return []
        try:
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            scores, indices = self.faiss_index.search(query_embedding, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks) and idx in self.chunk_metadata:
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
        except Exception as e:
            print(f"❌ Search failed: {e}")
            logging.error(f"Search failed: {e}")
            return []
    
    def save_system(self, filepath: str):
        """Save the entire RAG system to disk"""
        try:
            dirpath = os.path.dirname(filepath)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            save_data = {
                'chunks': self.chunks,
                'model_name': self.model_name,
            }
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            if self.faiss_index:
                faiss_path = filepath.replace('.pkl', '_faiss.index')
                faiss.write_index(self.faiss_index, faiss_path)
            print(f"✅ System saved to {filepath}")
        except Exception as e:
            print(f"❌ Failed to save system: {e}")
            logging.error(f"Failed to save system: {e}")

    @classmethod
    def load_system(cls, filepath: str) -> "FinanceRAGSystem":
        """
        Load a system from disk. If missing, return a fresh initialized system.
        Rebuilds FAISS from saved index or from embeddings if needed.
        """
        if not os.path.exists(filepath):
            print(f"⚠️ System file not found at {filepath}. Starting with an empty system.")
            return cls()
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)  # dict with chunks + model_name
            system = cls(model_name=save_data.get('model_name', "sentence-transformers/all-MiniLM-L6-v2"))
            system.chunks = save_data.get('chunks', [])
            # Try to load FAISS index; if not there, rebuild from embeddings
            faiss_path = filepath.replace('.pkl', '_faiss.index')
            if os.path.exists(faiss_path) and HAS_FAISS:
                system.faiss_index = faiss.read_index(faiss_path)
                system.chunk_metadata = {i: ch for i, ch in enumerate(system.chunks)}
                print(f"✅ System loaded from {filepath} ({len(system.chunks)} chunks) with FAISS index")
            else:
                if system.chunks and HAS_FAISS:
                    system._rebuild_faiss_index()
                    print(f"✅ System loaded from {filepath} ({len(system.chunks)} chunks) and rebuilt FAISS")
                else:
                    print(f"✅ System loaded from {filepath} ({len(system.chunks)} chunks) (no FAISS available)")
            return system
        except Exception as e:
            print(f"❌ Failed to load system: {e}")
            logging.error(f"Failed to load system: {e}")
            # Fall back to empty system so the process mode can still run
            return cls()

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Finance RAG System (Fixed)")
    parser.add_argument("--mode", choices=['process', 'search', 'test'], required=True,
                       help="Mode: process new filing, search existing, or test system")
    parser.add_argument("--cik", help="Company CIK number")
    parser.add_argument("--accession", help="Accession number")
    parser.add_argument("--company", help="Company name")
    parser.add_argument("--filing_type", help="Filing type (10-K, 10-Q)")
    parser.add_argument("--filing_date", help="Filing date (YYYY-MM-DD)")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--save_path", default="models/finance_rag_system.pkl", 
                       help="Path to save/load system")
    args = parser.parse_args()

    # Ensure logs folder exists before FileHandler
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/finance_rag.log'),
            logging.StreamHandler()
        ]
    )
    
    # Always load (fresh if not present)
    rag_system = FinanceRAGSystem.load_system(args.save_path)

    if args.mode == 'process':
        if not all([args.cik, args.accession, args.company, args.filing_type, args.filing_date]):
            print("❌ Missing required arguments for processing")
            print("Required: --cik, --accession, --company, --filing_type, --filing_date")
            return
        chunks = rag_system.process_filing(
            args.cik, args.accession, args.company, 
            args.filing_type, args.filing_date
        )
        if chunks:
            rag_system.add_chunks(chunks)
            rag_system.save_system(args.save_path)
            print(f"✅ Processed and saved {len(chunks)} chunks")
        return
    
    if args.mode == 'search':
        if not args.query:
            print("❌ Query required for search mode")
            return
        results = rag_system.search(args.query)
        print(f"\n🔍 Search Results for: '{args.query}'\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['metadata']['company']} - {result['metadata']['section']}")
            print(f"   Filing: {result['metadata']['filing_type']} ({result['metadata']['filing_date']})")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Text: {result['text']}\n")
        return
    
    if args.mode == 'test':
        print("🧪 Testing system components...")
        print("\n1. Checking dependencies...")
        deps = {
            'sentence_transformers': HAS_SENTENCE_TRANSFORMERS,
            'faiss': HAS_FAISS,
        }
        for dep, available in deps.items():
            status = "✅" if available else "❌"
            print(f"   {status} {dep}")
        
        print("\n2. Testing embedding model...")
        if rag_system.embedding_model:
            try:
                test_text = "This is a test sentence for embedding."
                embedding = rag_system.embedding_model.encode([test_text])
                print(f"   ✅ Generated embedding shape: {embedding.shape}")
            except Exception as e:
                print(f"   ❌ Embedding test failed: {e}")
        else:
            print("   ❌ No embedding model available")
        
        print("\n3. Checking file structure...")
        required_dirs = ['data/raw', 'data/structured', 'models', 'logs']
        for directory in required_dirs:
            if os.path.exists(directory):
                print(f"   ✅ {directory}")
            else:
                print(f"   ❌ {directory} (missing)")

# inside main_2_with_vector_database.py

import pandas as pd

class StockDataProcessor:
    """Dummy stock processor until implemented"""
    def fetch_stock_data(self, symbol: str) -> pd.DataFrame:
        print(f"⚠️ Stock data fetch not implemented for {symbol}")
        return pd.DataFrame()

    def create_stock_summary(self, symbol: str, data: pd.DataFrame) -> str:
        return f"No stock data available for {symbol}"

if __name__ == "__main__":
    main()
