# Finance RAG Query Interface - FIXED VERSION
# File: src/query_interface.py

import os
import pickle
from typing import List, Dict
from main_2_with_vector_database import FinanceRAGSystem, StockDataProcessor
from main_2_with_vector_database import DocumentChunk

# ✅ Hugging Face integration
from transformers import pipeline

class FinanceRAGQuery:
    """Complete Finance RAG Query System"""
    
    def __init__(self, system_path: str, use_hf: bool = True):
        self.stock_processor = StockDataProcessor()

        # Load FAISS-based RAG system - FIXED to use the new load method
        if os.path.exists(system_path):
            try:
                # Use the class method from FinanceRAGSystem
                self.rag_system = FinanceRAGSystem.load_system(system_path)
                print(f"✅ Loaded FAISS system from {system_path}")
            except Exception as e:
                print(f"❌ Error loading system: {e}")
                print("💡 If this was saved with the old method, you'll need to reprocess your filings")
                raise
        else:
            print(f"⚠️ System file not found: {system_path}")
            print("💡 Creating new empty system - process some filings first!")
            self.rag_system = FinanceRAGSystem()
        
        # Setup LLM (Hugging Face)
        if use_hf:
            try:
                print("🤗 Loading Hugging Face model: google/flan-t5-small...")
                self.llm = pipeline("text2text-generation", model="google/flan-t5-base")
                self.use_llm = True
                print("✅ LLM loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load LLM: {e}")
                print("💡 Continuing without LLM - showing raw context")
                self.llm = None
                self.use_llm = False
        else:
            self.llm = None
            self.use_llm = False
            print("💡 Running without LLM - showing raw retrieved context")
    
    def query(self, question: str, include_stock_data: bool = True, k: int = 5) -> Dict:
        """Process a query and return comprehensive answer"""
        
        print(f"\n🤔 Processing query: '{question}'")
        
        # Check if system has any chunks
        if not self.rag_system.chunks:
            return {
                "question": question,
                "answer": "❌ No filings have been processed yet. Please run the system in 'process' mode first to add SEC filings.",
                "filing_sources": 0,
                "stock_data_included": False,
                "context": "",
                "sources": []
            }
        
        # Step 1: Retrieve relevant filing chunks
        filing_results = self.rag_system.search(question, k=k)
        
        # Step 2: Check if stock data is relevant
        stock_context = ""
        if include_stock_data and self._mentions_stock_symbols(question):
            symbols = self._extract_stock_symbols(question, filing_results)
            for symbol in symbols:
                stock_data = self.stock_processor.fetch_stock_data(symbol)
                if not stock_data.empty:
                    stock_context += self.stock_processor.create_stock_summary(symbol, stock_data) + "\n\n"
        
        # Step 3: Prepare context
        filing_context = self._format_filing_context(filing_results)
        full_context = f"{filing_context}\n\n{stock_context}".strip()
        
        # Step 4: Generate answer
        if self.use_llm and full_context:
            answer = self._generate_llm_answer(question, full_context)
        else:
            answer = f"📚 Retrieved {len(filing_results)} relevant sections:\n\n{filing_context[:1000]}{'...' if len(filing_context) > 1000 else ''}"
        
        return {
            "question": question,
            "answer": answer,
            "filing_sources": len(filing_results),
            "stock_data_included": bool(stock_context),
            "context": full_context,
            "sources": [r['metadata'] for r in filing_results]
        }
    
    def _mentions_stock_symbols(self, question: str) -> bool:
        """Check if question mentions stock-related terms"""
        stock_keywords = ['stock', 'price', 'performance', 'return', 'volatility', 'market']
        return any(keyword in question.lower() for keyword in stock_keywords)
    
    def _extract_stock_symbols(self, question: str, filing_results: List) -> List[str]:
        """Extract relevant stock symbols from query and filing results"""
        symbols = []
        
        # Common symbol mappings
        company_symbols = {
            'tesla': 'TSLA', 'apple': 'AAPL', 'microsoft': 'MSFT', 
            'google': 'GOOGL', 'amazon': 'AMZN', 'meta': 'META'
        }
        
        # Check question
        question_lower = question.lower()
        for company, symbol in company_symbols.items():
            if company in question_lower:
                symbols.append(symbol)
        
        # Check filing companies
        for result in filing_results:
            company = result['metadata']['company'].lower()
            for company_name, symbol in company_symbols.items():
                if company_name in company:
                    symbols.append(symbol)
                    break
        
        return list(set(symbols))  # Remove duplicates
    
    def _format_filing_context(self, results: List[Dict]) -> str:
        """Format filing results into context"""
        if not results:
            return "No relevant filing information found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            meta = result['metadata']
            context_parts.append(
                f"[Source {i}: {meta['company']} {meta['filing_type']} - {meta['section']} (Score: {result['score']:.3f})]\n"
                f"{result['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def _generate_llm_answer(self, question: str, context: str) -> str:
        """Generate answer using Hugging Face"""
        
        # Truncate context if too long
        max_context_length = 1500  # Shorter context for better processing
        if len(context) > max_context_length:
            context = context[:max_context_length] + "...[content truncated]"
        
        # Better prompt for T5 model
        prompt = f"""Answer the question based on the SEC filing context below. Be specific and cite relevant details.

Context: {context}

Question: {question}

Provide a detailed answer:"""

        try:
            response = self.llm(
                prompt, 
                max_new_tokens=500,  # More tokens for detailed answers
                truncation=True, 
                do_sample=True,      # Enable sampling for more natural responses
                temperature=0.7,     # Add some creativity
                top_p=0.9
            )
            
            generated_text = response[0]['generated_text']
            
            # If response is too short or generic, show context too
            if len(generated_text.strip()) < 20 or generated_text.lower() in ['no', 'yes', 'none']:
                return f"🤖 LLM Response: {generated_text}\n\n📚 Relevant Context:\n{context[:800]}{'...' if len(context) > 800 else ''}"
            
            return generated_text
            
        except Exception as e:
            return f"❌ Error generating LLM response: {str(e)}\n\n📚 Raw context:\n{context[:500]}..."

    def system_status(self) -> Dict:
        """Get status of the RAG system"""
        return {
            "total_chunks": len(self.rag_system.chunks),
            "has_faiss_index": self.rag_system.faiss_index is not None,
            "embedding_model_loaded": self.rag_system.embedding_model is not None,
            "llm_available": self.use_llm,
            "companies": list(set(chunk.company for chunk in self.rag_system.chunks)) if self.rag_system.chunks else []
        }

def interactive_session(system_path: str, use_hf: bool = True):
    """Run interactive query session"""
    
    try:
        query_system = FinanceRAGQuery(system_path, use_hf)
    except Exception as e:
        print(f"❌ Failed to initialize query system: {e}")
        return
    
    # Show system status
    status = query_system.system_status()
    print("\n" + "="*60)
    print("🏦 Finance RAG Query System")
    print("="*60)
    print(f"📊 System Status:")
    print(f"   • Chunks loaded: {status['total_chunks']}")
    print(f"   • FAISS index: {'✅' if status['has_faiss_index'] else '❌'}")
    print(f"   • LLM available: {'✅' if status['llm_available'] else '❌'}")
    print(f"   • Companies: {', '.join(status['companies']) if status['companies'] else 'None'}")
    
    if status['total_chunks'] == 0:
        print("\n⚠️ No filings loaded! Please process some filings first:")
        print("   python main_2_with_vector_database.py --mode process --cik 1318605 --accession 0001104659-25-042659 --company 'Tesla Inc' --filing_type '10-K' --filing_date '2025-04-26'")
        return
    
    print("\n💡 Ask questions about SEC filings and stock performance!")
    print("Examples:")
    print("- 'What risks did Tesla mention in their latest filing?'")
    print("- 'What are the main business segments?'")
    print("- 'What regulatory challenges does the company face?'")
    print("\nType 'quit' to exit.\n")
    
    while True:
        question = input("💬 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not question:
            continue
        
        if question.lower() == 'status':
            status = query_system.system_status()
            print(f"\n📊 System Status:")
            print(f"   • Total chunks: {status['total_chunks']}")
            print(f"   • Companies: {', '.join(status['companies'])}")
            continue
        
        if question.lower().startswith('debug '):
            # Debug mode - show raw search results
            debug_query = question[6:]  # Remove 'debug '
            results = query_system.rag_system.search(debug_query, k=3)
            print(f"\n🔍 Debug Search Results for: '{debug_query}'")
            print("=" * 50)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result['score']:.4f}")
                print(f"   Company: {result['metadata']['company']}")
                print(f"   Section: {result['metadata']['section']}")
                print(f"   Text preview: {result['text'][:200]}...")
            continue
        
        # Process query
        try:
            result = query_system.query(question)
            
            # Display results
            print(f"\n📊 Answer:")
            print("-" * 40)
            print(result['answer'])
            
            if result['filing_sources'] > 0:
                print(f"\n📚 Sources: {result['filing_sources']} filing sections")
                if result['stock_data_included']:
                    print("📈 Stock data included")
                
                print(f"\n🏢 Companies referenced:")
                companies = set(source['company'] for source in result['sources'])
                for company in companies:
                    print(f"  • {company}")
            
            print("\n" + "="*60)
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            print("Please try a different question.")

# Example usage and testing
def example_queries():
    """Example queries for testing"""
    return [
        "What are the main risk factors?",
        "What business segments does the company operate in?",
        "What are the competitive challenges?",
        "What regulatory risks are mentioned?",
        "How did the company perform financially?"
    ]

def test_system(system_path: str):
    """Test the system with example queries"""
    try:
        query_system = FinanceRAGQuery(system_path, use_hf=False)  # Test without LLM first
        status = query_system.system_status()
        
        print("🧪 Testing Finance RAG Query System")
        print("="*50)
        print(f"System has {status['total_chunks']} chunks from companies: {status['companies']}")
        
        if status['total_chunks'] == 0:
            print("❌ No data to test with. Process some filings first.")
            return
        
        for i, question in enumerate(example_queries()[:3], 1):  # Test first 3 queries
            print(f"\n🧪 Test {i}: {question}")
            print("-" * 30)
            result = query_system.query(question)
            print(f"Sources found: {result['filing_sources']}")
            if result['filing_sources'] > 0:
                print(f"Answer preview: {result['answer'][:200]}...")
            else:
                print("No relevant sources found")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Finance RAG Query Interface")
    parser.add_argument("--system_path", default="models/finance_rag_system.pkl",
                       help="Path to the FAISS system file")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run interactive session")
    parser.add_argument("--test", action="store_true",
                       help="Run example queries")
    parser.add_argument("--no_llm", action="store_true",
                       help="Disable LLM (faster startup)")
    
    args = parser.parse_args()
    
    use_hf = not args.no_llm
    
    if args.interactive:
        interactive_session(args.system_path, use_hf)
    elif args.test:
        test_system(args.system_path)
    else:
        print("Use --interactive or --test flag")
        print("\nQuick start:")
        print("1. Process a filing: python main_2_with_vector_database.py --mode process --cik 1318605 --accession 0001104659-25-042659 --company 'Tesla Inc' --filing_type '10-K' --filing_date '2025-04-26'")
        print("2. Query the system: python query_interface.py --interactive")