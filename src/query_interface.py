# Finance RAG Query Interface
# File: src/query_interface.py

import os
import sys
from typing import List, Dict
import openai
from main_2_with_vector_database import FinanceRAGSystem, StockDataProcessor

class FinanceRAGQuery:
    """Complete Finance RAG Query System"""
    
    def __init__(self, system_path: str, openai_api_key: str = None):
        self.rag_system = FinanceRAGSystem()
        self.stock_processor = StockDataProcessor()
        
        # Load existing system
        if os.path.exists(system_path):
            self.rag_system.load_system(system_path)
            print(f"✅ Loaded {len(self.rag_system.chunks)} chunks from system")
        else:
            print(f"⚠️ System file not found: {system_path}")
        
        # Setup OpenAI (optional)
        if openai_api_key:
            openai.api_key = openai_api_key
            self.use_llm = True
        else:
            self.use_llm = False
            print("💡 Running without LLM - showing raw retrieved context")
    
    def query(self, question: str, include_stock_data: bool = True, k: int = 5) -> Dict:
        """Process a query and return comprehensive answer"""
        
        print(f"\n🤔 Processing query: '{question}'")
        
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
            answer = "Raw context retrieved (no LLM configured):"
        
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
                f"[Source {i}: {meta['company']} {meta['filing_type']} - {meta['section']}]\n"
                f"{result['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def _generate_llm_answer(self, question: str, context: str) -> str:
        """Generate answer using OpenAI"""
        
        prompt = f"""You are a financial analyst assistant. Use the provided context from SEC filings and stock data to answer the user's question accurately and concisely.

Context:
{context}

Question: {question}

Instructions:
- Base your answer primarily on the provided context
- If the context doesn't contain enough information, say so
- Cite specific sections when relevant (e.g., "According to Tesla's 10-K Item 1A...")
- Be factual and avoid speculation
- If stock data is provided, incorporate it appropriately

Answer:"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating LLM response: {str(e)}"

def interactive_session(system_path: str, openai_key: str = None):
    """Run interactive query session"""
    
    query_system = FinanceRAGQuery(system_path, openai_key)
    
    print("\n" + "="*60)
    print("🏦 Finance RAG Query System")
    print("="*60)
    print("Ask questions about SEC filings and stock performance!")
    print("Examples:")
    print("- 'What risks did Tesla mention in their latest filing?'")
    print("- 'How is Apple's stock performing?'")
    print("- 'Compare revenue growth between companies'")
    print("\nType 'quit' to exit.\n")
    
    while True:
        question = input("💬 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not question:
            continue
        
        # Process query
        result = query_system.query(question)
        
        # Display results
        print(f"\n📊 Answer:")
        print("-" * 40)
        print(result['answer'])
        print(f"\n📚 Sources: {result['filing_sources']} filing sections")
        if result['stock_data_included']:
            print("📈 Stock data included")
        
        print(f"\n🏢 Companies referenced:")
        companies = set(source['company'] for source in result['sources'])
        for company in companies:
            print(f"  • {company}")
        print("\n" + "="*60)

# Example usage and testing
def example_queries():
    """Example queries for testing"""
    return [
        "What are the main risk factors for Tesla?",
        "How did Apple perform financially last quarter?", 
        "What business segments does Microsoft operate in?",
        "Compare the competitive landscape for tech companies",
        "What regulatory risks do these companies face?"
    ]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Finance RAG Query Interface")
    parser.add_argument("--system_path", default="../models/finance_rag_system.pkl",
                       help="Path to the RAG system file")
    parser.add_argument("--openai_key", help="OpenAI API key for LLM responses")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run interactive session")
    parser.add_argument("--test", action="store_true",
                       help="Run example queries")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_session(args.system_path, args.openai_key)
    elif args.test:
        query_system = FinanceRAGQuery(args.system_path, args.openai_key)
        for question in example_queries():
            print(f"\n{'='*60}")
            result = query_system.query(question)
            print(f"Q: {result['question']}")
            print(f"A: {result['answer'][:200]}...")
    else:
        print("Use --interactive or --test flag")