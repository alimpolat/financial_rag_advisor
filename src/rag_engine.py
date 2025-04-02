from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from embedding_engine import EmbeddingEngine
from typing import List, Dict, Any
import logging
import os
import re


class RAGEngine:
    """
    Financial-specific RAG pipeline for querying financial documents.
    Optimized for financial terminology and metrics analysis.
    """
    
    def __init__(self, docs: List, persist_dir: str = "./chroma_db"):
        """
        Initializes the RAG pipeline with document chunks optimized for financial queries.
        
        Args:
            docs: List of document chunks
            persist_dir: Directory to persist the vector store
        """
        self.logger = logging.getLogger(__name__)
        self.embedding_engine = EmbeddingEngine()
        
        try:
            self.vectorstore = Chroma.from_documents(
                docs, 
                self.embedding_engine.embeddings, 
                persist_directory=persist_dir
            )
            
            # Use MMR retrieval for balancing relevance and diversity in financial data
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr", 
                search_kwargs={"k": 4, "fetch_k": 10}
            )
            
            self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)
            
            # Create a financial-specific prompt template
            template = """You are a financial analyst assistant. Use the following context from financial documents to answer the question accurately. When mentioning financial figures, always specify the currency, time period, and other relevant qualifiers.
            
Context: {context}

Question: {question}

Answer with factual information only. If the answer isn't in the context, say "The information is not available in the provided documents." Include relevant financial metrics when applicable."""
            
            self.prompt = PromptTemplate.from_template(template)
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            raise ValueError(f"Failed to initialize RAG pipeline: {str(e)}")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Queries the RAG system with a financial question and returns an answer with sources.
        
        Args:
            question: The financial question to answer
            
        Returns:
            Dictionary with answer and source documents
        """
        try:
            # Expand financial query for better retrieval
            expanded_query = self.expand_financial_query(question)
            self.logger.info(f"Expanded query: {expanded_query}")
            
            result = self.qa_chain({"query": expanded_query})
            
            sources = []
            for doc in result["source_documents"]:
                sources.append({
                    "content": doc.page_content[:200] + "...",
                    "page": doc.metadata.get("page", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "company": doc.metadata.get("company", "Unknown"),
                    "date": doc.metadata.get("date", "Unknown")
                })
                
            return {"answer": result["result"], "sources": sources}
            
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            return {"answer": f"Query failed: {str(e)}", "sources": []}
    
    def expand_financial_query(self, query: str) -> str:
        """
        Expands financial queries with domain-specific terms to improve retrieval.
        
        Args:
            query: Original financial query
            
        Returns:
            Expanded query with financial synonyms
        """
        # Dictionary of financial terms and their synonyms
        financial_synonyms = {
            r'\bprofit\b': ['profit', 'net income', 'earnings', 'bottom line'],
            r'\brevenue\b': ['revenue', 'sales', 'top line', 'turnover'],
            r'\bmargin\b': ['margin', 'profit margin', 'gross margin', 'operating margin'],
            r'\bgrowth\b': ['growth', 'increase', 'expansion', 'rise'],
            r'\bdebt\b': ['debt', 'liabilities', 'obligations', 'loans'],
            r'\bassets\b': ['assets', 'holdings', 'resources', 'property'],
            r'\bEPS\b': ['EPS', 'earnings per share', 'profit per share'],
            r'\bP/E\b': ['P/E', 'price to earnings', 'price-earnings ratio'],
        }
        
        expanded_query = query
        
        # Check if query contains any of our financial terms
        for term_pattern, synonyms in financial_synonyms.items():
            if re.search(term_pattern, query, re.IGNORECASE):
                # Add synonyms to the query
                synonym_string = " OR ".join([f'"{syn}"' for syn in synonyms if syn.lower() not in query.lower()])
                if synonym_string:
                    expanded_query += f" ({synonym_string})"
        
        return expanded_query 