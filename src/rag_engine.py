from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from embedding_engine import EmbeddingEngine
from typing import List, Dict, Any
import logging
import os
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGEngine:
    """
    Financial-specific RAG pipeline for querying financial documents.
    Optimized for financial terminology and metrics analysis.
    """
    
    def __init__(self, docs: List, 
                model: str = "ft:gpt-3.5-turbo-0125:handelsbanken::BIB4bXrP",  # Use the fine-tuned model by default
                embeddings = None):
        """
        Initializes the RAG pipeline with document chunks optimized for financial queries.
        
        Args:
            docs: List of document chunks
            model: Model to use for generation (defaults to the fine-tuned model)
            embeddings: Embedding function to use (if None, will use default)
        """
        self.logger = logging.getLogger(__name__)
        
        try:
            if embeddings is None:
                # Use specialized financial embeddings if not provided
                embedding_engine = EmbeddingEngine()
                embeddings = embedding_engine.get_embeddings()
            
            # Create vector store with financial documents
            self.vector_store = Chroma.from_documents(docs, embeddings)
            
            # Initialize language model with financial expertise
            self.llm = ChatOpenAI(
                temperature=0,
                model=model,  # Using the fine-tuned model
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Setup financial RAG pipeline
            template = """
            You are a financial expert analyzing documents.
            Answer the question based only on the following financial context.
            Format financial values consistently with dollar signs and decimal points where appropriate.
            If the answer is not in the context, say that you don't have that information.
            Highlight key financial metrics and trends when relevant.
            
            Context: {context}
            Question: {query}
            """
            
            self.prompt = PromptTemplate.from_template(template)
            
            # Financial RAG chain
            self.qa_chain = (
                {"context": self.vector_store.as_retriever(), "query": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG: {str(e)}")
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
            
            # Use MMR retrieval for financial documents to balance relevance with diversity
            retrieved_docs = self.vector_store.similarity_search_with_relevance_scores(
                expanded_query, 
                k=4
            )
            
            # Format the context with financial document metadata
            context_texts = []
            sources = []
            
            for doc, score in retrieved_docs:
                if score < 0.65:  # Only use relevant results for financial accuracy
                    continue
                    
                context_texts.append(doc.page_content)
                sources.append({
                    "content": doc.page_content,
                    "page": doc.metadata.get("page", "Unknown"),
                    "company": doc.metadata.get("company", "Unknown"),
                    "date": doc.metadata.get("date", "Unknown"),
                    "report_type": doc.metadata.get("report_type", "Unknown")
                })
            
            if not context_texts:
                return {"answer": "I couldn't find relevant information to answer your question about Tesla's finances.", "sources": []}
            
            # Format prompt with financial context
            prompt = f"""
            You are a financial expert assistant analyzing documents. 
            Answer the question based on the following financial context. 
            Format financial values consistently with dollar signs and decimal points where appropriate.
            If the answer is not in the context, say that you don't have that information.
            Highlight key financial metrics and trends when relevant.
            
            Context:
            {' '.join(context_texts)}
            
            Question: {question}
            """
            
            # Generate answer using the fine-tuned model
            response = self.llm.generate([prompt])
            answer = response.generations[0][0].text.strip()
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            return {"answer": f"Query failed: {str(e)}", "sources": []}
    
    def expand_financial_query(self, query: str) -> str:
        """
        Expands financial queries with domain-specific terminology.
        
        Args:
            query: Original financial query
            
        Returns:
            Expanded query with additional financial terms
        """
        # Financial term expansion dictionary
        financial_expansions = {
            "revenue": ["income", "sales", "top line"],
            "profit": ["earnings", "net income", "bottom line", "margin"],
            "eps": ["earnings per share", "income per share"],
            "margin": ["profit margin", "gross margin", "operating margin"],
            "debt": ["liabilities", "leverage", "borrowings"],
            "assets": ["resources", "holdings", "property"],
            "guidance": ["forecast", "outlook", "projection", "estimate"],
            "dividend": ["payout", "distribution"],
            "cost": ["expense", "expenditure"],
            "growth": ["increase", "expansion", "rise", "improvement"]
        }
        
        expanded_terms = query
        for term, expansions in financial_expansions.items():
            if term.lower() in query.lower():
                for expansion in expansions:
                    if expansion.lower() not in query.lower():
                        expanded_terms += f" OR {expansion}"
        
        return expanded_terms 