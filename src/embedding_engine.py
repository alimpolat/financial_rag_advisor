from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Any
import logging
import os
import re


class EmbeddingEngine:
    """
    Specialized embedding engine for financial text.
    Optimized for financial terminology and numerical data.
    """
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initializes the embedding engine with optimal settings for financial text.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.logger = logging.getLogger(__name__)
        
        try:
            # Simple initialization - the latest version has different parameters
            self.embeddings = OpenAIEmbeddings(model=model_name)
        except Exception as e:
            self.logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise ValueError(f"Failed to initialize embeddings: {str(e)}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts.
        Preprocesses financial text before embedding for better results.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Preprocess texts for financial context
            processed_texts = [self.preprocess_financial_text(text) for text in texts]
            return self.embeddings.embed_documents(processed_texts)
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            raise ValueError(f"Embedding generation failed: {str(e)}")
    
    def preprocess_financial_text(self, text: str) -> str:
        """
        Preprocesses financial text to improve embedding quality.
        Normalizes financial notation, numbers, and abbreviations.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text optimized for financial embeddings
        """
        if not text:
            return text
            
        # Remove currency symbols, standardize number formats
        text = re.sub(r'[$€£¥]', '', text)
        
        # Normalize financial abbreviations
        financial_abbr = {
            r'\bm\b': 'million',
            r'\bmm\b': 'million',
            r'\bbn\b': 'billion',
            r'\bb\b': 'billion',
            r'\bk\b': 'thousand',
            r'\bQ[1-4]\b': lambda m: f"Quarter {m.group(0)[1]}",
            r'\bFY\b': 'Fiscal Year',
            r'\bYOY\b': 'Year over Year',
            r'\bQOQ\b': 'Quarter over Quarter',
            r'\bEPS\b': 'Earnings Per Share',
            r'\bP/E\b': 'Price to Earnings',
            r'\bROI\b': 'Return on Investment',
            r'\bROE\b': 'Return on Equity',
            r'\bROA\b': 'Return on Assets',
            r'\bEBITDA\b': 'Earnings Before Interest Taxes Depreciation and Amortization',
            r'\bCAPEX\b': 'Capital Expenditure'
        }
        
        for abbr, expanded in financial_abbr.items():
            if callable(expanded):
                text = re.sub(abbr, expanded, text)
            else:
                text = re.sub(abbr, expanded, text)
        
        # Format percentages consistently
        text = re.sub(r'(\d+)%', r'\1 percent', text)
        
        # Format large numbers consistently
        text = re.sub(r'(\d+),(\d{3})', r'\1\2', text)
        
        return text 