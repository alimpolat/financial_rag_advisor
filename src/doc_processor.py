from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import re
import logging
import os
from pypdf import PdfReader


class DocProcessor:
    """
    Specialized document processor for financial documents.
    Handles loading, chunking, and metadata extraction with focus on financial context.
    """
    
    def load_and_chunk(self, file_path: str) -> List:
        """
        Loads a financial PDF document and splits it into semantically meaningful chunks.
        Preserves financial context and extracts metadata specific to financial documents.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of document chunks with financial metadata
        """
        logger = logging.getLogger(__name__)
        
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Financial-specific metadata extraction
            metadata = self._extract_financial_metadata(file_path)
            
            # Use RecursiveCharacterTextSplitter with financial-specific settings
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200, 
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = splitter.split_documents(docs)
            
            # Add metadata to each chunk
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "source": file_path,
                    "page": chunk.metadata["page"],
                    "chunk_id": i,
                    **metadata
                })
                
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load PDF: {str(e)}")
            raise ValueError(f"Failed to load PDF: {str(e)}")
            
    def _extract_financial_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extracts financial metadata from the document like date, company, report type.
        Uses regex patterns tailored for financial documents.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary of financial metadata
        """
        reader = PdfReader(file_path)
        text = "".join([page.extract_text() for page in reader.pages[:5]])
        
        # Extract company name, date, report type using regex
        company = re.search(r"([A-Z][A-Za-z\s]+(?:Inc\.|Corp\.|Ltd\.|LLC))", text)
        date = re.search(r"(?:Q[1-4]|Quarter [1-4]|Annual).+?20[0-9]{2}", text)
        report_type = re.search(r"(?:Annual Report|10-K|10-Q|Earnings Release|Financial Statement)", text)
        
        return {
            "company": company.group(1) if company else "Unknown",
            "date": date.group(0) if date else "Unknown",
            "report_type": report_type.group(0) if report_type else "Unknown"
        }
        
    def extract_tables(self, file_path: str) -> List[Dict]:
        """
        Extracts tables from financial documents which often contain the most valuable numeric data.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of extracted tables with metadata
        """
        # Simple implementation that looks for patterns of numbers and whitespace that suggest tabular data
        tables = []
        
        try:
            reader = PdfReader(file_path)
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                
                # Look for patterns that suggest financial tables
                # This is a simplified approach - production would use more sophisticated table extraction
                # Looking for rows of numbers with consistent spacing
                potential_table_rows = re.findall(r"(\s*[\d,]+\s+[\d,.]+\s+[\d,.]+\s+[\d,.%]+\s*\n){3,}", text)
                
                for i, table_text in enumerate(potential_table_rows):
                    tables.append({
                        "page": page_num,
                        "content": table_text,
                        "table_id": f"page_{page_num}_table_{i}"
                    })
                    
        except Exception as e:
            logger.error(f"Table extraction failed: {str(e)}")
            
        return tables

# Example usage:
# processor = DocProcessor()
# chunks = processor.load_and_chunk("data/annual_report.pdf") 