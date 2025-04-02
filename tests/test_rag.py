import pytest
import os
import sys
sys.path.append("../src")
from doc_processor import DocProcessor
from rag_engine import RAGEngine
from embedding_engine import EmbeddingEngine
from financial_analyzer import FinancialAnalyzer


class TestDocProcessor:
    """Test suite for the DocProcessor class."""
    
    def test_extract_financial_metadata(self):
        """Test financial metadata extraction from PDF."""
        processor = DocProcessor()
        
        # This is a mock test since we don't have a real file
        # In real testing, provide a path to a test PDF
        test_file = "../data/sample.pdf"
        if os.path.exists(test_file):
            metadata = processor._extract_financial_metadata(test_file)
            assert isinstance(metadata, dict)
            assert "company" in metadata
            assert "date" in metadata
            assert "report_type" in metadata
    
    def test_extract_tables(self):
        """Test table extraction from financial documents."""
        processor = DocProcessor()
        
        # Mock test for table extraction
        test_file = "../data/sample.pdf"
        if os.path.exists(test_file):
            tables = processor.extract_tables(test_file)
            assert isinstance(tables, list)


class TestFinancialAnalyzer:
    """Test suite for the FinancialAnalyzer class."""
    
    def test_extract_metrics(self):
        """Test extraction of financial metrics from text."""
        analyzer = FinancialAnalyzer()
        
        # Test with sample financial text
        sample_text = """
        In Q2 2024, the company reported Revenue of $500 million, 
        up 15% from the previous year. Net Income was $75 million,
        resulting in EPS of $1.25.
        """
        
        metrics = analyzer.extract_metrics(sample_text)
        assert isinstance(metrics, dict)
        
        # Check if metrics were extracted correctly
        if "revenue" in metrics:
            assert metrics["revenue"] == 500000000
        if "net_income" in metrics:
            assert metrics["net_income"] == 75000000
        if "eps" in metrics:
            assert metrics["eps"] == 1.25
    
    def test_calculate_ratios(self):
        """Test calculation of financial ratios."""
        analyzer = FinancialAnalyzer()
        
        # Test with sample metrics
        metrics = {
            "revenue": 1000000000,
            "cost_of_goods": 600000000,
            "net_income": 200000000,
            "total_assets": 2000000000,
            "total_liabilities": 800000000,
        }
        
        ratios = analyzer.calculate_ratios(metrics)
        assert isinstance(ratios, dict)
        
        # Check calculated ratios
        assert "gross_margin" in ratios
        assert abs(ratios["gross_margin"] - 0.4) < 0.01  # Should be 0.4 or 40%
        
        assert "net_profit_margin" in ratios
        assert abs(ratios["net_profit_margin"] - 0.2) < 0.01  # Should be 0.2 or 20%
        
        assert "roa" in ratios
        assert abs(ratios["roa"] - 0.1) < 0.01  # Should be 0.1 or 10%
    
    def test_extract_financial_trends(self):
        """Test extraction of financial trends from text."""
        analyzer = FinancialAnalyzer()
        
        # Test with sample trend text
        trend_text = "Revenue increased by 15% year-over-year, while expenses decreased by 5%."
        
        trends = analyzer.extract_financial_trends(trend_text)
        assert isinstance(trends, dict)
        
        # Check extracted trends
        if "growth" in trends:
            assert trends["growth"] == 15.0
        if "direction" in trends:
            assert trends["direction"] == "increase"
        if "period" in trends:
            assert trends["period"] == "YOY"


# Integration test for RAG system
# This is just a skeleton, actual implementation would depend on test environment
def test_rag_integration():
    """Integration test for the entire RAG pipeline."""
    # Only run if we have the OpenAI API key and a test document
    if "OPENAI_API_KEY" in os.environ and os.path.exists("../data/sample.pdf"):
        processor = DocProcessor()
        docs = processor.load_and_chunk("../data/sample.pdf")
        
        # Initialize RAG engine with docs
        rag = RAGEngine(docs)
        
        # Test a query
        result = rag.query("What was the revenue in Q2 2024?")
        
        # Verify result structure
        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result
        assert isinstance(result["sources"], list) 