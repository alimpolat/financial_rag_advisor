# Financial RAG Advisor

A sophisticated Retrieval-Augmented Generation (RAG) system for analyzing and querying financial documents, built with LangChain, OpenAI API, and ChromaDB. This project demonstrates domain-specific RAG implementation for financial text analysis, with specialized components for extracting, analyzing, and querying financial data.

## Overview

The Financial RAG Advisor enables users to upload financial documents (such as annual reports, earnings releases, 10-K/10-Q filings) and query them using natural language. The system extracts and processes financial-specific data, understands financial terminology, recognizes numerical data patterns, and provides accurate answers with source attribution. It also automatically identifies financial metrics and calculates financial ratios from the content.

## Features

- **Specialized Financial Document Processing**
  - Optimized chunking algorithms preserving financial context and tabular data
  - Automatic detection of company names, report types, and dates
  - Special handling for financial tables and numerical data

- **Financial Metadata Extraction**
  - Company identification (including legal entity types like Inc., Ltd., Corp.)
  - Financial period detection (Q1-Q4, Annual, Fiscal Year)
  - Report type classification (10-K, 10-Q, Earnings Release, Financial Statement)

- **Financial-Aware Semantic Search**
  - Domain-specific embeddings optimized for financial terminology
  - Query expansion with financial synonyms (e.g., "profit" expands to "earnings", "net income")
  - Maximum Marginal Relevance (MMR) retrieval for balanced and diverse results

- **Comprehensive Metrics Analysis**
  - Automatic extraction of financial KPIs (revenue, EPS, profit margins, etc.)
  - Financial ratio calculation (P/E, ROA, ROE, debt-to-equity)
  - Trend identification in financial data (YoY and QoQ changes)

- **Interactive Streamlit UI**
  - Intuitive document upload and management
  - Real-time financial metrics visualization
  - Source attribution with page references and company context
  - Persistent chat history for continuous analysis

## Architecture

The system is built with a modular architecture focused on financial domain expertise:

1. **DocProcessor (`doc_processor.py`)**
   - Specialized PDF loading with `PyPDFLoader`
   - Financial-specific chunking with `RecursiveCharacterTextSplitter`
   - Metadata extraction using regex patterns optimized for financial documents
   - Table extraction for numerical financial data

2. **EmbeddingEngine (`embedding_engine.py`)**
   - Financial text preprocessing tailored for embedding quality
   - Normalization of financial notation, numbers, and abbreviations
   - Integration with OpenAI's text-embedding-3-small model
   - Optimized for financial terminology and concepts

3. **RAGEngine (`rag_engine.py`)**
   - Vector storage with `ChromaDB` for efficient semantic retrieval
   - MMR search for balancing relevance and diversity in financial data
   - Financial-specific prompt templates for accurate responses
   - Query expansion with financial domain knowledge
   - Integration with `ChatOpenAI` (GPT-4) for precise financial responses

4. **FinancialAnalyzer (`financial_analyzer.py`)**
   - Regex-based financial metric extraction
   - Standardization of financial values (millions, billions)
   - Calculation of common financial ratios and indicators
   - Financial trend detection and analysis

5. **Streamlit UI (`app.py`)**
   - Interactive document upload and management
   - Real-time question answering with source attribution
   - Financial metrics visualization and tabulation
   - Persistent state management and chat history

## Technical Details

### Financial Text Processing

The system employs specialized techniques for handling financial text:

- **Financial Abbreviation Normalization**: Expands common financial abbreviations (EPS, P/E, ROI, EBITDA, etc.)
- **Currency and Number Standardization**: Normalizes different formats of monetary values
- **Percentage Formatting**: Standardizes percentage representations for consistent analysis
- **Financial Entity Recognition**: Identifies companies, time periods, and report types

### Semantic Retrieval & RAG Pipeline

The RAG implementation uses several advanced techniques:

- **Vector Store**: ChromaDB for efficient similarity search of document chunks
- **Retrieval Strategy**: MMR to balance similarity with information diversity
- **Query Expansion**: Financial domain-specific expansion of search terms
- **Prompt Engineering**: Specialized financial analyst prompts for the LLM
- **Source Attribution**: Detailed metadata tracking for result verification

### Financial Analysis

The system performs automated financial analysis:

- **Metric Extraction**: Identifies key financial figures like revenue, profit, EPS
- **Ratio Calculation**: Computes financial indicators like gross margin, net profit margin, ROA
- **Trend Analysis**: Detects growth/decline patterns and YoY/QoQ changes

## Setup and Installation

### Prerequisites

- Python 3.9+ 
- OpenAI API key

### Environment Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd financial-rag-advisor
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   ```

### Running the Application

You can run the application using either:

1. The included run script:
   ```bash
   chmod +x run.sh  # Make executable (first time only)
   ./run.sh
   ```

2. Or directly with Streamlit:
   ```bash
   streamlit run src/app.py
   ```

The application will be available at http://localhost:8501 (or another port if 8501 is in use).

## Usage Guide

1. **Upload Financial Documents**:
   - Use the sidebar uploader to select a financial PDF document
   - The system will process the document, extract metadata and chunk it appropriately
   - You can upload multiple documents for comparative analysis

2. **Ask Financial Questions**:
   - Type questions in the input field, focusing on financial aspects
   - Example queries:
     - "What was the revenue in Q4 2023?"
     - "How did the gross margin change year-over-year?"
     - "What were the key factors affecting operating expenses?"
     - "What is the company's debt-to-equity ratio?"

3. **Analyze Results**:
   - Review the AI's answer which includes properly formatted financial figures
   - Examine extracted financial metrics and calculated ratios
   - Check source attribution to verify information accuracy
   - Explore cited document sections for context

4. **Extract Financial Metrics**:
   - Use the "Extract Financial Metrics" button to automatically identify KPIs
   - View calculated financial ratios based on extracted data
   - Analyze financial trends identified in the documents

## Implementation Challenges & Solutions

During development, several challenges were addressed:

1. **Library Compatibility**: 
   - Challenge: LangChain's rapid development cycle led to API changes and deprecations
   - Solution: Updated imports and method calls to work with the latest versions (langchain 0.3.x, langchain-openai 0.3.x)

2. **Financial Text Processing**:
   - Challenge: Financial documents contain specialized notation and formats
   - Solution: Custom regex patterns and normalization functions for financial text

3. **Embedding Quality**:
   - Challenge: General purpose embeddings miss financial nuances
   - Solution: Financial-specific preprocessing and OpenAI's powerful embedding models

4. **Metric Extraction Accuracy**:
   - Challenge: Diverse formats for presenting financial data
   - Solution: Multiple regex patterns to handle various financial reporting styles

5. **Error Handling**:
   - Challenge: Robust processing of various financial document formats
   - Solution: Comprehensive exception handling and graceful fallbacks

## Tech Stack

- **LangChain (0.3.x)**: Orchestrates the RAG pipeline
- **LangChain-OpenAI (0.3.x)**: Integration with OpenAI's embeddings and completions
- **OpenAI API**: Powers embeddings (text-embedding-3-small) and generation (GPT-4)
- **ChromaDB**: Vector storage for efficient similarity search
- **Streamlit**: Interactive user interface for document analysis
- **PyPDF**: PDF processing and text extraction
- **Pandas**: Financial data analysis and visualization
- **Matplotlib**: Visualization components for financial trends
- **Python-dotenv**: Environment management for API keys

## Project Structure

```
financial-rag-advisor/
├── README.md                # Project documentation
├── requirements.txt         # Dependencies
├── .gitignore               # Git ignore patterns
├── LICENSE                  # MIT License
├── .env                     # Environment variables (create this file)
├── run.sh                   # Convenience script to run the application
├── data/                    # Directory for sample PDFs
│   └── .gitkeep             # Ensures directory is tracked by git
├── src/
│   ├── doc_processor.py     # PDF loading and chunking with financial extraction
│   ├── embedding_engine.py  # Embeddings optimized for financial text
│   ├── rag_engine.py        # RAG pipeline with financial-aware retrieval
│   ├── financial_analyzer.py # Financial metrics extraction and analysis
│   ├── model_fine_tuner.py  # Fine-tuning OpenAI models with financial data
│   └── app.py               # Streamlit UI with financial visualization
├── fine_tune.py             # CLI tool for model fine-tuning operations
├── tests/
│   └── test_rag.py          # Test suite for components
└── uploads/                 # Directory for user uploaded files
```

## Future Enhancements

Potential improvements to consider:

- **Advanced Financial Metrics**: More complex financial ratio calculations and industry benchmarking
- **Multi-document Analysis**: Comparative analysis across multiple financial documents
- **Historical Trending**: Visualization of financial metrics over time
- **Sentiment Analysis**: Detecting sentiment in financial narratives and forward-looking statements
- **Regulatory Compliance**: Identifying disclosures and compliance-related statements

## Model Fine-Tuning

The Financial RAG Advisor includes capabilities for fine-tuning LLMs to further improve performance on financial text:

### Fine-Tuning Features

- **Domain-Specific Training**: Generate synthetic training data from financial documents
- **Specialized Financial QA**: Improve model responses for financial terminology and metrics
- **Performance Optimization**: Enhance accuracy for financial calculations and analyses

### Using Fine-Tuning

The system includes a command-line tool for managing fine-tuning operations:

1. **Create Training Data**:
   ```bash
   ./fine_tune.py create-data --docs data/financial_report1.pdf data/financial_report2.pdf
   ```

2. **Start Fine-Tuning Job**:
   ```bash
   ./fine_tune.py start --training-file ./fine_tuning/financial_training_data.jsonl
   ```

3. **Check Fine-Tuning Status**:
   ```bash
   ./fine_tune.py status
   ```

4. **Test Fine-Tuned Model**:
   ```bash
   ./fine_tune.py test
   ```

The fine-tuned model enhances the system's ability to understand financial context, extract relevant metrics, and provide more accurate answers to financial queries.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 