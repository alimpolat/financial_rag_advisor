import streamlit as st
from doc_processor import DocProcessor
from rag_engine import RAGEngine
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from financial_analyzer import FinancialAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup page config
st.set_page_config(page_title="Financial RAG Advisor", layout="wide")

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# State Management
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "processor" not in st.session_state:
    st.session_state.processor = DocProcessor()
if "current_answer" not in st.session_state:
    st.session_state.current_answer = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analyzer" not in st.session_state:
    st.session_state.analyzer = FinancialAnalyzer()

# Main Layout
st.title("Financial RAG Advisor")
st.markdown("Upload financial documents and ask questions to get insights based on their content.")

# Sidebar for document upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a financial PDF", type="pdf", accept_multiple_files=False)
    if uploaded_file:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")
        
        # Process the document
        try:
            logger.info(f"Processing {file_path}")
            with st.spinner("Processing document..."):
                docs = st.session_state.processor.load_and_chunk(file_path)
                st.session_state.rag_engine = RAGEngine(docs)
                st.session_state.uploaded_files.append(uploaded_file.name)
                st.success(f"Processed {len(docs)} chunks from the document.")
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            st.error(f"Failed to process document: {str(e)}")
    
    # Show uploaded files
    if st.session_state.uploaded_files:
        st.subheader("Uploaded Documents")
        for file in st.session_state.uploaded_files:
            st.write(f"- {file}")
    
    # Financial metrics extraction option
    st.subheader("Financial Analysis")
    if st.session_state.rag_engine and st.button("Extract Financial Metrics"):
        with st.spinner("Analyzing financial metrics..."):
            try:
                # Query for financial metrics
                result = st.session_state.rag_engine.query("What are the key financial metrics like revenue, profit, and EPS?")
                metrics = st.session_state.analyzer.extract_metrics(result["answer"])
                if metrics:
                    st.write("Extracted Metrics:")
                    for key, value in metrics.items():
                        st.write(f"- {key.replace('_', ' ').title()}: ${value:,.2f}")
                    
                    # Calculate ratios if possible
                    ratios = st.session_state.analyzer.calculate_ratios(metrics)
                    if ratios:
                        st.write("Financial Ratios:")
                        for key, value in ratios.items():
                            if key.endswith("_pct"):
                                st.write(f"- {key.replace('_pct', '').replace('_', ' ').title()}: {value}")
                else:
                    st.info("No financial metrics found in the document.")
            except Exception as e:
                logger.error(f"Error extracting metrics: {str(e)}")
                st.error(f"Failed to extract metrics: {str(e)}")

# Main area for queries and results
st.header("Financial Document Q&A")
if st.session_state.rag_engine:
    # Query input
    query = st.text_input("Ask a question about the financial documents:", placeholder="E.g., What was the revenue in Q2 2024?")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit Query"):
            if query:
                with st.spinner("Processing your question..."):
                    try:
                        result = st.session_state.rag_engine.query(query)
                        st.session_state.current_answer = result
                        st.session_state.chat_history.append((query, result["answer"]))
                    except Exception as e:
                        logger.error(f"Query failed: {str(e)}")
                        st.error(f"Query failed: {str(e)}")
            else:
                st.warning("Please enter a question.")
    
    with col2:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.current_answer = None
    
    # Display current answer if available
    if st.session_state.current_answer:
        st.subheader("Answer")
        st.markdown(st.session_state.current_answer["answer"])
        
        # Extract and display financial metrics in the answer if any
        metrics = st.session_state.analyzer.extract_metrics(st.session_state.current_answer["answer"])
        if metrics:
            st.subheader("Financial Metrics Detected")
            metrics_df = pd.DataFrame({
                "Metric": [k.replace("_", " ").title() for k in metrics.keys()],
                "Value": [f"${v:,.2f}" for v in metrics.values()]
            })
            st.table(metrics_df)
        
        # Extract and display trends if any
        trends = st.session_state.analyzer.extract_financial_trends(st.session_state.current_answer["answer"])
        if trends:
            st.subheader("Financial Trends Detected")
            st.write(f"Direction: {trends.get('direction', 'N/A').title()}")
            if 'growth' in trends:
                st.write(f"Change: {abs(trends['growth']):.2f}%")
            if 'period' in trends:
                st.write(f"Period: {trends['period']}")
        
        st.subheader("Sources")
        for i, source in enumerate(st.session_state.current_answer["sources"]):
            with st.expander(f"Source {i+1} - Page {source['page']} - {source['company']} {source['date']}"):
                st.write(source["content"])
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for q, a in st.session_state.chat_history:
            with st.expander(f"Q: {q}"):
                st.markdown(f"A: {a}")
                st.markdown("---")
    
else:
    st.info("Please upload a financial document to start asking questions.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Built for A.Team interview by [Your Name]")
st.sidebar.info("This showcase project demonstrates RAG implementation with financial domain expertise.") 