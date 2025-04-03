"""
Model fine-tuning module for Financial RAG Advisor.
Handles the creation of training data, fine-tuning process, 
and integration with the main application.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from openai import OpenAI
from financial_analyzer import FinancialAnalyzer


class ModelFineTuner:
    """
    Handles fine-tuning of language models for financial domain expertise.
    Creates training data from financial documents and manages the fine-tuning process.
    """
    
    def __init__(self, 
                 base_model: str = "gpt-3.5-turbo", 
                 output_dir: str = "./fine_tuning",
                 n_epochs: int = 3):
        """
        Initialize the fine-tuning manager.
        
        Args:
            base_model: Base model to fine-tune
            output_dir: Directory to save fine-tuning files
            n_epochs: Number of epochs for fine-tuning
        """
        self.logger = logging.getLogger(__name__)
        self.client = OpenAI()
        self.base_model = base_model
        self.output_dir = output_dir
        self.n_epochs = n_epochs
        self.analyzer = FinancialAnalyzer()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_training_data(self, 
                              financial_texts: List[str], 
                              financial_metrics: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate training data from financial texts and extracted metrics.
        
        Args:
            financial_texts: List of financial document texts
            financial_metrics: Optional pre-extracted financial metrics
            
        Returns:
            Path to the generated JSONL training file
        """
        training_data = []
        
        # If metrics not provided, extract them
        if not financial_metrics:
            financial_metrics = []
            for text in financial_texts:
                metrics = self.analyzer.extract_metrics(text)
                if metrics:
                    financial_metrics.append(metrics)
        
        # Generate question-answer pairs about financial metrics
        for i, (text, metrics) in enumerate(zip(financial_texts, financial_metrics)):
            if not metrics:
                continue
                
            # Generate revenue question
            if "revenue" in metrics:
                revenue_q = {
                    "messages": [
                        {"role": "system", "content": "You are a financial analyst assistant. Provide accurate, concise answers about financial metrics with proper formatting and units."},
                        {"role": "user", "content": f"What was the revenue reported in this document?"},
                        {"role": "assistant", "content": f"The revenue reported was ${metrics['revenue']:,.2f}. This figure represents the total income generated from the company's main business activities."}
                    ]
                }
                training_data.append(revenue_q)
            
            # Generate profit margin question if we have both revenue and net_income
            if "revenue" in metrics and "net_income" in metrics:
                margin = metrics["net_income"] / metrics["revenue"]
                margin_q = {
                    "messages": [
                        {"role": "system", "content": "You are a financial analyst assistant. Provide accurate, concise answers about financial metrics with proper formatting and units."},
                        {"role": "user", "content": f"What is the profit margin shown in this document?"},
                        {"role": "assistant", "content": f"The profit margin is {margin:.2%}. This represents the company's net income as a percentage of total revenue, indicating how much of each dollar of revenue is converted to profit."}
                    ]
                }
                training_data.append(margin_q)
            
            # Generate EPS question
            if "eps" in metrics:
                eps_q = {
                    "messages": [
                        {"role": "system", "content": "You are a financial analyst assistant. Provide accurate, concise answers about financial metrics with proper formatting and units."},
                        {"role": "user", "content": f"What was the reported earnings per share (EPS)?"},
                        {"role": "assistant", "content": f"The reported earnings per share (EPS) was ${metrics['eps']:.2f}. This represents the portion of the company's profit allocated to each outstanding share of common stock."}
                    ]
                }
                training_data.append(eps_q)
                
            # Generate debt-to-equity question
            if "debt_to_equity" in metrics:
                debt_q = {
                    "messages": [
                        {"role": "system", "content": "You are a financial analyst assistant. Provide accurate, concise answers about financial metrics with proper formatting and units."},
                        {"role": "user", "content": f"What is the debt-to-equity ratio?"},
                        {"role": "assistant", "content": f"The debt-to-equity ratio is {metrics['debt_to_equity']:.2f}. This indicates the relative proportion of shareholders' equity and debt used to finance the company's assets."}
                    ]
                }
                training_data.append(debt_q)
                
            # Generate trend analysis if available
            trends = self.analyzer.extract_financial_trends(text)
            if trends and "growth" in trends and "direction" in trends:
                trend_q = {
                    "messages": [
                        {"role": "system", "content": "You are a financial analyst assistant. Provide accurate, concise answers about financial metrics with proper formatting and units."},
                        {"role": "user", "content": f"What is the trend in financial performance shown in this document?"},
                        {"role": "assistant", "content": f"The financial performance shows a {trends['direction']} of {abs(trends['growth']):.2f}%. " + 
                                                        (f"This is measured on a {trends['period']} basis." if 'period' in trends else "")}
                    ]
                }
                training_data.append(trend_q)
                
            # Add general financial analysis question
            metrics_str = ", ".join([f"{k.replace('_', ' ')}: ${v:,.2f}" if isinstance(v, (int, float)) and k != "eps" else f"{k.replace('_', ' ')}: {v}" 
                                   for k, v in metrics.items() if not k.endswith("_pct")])
            
            general_q = {
                "messages": [
                    {"role": "system", "content": "You are a financial analyst assistant. Provide accurate, concise answers about financial metrics with proper formatting and units."},
                    {"role": "user", "content": f"Can you analyze the key financial metrics in this report?"},
                    {"role": "assistant", "content": f"Based on my analysis, the key financial metrics from this report are:\n\n{metrics_str}\n\nThese figures indicate " +
                                                   ("a strong financial position " if "net_income" in metrics and metrics["net_income"] > 0 else "financial challenges ") +
                                                   "for the company during this reporting period."}
                ]
            }
            training_data.append(general_q)
                
        # Create additional training examples for common financial queries
        common_financial_questions = [
            {
                "messages": [
                    {"role": "system", "content": "You are a financial analyst assistant. Provide accurate, concise answers about financial metrics with proper formatting and units."},
                    {"role": "user", "content": "What does EPS stand for and why is it important?"},
                    {"role": "assistant", "content": "EPS stands for Earnings Per Share. It represents a company's net profit divided by the number of outstanding shares. EPS is important because it indicates profitability on a per-share basis, allowing investors to compare companies of different sizes. Higher EPS generally indicates stronger financial performance, making it a key metric for investment decisions."}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a financial analyst assistant. Provide accurate, concise answers about financial metrics with proper formatting and units."},
                    {"role": "user", "content": "What's the difference between gross margin and net profit margin?"},
                    {"role": "assistant", "content": "Gross margin is revenue minus cost of goods sold (COGS) divided by revenue, showing the percentage of revenue retained after direct production costs. Net profit margin is net income divided by revenue, showing the percentage of revenue converted to profit after all expenses. Gross margin focuses on production efficiency, while net profit margin reflects overall business profitability including all operating costs, taxes, and interest."}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a financial analyst assistant. Provide accurate, concise answers about financial metrics with proper formatting and units."},
                    {"role": "user", "content": "How do you interpret a company's P/E ratio?"},
                    {"role": "assistant", "content": "The P/E (Price-to-Earnings) ratio is calculated by dividing a company's current share price by its earnings per share. A high P/E suggests investors expect higher future earnings growth, while a low P/E may indicate undervaluation or limited growth prospects. When interpreting a P/E ratio: (1) Compare it to industry averages, (2) Consider current market conditions, (3) Evaluate alongside other metrics like revenue growth and debt levels, and (4) Look at historical P/E trends for context."}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a financial analyst assistant. Provide accurate, concise answers about financial metrics with proper formatting and units."},
                    {"role": "user", "content": "What financial metrics should I look at to assess a company's debt burden?"},
                    {"role": "assistant", "content": "To assess a company's debt burden, examine these key metrics: (1) Debt-to-Equity Ratio: Total debt divided by shareholders' equity, (2) Debt-to-EBITDA Ratio: Debt relative to earnings before interest, taxes, depreciation, and amortization, (3) Interest Coverage Ratio: How many times operating income can cover interest expenses, (4) Current Ratio: Short-term debt obligations versus current assets, and (5) Free Cash Flow: Cash available after capital expenditures to service debt. Lower D/E and D/EBITDA ratios and higher interest coverage generally indicate healthier debt levels."}
                ]
            }
        ]
        
        training_data.extend(common_financial_questions)
        
        # Write training data to JSONL file
        training_file_path = os.path.join(self.output_dir, "financial_training_data.jsonl")
        with open(training_file_path, "w") as f:
            for entry in training_data:
                f.write(json.dumps(entry) + "\n")
                
        self.logger.info(f"Generated {len(training_data)} training examples in {training_file_path}")
        return training_file_path
    
    def start_fine_tuning(self, training_file_path: str) -> Dict[str, Any]:
        """
        Start the fine-tuning process using the OpenAI API.
        
        Args:
            training_file_path: Path to the JSONL training file
            
        Returns:
            Response from the fine-tuning creation API
        """
        try:
            # Upload the training file
            with open(training_file_path, "rb") as f:
                response = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
                file_id = response.id
                
            self.logger.info(f"Uploaded training file with ID: {file_id}")
            
            # Wait for file to be processed
            time.sleep(5)
            
            # Create fine-tuning job
            response = self.client.fine_tuning.jobs.create(
                training_file=file_id,
                model=self.base_model,
                hyperparameters={
                    "n_epochs": self.n_epochs
                }
            )
            
            job_id = response.id
            self.logger.info(f"Created fine-tuning job with ID: {job_id}")
            
            # Save job details
            job_details_path = os.path.join(self.output_dir, "job_details.json")
            with open(job_details_path, "w") as f:
                json.dump({
                    "job_id": job_id,
                    "base_model": self.base_model,
                    "file_id": file_id,
                    "created_at": time.time()
                }, f)
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error starting fine-tuning job: {str(e)}")
            raise ValueError(f"Fine-tuning failed: {str(e)}")
    
    def check_fine_tuning_status(self, job_id: str = None) -> Dict[str, Any]:
        """
        Check the status of a fine-tuning job.
        
        Args:
            job_id: ID of the fine-tuning job. If None, read from job_details.json
            
        Returns:
            Current status of the fine-tuning job
        """
        if job_id is None:
            job_details_path = os.path.join(self.output_dir, "job_details.json")
            if not os.path.exists(job_details_path):
                raise ValueError("No job ID provided and no job details found")
                
            with open(job_details_path, "r") as f:
                details = json.load(f)
                job_id = details.get("job_id")
                
            if not job_id:
                raise ValueError("No job ID found in job details")
        
        try:
            response = self.client.fine_tuning.jobs.retrieve(job_id)
            
            # Log status details
            status = response.status
            self.logger.info(f"Fine-tuning job {job_id} status: {status}")
            
            if hasattr(response, 'fine_tuned_model') and response.fine_tuned_model:
                # Save the fine-tuned model ID
                with open(os.path.join(self.output_dir, "fine_tuned_model.txt"), "w") as f:
                    f.write(response.fine_tuned_model)
                self.logger.info(f"Fine-tuned model ID: {response.fine_tuned_model}")
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error checking fine-tuning status: {str(e)}")
            raise ValueError(f"Status check failed: {str(e)}")
    
    def get_fine_tuned_model(self) -> str:
        """
        Get the ID of the fine-tuned model if available.
        
        Returns:
            ID of the fine-tuned model or None if not available
        """
        model_file_path = os.path.join(self.output_dir, "fine_tuned_model.txt")
        if not os.path.exists(model_file_path):
            return None
            
        with open(model_file_path, "r") as f:
            model_id = f.read().strip()
            
        return model_id if model_id else None
    
    def generate_sample_questions(self, n: int = 5) -> List[str]:
        """
        Generate sample financial questions to test the fine-tuned model.
        
        Args:
            n: Number of questions to generate
            
        Returns:
            List of financial questions
        """
        questions = [
            "What are the key components of a company's balance sheet?",
            "How do you calculate Return on Equity (ROE)?",
            "What's the difference between operating income and net income?",
            "How should I interpret a negative free cash flow?",
            "What financial metrics best indicate a company's liquidity?",
            "How does depreciation affect a company's financial statements?",
            "What is EBITDA and why is it important?",
            "How do stock buybacks affect EPS?",
            "What's a good debt-to-equity ratio for a technology company?",
            "How do changes in interest rates affect corporate bonds?"
        ]
        
        return questions[:min(n, len(questions))]
    
    def test_fine_tuned_model(self, questions: List[str] = None) -> Dict[str, str]:
        """
        Test the fine-tuned model with financial questions.
        
        Args:
            questions: List of questions to test with, default to sample questions
            
        Returns:
            Dictionary of questions and responses
        """
        model_id = self.get_fine_tuned_model()
        if not model_id:
            self.logger.warning("No fine-tuned model available for testing")
            return {"error": "No fine-tuned model available"}
            
        if not questions:
            questions = self.generate_sample_questions()
            
        results = {}
        
        for question in questions:
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are a financial analyst assistant. Provide accurate, concise answers about financial metrics with proper formatting and units."},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.1,
                    max_tokens=300
                )
                answer = response.choices[0].message.content
                results[question] = answer
                
            except Exception as e:
                self.logger.error(f"Error testing question '{question}': {str(e)}")
                results[question] = f"Error: {str(e)}"
                
        return results


if __name__ == "__main__":
    """
    Example usage of the ModelFineTuner.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Sample financial text for demonstration
    sample_text = """
    Q4 2023 Financial Results
    
    Revenue: $500 million, up 15% year-over-year
    Net Income: $75 million, resulting in an EPS of $1.25
    Gross Margin: 60%, up from 58% in Q4 2022
    Operating Expenses: $150 million
    Cash and Cash Equivalents: $1.2 billion
    Total Assets: $2.5 billion
    Total Liabilities: $800 million
    """
    
    # Initialize the fine-tuner
    fine_tuner = ModelFineTuner()
    
    # Generate training data
    training_file = fine_tuner.generate_training_data([sample_text])
    
    # To start actual fine-tuning (commented out to avoid accidental API usage)
    # response = fine_tuner.start_fine_tuning(training_file)
    # print(f"Started fine-tuning job: {response.id}")
    
    # To check status of a job
    # status = fine_tuner.check_fine_tuning_status("job_id_here")
    # print(f"Job status: {status.status}")
    
    # To test a fine-tuned model (if available)
    # results = fine_tuner.test_fine_tuned_model()
    # for q, a in results.items():
    #     print(f"Q: {q}")
    #     print(f"A: {a}")
    #     print("---") 