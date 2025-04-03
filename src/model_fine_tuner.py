"""
Model fine-tuning module for Financial RAG Advisor.
Handles the creation of training data, fine-tuning process, 
and integration with the main application.
"""

import os
import json
import time
import logging
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import jsonlines
import tqdm
import openai
from dotenv import load_dotenv
from src.financial_analyzer import FinancialAnalyzer

# Load environment variables from .env file
load_dotenv()

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
        self.base_model = base_model
        self.output_dir = output_dir
        self.n_epochs = n_epochs
        self.analyzer = FinancialAnalyzer()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize OpenAI API
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Set API key directly instead of using client
        openai.api_key = api_key
        
    def generate_training_data(self, 
                              financial_texts: List[str], 
                              output_file: str = None) -> str:
        """
        Generate training data from financial texts.
        
        Args:
            financial_texts: List of financial texts
            output_file: Path to output file (default: ./fine_tuning/financial_training_data.jsonl)
            
        Returns:
            Path to the generated training data file
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, "financial_training_data.jsonl")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Extract metrics and generate question-answer pairs
        training_data = []
        
        for text in tqdm.tqdm(financial_texts, desc="Generating training data"):
            # Extract metrics from text
            metrics = self.analyzer.extract_metrics(text)
            
            if metrics:
                # Generate question-answer pairs for each metric
                for metric_name, value in metrics.items():
                    # Format the metric name for natural language
                    formatted_name = metric_name.replace('_', ' ').title()
                    
                    # Generate questions about this metric
                    questions = [
                        f"What is the {formatted_name}?",
                        f"Can you tell me the {formatted_name}?",
                        f"What was the company's {formatted_name}?",
                        f"What did the {formatted_name} amount to?"
                    ]
                    
                    # Format the answer
                    if isinstance(value, (int, float)):
                        answer = f"The {formatted_name} is ${value:,.2f}."
                    else:
                        answer = f"The {formatted_name} is {value}."
                    
                    # Add to training data
                    for question in questions:
                        training_data.append({
                            "messages": [
                                {"role": "system", "content": "You are a financial expert assistant that provides accurate information about financial data."},
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": answer}
                            ]
                        })
            
            # Generate common financial questions
            common_financial_metrics = [
                "revenue", "profit margin", "EPS", "debt-to-equity ratio", 
                "gross profit", "net income", "operating expenses"
            ]
            
            for metric in common_financial_metrics:
                if random.random() < 0.7:  # 70% chance to include each metric
                    # Create a generic question about financial metrics
                    question = f"What was the company's {metric}?"
                    answer = f"Based on the financial documents, the company's {metric} information is not directly specified in this section."
                    
                    training_data.append({
                        "messages": [
                            {"role": "system", "content": "You are a financial expert assistant that provides accurate information about financial data."},
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer}
                        ]
                    })
        
        # Add some generic financial analysis questions
        generic_qa_pairs = [
            {
                "question": "How did the company perform financially last quarter?",
                "answer": "To analyze the company's financial performance last quarter, I would need to examine key metrics such as revenue growth, profit margins, EPS, and compare them to previous quarters and industry benchmarks. I can provide this analysis if you share the specific financial statements."
            },
            {
                "question": "What does the balance sheet tell us about the company's financial health?",
                "answer": "A company's balance sheet provides insights into its financial health by showing assets, liabilities, and shareholders' equity. Key indicators include the debt-to-equity ratio, current ratio, and working capital. These metrics help assess liquidity, solvency, and overall financial stability."
            },
            {
                "question": "How do I calculate the P/E ratio?",
                "answer": "The Price-to-Earnings (P/E) ratio is calculated by dividing the current market price per share by the earnings per share (EPS). For example, if a company's stock is trading at $50 and its EPS is $5, the P/E ratio would be 10. This ratio helps investors assess whether a stock is overvalued or undervalued relative to its earnings."
            }
        ]
        
        for qa in generic_qa_pairs:
            training_data.append({
                "messages": [
                    {"role": "system", "content": "You are a financial expert assistant that provides accurate information about financial data."},
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]}
                ]
            })
        
        # Write to JSONL file
        with jsonlines.open(output_file, 'w') as writer:
            for item in training_data:
                writer.write(item)
        
        self.logger.info(f"Generated {len(training_data)} training examples")
        return output_file
    
    def start_fine_tuning(self, training_file: str) -> Any:
        """
        Start the fine-tuning process using the OpenAI API.
        
        Args:
            training_file: Path to the JSONL training file
            
        Returns:
            Response from the fine-tuning creation API
        """
        try:
            # Upload the training file
            with open(training_file, "rb") as f:
                response = openai.File.create(
                    file=f,
                    purpose="fine-tune"
                )
                file_id = response.id
                
            self.logger.info(f"Uploaded training file with ID: {file_id}")
            
            # Wait for file to be processed
            time.sleep(5)
            
            # Create fine-tuning job
            response = openai.FineTuningJob.create(
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
                    "status": response.status,
                    "model": self.base_model,
                    "fine_tuned_model": response.get("fine_tuned_model", None),
                    "training_file": file_id,
                    "created_at": response.created_at
                }, f, indent=2)
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error starting fine-tuning job: {str(e)}")
            raise ValueError(f"Fine-tuning failed: {str(e)}")
    
    def check_fine_tuning_status(self, job_id: str = None) -> Any:
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
            response = openai.FineTuningJob.retrieve(job_id)
            
            # Log status details
            status = response.status
            self.logger.info(f"Fine-tuning job {job_id} status: {status}")
            
            if hasattr(response, 'fine_tuned_model') and response.fine_tuned_model:
                # Save the fine-tuned model ID
                with open(os.path.join(self.output_dir, "fine_tuned_model.txt"), "w") as f:
                    f.write(response.fine_tuned_model)
                self.logger.info(f"Fine-tuned model ID: {response.fine_tuned_model}")
                
            # Update job details
            job_details_path = os.path.join(self.output_dir, "job_details.json")
            if os.path.exists(job_details_path):
                with open(job_details_path, 'r') as f:
                    job_details = json.load(f)
                
                job_details.update({
                    "status": response.status,
                    "fine_tuned_model": response.get("fine_tuned_model", None),
                    "finished_at": response.get("finished_at", None)
                })
                
                with open(job_details_path, 'w') as f:
                    json.dump(job_details, f, indent=2)
            
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
            "What was the company's revenue in the last quarter?",
            "How did the profit margin change year-over-year?",
            "What is the current EPS?",
            "What factors affected the company's operating expenses?",
            "How does the company's debt-to-equity ratio compare to industry standards?",
            "What are the key financial metrics to focus on in this report?",
            "Can you summarize the company's financial performance?",
            "What is the gross profit margin?",
            "How much did R&D expenses increase compared to last year?",
            "What are the most significant risks mentioned in the financial report?"
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
                response = openai.ChatCompletion.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are a financial expert assistant that provides accurate information about financial data."},
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