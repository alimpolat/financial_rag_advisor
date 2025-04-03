#!/usr/bin/env python3
"""
Command-line script for fine-tuning the Financial RAG Advisor model.
"""

import os
import sys
import argparse
import logging
import time
from src.model_fine_tuner import ModelFineTuner
from src.doc_processor import DocProcessor


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("fine_tuning.log")
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a model for the Financial RAG Advisor")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Create training data command
    train_parser = subparsers.add_parser("create-data", help="Create training data from financial documents")
    train_parser.add_argument("--docs", "-d", nargs="+", required=True, help="Paths to financial documents")
    train_parser.add_argument("--output-dir", "-o", default="./fine_tuning", help="Output directory for training data")
    
    # Start fine-tuning command
    fine_tune_parser = subparsers.add_parser("start", help="Start a fine-tuning job")
    fine_tune_parser.add_argument("--training-file", "-t", required=True, help="Path to training data JSONL file")
    fine_tune_parser.add_argument("--model", "-m", default="gpt-3.5-turbo", help="Base model to fine-tune")
    fine_tune_parser.add_argument("--epochs", "-e", type=int, default=3, help="Number of epochs")
    fine_tune_parser.add_argument("--output-dir", "-o", default="./fine_tuning", help="Output directory for fine-tuning data")
    
    # Check status command
    status_parser = subparsers.add_parser("status", help="Check status of a fine-tuning job")
    status_parser.add_argument("--job-id", "-j", help="Fine-tuning job ID (optional, will read from job_details.json if not provided)")
    status_parser.add_argument("--output-dir", "-o", default="./fine_tuning", help="Directory containing job details")
    
    # Test model command
    test_parser = subparsers.add_parser("test", help="Test a fine-tuned model")
    test_parser.add_argument("--questions", "-q", nargs="+", help="List of questions to test (optional)")
    test_parser.add_argument("--output-dir", "-o", default="./fine_tuning", help="Directory containing model details")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    logger = setup_logging()
    
    if not args.command:
        logger.error("No command specified. Use --help for usage information.")
        return 1
    
    try:
        if args.command == "create-data":
            logger.info(f"Creating training data from {len(args.docs)} documents")
            docs_processor = DocProcessor()
            fine_tuner = ModelFineTuner(output_dir=args.output_dir)
            
            all_texts = []
            for doc_path in args.docs:
                if not os.path.exists(doc_path):
                    logger.error(f"Document not found: {doc_path}")
                    continue
                    
                try:
                    chunks = docs_processor.load_and_chunk(doc_path)
                    texts = [chunk.page_content for chunk in chunks]
                    all_texts.extend(texts)
                    logger.info(f"Processed {len(chunks)} chunks from {doc_path}")
                except Exception as e:
                    logger.error(f"Error processing {doc_path}: {str(e)}")
            
            if not all_texts:
                logger.error("No text extracted from documents")
                return 1
                
            training_file = fine_tuner.generate_training_data(all_texts)
            logger.info(f"Training data created and saved to {training_file}")
            
        elif args.command == "start":
            logger.info(f"Starting fine-tuning job with {args.model}")
            fine_tuner = ModelFineTuner(
                base_model=args.model,
                output_dir=args.output_dir,
                n_epochs=args.epochs
            )
            
            if not os.path.exists(args.training_file):
                logger.error(f"Training file not found: {args.training_file}")
                return 1
                
            response = fine_tuner.start_fine_tuning(args.training_file)
            logger.info(f"Fine-tuning job created with ID: {response.id}")
            logger.info(f"Check status with: python fine_tune.py status --job-id {response.id}")
            
        elif args.command == "status":
            logger.info("Checking fine-tuning status")
            fine_tuner = ModelFineTuner(output_dir=args.output_dir)
            
            try:
                response = fine_tuner.check_fine_tuning_status(args.job_id)
                logger.info(f"Job status: {response.status}")
                
                if hasattr(response, 'fine_tuned_model') and response.fine_tuned_model:
                    logger.info(f"Fine-tuned model ID: {response.fine_tuned_model}")
                    
                if hasattr(response, 'finished_at') and response.finished_at:
                    logger.info(f"Finished at: {response.finished_at}")
                elif hasattr(response, 'created_at'):
                    logger.info(f"Created at: {response.created_at}")
                    
            except ValueError as e:
                logger.error(str(e))
                return 1
                
        elif args.command == "test":
            logger.info("Testing fine-tuned model")
            fine_tuner = ModelFineTuner(output_dir=args.output_dir)
            
            model_id = fine_tuner.get_fine_tuned_model()
            if not model_id:
                logger.error("No fine-tuned model available for testing")
                return 1
                
            questions = args.questions if args.questions else fine_tuner.generate_sample_questions()
            logger.info(f"Testing with {len(questions)} questions")
            
            results = fine_tuner.test_fine_tuned_model(questions)
            for q, a in results.items():
                logger.info(f"Q: {q}")
                logger.info(f"A: {a}")
                logger.info("---")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main()) 