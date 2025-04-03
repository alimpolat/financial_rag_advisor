#!/usr/bin/env python3
"""
Simple script to test OpenAI API access
"""

import os
from dotenv import load_dotenv
import openai  # Import as a module instead of from openai import OpenAI

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable not set")
    exit(1)

print(f"API Key found: {api_key[:5]}...")

try:
    # Set API key directly
    openai.api_key = api_key
    
    # Test with a simple completion using the older API style
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
    )
    
    print("API Test successful!")
    print(f"Response: {response.choices[0].message.content}")

except Exception as e:
    print(f"ERROR: {str(e)}")
    print("Full exception:")
    import traceback
    traceback.print_exc() 