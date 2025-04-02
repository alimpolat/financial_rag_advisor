#!/bin/bash

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
    else
        echo "Warning: OPENAI_API_KEY is not set. Please set it in your environment or create a .env file."
    fi
fi

# Run the Streamlit app
streamlit run src/app.py 