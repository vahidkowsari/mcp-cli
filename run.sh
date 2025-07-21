#!/bin/bash
# AI MCP Host Runner Script

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

printf "${GREEN}🚀 AI MCP Host${NC}\n"
echo "===================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    printf "${YELLOW}⚠️  Python 3.11 virtual environment not found. Running setup...${NC}\n"
    make setup
fi

# Check for API keys (either in env vars or .env file)
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ] && [ ! -f ".env" ]; then
    printf "${RED}❌ No AI API keys found!${NC}\n"
    echo "Please set one of the following environment variables:"
    echo "  export OPENAI_API_KEY='your-openai-api-key'"
    echo "  export ANTHROPIC_API_KEY='your-anthropic-api-key'"
    echo ""
    echo "Or create a .env file with your API keys (see .env.example)"
    exit 1
fi

# Activate virtual environment and run the application
printf "${GREEN}🎯 Starting AI MCP Host...${NC}\n"
echo ""

# Set PYTHONPATH to current directory
export PYTHONPATH="$PWD:$PYTHONPATH"

# Run the application with Python 3.11
# Pass all command line arguments to main.py
./.venv/bin/python main.py "$@"
