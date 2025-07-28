#!/bin/bash
# Run the new mcp-use implementation

# Activate virtual environment
source mcp-use-env/bin/activate

# Run the new CLI
python mcp-use-cli.py "$@"
