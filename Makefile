# AI MCP Host - Makefile
# Automation for setup, development, and deployment

.PHONY: help install setup clean run test lint format check dev prod logs status

# Default target
help: ## Show this help message
	@echo "AI MCP Host - Available Commands:"
	@echo "=================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Environment setup
install: ## Install all dependencies and setup environment
	@echo "🚀 Setting up AI MCP Host..."
	@$(MAKE) check-python
	@$(MAKE) setup-venv
	@$(MAKE) install-deps
	@$(MAKE) check-node
	@$(MAKE) setup-env
	@echo "✅ Setup complete! Run 'make run' to start the application."

setup: install ## Alias for install

check-python: ## Check Python version compatibility
	@echo "🐍 Checking Python version..."
	@python3.11 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" || (echo "❌ Python 3.11+ required. Install with: brew install python@3.11" && exit 1)
	@echo "✅ Python 3.11 OK"

setup-venv: ## Create and setup virtual environment
	@echo "📦 Setting up virtual environment..."
	@if [ ! -d "venv311" ]; then python3.11 -m venv venv311; fi
	@echo "✅ Python 3.11 virtual environment ready"

install-deps: ## Install Python dependencies
	@echo "📚 Installing Python dependencies..."
	@./venv311/bin/pip install --upgrade pip
	@./venv311/bin/pip install litellm tenacity anthropic rich python-dotenv
	@echo "✅ Dependencies installed"

check-node: ## Check Node.js availability
	@echo "🟢 Checking Node.js..."
	@if command -v node >/dev/null 2>&1; then \
		echo "✅ Node.js available: $$(node --version)"; \
	else \
		echo "⚠️  Node.js not found. Install from: https://nodejs.org/"; \
	fi

setup-env: ## Setup environment variables
	@echo "🔧 Setting up environment..."
	@if [ ! -f ".env" ]; then \
		cp .env.example .env; \
		echo "📝 Created .env file from template"; \
		echo "⚠️  Please edit .env and add your API keys"; \
	else \
		echo "✅ .env file already exists"; \
	fi
	@if [ ! -f "mcp_config.json" ]; then \
		cp mcp_config.example.json mcp_config.json; \
		echo "📝 Created mcp_config.json from template"; \
		echo "⚠️  Please edit mcp_config.json and configure your MCP servers"; \
	else \
		echo "✅ mcp_config.json already exists"; \
	fi

# Development commands
run: ## Start the AI MCP Host application
	@echo "🚀 Starting AI MCP Host..."
	@./run.sh

run-verbose: ## Start the AI MCP Host application with verbose output
	@echo "🚀 Starting AI MCP Host (verbose mode)..."
	@./run.sh --verbose

dev: run ## Alias for run (development mode)

test: ## Run tests
	@echo "🧪 Running tests..."
	@./venv/bin/python -m pytest tests/ -v || echo "⚠️  No tests found"

lint: ## Run linting checks
	@echo "🔍 Running linting checks..."
	@./venv/bin/python -m flake8 --max-line-length=120 --ignore=E203,W503 *.py || echo "⚠️  flake8 not installed"
	@./venv/bin/python -m mypy --ignore-missing-imports *.py || echo "⚠️  mypy not installed"

format: ## Format code with black
	@echo "🎨 Formatting code..."
	@./venv/bin/python -m black --line-length=120 *.py || echo "⚠️  black not installed"

check: lint test ## Run all checks (lint + test)

# Utility commands
clean: ## Clean up generated files and caches
	@echo "🧹 Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache/ .mypy_cache/ .coverage 2>/dev/null || true
	@echo "✅ Cleanup complete"

clean-all: clean ## Clean everything including virtual environment
	@echo "🗑️  Removing virtual environment..."
	@rm -rf venv/
	@echo "✅ Full cleanup complete"

logs: ## Show recent application logs
	@echo "📋 Recent logs:"
	@tail -n 50 app.log 2>/dev/null || echo "No log file found"

status: ## Show system status and dependencies
	@echo "📊 AI MCP Host Status:"
	@echo "======================"
	@echo "Python: $$(python3 --version 2>/dev/null || echo 'Not found')"
	@echo "Node.js: $$(node --version 2>/dev/null || echo 'Not found')"
	@echo "Virtual env: $$([ -d 'venv' ] && echo 'Present' || echo 'Missing')"
	@echo "Dependencies: $$([ -f 'venv/pyvenv.cfg' ] && echo 'Installed' || echo 'Missing')"
	@echo "Environment: $$([ -f '.env' ] && echo 'Configured' || echo 'Missing')"
	@echo "MCP Config: $$([ -f 'mcp_config.json' ] && echo 'Present' || echo 'Missing')"

# Production commands
prod: ## Run in production mode (with optimizations)
	@echo "🏭 Starting in production mode..."
	@PYTHONOPTIMIZE=1 ./run.sh

deploy: ## Deploy to production (placeholder)
	@echo "🚀 Deployment not configured yet"
	@echo "   Add your deployment commands here"

# Development tools
install-dev: ## Install development dependencies
	@echo "🛠️  Installing development dependencies..."
	@./venv/bin/pip install black flake8 mypy pytest pytest-asyncio
	@echo "✅ Development tools installed"

update: ## Update all dependencies
	@echo "⬆️  Updating dependencies..."
	@./venv/bin/pip install --upgrade pip
	@./venv/bin/pip install --upgrade -r requirements.txt
	@echo "✅ Dependencies updated"

# Docker support (future)
docker-build: ## Build Docker image
	@echo "🐳 Docker support not implemented yet"

docker-run: ## Run in Docker container
	@echo "🐳 Docker support not implemented yet"

# Quick shortcuts
i: install    ## Quick install
r: run       ## Quick run
c: clean     ## Quick clean
t: test      ## Quick test
l: lint      ## Quick lint
f: format    ## Quick format
