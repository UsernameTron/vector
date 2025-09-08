# Vector RAG Database - Makefile
# Enterprise-ready automation for RAG operations

.PHONY: help setup install clean test lint format serve ingest index eval bench backup restore docker-build docker-up docker-down

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
VENV := venv
APP := app.py
PORT := 5001
WORKERS := 4

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

# Default target
help: ## Show this help message
	@echo "$(GREEN)Vector RAG Database - Make Targets$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# Setup and Installation
setup: ## Complete initial setup
	@echo "$(GREEN)Setting up Vector RAG Database...$(NC)"
	$(MAKE) install
	$(MAKE) init-db
	$(MAKE) verify
	@echo "$(GREEN)Setup complete!$(NC)"

install: ## Install dependencies
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && $(PIP) install --upgrade pip
	. $(VENV)/bin/activate && $(PIP) install -r requirements.txt
	. $(VENV)/bin/activate && $(PIP) install -r requirements-prod.txt 2>/dev/null || true
	@echo "$(GREEN)Dependencies installed$(NC)"

init-db: ## Initialize vector database
	@echo "$(GREEN)Initializing ChromaDB...$(NC)"
	mkdir -p ./chroma_db
	mkdir -p ./logs
	mkdir -p ./data/documents
	mkdir -p ./data/processed
	. $(VENV)/bin/activate && $(PYTHON) -c "from vector_db import VectorDatabase; VectorDatabase()"
	@echo "$(GREEN)Database initialized$(NC)"

clean: ## Clean up generated files and caches
	@echo "$(YELLOW)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	@echo "$(GREEN)Cleanup complete$(NC)"

# Development
dev: ## Run development server with auto-reload
	@echo "$(GREEN)Starting development server...$(NC)"
	. $(VENV)/bin/activate && FLASK_ENV=development FLASK_DEBUG=true $(PYTHON) $(APP) --mode development --port $(PORT)

serve: ## Run production server
	@echo "$(GREEN)Starting production server...$(NC)"
	. $(VENV)/bin/activate && gunicorn -c gunicorn.conf.py wsgi:app || $(PYTHON) $(APP) --mode production --port $(PORT)

shell: ## Start interactive Python shell with app context
	. $(VENV)/bin/activate && $(PYTHON) -i -c "from app import app; from vector_db import VectorDatabase; db = VectorDatabase()"

# Document Processing
ingest: ## Ingest documents from data/documents directory
	@echo "$(GREEN)Ingesting documents...$(NC)"
	. $(VENV)/bin/activate && $(PYTHON) scripts/ingest.py --input-dir ./data/documents --dedupe

ingest-file: ## Ingest a specific file (usage: make ingest-file FILE=path/to/file.pdf)
	@echo "$(GREEN)Ingesting file: $(FILE)$(NC)"
	. $(VENV)/bin/activate && $(PYTHON) scripts/ingest.py --file $(FILE)

index: ## Build/rebuild vector index
	@echo "$(GREEN)Building vector index...$(NC)"
	. $(VENV)/bin/activate && $(PYTHON) scripts/build_index.py

index-optimize: ## Optimize vector index for production
	@echo "$(GREEN)Optimizing index...$(NC)"
	. $(VENV)/bin/activate && $(PYTHON) scripts/optimize_index.py

# Evaluation and Testing
test: ## Run test suite
	@echo "$(GREEN)Running tests...$(NC)"
	. $(VENV)/bin/activate && pytest tests/ -v --cov=. --cov-report=term-missing

test-unit: ## Run unit tests only
	. $(VENV)/bin/activate && pytest tests/unit/ -v

test-integration: ## Run integration tests
	. $(VENV)/bin/activate && pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests
	. $(VENV)/bin/activate && pytest tests/e2e/ -v

eval: ## Run RAG evaluation suite
	@echo "$(GREEN)Running evaluation suite...$(NC)"
	. $(VENV)/bin/activate && $(PYTHON) scripts/eval.py --metrics all --output reports/eval_$(shell date +%Y%m%d_%H%M%S).json

bench: ## Run performance benchmarks
	@echo "$(GREEN)Running benchmarks...$(NC)"
	. $(VENV)/bin/activate && $(PYTHON) scripts/bench.py --queries 100 --output reports/bench_$(shell date +%Y%m%d_%H%M%S).csv

# Code Quality
lint: ## Run linting checks
	@echo "$(GREEN)Running linters...$(NC)"
	. $(VENV)/bin/activate && flake8 . --exclude=$(VENV),__pycache__,.git || true
	. $(VENV)/bin/activate && pylint *.py src/ --disable=C0114,C0115,C0116 || true
	. $(VENV)/bin/activate && mypy . --ignore-missing-imports || true

format: ## Format code with black
	@echo "$(GREEN)Formatting code...$(NC)"
	. $(VENV)/bin/activate && black . --exclude=$(VENV)
	. $(VENV)/bin/activate && isort . --skip=$(VENV)

security: ## Run security checks
	@echo "$(GREEN)Running security checks...$(NC)"
	. $(VENV)/bin/activate && pip-audit || true
	. $(VENV)/bin/activate && bandit -r . -ll || true
	. $(VENV)/bin/activate && safety check || true

# Docker Operations
docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t vector-rag:latest .

docker-up: ## Start services with docker-compose
	@echo "$(GREEN)Starting Docker services...$(NC)"
	docker-compose up -d

docker-down: ## Stop Docker services
	@echo "$(GREEN)Stopping Docker services...$(NC)"
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

# Backup and Restore
backup: ## Create backup of vector database
	@echo "$(GREEN)Creating backup...$(NC)"
	mkdir -p backups
	tar -czf backups/chroma_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz ./chroma_db
	@echo "$(GREEN)Backup created in backups/$(NC)"

restore: ## Restore from latest backup
	@echo "$(YELLOW)Restoring from latest backup...$(NC)"
	@latest=$$(ls -t backups/*.tar.gz | head -1); \
	if [ -z "$$latest" ]; then \
		echo "$(RED)No backup found$(NC)"; \
	else \
		echo "Restoring from $$latest"; \
		rm -rf ./chroma_db.old; \
		mv ./chroma_db ./chroma_db.old 2>/dev/null || true; \
		tar -xzf $$latest; \
		echo "$(GREEN)Restore complete$(NC)"; \
	fi

# Monitoring and Health
health: ## Check system health
	@echo "$(GREEN)Checking system health...$(NC)"
	curl -s http://localhost:$(PORT)/health | python -m json.tool || echo "$(RED)Service not running$(NC)"

logs: ## Tail application logs
	tail -f logs/app.log

stats: ## Show database statistics
	. $(VENV)/bin/activate && $(PYTHON) -c "from vector_db import VectorDatabase; db = VectorDatabase(); print(db.get_stats())"

# Deployment
deploy-dev: ## Deploy to development environment
	@echo "$(GREEN)Deploying to development...$(NC)"
	./deployment/deploy.sh development

deploy-staging: ## Deploy to staging environment
	@echo "$(GREEN)Deploying to staging...$(NC)"
	./deployment/deploy.sh staging

deploy-prod: ## Deploy to production environment
	@echo "$(RED)Deploying to PRODUCTION...$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && ./deployment/deploy.sh production

# Utility
verify: ## Verify installation and configuration
	@echo "$(GREEN)Verifying installation...$(NC)"
	@$(PYTHON) --version
	@. $(VENV)/bin/activate && $(PYTHON) -c "import chromadb; print('ChromaDB:', chromadb.__version__)"
	@. $(VENV)/bin/activate && $(PYTHON) -c "import openai; print('OpenAI: OK')"
	@. $(VENV)/bin/activate && $(PYTHON) -c "import flask; print('Flask:', flask.__version__)"
	@test -f .env && echo "$(GREEN)✓ .env file exists$(NC)" || echo "$(RED)✗ .env file missing$(NC)"
	@test -d ./chroma_db && echo "$(GREEN)✓ ChromaDB directory exists$(NC)" || echo "$(RED)✗ ChromaDB directory missing$(NC)"
	@echo "$(GREEN)Verification complete$(NC)"

env-setup: ## Create .env from template
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN).env file created from template$(NC)"; \
		echo "$(YELLOW)Please edit .env and add your API keys$(NC)"; \
	else \
		echo "$(YELLOW).env file already exists$(NC)"; \
	fi

docs: ## Generate documentation
	@echo "$(GREEN)Generating documentation...$(NC)"
	. $(VENV)/bin/activate && pdoc --html --output-dir docs/api src/
	@echo "$(GREEN)Documentation generated in docs/api/$(NC)"

# CI/CD targets
ci: lint test security ## Run CI pipeline locally

cd: docker-build docker-up health ## Run CD pipeline locally

.DEFAULT_GOAL := help
