.PHONY: test lint format typecheck build publish clean install dev

# Development setup
install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -m "not slow and not gpu" --cov=usn -q

test-all:
	pytest tests/ --cov=usn -q

test-slow:
	pytest tests/ -m slow -q

# Code quality
lint:
	ruff check usn/ tests/

format:
	ruff format usn/ tests/

format-check:
	ruff format --check usn/ tests/

typecheck:
	mypy usn/ --strict

# Build and publish
build: clean
	python -m build

publish: build
	twine upload dist/*

publish-test: build
	twine upload --repository testpypi dist/*

# Cleanup
clean:
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -f coverage.xml .coverage
