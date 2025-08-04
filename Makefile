.PHONY: help install install-dev test lint format clean build run docker-build docker-run docker-stop deploy-local

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  test          Run tests"
	@echo "  lint          Run linting"
	@echo "  format        Format code"
	@echo "  clean         Clean cache and temp files"
	@echo "  build         Build the package"
	@echo "  run           Run the API server locally"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run with Docker Compose"
	@echo "  docker-stop   Stop Docker containers"
	@echo "  deploy-local  Deploy locally with Docker"
	@echo "  train         Train the model"
	@echo "  predict       Run inference"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Testing and Quality
test:
	pytest src/tests/ -v --cov=src

lint:
	flake8 src/
	mypy src/

format:
	black src/
	isort src/

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/

# Building
build: clean
	python setup.py sdist bdist_wheel

# Running
run:
	python -m uvicorn src.services.api:app --reload --host 0.0.0.0 --port 8000

# Docker operations
docker-build:
	docker build -t smart-product-classifier:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

deploy-local: docker-build docker-run

# ML Operations
train:
	python -m src.training.train

predict:
	python -m src.inference.predict

# Development setup
setup-dev: install-dev
	pre-commit install

# CI/CD
ci: install-dev lint test

# Deploy to production (customize for your deployment platform)
deploy-prod:
	@echo "Implement your production deployment here"
	@echo "e.g., docker push, kubectl apply, etc."