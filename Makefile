PYTHON ?= python

.PHONY: install dev run test coverage lint format clean

install:
	$(PYTHON) -m pip install .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

run:
	$(PYTHON) main.py

test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m pytest --cov=iris_nn --cov-report=term-missing --cov-report=xml

lint:
	ruff check .
	ruff format --check .
	mypy iris_nn main.py

format:
	ruff check --fix .
	ruff format .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov build dist
	rm -rf ./*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
