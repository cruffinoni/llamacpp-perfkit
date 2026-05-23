.PHONY: build typecheck lint test clean deps install

build:
	python -m build

typecheck:
	python -m mypy src/llamacpp_perfkit/ tests/

lint:
	python -m ruff check src/llamacpp_perfkit/ tests/
	python -m ruff format --check src/llamacpp_perfkit/ tests/

test: typecheck lint
	python -m pytest tests/ -v

clean:
	rm -rf build/ dist/ *.egg-info/ src/*.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete

deps:
	pip install -e ".[dev]"

install: build
	pip install .
