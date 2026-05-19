.PHONY: typecheck lint test build clean install

typecheck:
	python -m mypy src/llamacpp_perfkit/ tests/

lint:
	python -m ruff check src/llamacpp_perfkit/ tests/
	python -m ruff format --check src/llamacpp_perfkit/ tests/

test: typecheck lint
	python -m pytest tests/ -v

build: test
	python -m build

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete

install:
	pip install -e ".[dev]"
	pre-commit install
