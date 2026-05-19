# Repository Guidelines

## Scope

This is a terminal-first Python toolkit for benchmarking llama.cpp MoE GGUF settings.

Core code lives in `src/llamacpp_perfkit/`:
- `cli.py`: Typer CLI
- `planner.py`: benchmark plan generation
- `server_runner.py`: llama-server execution
- `reporting.py`: result summaries
- `benchlib.py`: shared helpers

Configuration presets live in `config/`. Prompts live in `prompts/`.

Treat `results/` and `logs/` as generated output unless the task explicitly involves benchmark artifacts.

Do not edit documentation files, including `README.md`, unless explicitly asked.

## Commands

Use default config behavior unless a non-default config is required.

- `make test` to run type checking, linting, and unit tests
- `make typecheck` to run only mypy
- `make lint` to run only ruff
- `lcpk detect`: inspect llama.cpp binaries and write feature output.
- `lcpk bench --dry-run`: validate planning without launching models.
- `lcpk bench --mode smoke`: run the smallest practical benchmark.
- `lcpk bench --retry-failed`: retry failed, OOM, timeout, or unsupported runs.
- `lcpk report summary --results results/runs.jsonl`: summarize runs.
- `lcpk report recommend --results results/runs.jsonl`: print measured command recommendations.

Use `--config` only for temporary, alternate, or user-specified config files.

## Style

Use Python 3, four-space indentation, Typer for CLI entrypoints, `snake_case` names, `pathlib.Path`, and structured JSON/YAML handling.

Keep functions small and use explicit dictionary keys matching benchmark result fields.

## Testing

Run `make test` to validate changes. This runs mypy, ruff, and pytest sequentially.

The type checker (mypy) is configured for strict mode with exceptions for dynamic dict patterns.
The linter (ruff) is configured for PEP 8, import sorting, naming conventions, and pyupgrade.

Pre-commit hooks enforce these checks automatically before every git commit.
Run `pre-commit run --all-files` to run them manually at any time.

## Results

Benchmark runs append to `results/runs.jsonl`; do not delete older rows unless explicitly asked. `results/runs.csv` is regenerated from the JSONL after benchmark runs.

When `budget.reuse_existing_results: true`, successful matching configs may be reused. Failed, OOM, timeout, or unsupported rows can be retried with `--retry-failed`.

## Security

Do not commit private model paths, tokens, machine-specific absolute paths, or unnecessarily large raw logs.
