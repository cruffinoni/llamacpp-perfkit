# Repository Guidelines

## Scope

This is a terminal-first Go toolkit for benchmarking llama.cpp MoE GGUF settings.

Core code lives in `internal/`:
- `cli/`: Cobra CLI
- `runner/`: benchmark execution
- `tui/`: terminal UI rendering
- `report/`: result summaries and reporting
- `config/`: configuration loading
- `bench/`: benchmark planning
- `llamacpp/`: llama.cpp integration

Configuration presets live in `config/`. Prompts live in `prompts/`.

Treat `runs/` and `logs/` as generated output unless the task explicitly involves benchmark artifacts.

Do not edit documentation files, including README.md, unless explicitly asked.

## Commands

Use default config behavior unless a non-default config is required.

- `make test` to run tests
- `llama-cpp-perfkit run <config>`: execute a benchmark matrix
- `llama-cpp-perfkit report summary`: summarize benchmark runs
- `llama-cpp-perfkit dev tui`: render a static fake TUI

Use `--retry-failed` and `--dry-run` flags as needed.

## Testing

Run `make test` to validate changes.

## Results

Benchmark runs are stored in `runs/`; do not delete older data unless explicitly asked.

When `budget.reuse_existing_results: true` in a config, successful matching configs may be reused. Failed, OOM, timeout, or unsupported configs can be retried with `--retry-failed`.

## Security

Do not commit private model paths, tokens, machine-specific absolute paths, or unnecessarily large raw logs.
