# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Terminal-first Python toolkit for benchmarking llama.cpp MoE GGUF inference settings. Discovers feature flags from local llama.cpp binaries, generates a benchmark plan from a YAML config matrix, runs llama-server with varying parameters, and reports aggregated results.

## Commands

- `lcpk detect`: probe llama.cpp binaries and write feature output
- `lcpk bench --dry-run`: validate planning without launching models
- `lcpk bench --mode smoke`: run the smallest practical benchmark
- `lcpk bench --retry-failed`: retry failed, OOM, timeout, or unsupported runs
- `lcpk report summary --runs runs`: summarize runs
- `lcpk report by-profile --runs runs`: show observations split by prompt profile
- `lcpk report compare --baseline runs/OLD --runs runs/NEW`: compare candidate configs against a baseline
- `lcpk report recommend --runs runs`: print recommended llama-server command
- Run tests: `python -m pytest tests/ -v`
- Run a single test: `python -m pytest tests/test_stats.py -v`

Entry points: `llamacpp-perfkit` or `lcpk` (both map to `llamacpp_perfkit.cli:main`).

## Architecture

### Configuration → Execution pipeline

```
YAML config → ConfigLoader (validates via Pydantic BenchmarkConfig)
  → FeatureDetector (probes llama.cpp binaries for supported flags, writes features.json)
  → make_plan() (Cartesian product of matrix dimensions, filtered by budget mode)
  → LlamaServerBenchmarkRunner (groups jobs by server config, launches llama-server, sends HTTP requests, collects metrics)
```

### Budget modes (how the Cartesian matrix is filtered)

Each mode in `planner.py` applies different selection heuristics on the full job matrix:

- **smoke**: 1-2 jobs — smallest context, first KV type, max n_cpu_moe, optionally 1 MTP variant
- **quick**: targets a single (context, KV, n_cpu_moe) tuple, prefers non-MTP baselines with MTP comparisons where safe baselines exist
- **focused**: refines around the best existing safe baseline, expands to neighboring n_cpu_moe values and top 2 KV types
- **full**: no filtering — runs the entire Cartesian matrix

### Result deduplication

Each job gets a `config_hash` (first 16 chars of SHA-256 of deterministic config payload including model HF, binary identity, prompt file identity, all matrix params). When `budget.reuse_existing_results: true`, successful matching hashes are reused. Failed/OOM/timeout rows can be retried with `--retry-failed`.

### Server batching

Jobs are grouped by `server_group_key` (context_size, kv_type, n_cpu_moe, mtp_enabled, mtp_spec_type, mtp_draft_n_max, mtp_draft_p_min). One llama-server process is launched per group and serves multiple HTTP requests with different prompt profiles. This avoids restarting the server for every parameter combination.

### Monitoring

Two parallel samplers run during benchmarks:
- **Monitor** (`benchlib.py:Monitor`): system metrics — GPU VRAM/utilization/power/temp via `nvidia-smi`, RAM via `/proc/meminfo`
- **LlamaCppMetricsSampler** (`server_runner.py`): llama.cpp metrics via `/slots` and `/metrics` endpoints

### Output structure

Each run writes to `runs/<run_id>/`:
- `summary.json`: full run record (config, parsed output, monitor summary, server info)
- `metrics/system.jsonl`: per-interval system monitoring samples
- `metrics/llamacpp.jsonl`: per-interval llama.cpp metric samples

### Reporting architecture

```
run_storage.py: discover_run_dirs() → load_run_rows() (reads summary.json + metrics per run dir)
  → reporting.py: run_observation_from_row() → aggregate_server_config_reports()
    (groups by ServerConfigKey, computes MetricSummary per group)
  → stats.py: MetricSummary (geometric_mean, p10, stddev, min/max)
  → tabular CLI output via benchlib.table()
```

### Feature detection (`benchlib.py:detect_features`)

Runs `llama-server --help` and `llama-bench --help`, parses output to discover supported flags (n_cpu_moe, cache types, spec types, `--host`, `--port`, etc.). Writes `features.json` + `features.txt` to the results directory. Feature results are reused across runs unless the binary path, extra args, or llama.cpp commit changes.

### MTP (Multi-Token Prediction)

Dual model support: `models.baseline_hf` (standard) and `models.mtp_hf` (MTP variant). When MTP is enabled for a job, the MTP model is loaded and spec_type/draft_n_max/draft_p_min are passed to llama-server. Supported spec types are auto-detected from `--spec-type` help output.

### CLI entry point

`cli.py:main()` manually parses `--color`/`--no-color` flags before delegating to Typer, because Typer processes its own callbacks after argument parsing. Without this pre-parsing, color config wouldn't take effect during early output.

## Key modules

| Module | Role |
|---|---|
| `cli.py` | Typer CLI with `report` sub-command group |
| `models.py` | Pydantic config models (`BenchmarkConfig`, `ReportFilters`, etc.) |
| `benchlib.py` | Shared helpers: config loading, feature detection, hash, monitoring, HTTP, parsing |
| `planner.py` | Benchmark plan generation: matrix expansion, mode selection, risk assessment |
| `server_runner.py` | llama-server process lifecycle, HTTP request execution, metric collection |
| `services.py` | Orchestration: `BenchmarkService`, `FeatureDetector`, `ConfigLoader`, `RecommendationService` |
| `reporting.py` | Result aggregation, tabular display, compare logic |
| `run_storage.py` | Run directory layout, summary/metric read/write, row assembly |
| `stats.py` | Statistical summaries (geometric mean, percentiles) |
| `output.py` | Terminal coloring, duration formatting, status styles |

## Development

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

### Commands

- `make typecheck` — Run mypy type checker on source and tests
- `make lint` — Run ruff linter and format checker
- `make test` — Run typecheck + lint + pytest (full validation before committing)
- `make build` — Run full validation then build the package
- `make clean` — Remove build artifacts and cache files

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. They run:
1. trailing-whitespace, end-of-file-fixer, check-yaml
2. ruff linter with auto-fix, ruff formatter
3. mypy type checker (full project)

To run hooks manually: `pre-commit run --all-files`

### Coding Standards

- All functions must have type annotations (enforced by mypy `disallow_untyped_defs`)
- Use `from __future__ import annotations` for PEP 604 syntax
- For dynamic dict patterns (config, JSON results), use `dict[str, Any]`
- Add `# type: ignore[code]` for legitimate false positives only
- Run `make test` before pushing any changes
