# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A terminal-first Go toolkit for benchmarking llama.cpp MoE GGUF server configurations. It expands a YAML config matrix, starts llama-server processes, sends HTTP completion/chat requests, collects GPU/RAM and llama.cpp metrics, and produces comparison reports.

## Commands

```sh
make build        # go build -o llama-cpp-perfkit ./cmd/llama-cpp-perfkit
make test         # go test ./...
make fmt          # go fmt ./...
make dev          # build + ./llama-cpp-perfkit dev tui (simulation)

# Run a single test
go test ./internal/bench/ -run TestMakePlan -v

# Lint changed Go files (gg-lint must be installed)
gg-lint ./... --git --changed-lines
```

## Architecture

### CLI entry point (`cmd/llama-cpp-perfkit/main.go`)
Sets up TrueColor output and signal-aware context, then delegates to the Cobra root command.

### Subcommands (`internal/cli/root.go`)
- `run <config>` — execute a benchmark matrix. Flags: `--retry-failed`, `--force` (`-f`), `--dry-run`
- `report summary|by-profile|compare` — inspect benchmark results
- `dev tui` — animated fake benchmark simulation with `--bar-style` and `--loop` flags

### Benchmark pipeline

1. **Config** (`internal/config/`) — YAML config with defaults. `Config.Matrix` defines the parameter space (context sizes, KV types, batch sizes, speculative decoding params, etc.). `Config.ApplyDefaults()` fills unset fields.

2. **Feature detection** (`internal/llamacpp/features.go`) — Probes the local `llama-server` and `llama-bench` binaries by running `--help` to discover available flags and their allowed values. Results cached in `features.json`. This determines which matrix dimensions are actually usable.

3. **Plan generation** (`internal/bench/plan.go`) — Expands the config matrix into `BenchmarkJob` candidates, computes a SHA256-based `ConfigHash` for each, deduplicates, checks against existing run results to decide `run`/`reuse`/`skip` actions, and assigns OOM risk levels. Output is a `BenchmarkPlan`.

4. **Execution** (`internal/runner/`) — Groups planned runs by shared server config. For each group: allocates a free TCP port, starts `llama-server`, waits for health check, sends HTTP requests per prompt profile, collects system and llama.cpp metrics in parallel goroutines, writes results to `runs/`. Progress is streamed to the TUI via a `chan viewmodel.StateUpdate`.

5. **Reporting** (`internal/report/`) — Loads all runs from disk, flattens them into `RunObservation` values, groups by `ServerConfigKey`, computes statistical summaries (geometric mean, percentiles), and renders comparison tables with delta percentages against a baseline.

### TUI (`internal/tui/`)

Built on [Bubble Tea](https://github.com/charmbracelet/bubbletea). The TUI receives `StateUpdate` closures on a channel and applies them to a `BenchmarkTUIState` struct. Rendering is separated from state:

- `model.go` — Bubble Tea model: receives updates, handles keybindings (q/esc/ctrl+c to quit, space to pause, r to reset in simulation)
- `app/app.go` — `Run()` bridges the benchmark function to the TUI program, managing context cancellation and error propagation
- `viewmodel/state.go` — `BenchmarkTUIState` and `StateUpdate` (functional state transitions)
- `views/dashboard.go` — layout rendering from state
- `components/` — reusable widgets (progress bars, panels, tables, charts)
- `theme/` — lipgloss style definitions (Solarized Dark)
- `sim/` — deterministic simulation for `dev tui`, driven by predefined `Scenario` data

### Key domain types (`internal/domain/types.go`)

`ServerConfig`, `BenchmarkJob`, `PlannedRun`, `BenchmarkPlan`, `RunSummary`, `RunStatusInfo`, `SystemMetricSample`, `LlamaCppMetricSample`, `LoadedRun`. Types in `domain/` have no dependencies on other internal packages.

### Metrics collection (`internal/metrics/`)

- `SystemCollector` — polls GPU via `nvidia-smi` and RAM via `/proc/meminfo` at a configurable interval
- `LlamaCollector` — polls `/slots` and `/metrics` endpoints of the running llama-server

### Data storage (`internal/storage/`)

Runs stored under `runs/<run-id>/` with `summary.json`, and JSONL files under `metrics/system.jsonl` and `metrics/llamacpp.jsonl`. Uses generic `ReadJSONL[T]` for typed deserialization.

### Clock interface (`internal/domain/clock.go`)

`domain.Clock` abstracts `time.Now()` and `time.Since()` for deterministic testing. Production code uses `domain.RealClock{}`.

## Testing

Tests use `github.com/stretchr/testify` assertions. The `Clock` interface enables deterministic time in tests. The TUI simulation (`dev tui`) provides a visual smoke test without needing a real llama.cpp binary.
