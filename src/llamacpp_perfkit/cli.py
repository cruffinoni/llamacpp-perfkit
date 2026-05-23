from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from typer.main import get_command

from .benchlib import DEFAULT_CONFIG_PATH
from .models import BenchOptions, GenerateOptions, Mode, ReportFilters, SortKey, Status
from .output import configure_color, error
from .services import BenchmarkService, ConfigError, ConfigLoader, FeatureDetector, GenerateService

app = typer.Typer(
    name="lcpk",
    add_completion=False,
    no_args_is_help=True,
    help="Benchmark llama.cpp MoE GGUF settings.",
)
report_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Inspect benchmark results.")
app.add_typer(report_app, name="report")


ConfigPath = Annotated[Path, typer.Option("--config", "-c", help="Benchmark YAML config.")]
RunsPath = Annotated[Path, typer.Option("--runs", "--results", "-r", help="Path to run root or run directory.")]


@app.callback()
def main_options(
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable colored terminal output.")] = False,
    color: Annotated[str, typer.Option("--color", help="Color mode: auto, always, never.")] = "auto",
) -> None:
    configure_color(color=color, no_color=no_color)


def _cfg(path: Path) -> dict[str, Any]:
    try:
        return ConfigLoader().load(path)
    except ConfigError as exc:
        print(error(f"Error: {exc}"), file=sys.stderr)
        raise typer.Exit(2) from exc


def _filters(
    runs: Path,
    model: str | None,
    quant: str | None,
    context_size: int | None,
    kv_type: str | None,
    batch_size: int | None,
    ubatch_size: int | None,
    prompt_profile: str | None,
    n_cpu_moe: int | None,
    status: Status | None,
    sort: SortKey,
    limit: int,
) -> ReportFilters:
    return ReportFilters(
        runs=runs,
        model=model,
        quant=quant,
        context_size=context_size,
        kv_type=kv_type,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        prompt_profile=prompt_profile,
        n_cpu_moe=n_cpu_moe,
        status=status,
        sort=sort,
        limit=limit,
    )


@app.command()
def detect(config: Annotated[Path | None, typer.Option("--config", "-c", help="Benchmark YAML config.")] = None) -> None:
    """Detect local llama.cpp feature support."""
    if config is not None:
        cfg = _cfg(config)
    else:
        cfg = {
            "models": {},
            "llama": {"extra_args": {}, "server": {}, "bin_dir": "../llama.cpp/build/bin"},
            "prompt": {},
            "run": {"cache_prompt": False, "endpoint": "chat"},
            "budget": {"mode": "quick", "max_runs": 8, "reuse_existing_results": True, "stop_if_all_remaining_are_risky": True},
            "matrix": {"kv_type": [], "spec_type": []},
            "output": {"logs_dir": "logs", "results_dir": "runs"},
        }
    raise typer.Exit(FeatureDetector().run(cfg))


@app.command()
def generate(
    model: Annotated[str, typer.Option("-m", "--model", help="Model HF identifier (e.g. org/model:quant).")] = ...,  # type: ignore[assignment]
    llama_bin: Annotated[Path, typer.Option("--llama-bin", help="Path to llama.cpp bin directory.")] = ...,  # type: ignore[assignment]
    output: Annotated[Path, typer.Option("-o", "--output", help="Output directory.")] = Path("config"),
    name: Annotated[str | None, typer.Option("-n", "--name", help="Config filename (auto-derived from model if not set).")] = None,
    temp: Annotated[float, typer.Option("--temp")] = 1.0,
    top_p: Annotated[float, typer.Option("--top-p")] = 0.95,
    top_k: Annotated[int, typer.Option("--top-k")] = 20,
    presence_penalty: Annotated[float, typer.Option("--presence-penalty")] = 0.0,
    min_p: Annotated[float, typer.Option("--min-p")] = 0.0,
    n_gpu_layers: Annotated[int, typer.Option("--n-gpu-layers")] = 99,
    split_mode: Annotated[str, typer.Option("--split-mode")] = "none",
    parallel: Annotated[int, typer.Option("--parallel")] = 1,
) -> None:
    """Generate a benchmark config file."""
    options = GenerateOptions(
        output_dir=output,
        model=model,
        llama_bin=llama_bin,
        name=name,
        temp=temp,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        min_p=min_p,
        n_gpu_layers=n_gpu_layers,
        split_mode=split_mode,
        parallel=parallel,
    )
    raise typer.Exit(GenerateService().run(options))


@app.command()
def bench(
    config: ConfigPath = DEFAULT_CONFIG_PATH,
    limit: Annotated[int | None, typer.Option("--limit", help="Backward-compatible alias for --max-runs.")] = None,
    max_runs: Annotated[int | None, typer.Option("--max-runs", help="Run at most N new benchmark requests.")] = None,
    mode: Annotated[Mode | None, typer.Option("--mode", help="Override budget.mode.")] = None,
    retry_failed: Annotated[
        bool, typer.Option("--retry-failed", help="Rerun failed, OOM, timeout, or unsupported configurations.")
    ] = False,
    force: Annotated[bool, typer.Option("--force", "-f", help="Force rerun of all configs, ignoring existing results.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print planned server commands without running models.")] = False,
) -> None:
    """Plan and run benchmark requests."""
    options = BenchOptions(mode=mode, max_runs=max_runs, limit=limit, retry_failed=retry_failed, force=force, dry_run=dry_run)
    raise typer.Exit(BenchmarkService().run(_cfg(config), options))


@report_app.command()
def summary(
    runs: RunsPath = Path("runs"),
    details: Annotated[bool, typer.Option("--details", help="Show expanded server config columns.")] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable colored terminal output.")] = False,
    color: Annotated[str, typer.Option("--color", help="Color mode: auto, always, never.")] = "auto",
    model: Annotated[str | None, typer.Option("--model")] = None,
    quant: Annotated[str | None, typer.Option("--quant")] = None,
    context_size: Annotated[int | None, typer.Option("--context-size")] = None,
    kv_type: Annotated[str | None, typer.Option("--kv-type")] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size")] = None,
    ubatch_size: Annotated[int | None, typer.Option("--ubatch-size")] = None,
    prompt_profile: Annotated[str | None, typer.Option("--prompt-profile")] = None,
    n_cpu_moe: Annotated[int | None, typer.Option("--n-cpu-moe")] = None,
    status: Annotated[Status | None, typer.Option("--status")] = None,
    sort: Annotated[SortKey, typer.Option("--sort")] = SortKey.balanced,
    limit: Annotated[int, typer.Option("--limit")] = 20,
) -> None:
    """Show compact run status."""
    filters = _filters(runs, model, quant, context_size, kv_type, batch_size, ubatch_size, prompt_profile, n_cpu_moe, status, sort, limit)
    args = filters.as_report_namespace()
    args.details = details
    from .reporting import load_rows, print_summary

    print_summary(load_rows(args), args)


@report_app.command("by-profile")
def by_profile(
    runs: RunsPath = Path("runs"),
    details: Annotated[bool, typer.Option("--details", help="Show expanded server config columns.")] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable colored terminal output.")] = False,
    color: Annotated[str, typer.Option("--color", help="Color mode: auto, always, never.")] = "auto",
    model: Annotated[str | None, typer.Option("--model")] = None,
    quant: Annotated[str | None, typer.Option("--quant")] = None,
    context_size: Annotated[int | None, typer.Option("--context-size")] = None,
    kv_type: Annotated[str | None, typer.Option("--kv-type")] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size")] = None,
    ubatch_size: Annotated[int | None, typer.Option("--ubatch-size")] = None,
    prompt_profile: Annotated[str | None, typer.Option("--prompt-profile")] = None,
    n_cpu_moe: Annotated[int | None, typer.Option("--n-cpu-moe")] = None,
    status: Annotated[Status | None, typer.Option("--status")] = None,
    sort: Annotated[SortKey, typer.Option("--sort")] = SortKey.generation_tok_s,
    limit: Annotated[int, typer.Option("--limit")] = 20,
) -> None:
    """Show observations split by prompt profile."""
    filters = _filters(runs, model, quant, context_size, kv_type, batch_size, ubatch_size, prompt_profile, n_cpu_moe, status, sort, limit)
    args = filters.as_report_namespace()
    args.details = details
    from .reporting import load_rows, print_by_profile

    print_by_profile(load_rows(args), args)


@report_app.command()
def compare(
    baseline: Annotated[Path, typer.Option("--baseline", help="Baseline run directory or run root.")],
    candidates: Annotated[list[Path], typer.Argument(help="Candidate run directories or run roots.")],
    details: Annotated[bool, typer.Option("--details", help="Show expanded server config columns.")] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable colored terminal output.")] = False,
    color: Annotated[str, typer.Option("--color", help="Color mode: auto, always, never.")] = "auto",
    limit: Annotated[int, typer.Option("--limit")] = 20,
) -> None:
    """Compare candidate run configs against a baseline."""
    from types import SimpleNamespace

    from .reporting import print_compare

    try:
        print_compare(SimpleNamespace(baseline=str(baseline), candidates=[str(path) for path in candidates], limit=limit, details=details))
    except ValueError as exc:
        print(error(str(exc)), file=sys.stderr)
        raise typer.Exit(2) from exc


def main() -> None:
    args = sys.argv[1:]
    no_color = "--no-color" in args or "--no-colour" in args
    color = "auto"
    if "--color" in args:
        index = args.index("--color")
        if index + 1 < len(args):
            color = args[index + 1]
    cleaned = []
    skip_next = False
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in {"--no-color", "--no-colour"}:
            continue
        if arg == "--color":
            skip_next = True
            continue
        cleaned.append(arg)
    configure_color(color=color, no_color=no_color)
    command = get_command(app)
    prog_name = Path(sys.argv[0]).name
    if not cleaned:
        command.main(args=[], prog_name=prog_name, standalone_mode=True, color=not no_color)
        return
    prefix = ["--color", color]
    if no_color:
        prefix.insert(0, "--no-color")
    args = prefix + cleaned
    command.main(args=args, prog_name=prog_name, standalone_mode=True, color=not no_color)


if __name__ == "__main__":
    main()
