from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from typer.main import get_command

from .benchlib import DEFAULT_CONFIG_PATH
from .models import BenchOptions, Mode, MtpMode, ReportFilters, SortKey, Status
from .output import configure_color, error
from .services import BenchmarkService, ConfigError, ConfigLoader, FeatureDetector

app = typer.Typer(
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
    mtp_mode: MtpMode | None,
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
        mtp_mode=mtp_mode,
        n_cpu_moe=n_cpu_moe,
        status=status,
        sort=sort,
        limit=limit,
    )


@app.command()
def detect(config: ConfigPath = DEFAULT_CONFIG_PATH) -> None:
    """Detect local llama.cpp feature support."""
    raise typer.Exit(FeatureDetector().run(_cfg(config)))


@app.command()
def bench(
    config: ConfigPath = DEFAULT_CONFIG_PATH,
    limit: Annotated[int | None, typer.Option("--limit", help="Backward-compatible alias for --max-runs.")] = None,
    max_runs: Annotated[int | None, typer.Option("--max-runs", help="Run at most N new benchmark requests.")] = None,
    mode: Annotated[Mode | None, typer.Option("--mode", help="Override budget.mode.")] = None,
    retry_failed: Annotated[
        bool, typer.Option("--retry-failed", help="Rerun failed, OOM, timeout, or unsupported configurations.")
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print planned server commands without running models.")] = False,
) -> None:
    """Plan and run benchmark requests."""
    options = BenchOptions(mode=mode, max_runs=max_runs, limit=limit, retry_failed=retry_failed, dry_run=dry_run)
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
    mtp_mode: Annotated[MtpMode | None, typer.Option("--mtp-mode")] = None,
    n_cpu_moe: Annotated[int | None, typer.Option("--n-cpu-moe")] = None,
    status: Annotated[Status | None, typer.Option("--status")] = None,
    sort: Annotated[SortKey, typer.Option("--sort")] = SortKey.balanced,
    limit: Annotated[int, typer.Option("--limit")] = 20,
) -> None:
    """Show compact run status."""
    filters = _filters(
        runs, model, quant, context_size, kv_type, batch_size, ubatch_size, prompt_profile, mtp_mode, n_cpu_moe, status, sort, limit
    )
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
    mtp_mode: Annotated[MtpMode | None, typer.Option("--mtp-mode")] = None,
    n_cpu_moe: Annotated[int | None, typer.Option("--n-cpu-moe")] = None,
    status: Annotated[Status | None, typer.Option("--status")] = None,
    sort: Annotated[SortKey, typer.Option("--sort")] = SortKey.generation_tok_s,
    limit: Annotated[int, typer.Option("--limit")] = 20,
) -> None:
    """Show observations split by prompt profile."""
    filters = _filters(
        runs, model, quant, context_size, kv_type, batch_size, ubatch_size, prompt_profile, mtp_mode, n_cpu_moe, status, sort, limit
    )
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
    prefix = ["--color", color]
    if no_color:
        prefix.insert(0, "--no-color")
    args = prefix + cleaned
    configure_color(color=color, no_color=no_color)
    command = get_command(app)
    command.main(args=args, prog_name=sys.argv[0], standalone_mode=True, color=not no_color)


if __name__ == "__main__":
    main()
