#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .benchlib import PROJECT_ROOT, filter_rows, fmt_value, get_field, table
from .output import (
    colorize_config_label,
    colorize_delta,
    colorize_evidence,
    configure_color,
    format_duration,
    heading,
)
from .run_storage import load_run_rows
from .stats import MetricSummary, summarize


@dataclass(frozen=True)
class ColumnSpec:
    key: str
    short_visible: bool = True
    detail_visible: bool = True
    gradient: bool = False
    gradient_higher_is_better: bool = True


COLUMN_CATALOG: dict[str, ColumnSpec] = {
    "profile": ColumnSpec("profile", short_visible=True, detail_visible=True),
    "score": ColumnSpec("score"),
    "config": ColumnSpec("config", short_visible=True, detail_visible=False),
    "model": ColumnSpec("model", short_visible=False),
    "ctx": ColumnSpec("ctx", short_visible=False),
    "kv": ColumnSpec("kv", short_visible=False),
    "moe": ColumnSpec("moe", short_visible=False),
    "spec": ColumnSpec("spec", short_visible=False),
    "draft": ColumnSpec("draft", short_visible=False),
    "pmin": ColumnSpec("pmin", short_visible=False),
    "batch": ColumnSpec("batch", short_visible=False),
    "ubatch": ColumnSpec("ubatch", short_visible=False),
    "gen tok/s": ColumnSpec("gen tok/s", gradient=True),
    "prompt tok/s": ColumnSpec("prompt tok/s", gradient=True),
    "ttft": ColumnSpec("ttft", gradient=True, gradient_higher_is_better=False),
    "time": ColumnSpec("time", gradient=True, gradient_higher_is_better=False),
    "vram": ColumnSpec("vram", gradient=True),
    "status": ColumnSpec("status"),
}

SUMMARY_COLS = ["score", "config", "gen tok/s", "prompt tok/s", "ttft", "time", "vram", "status"]
SUMMARY_DETAIL_COLS = [
    "score",
    "model",
    "ctx",
    "kv",
    "moe",
    "spec",
    "draft",
    "pmin",
    "batch",
    "ubatch",
    "gen tok/s",
    "prompt tok/s",
    "ttft",
    "time",
    "vram",
    "status",
]
BY_PROFILE_COLS = ["profile", "config", "gen tok/s", "prompt tok/s", "time", "vram", "status"]
BY_PROFILE_DETAIL_COLS = [
    "profile",
    "model",
    "ctx",
    "kv",
    "moe",
    "spec",
    "draft",
    "pmin",
    "batch",
    "ubatch",
    "gen tok/s",
    "prompt tok/s",
    "time",
    "vram",
    "status",
]
COMPARE_COLS = ["score", "config", "gen tok/s", "prompt tok/s", "ttft", "time", "vram", "status"]
COMPARE_DETAIL_COLS = [
    "score",
    "model",
    "ctx",
    "kv",
    "moe",
    "spec",
    "draft",
    "pmin",
    "batch",
    "ubatch",
    "gen tok/s",
    "prompt tok/s",
    "ttft",
    "time",
    "vram",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report on llamacpp-perfkit runs.")
    parser.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--no-color", action="store_true", help="Disable colored terminal output.")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("summary", "by-profile"):
        p = sub.add_parser(name)
        add_common_report_args(p)
        if name == "summary":
            p.add_argument("--details", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("candidates", nargs="+")
    compare.add_argument("--color", choices=["auto", "always", "never"], default=None)
    compare.add_argument("--no-color", action="store_true")
    compare.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def add_common_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--results", help="Compatibility alias for --runs; value is interpreted as a run root.")
    parser.add_argument("--model")
    parser.add_argument("--quant")
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--kv-type")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--ubatch-size", type=int)
    parser.add_argument("--prompt-profile")
    parser.add_argument("--mtp-mode", choices=["on", "off"])
    parser.add_argument("--n-cpu-moe", type=int)
    parser.add_argument("--status", choices=["success", "failed", "oom", "timeout", "unsupported"])
    parser.add_argument(
        "--sort",
        choices=["balanced", "latest", "generation_tok_s", "vram_headroom", "peak_vram", "context_size", "mtp_speedup"],
        default="balanced",
    )
    parser.add_argument("--limit", type=int, default=20)


def path_value(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_args(args: Any) -> Any:
    if getattr(args, "mtp_mode", None) == "on":
        args.mtp_mode = True
    elif getattr(args, "mtp_mode", None) == "off":
        args.mtp_mode = False
    elif not isinstance(getattr(args, "mtp_mode", None), bool):
        args.mtp_mode = None
    return args


def load_rows(args: Any) -> list[dict[str, Any]]:
    args = normalize_args(args)
    source: str = getattr(args, "results", None) or getattr(args, "runs", "runs")  # type: ignore[assignment]
    rows = load_run_rows(path_value(source))
    return filter_rows(rows, args)


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def command_list(row: dict[str, Any]) -> list[str]:
    server = row.get("server") or {}
    command = server.get("server_command")
    if isinstance(command, list) and command:
        return command
    command = row.get("command")
    if isinstance(command, list):
        return command
    if isinstance(command, str):
        return command.split()
    return []


def command_flag_value(cmd: list[str], flag: str) -> str | None:
    if not isinstance(cmd, list):
        return None
    for index, item in enumerate(cmd):
        if item != flag:
            continue
        if index + 1 >= len(cmd):
            return None
        value = cmd[index + 1]
        if isinstance(value, str) and value.startswith("-"):
            return None
        return value
    return None


def command_numeric_value(cmd: list[str], *flags: str) -> int | float | str | None:
    for flag in flags:
        value = command_flag_value(cmd, flag)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value
        return value
    return None


@dataclass(frozen=True)
class ServerConfigKey:
    model: str | None
    context_size: int | None
    kv_type: str | None
    n_cpu_moe: int | None
    mtp_enabled: bool | None
    batch_size: int | float | str | None
    ubatch_size: int | float | str | None
    spec_type: str | None
    spec_draft_n_max: int | float | str | None
    spec_draft_p_min: int | float | str | None
    parallel: int | float | str | None
    n_gpu_layers: int | float | str | None
    split_mode: str | None


@dataclass(frozen=True)
class RunObservation:
    row: dict[str, Any]
    key: ServerConfigKey
    prompt_profile: str
    status: str
    created_at: str | None
    duration_seconds: float | None
    generation_values: tuple[float, ...] = field(default_factory=tuple)
    prompt_values: tuple[float, ...] = field(default_factory=tuple)
    ttft_values: tuple[float, ...] = field(default_factory=tuple)
    total_time_values: tuple[float, ...] = field(default_factory=tuple)
    free_vram_values: tuple[float, ...] = field(default_factory=tuple)
    peak_vram_values: tuple[float, ...] = field(default_factory=tuple)

    @property
    def generation_tok_s(self) -> float | None:
        return summarize(self.generation_values).geometric_mean

    @property
    def prompt_tok_s(self) -> float | None:
        return summarize(self.prompt_values).geometric_mean

    @property
    def min_free_vram(self) -> float | None:
        return min(self.free_vram_values) if self.free_vram_values else None


@dataclass(frozen=True)
class AggregatedServerConfigReport:
    key: ServerConfigKey
    observations: tuple[RunObservation, ...] = field(default_factory=tuple)
    total_runs: int = 0
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    oom_count: int = 0
    profiles_seen: tuple[str, ...] = field(default_factory=tuple)
    score: MetricSummary = field(default_factory=MetricSummary.empty)
    generation_tok_s: MetricSummary = field(default_factory=MetricSummary.empty)
    prompt_tok_s: MetricSummary = field(default_factory=MetricSummary.empty)
    ttft_seconds: MetricSummary = field(default_factory=MetricSummary.empty)
    duration_seconds: MetricSummary = field(default_factory=MetricSummary.empty)
    free_vram_mib: MetricSummary = field(default_factory=MetricSummary.empty)
    peak_vram_mib: MetricSummary = field(default_factory=MetricSummary.empty)
    latest_created_at: str | None = None
    status: str = "unknown"
    evidence: str = "-"

    @property
    def profiles(self) -> int:
        return len(self.profiles_seen)

    @property
    def min_free_vram(self) -> float | None:
        return self.free_vram_mib.min

    @property
    def mean_free_vram(self) -> float | None:
        return self.free_vram_mib.mean

    @property
    def decode_tok_s(self) -> MetricSummary:
        return self.generation_tok_s

    @property
    def prefill_tok_s(self) -> MetricSummary:
        return self.prompt_tok_s

    @property
    def evidence_display(self) -> str:
        return self.evidence

    @classmethod
    def from_observations(
        cls,
        key: ServerConfigKey,
        observations: list[RunObservation],
        context: dict[str, Any] | None = None,
        min_headroom_gb: float = 1.5,
    ) -> AggregatedServerConfigReport:
        total_runs = len(observations)
        success_count = sum(obs.status == "success" for obs in observations)
        timeout_count = sum(obs.status == "timeout" for obs in observations)
        oom_count = sum(obs.status == "oom" for obs in observations)
        failure_count = total_runs - success_count - timeout_count
        profiles_seen = tuple(sorted({obs.prompt_profile for obs in observations if obs.prompt_profile}))
        generation_values = tuple(value for obs in observations for value in obs.generation_values)
        prompt_values = tuple(value for obs in observations for value in obs.prompt_values)
        ttft_values = tuple(value for obs in observations for value in obs.ttft_values)
        duration_values = tuple(
            value
            for obs in observations
            for value in (obs.total_time_values or ((obs.duration_seconds,) if obs.duration_seconds is not None else ()))
        )
        free_vram_values = tuple(value for obs in observations for value in obs.free_vram_values)
        peak_vram_values = tuple(value for obs in observations for value in obs.peak_vram_values)
        status = aggregate_status(observations, min_headroom_gb)
        score_values = [score_observation(obs, context or {}) for obs in observations]
        evidence = f"{status} ({success_count}/{total_runs})" if total_runs else "-"
        return cls(
            key=key,
            observations=tuple(observations),
            total_runs=total_runs,
            success_count=success_count,
            failure_count=failure_count,
            timeout_count=timeout_count,
            oom_count=oom_count,
            profiles_seen=profiles_seen,
            score=summarize(score_values),
            generation_tok_s=summarize(generation_values),
            prompt_tok_s=summarize(prompt_values),
            ttft_seconds=summarize(ttft_values),
            duration_seconds=summarize(duration_values),
            free_vram_mib=summarize(free_vram_values),
            peak_vram_mib=summarize(peak_vram_values),
            latest_created_at=max((obs.created_at for obs in observations if obs.created_at), default=None),
            status=status,
            evidence=evidence,
        )


def numeric_samples(samples: Iterable[dict[str, Any]], key: str) -> tuple[float, ...]:
    return tuple(float(sample[key]) for sample in samples if isinstance(sample.get(key), (int, float)))


def fallback_tuple(value: Any) -> tuple[float, ...]:
    return (float(value),) if isinstance(value, (int, float)) else ()


def summary_profile(row: dict[str, Any]) -> str:
    return get_field(row, "config.prompt_profile") or row.get("prompt_profile") or "default"


def server_config_key_from_row(row: dict[str, Any]) -> ServerConfigKey:
    cfg = row.get("config") or {}
    server_cfg = row.get("server_config") or {}
    cmd = command_list(row)
    return ServerConfigKey(
        model=coalesce(cfg.get("model_hf"), cfg.get("model"), row.get("model"), server_cfg.get("model")),
        context_size=coalesce(cfg.get("context_size"), cfg.get("ctx"), server_cfg.get("context_size")),
        kv_type=coalesce(cfg.get("kv_type"), server_cfg.get("kv_type")),
        n_cpu_moe=coalesce(cfg.get("n_cpu_moe"), server_cfg.get("n_cpu_moe")),
        mtp_enabled=coalesce(cfg.get("mtp_enabled"), server_cfg.get("mtp_enabled")),
        batch_size=coalesce(
            server_cfg.get("batch_size"), cfg.get("batch_size"), command_numeric_value(cmd, "--batch-size", "--batch_size", "-b")
        ),
        ubatch_size=coalesce(
            server_cfg.get("ubatch_size"), cfg.get("ubatch_size"), command_numeric_value(cmd, "--ubatch-size", "--ubatch_size", "-ub")
        ),
        spec_type=coalesce(server_cfg.get("spec_type"), cfg.get("spec_type"), cfg.get("mtp_spec_type")),
        spec_draft_n_max=coalesce(server_cfg.get("spec_draft_n_max"), cfg.get("spec_draft_n_max"), cfg.get("mtp_draft_n_max")),
        spec_draft_p_min=coalesce(server_cfg.get("spec_draft_p_min"), cfg.get("spec_draft_p_min"), cfg.get("mtp_draft_p_min")),
        parallel=coalesce(server_cfg.get("parallel"), cfg.get("parallel"), command_numeric_value(cmd, "--parallel", "-np")),
        n_gpu_layers=coalesce(
            server_cfg.get("n_gpu_layers"), cfg.get("n_gpu_layers"), command_numeric_value(cmd, "--n-gpu-layers", "-ngl")
        ),
        split_mode=coalesce(server_cfg.get("split_mode"), cfg.get("split_mode"), command_flag_value(cmd, "--split-mode")),
    )


def run_observation_from_row(row: dict[str, Any]) -> RunObservation:
    metrics = row.get("_metrics") or {}
    llama = metrics.get("llamacpp") or []
    system = metrics.get("system") or []
    generation = numeric_samples(llama, "generation_tok_s") or fallback_tuple(get_field(row, "parsed.generation_tok_s"))
    prompt = numeric_samples(llama, "prompt_eval_tok_s") or fallback_tuple(get_field(row, "parsed.prompt_eval_tok_s"))
    ttft = numeric_samples(llama, "ttft_seconds")
    total_time = numeric_samples(llama, "total_time_seconds")
    free_vram = numeric_samples(system, "vram_free_mib") or fallback_tuple(get_field(row, "monitor.min_vram_free_mib"))
    peak_vram = numeric_samples(system, "vram_used_mib") or fallback_tuple(get_field(row, "monitor.peak_vram_mib"))
    return RunObservation(
        row=row,
        key=server_config_key_from_row(row),
        prompt_profile=summary_profile(row),
        status=row.get("status") or "unknown",
        created_at=row.get("created_at"),
        duration_seconds=row.get("duration_seconds"),
        generation_values=generation,
        prompt_values=prompt,
        ttft_values=ttft,
        total_time_values=total_time,
        free_vram_values=free_vram,
        peak_vram_values=peak_vram,
    )


def aggregate_status(observations: list[RunObservation], min_headroom_gb: float) -> str:
    statuses = [obs.status for obs in observations]
    for status in ("timeout", "oom", "failed", "unsupported"):
        if status in statuses:
            return status
    if statuses and all(status == "success" for status in statuses):
        free_values = [value for obs in observations for value in obs.free_vram_values]
        if free_values and min(free_values) < min_headroom_gb * 1024:
            return "tight"
        return "stable"
    return statuses[0] if statuses else "unknown"


def summary_context_from_observations(observations: list[RunObservation]) -> dict[str, float | None]:
    speeds = [obs.generation_tok_s for obs in observations if isinstance(obs.generation_tok_s, (int, float))]
    vram_values = [obs.min_free_vram for obs in observations if isinstance(obs.min_free_vram, (int, float))]
    return {
        "fastest_speed": max(speeds, default=None),
        "max_vram": max(vram_values, default=None),
    }


def status_weight(status: str) -> float:
    if status == "success":
        return 1.0
    if status in {"timeout", "oom", "failed"}:
        return 0.0
    return 0.4


def score_observation(obs: RunObservation, context: dict[str, Any]) -> float | None:
    speed = obs.generation_tok_s
    if not isinstance(speed, (int, float)) or speed <= 0:
        return None
    speed_norm = speed / context["fastest_speed"] if context.get("fastest_speed") else 0
    vram = obs.min_free_vram
    vram_norm = max(vram, 0) / context["max_vram"] if isinstance(vram, (int, float)) and context.get("max_vram") else 0
    return (speed_norm * 0.70) + (status_weight(obs.status) * 0.20) + (vram_norm * 0.10)


def aggregate_server_config_reports(
    rows: list[dict[str, Any]], min_headroom_gb: float = 1.5
) -> tuple[list[AggregatedServerConfigReport], dict[str, float | None]]:
    observations = [run_observation_from_row(row) for row in rows]
    context = summary_context_from_observations(observations)
    by_key: dict[ServerConfigKey, list[RunObservation]] = defaultdict(list)
    order: list[ServerConfigKey] = []
    for observation in observations:
        if observation.key not in by_key:
            order.append(observation.key)
        by_key[observation.key].append(observation)
    reports = [AggregatedServerConfigReport.from_observations(key, by_key[key], context, min_headroom_gb) for key in order]
    return reports, context


def enforce_prompt_profile_comparability(reports: list[AggregatedServerConfigReport]) -> None:
    if len(reports) < 2:
        return
    baseline = reports[0].profiles_seen
    for report in reports[1:]:
        if report.profiles_seen != baseline:
            raise ValueError(
                "cannot compare configs: prompt profile sets differ\n"
                f"baseline profiles: {list(baseline)}\n"
                f"candidate profiles: {list(report.profiles_seen)}"
            )


def report_sort_key(report: AggregatedServerConfigReport) -> tuple[bool, float, float, str]:
    score = report.score.mean
    scored = isinstance(score, (int, float))
    return (
        scored,
        float(score) if scored else -1.0,  # type: ignore[arg-type]
        float(report.generation_tok_s.geometric_mean) if isinstance(report.generation_tok_s.geometric_mean, (int, float)) else -1.0,
        report.latest_created_at or "",
    )


def sort_reports(reports: list[AggregatedServerConfigReport], sort_key: str) -> list[AggregatedServerConfigReport]:
    if sort_key == "balanced":
        return sorted(reports, key=report_sort_key, reverse=True)
    if sort_key == "latest":
        return sorted(reports, key=lambda r: (r.latest_created_at is not None, r.latest_created_at or ""), reverse=True)
    if sort_key == "generation_tok_s":
        return sorted(reports, key=lambda r: r.generation_tok_s.geometric_mean or -1, reverse=True)
    if sort_key == "vram_headroom":
        return sorted(reports, key=lambda r: r.min_free_vram or -1, reverse=True)
    if sort_key == "peak_vram":
        return sorted(reports, key=lambda r: r.peak_vram_mib.max or -1, reverse=True)
    if sort_key == "context_size":
        return sorted(reports, key=lambda r: r.key.context_size or -1, reverse=True)
    return reports


def format_metric(summary: MetricSummary) -> str:
    if not isinstance(summary.geometric_mean, (int, float)):
        return "-"
    text = f"g{summary.geometric_mean:.1f}"
    if isinstance(summary.p10, (int, float)):
        text += f" p10:{summary.p10:.1f}"
    return text


def format_seconds(summary: MetricSummary) -> str:
    if isinstance(summary.geometric_mean, (int, float)):
        return format_duration(summary.geometric_mean)
    if isinstance(summary.mean, (int, float)):
        return format_duration(summary.mean)
    return "-"


def format_vram(report: AggregatedServerConfigReport, *, details: bool = False) -> str:
    if not isinstance(report.min_free_vram, (int, float)):
        return "-"
    if details and isinstance(report.mean_free_vram, (int, float)):
        return f"{report.min_free_vram / 1024:.1f}G min {report.mean_free_vram / 1024:.1f}G mean"
    return f"min {report.min_free_vram / 1024:.1f}G"


def format_score(summary: MetricSummary) -> str:
    return f"{summary.mean:.3f}" if isinstance(summary.mean, (int, float)) else "-"


VARIANT_SUFFIXES = ("-it", "-instruct", "-chat", "-mtp")


def short_model_name(full: str | None) -> str:
    if not full:
        return "-"
    name = full.split(":", 1)[0]
    if name.lower().endswith("-gguf"):
        name = name[:-5]
    lower = name.lower()
    for suffix in VARIANT_SUFFIXES:
        if lower.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name or full


def config_label_from_key(key: ServerConfigKey) -> str:
    mtp = "mtp" if key.mtp_enabled else "base"
    spec = key.spec_type or "-"
    batch = f"b{fmt_value(key.batch_size)}" if key.batch_size is not None else ""
    ubatch = f"u{fmt_value(key.ubatch_size)}" if key.ubatch_size is not None else ""
    sizes = f" {batch}/{ubatch}" if batch or ubatch else ""
    model = short_model_name(key.model)
    return f"{model} ctx{fmt_value(key.context_size) if key.context_size is not None else '-'} {key.kv_type or '-'} moe{key.n_cpu_moe if key.n_cpu_moe is not None else '-'} {mtp} {spec}{sizes}"


def config_label(report: AggregatedServerConfigReport) -> str:
    return config_label_from_key(report.key)


class TableBuilder:
    """Builds table rows from AggregatedServerConfigReport, RunObservation, or compare pairs."""

    def __init__(self, columns: list[str], *, details: bool = False, compare: bool = False) -> None:
        self._columns = list(columns)
        self._details = details
        self._compare = compare

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    @property
    def gradients(self) -> dict[str, bool]:
        if self._compare:
            return {}
        return {spec.key: spec.gradient_higher_is_better for spec in COLUMN_CATALOG.values() if spec.key in self._columns and spec.gradient}

    # -- config identity columns -------------------------------------------------

    def _config_cell(self, col: str, key: ServerConfigKey) -> str:
        if col == "config":
            return colorize_config_label(config_label_from_key(key))
        if col == "model":
            return key.model or "-"
        if col == "ctx":
            return str(key.context_size) if key.context_size is not None else "-"
        if col == "kv":
            return key.kv_type or "-"
        if col == "moe":
            return str(key.n_cpu_moe) if key.n_cpu_moe is not None else "-"
        if col == "spec":
            return key.spec_type or ("mtp" if key.mtp_enabled else "base")
        if col == "draft":
            return str(key.spec_draft_n_max) if key.spec_draft_n_max is not None else "-"
        if col == "pmin":
            return str(key.spec_draft_p_min) if key.spec_draft_p_min is not None else "-"
        if col == "batch":
            return str(key.batch_size) if key.batch_size is not None else "-"
        if col == "ubatch":
            return str(key.ubatch_size) if key.ubatch_size is not None else "-"
        return "-"

    # -- row builders -----------------------------------------------------------

    def row_from_report(self, report: AggregatedServerConfigReport) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for col in self._columns:
            if col == "score":
                row["score"] = format_score(report.score)
            elif col == "gen tok/s":
                row["gen tok/s"] = format_metric(report.generation_tok_s)
                row["_raw_gen tok/s"] = report.generation_tok_s.geometric_mean
            elif col == "prompt tok/s":
                row["prompt tok/s"] = format_metric(report.prompt_tok_s)
                row["_raw_prompt tok/s"] = report.prompt_tok_s.geometric_mean
            elif col == "ttft":
                row["ttft"] = format_seconds(report.ttft_seconds)
            elif col == "time":
                row["time"] = format_seconds(report.duration_seconds)
            elif col == "vram":
                row["vram"] = format_vram(report, details=self._details)
                row["_raw_vram"] = report.min_free_vram
            elif col == "status":
                row["status"] = colorize_evidence(report.evidence)
            else:
                row[col] = self._config_cell(col, report.key)
        return row

    def row_from_observation(self, obs: RunObservation) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for col in self._columns:
            if col == "profile":
                row["profile"] = obs.prompt_profile
            elif col == "gen tok/s":
                summary = summarize(obs.generation_values)
                row["gen tok/s"] = format_metric(summary)
                row["_raw_gen tok/s"] = summary.geometric_mean
            elif col == "prompt tok/s":
                summary = summarize(obs.prompt_values)
                row["prompt tok/s"] = format_metric(summary)
                row["_raw_prompt tok/s"] = summary.geometric_mean
            elif col == "time":
                row["time"] = format_duration(obs.duration_seconds) if obs.duration_seconds is not None else "-"
            elif col == "vram":
                vram = obs.min_free_vram
                row["vram"] = f"{vram / 1024:.1f}G" if isinstance(vram, (int, float)) else "-"
                row["_raw_vram"] = vram
            elif col == "status":
                row["status"] = colorize_evidence(obs.status)
            else:
                row[col] = self._config_cell(col, obs.key)
        return row

    def compare_row(self, report: AggregatedServerConfigReport, baseline: AggregatedServerConfigReport | None = None) -> dict[str, Any]:
        b = baseline
        gen_base: Any = b.generation_tok_s.geometric_mean if b else None
        prompt_base: Any = b.prompt_tok_s.geometric_mean if b else None
        ttft_base: Any = b.ttft_seconds.geometric_mean or b.ttft_seconds.mean if b else None
        time_base: Any = b.duration_seconds.geometric_mean or b.duration_seconds.mean if b else None
        vram_base: Any = b.min_free_vram if b else None

        row: dict[str, Any] = {}
        for col in self._columns:
            if col == "score":
                row["score"] = format_score(report.score)
            elif col == "gen tok/s":
                row["gen tok/s"] = self._pct_cell(format_metric(report.generation_tok_s), report.generation_tok_s.geometric_mean, gen_base)
            elif col == "prompt tok/s":
                row["prompt tok/s"] = self._pct_cell(format_metric(report.prompt_tok_s), report.prompt_tok_s.geometric_mean, prompt_base)
            elif col == "ttft":
                cur = report.ttft_seconds.geometric_mean or report.ttft_seconds.mean
                row["ttft"] = self._sec_cell(format_seconds(report.ttft_seconds), cur, ttft_base)
            elif col == "time":
                cur = report.duration_seconds.geometric_mean or report.duration_seconds.mean
                row["time"] = self._sec_cell(format_seconds(report.duration_seconds), cur, time_base)
            elif col == "vram":
                row["vram"] = self._gib_cell(format_vram(report), report.min_free_vram, vram_base)
            elif col == "status":
                row["status"] = colorize_evidence(report.evidence)
            else:
                row[col] = self._config_cell(col, report.key)
        return row

    # -- delta helpers ----------------------------------------------------------

    @staticmethod
    def _pct_cell(fmt_value: str, current: Any, baseline_value: Any) -> str:
        if not isinstance(current, (int, float)) or not isinstance(baseline_value, (int, float)) or baseline_value == 0:
            return fmt_value
        pct = ((current - baseline_value) / baseline_value) * 100.0
        return f"{fmt_value} {colorize_delta(f'{pct:+.1f}%', pct, higher_is_better=True)}"

    @staticmethod
    def _sec_cell(fmt_value: str, current: Any, baseline_value: Any) -> str:
        if not isinstance(current, (int, float)) or not isinstance(baseline_value, (int, float)):
            return fmt_value
        diff = current - baseline_value
        return f"{fmt_value} {colorize_delta(f'{diff:+.2f}s', diff, higher_is_better=False)}"

    @staticmethod
    def _gib_cell(fmt_value: str, current: Any, baseline_value: Any) -> str:
        if not isinstance(current, (int, float)) or not isinstance(baseline_value, (int, float)):
            return fmt_value
        diff = (current - baseline_value) / 1024.0
        return f"{fmt_value} {colorize_delta(f'{diff:+.1f}G', diff, higher_is_better=True, mild_threshold=1.0)}"

    # -- rendering --------------------------------------------------------------

    def render(self, rows: list[dict[str, Any]], limit: int | None = None) -> str:
        return table(rows, self._columns, limit=limit, gradients=self.gradients)


def summary_overview(rows: list[dict[str, Any]], reports: list[AggregatedServerConfigReport]) -> None:
    counts = Counter(row.get("status") or "unknown" for row in rows)
    status_text = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
    suffix = f" ({status_text})" if status_text else ""
    print(heading(f"Groups: {len(reports)} from {len(rows)} runs{suffix}"))


def print_summary(rows: list[dict[str, Any]], args: Any) -> None:
    reports, _ = aggregate_server_config_reports(rows)
    reports = sort_reports(reports, args.sort)
    summary_overview(rows, reports)
    details = getattr(args, "details", False)
    columns = SUMMARY_DETAIL_COLS if details else SUMMARY_COLS
    builder = TableBuilder(columns, details=details)
    print(builder.render([builder.row_from_report(r) for r in reports], limit=args.limit))


def print_by_profile(rows: list[dict[str, Any]], args: Any) -> None:
    details = getattr(args, "details", False)
    columns = BY_PROFILE_DETAIL_COLS if details else BY_PROFILE_COLS
    builder = TableBuilder(columns, details=details)
    observations = [run_observation_from_row(row) for row in rows]
    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        by_profile[obs.prompt_profile].append(builder.row_from_observation(obs))
    for profile in sorted(by_profile):
        print(heading(profile))
        print(builder.render(by_profile[profile], limit=args.limit))


def load_reports_for_path(path: str) -> list[AggregatedServerConfigReport]:
    rows = load_run_rows(path_value(path))
    reports, _ = aggregate_server_config_reports(rows)
    return sort_reports(reports, "balanced")


def print_compare(args: Any) -> None:
    baseline_reports = load_reports_for_path(args.baseline)
    if not baseline_reports:
        raise ValueError(f"baseline has no runs: {args.baseline}")
    baseline = baseline_reports[0]
    candidate_reports: list[AggregatedServerConfigReport] = []
    for candidate in args.candidates:
        reports = load_reports_for_path(candidate)
        if not reports:
            raise ValueError(f"candidate has no runs: {candidate}")
        candidate_reports.extend(reports)
    enforce_prompt_profile_comparability([baseline, *candidate_reports])
    details = getattr(args, "details", False)
    columns = COMPARE_DETAIL_COLS if details else COMPARE_COLS
    builder = TableBuilder(columns, details=details, compare=True)
    rows: list[dict[str, Any]] = [builder.compare_row(baseline)]
    rows.extend(builder.compare_row(report, baseline) for report in candidate_reports)
    print(builder.render(rows, limit=args.limit))


def configure_color_from_args(args: Any) -> None:
    color_mode = getattr(args, "color", None) or "auto"
    no_color = bool(getattr(args, "no_color", False))
    configure_color(color=color_mode, no_color=no_color)


def main() -> int:
    args = parse_args()
    configure_color_from_args(args)
    try:
        if args.command == "summary":
            print_summary(load_rows(args), args)
        elif args.command == "by-profile":
            print_by_profile(load_rows(args), args)
        elif args.command == "compare":
            print_compare(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
