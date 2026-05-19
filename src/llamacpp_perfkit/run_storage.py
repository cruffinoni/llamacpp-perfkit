from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class RunStatus(StrEnum):
    success = "success"
    failed = "failed"
    oom = "oom"
    timeout = "timeout"
    unsupported = "unsupported"


@dataclass(frozen=True)
class ServerConfig:
    model: str | None = None
    context_size: int | None = None
    kv_type: str | None = None
    n_cpu_moe: int | None = None
    mtp_enabled: bool | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    spec_type: str | None = None
    spec_draft_n_max: int | None = None
    spec_draft_p_min: float | None = None
    parallel: int | None = None
    n_gpu_layers: int | None = None
    split_mode: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    batch_id: str | None
    created_at: str
    model: str | None
    prompt_profile: str
    server_config: dict[str, Any]
    status: str
    config_hash: str | None = None
    duration_seconds: float | None = None
    request: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    llama_cpp: dict[str, Any] = field(default_factory=dict)
    command: list[str] | str | None = None
    command_shell: str | None = None
    raw_log_path: str | None = None
    monitoring_log_path: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    parsed: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        extra = data.pop("extra", {}) or {}
        data.update(extra)
        return data


@dataclass(frozen=True)
class SystemMetricSample:
    time: str
    gpu_power_w: float | None = None
    gpu_temp_c: float | None = None
    gpu_util_pct: float | None = None
    vram_free_mib: float | None = None
    vram_used_mib: float | None = None
    ram_free_mib: float | None = None
    ram_used_mib: float | None = None


@dataclass(frozen=True)
class LlamaCppMetricSample:
    time: str
    prompt_tokens: float | None = None
    generated_tokens: float | None = None
    prompt_eval_tokens: float | None = None
    eval_tokens: float | None = None
    prompt_eval_tok_s: float | None = None
    generation_tok_s: float | None = None
    total_tokens: float | None = None
    slots_idle: int | None = None
    slots_processing: int | None = None
    ttft_seconds: float | None = None
    total_time_seconds: float | None = None


def run_dir(run_root: Path | str, run_id: str) -> Path:
    return Path(run_root) / run_id


def metrics_dir(path: Path | str) -> Path:
    return Path(path) / "metrics"


def summary_path(path: Path | str) -> Path:
    return Path(path) / "summary.json"


def system_metrics_path(path: Path | str) -> Path:
    return metrics_dir(path) / "system.jsonl"


def llamacpp_metrics_path(path: Path | str) -> Path:
    return metrics_dir(path) / "llamacpp.jsonl"


def _json_ready(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value


def write_run_summary(run_root: Path | str, summary: RunSummary | dict[str, Any]) -> Path:
    data = _json_ready(summary)
    path = run_dir(run_root, data["run_id"])
    metrics_dir(path).mkdir(parents=True, exist_ok=True)
    target = summary_path(path)
    target.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    system_metrics_path(path).touch(exist_ok=True)
    llamacpp_metrics_path(path).touch(exist_ok=True)
    return target


def append_jsonl(path: Path | str, row: dict[str, Any] | SystemMetricSample | LlamaCppMetricSample) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready(row), sort_keys=True, default=str) + "\n")


def append_system_metric(path: Path | str, sample: dict[str, Any] | SystemMetricSample) -> None:
    append_jsonl(system_metrics_path(path), sample)


def append_llamacpp_metric(path: Path | str, sample: dict[str, Any] | LlamaCppMetricSample) -> None:
    append_jsonl(llamacpp_metrics_path(path), sample)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_run_summary(path: Path | str) -> dict[str, Any]:
    return json.loads(summary_path(path).read_text(encoding="utf-8"))


def read_run_metrics(path: Path | str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return read_jsonl(system_metrics_path(path)), read_jsonl(llamacpp_metrics_path(path))


def discover_run_dirs(paths: Path | str | Iterable[Path | str]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        candidates: Iterable[Path | str] = [paths]
    else:
        candidates = paths
    out: list[Path] = []
    seen: set[Path] = set()
    for value in candidates:
        path = Path(value)
        if path.is_file() and path.name == "summary.json":
            possible = path.parent
            if possible not in seen:
                out.append(possible)
                seen.add(possible)
            continue
        if path.is_file():
            path = path.parent
        if not path.exists():
            continue
        if summary_path(path).exists():
            if path not in seen:
                out.append(path)
                seen.add(path)
            continue
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_dir() and summary_path(child).exists() and child not in seen:
                    out.append(child)
                    seen.add(child)
    return out


def _numeric_values(samples: list[dict[str, Any]], key: str) -> list[float]:
    return [float(sample[key]) for sample in samples if isinstance(sample.get(key), (int, float))]


def summarize_system_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    vram_free = _numeric_values(samples, "vram_free_mib")
    vram_used = _numeric_values(samples, "vram_used_mib")
    ram_free = _numeric_values(samples, "ram_free_mib")
    ram_used = _numeric_values(samples, "ram_used_mib")
    gpu_util = _numeric_values(samples, "gpu_util_pct")
    gpu_power = _numeric_values(samples, "gpu_power_w")
    gpu_temp = _numeric_values(samples, "gpu_temp_c")
    return {
        "peak_vram_mib": max(vram_used, default=None),
        "min_vram_free_mib": min(vram_free, default=None),
        "mean_vram_free_mib": (sum(vram_free) / len(vram_free)) if vram_free else None,
        "peak_ram_mib": max(ram_used, default=None),
        "min_ram_free_mib": min(ram_free, default=None),
        "avg_gpu_util_pct": (sum(gpu_util) / len(gpu_util)) if gpu_util else None,
        "peak_gpu_power_w": max(gpu_power, default=None),
        "peak_gpu_temp_c": max(gpu_temp, default=None),
    }


def summarize_llamacpp_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    generation = _numeric_values(samples, "generation_tok_s")
    prompt = _numeric_values(samples, "prompt_eval_tok_s")
    generated_tokens = _numeric_values(samples, "generated_tokens") or _numeric_values(samples, "eval_tokens")
    prompt_tokens = _numeric_values(samples, "prompt_tokens") or _numeric_values(samples, "prompt_eval_tokens")
    total_time = _numeric_values(samples, "total_time_seconds")
    return {
        "generation_tok_s": generation[-1] if generation else None,
        "prompt_eval_tok_s": prompt[-1] if prompt else None,
        "tokens_generated": max(generated_tokens, default=None),
        "tokens_prompt": max(prompt_tokens, default=None),
        "total_time_ms": (total_time[-1] * 1000.0) if total_time else None,
    }


def terminal_llamacpp_sample(time_value: str, parsed: dict[str, Any], duration_seconds: float | None = None) -> dict[str, Any]:
    timings = parsed.get("server_timings") if isinstance(parsed.get("server_timings"), dict) else {}
    prompt_tokens = parsed.get("tokens_prompt") if isinstance(parsed, dict) else None
    generated_tokens = parsed.get("tokens_generated") if isinstance(parsed, dict) else None
    return {
        "time": time_value,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "prompt_eval_tokens": timings.get("prompt_n") or prompt_tokens,  # type: ignore[union-attr]
        "eval_tokens": timings.get("predicted_n") or generated_tokens,  # type: ignore[union-attr]
        "prompt_eval_tok_s": parsed.get("prompt_eval_tok_s"),
        "generation_tok_s": parsed.get("generation_tok_s"),
        "total_tokens": (prompt_tokens + generated_tokens)
        if isinstance(prompt_tokens, (int, float)) and isinstance(generated_tokens, (int, float))
        else None,
        "slots_idle": None,
        "slots_processing": None,
        "total_time_seconds": duration_seconds,
    }


def row_from_run_dir(path: Path | str) -> dict[str, Any]:
    summary = read_run_summary(path)
    system_samples, llamacpp_samples = read_run_metrics(path)
    row = dict(summary)
    metrics = {
        "system": system_samples,
        "llamacpp": llamacpp_samples,
    }
    row["_metrics"] = metrics
    row["monitor"] = {**summarize_system_metrics(system_samples), **(row.get("monitor") or {})}
    parsed = summarize_llamacpp_metrics(llamacpp_samples)
    for key, value in (row.get("parsed") or {}).items():
        if value is not None:
            parsed[key] = value
    row["parsed"] = parsed
    row.setdefault("config", {})
    if not row["config"]:
        row["config"] = {
            "model_hf": row.get("model"),
            "prompt_profile": row.get("prompt_profile", "default"),
            **(row.get("server_config") or {}),
        }
    return row


def load_run_rows(paths: Path | str | Iterable[Path | str]) -> list[dict[str, Any]]:
    return [row_from_run_dir(path) for path in discover_run_dirs(paths)]
