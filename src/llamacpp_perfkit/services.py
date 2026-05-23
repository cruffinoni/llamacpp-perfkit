from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .benchlib import (
    build_server_cmd,
    command_template,
    command_to_shell,
    detect_features,
    fmt_value,
    llama_bin_dir,
    llama_cpp_git_metadata,
    load_config,
    load_features,
    normalize_extra_args,
    output_dirs,
    request_args_from_features,
    server_args_from_features,
    server_group_key,
    successful,
    write_features,
)
from .models import BenchmarkConfig, BenchOptions, GenerateOptions, ReportFilters
from .output import command, error, heading, skip, warning
from .output import note as info
from .planner import make_plan, write_plan
from .reporting import (
    load_rows,
    print_summary,
)
from .run_storage import load_run_rows
from .server_runner import LlamaServerBenchmarkRunner


class F(list[Any]):
    pass


yaml.add_representer(F, lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True))


class ConfigError(Exception):
    pass


class ConfigLoader:
    def resolve_path(self, value: Path | str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else Path.cwd() / path

    def load(self, value: Path | str) -> dict[str, Any]:
        path = self.resolve_path(value)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        if path.is_dir():
            raise ConfigError(f"Config path is a directory, expected a YAML file: {path}")
        try:
            cfg = load_config(path)
        except PermissionError as exc:
            raise ConfigError(f"Config file is not readable: {path}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"Config file is not valid YAML: {path}\n{exc}") from exc
        except OSError as exc:
            raise ConfigError(f"Could not read config file {path}: {exc}") from exc
        try:
            BenchmarkConfig.model_validate(cfg)
        except ValidationError as exc:
            raise ConfigError(f"Invalid config file {path}:\n{exc}") from exc
        cfg["llama"]["preferred_binary"] = "llama-server"
        return cfg


class FeatureDetector:
    def run(self, cfg: dict[str, Any]) -> int:
        _, results_dir, _, _ = output_dirs(cfg)
        features = detect_features(cfg)
        self.resolve_feature_args(cfg, features)
        json_path, txt_path = write_features(features, results_dir)
        print(info(f"Wrote {json_path}"))
        print(info(f"Wrote {txt_path}"))
        llama_cpp = features.get("llama_cpp") or {}
        if llama_cpp.get("commit_short"):
            print(info(f"llama.cpp commit: {llama_cpp['commit_short']}"))
        else:
            print(warning(f"llama.cpp commit unavailable: {llama_cpp.get('error') or 'unknown'}"))
        print(f"valid_for_bench: {features['valid_for_bench']}")
        for item in features["kv"]["skipped"]:
            print(skip(f"skip kv_type={item['value']}: {item['reason']}"))
        for item in features.get("spec", {}).get("skipped", []):
            print(skip(f"skip spec_type={item['value']}: {item['reason']}"))
        for item in features.get("extra_args", {}).get("skipped", []):
            print(skip(f"skip extra_arg={item['flag']}: {item['reason']}"))
        return 0 if features["valid_for_bench"] else 2

    @staticmethod
    def resolve_feature_args(cfg: dict[str, Any], features: dict[str, Any]) -> None:
        cfg["_resolved_server_args"] = server_args_from_features(features)
        cfg["_resolved_request_args"] = request_args_from_features(features)
        cfg["_llama_cpp"] = features.get("llama_cpp") or {}


class BenchmarkService:
    def __init__(self) -> None:
        self.features = FeatureDetector()

    def run(self, cfg: dict[str, Any], options: BenchOptions) -> int:
        _, results_dir, raw_dir, mon_dir = output_dirs(cfg)
        features = load_features(results_dir)
        requested_extra_args = normalize_extra_args(cfg)
        stale_extra_args = features and features.get("extra_args", {}).get("requested") != requested_extra_args
        stale_bin_dir = features and features.get("bin_dir") != str(llama_bin_dir(cfg))
        current_llama_cpp = llama_cpp_git_metadata(cfg)
        feature_llama_cpp = (features or {}).get("llama_cpp") or {}
        stale_llama_cpp = features and (
            "llama_cpp" not in features
            or feature_llama_cpp.get("commit") != current_llama_cpp.get("commit")
            or ("branch" not in feature_llama_cpp and current_llama_cpp.get("branch") is not None)
        )
        if (
            not features
            or not features.get("valid_for_bench")
            or stale_extra_args
            or stale_bin_dir
            or stale_llama_cpp
            or features.get("backend") != "server"
        ):
            if stale_extra_args:
                reason = "changed extra args"
            elif stale_bin_dir:
                reason = "changed llama.bin_dir"
            elif stale_llama_cpp:
                reason = "missing or changed llama.cpp commit metadata"
            else:
                reason = "missing, invalid, or not server-backed"
            print(info(f"Feature detection is {reason}; running detect first."))
            features = detect_features(cfg)
            write_features(features, results_dir)
        if not features.get("valid_for_bench"):
            print(error(f"Cannot benchmark: {features.get('invalid_reason')}"))
            return 2
        self.features.resolve_feature_args(cfg, features)

        max_runs = options.max_runs if options.max_runs is not None else options.limit
        plan_path = results_dir / "last_plan.json"
        rows = load_run_rows(results_dir)
        plan = make_plan(
            cfg,
            features,
            rows,
            mode=options.mode.value if options.mode else None,
            max_runs=max_runs,
            retry_failed=options.retry_failed,
            force=options.force,
        )
        write_plan(plan, plan_path)
        if options.dry_run:
            self.print_plan(plan, cfg, features, raw_dir)
            return 0

        planned_new = int(plan.get("estimated_runs", 0))
        if planned_new == 0:
            print(info("No runnable jobs after feature filtering, reuse, dependency, and risk checks."))
            return 0

        plan_limit = int(plan.get("max_runs") or 0)
        max_new_runs_val: int | None = max_runs if max_runs is not None else plan_limit
        if max_new_runs_val is not None:
            max_new_runs_val = int(max_new_runs_val)
            if max_new_runs_val <= 0:
                max_new_runs_val = None
        runner_args = options.model_dump()
        if options.mode:
            runner_args["mode"] = options.mode.value
        runner = LlamaServerBenchmarkRunner(cfg, _Namespace(runner_args), features, results_dir, raw_dir, mon_dir)
        runner.run(max_new_runs_val)
        return 0

    def print_plan(self, plan: dict[str, Any], cfg: dict[str, Any], features: dict[str, Any], raw_dir: Path) -> None:
        print(heading(f"Budget mode: {plan['mode']}"))
        max_runs_label = "unlimited" if int(plan["max_runs"]) <= 0 else str(plan["max_runs"])
        print(f"Max new requests: {max_runs_label}")
        print(f"Reuse existing results: {plan['reuse_existing_results']}")
        print(f"Candidate combinations: {fmt_value(plan.get('candidate_count'))}")
        print(f"Selected plan entries: {fmt_value(plan.get('selected_count'))}")
        print(f"Estimated new requests now: {fmt_value(plan['estimated_runs'])}")
        llama_cpp = features.get("llama_cpp") or {}
        if llama_cpp.get("commit_short"):
            print(info(f"llama.cpp commit: {llama_cpp['commit_short']}"))
        else:
            print(warning(f"llama.cpp commit unavailable: {llama_cpp.get('error') or 'unknown'}"))
        if server_args_from_features(features):
            rendered = []
            for item in server_args_from_features(features):
                value = item.get("value")
                rendered.append(item["flag"] if value is True else f"{item['flag']} {value}")
            print(info(f"Fixed llama-server args: {' '.join(rendered)}"))
        if request_args_from_features(features):
            rendered = [f"{key}={value}" for key, value in sorted(request_args_from_features(features).items())]
            print(info(f"Request args: {' '.join(rendered)}"))
        if plan.get("max_runs_capped"):
            print(info("note: max_runs capped the selected plan."))
        if plan.get("warning"):
            print(warning(f"WARNING: {plan['warning']}"))
        for plan_note in plan.get("notes", []):
            print(info(f"note: {plan_note}"))
        seen_skips: set[str] = set()
        for item in plan.get("skipped", []):
            skip_key = str(sorted(item.items()))
            if skip_key not in seen_skips:
                print(skip(f"skip {item.get('dimension')}={item.get('value', '*')}: {item['reason']}"))
                seen_skips.add(skip_key)
        print(heading("Planned requests:"))
        groups: list[list[dict[str, Any]]] = []
        by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for item in plan["planned"]:
            key = server_group_key(item["job"])
            if key not in by_group:
                by_group[key] = []
                groups.append(by_group[key])
            by_group[key].append(item)

        for group_index, group in enumerate(groups, 1):
            representative = group[0]["job"]
            server_run_id = f"dry-run-server-{group_index:04d}"
            counts = {name: 0 for name in ("run", "reuse", "skip", "blocked")}
            for item in group:
                if item["action"] in counts:
                    counts[item["action"]] += 1
            print(
                heading(
                    f"[server {group_index}/{len(groups)}] {server_run_id} entries={len(group)} "
                    f"run={counts['run']} reuse={counts['reuse']} skip={counts['skip']} blocked={counts['blocked']} "
                    f"ctx={fmt_value(representative['context_size'])} kv={representative['kv_type']} "
                    f"n_cpu_moe={representative['n_cpu_moe']} spec={representative.get('spec_type') or 'none'} "
                    f"batch={fmt_value(representative.get('batch_size'))} ubatch={fmt_value(representative.get('ubatch_size'))}"
                )
            )
            runnable = next((item for item in group if item["action"] == "run"), None)
            if runnable:
                raw_path = raw_dir / f"{server_run_id}.log"
                print(command(command_to_shell(build_server_cmd(cfg, features, runnable["job"], 0, raw_path))))
            else:
                print(info("  no llama-server would start for this group; all entries are non-runnable."))

            for item in group:
                job = item["job"]
                action_reason = f" reason={item['action_reason']}" if item.get("action_reason") else ""
                print(
                    f"  request plan_id={item['run_id']} action={item['action']} "
                    f"profile={job.get('prompt_profile', 'default')} hash={item['config_hash']} "
                    f"spec_type={job.get('spec_type') or 'none'} risk={item['risk_level']}{action_reason}"
                )


class ReportService:
    def _rows(self, filters: ReportFilters) -> tuple[list[dict[str, Any]], Any]:
        args = filters.as_report_namespace()
        return load_rows(args), args

    def summary(self, filters: ReportFilters) -> None:
        rows, args = self._rows(filters)
        print_summary(rows, args)


class RecommendationService:
    def run_simple(self, cfg: dict[str, Any]) -> int:
        _, results_dir, _, _ = output_dirs(cfg)
        rows = successful(load_run_rows(results_dir))
        if not rows:
            print(info("No successful benchmark rows found."))
            return 1
        safe = [r for r in rows if r.get("stability_status") == "stable"]
        best = max(safe or rows, key=lambda r: r["parsed"]["generation_tok_s"] or 0)
        fallback_pool = safe or rows
        safe_fallback = max(
            fallback_pool,
            key=lambda r: (r.get("monitor", {}).get("min_vram_free_mib") or 0, r["parsed"]["generation_tok_s"] or 0),
        )
        print(heading("Recommended llama-server command:"))
        print(command(command_template(best.get("server", {}).get("server_command") or best["command"], {"--port": "$PORT"})))
        print("\n" + heading("Safe fallback llama-server command:"))
        print(
            command(
                command_template(safe_fallback.get("server", {}).get("server_command") or safe_fallback["command"], {"--port": "$PORT"})
            )
        )
        return 0


class GenerateService:
    def run(self, options: GenerateOptions) -> int:
        filename = self._derive_filename(options.model, options.name)
        output_dir = options.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        config = self._build_config(options)
        output_path.write_text(yaml.dump(config, sort_keys=False), encoding="utf-8")
        print(info(f"Wrote {output_path}"))
        return 0

    @staticmethod
    def _derive_filename(model: str, name: str | None) -> str:
        if name:
            return name if name.endswith(".yaml") else f"{name}.yaml"
        owner_free = model.rsplit("/", 1)[-1]
        parts = owner_free.split(":", 1)
        stem = "-".join(parts).lower() if len(parts) > 1 else parts[0].lower()
        sanitized = "".join(c if c.isalnum() or c in ".-_" else "-" for c in stem).replace("--", "-")
        return f"{sanitized}.yaml"

    @staticmethod
    def _build_config(options: GenerateOptions) -> dict[str, Any]:
        return {
            "models": {"hf": options.model},
            "llama": {
                "bin_dir": str(options.llama_bin),
                "preferred_binary": "llama-server",
                "server": {
                    "host": "127.0.0.1",
                    "startup_timeout_seconds": 300,
                    "shutdown_timeout_seconds": 15,
                },
                "extra_args": {
                    "--temp": options.temp,
                    "--top-p": options.top_p,
                    "--top-k": options.top_k,
                    "--presence-penalty": options.presence_penalty,
                    "--min-p": options.min_p,
                    "--n-gpu-layers": options.n_gpu_layers,
                    "--split-mode": options.split_mode,
                    "--parallel": options.parallel,
                },
            },
            "prompt": {
                "profiles": [
                    {"name": "code_python", "file": "prompts/profiles/code_python.txt"},
                    {"name": "code_cpp", "file": "prompts/profiles/code_cpp.txt"},
                    {"name": "long_code_review", "file": "prompts/profiles/long_code_review.txt"},
                    {"name": "long_prefill_8k", "file": "prompts/profiles/long_prefill_8k.txt"},
                    {"name": "long_prefill_16k", "file": "prompts/profiles/long_prefill_16k.txt"},
                    {"name": "long_prefill_32k", "file": "prompts/profiles/long_prefill_32k.txt"},
                    {"name": "long_prefill_48k", "file": "prompts/profiles/long_prefill_48k.txt"},
                    {"name": "long_prefill_60k", "file": "prompts/profiles/long_prefill_60k.txt"},
                ],
            },
            "run": {
                "endpoint": "chat",
                "generation_tokens": 192,
                "seed": 42,
                "min_vram_headroom_gb": 0.5,
                "monitor_interval_seconds": 1,
                "timeout_seconds": 900,
                "cache_prompt": False,
            },
            "budget": {
                "mode": "full",
                "max_runs": 0,
                "reuse_existing_results": True,
                "stop_if_all_remaining_are_risky": True,
            },
            "matrix": {
                "n_cpu_moe": F([0, 6, 12, 18, 24]),
                "context_size": F([4096, 8192, 16384, 32768]),
                "kv_type": F(["q8_0"]),
                "batch_size": F([512]),
                "ubatch_size": F([512]),
                "spec_type": F(["draft-mtp"]),
                "spec_draft_n_max": F([2]),
                "spec_draft_p_min": F([0.75]),
            },
            "output": {
                "logs_dir": "logs",
                "results_dir": "runs",
            },
        }


class _Namespace:
    def __init__(self, values: dict[str, Any]) -> None:
        self.__dict__.update(values)
