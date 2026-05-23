from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from .benchlib import (
    benchmark_invalid_reason,
    config_hash,
    get_field,
    prompt_profiles,
    row_config_hash,
    successful,
)


def budget_config(cfg: dict[str, Any], mode: str | None = None, max_runs: int | None = None) -> dict[str, Any]:
    budget = dict(cfg.get("budget", {}))
    mode_defaults = {"smoke": 2, "quick": 8, "focused": 16, "full": 0}
    if mode:
        budget["mode"] = mode
        if max_runs is None:
            budget["max_runs"] = mode_defaults.get(mode, 8)
    if max_runs is not None:
        budget["max_runs"] = max_runs
    if budget.get("max_runs") is None:
        budget["max_runs"] = mode_defaults.get(budget.get("mode", "quick"), 8)
    return budget


def candidate_jobs(cfg: dict[str, Any], features: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrix = cfg.get("matrix", {})
    flags = features["flags"]["llama_server"]
    skipped: list[dict[str, Any]] = []
    kv_values = usable_kv_values(cfg, features)
    if not kv_values:
        skipped.append({"dimension": "kv_type", "reason": "no configured KV values are supported"})
    supported_kv_values = features.get("kv", {}).get("supported_values") or []
    for value in matrix.get("kv_type", []):
        if supported_kv_values and value not in supported_kv_values:
            skipped.append({"dimension": "kv_type", "value": value, "reason": "not listed in local cache-type allowed values"})

    n_cpu_moe_values: list[int | None] = matrix.get("n_cpu_moe", [0])
    if not flags.get("n_cpu_moe"):
        skipped.append({"dimension": "n_cpu_moe", "reason": "local llama-server lacks --n-cpu-moe"})
        n_cpu_moe_values = [None]

    batch_size_values = matrix.get("batch_size", [1024])
    ubatch_size_values = matrix.get("ubatch_size", [1024])

    spec_type_values: list[Any] = matrix.get("spec_type", [])
    if not spec_type_values:
        spec_type_values = [None]

    supported_spec_types = features.get("spec", {}).get("supported_values") or []
    for value in spec_type_values:
        if value is not None and supported_spec_types and value not in supported_spec_types:
            skipped.append({"dimension": "spec_type", "value": value, "reason": "not listed in local --spec-type allowed values"})

    jobs: list[dict[str, Any]] = []
    profiles = prompt_profiles(cfg)
    base_product = itertools.product(
        profiles,
        n_cpu_moe_values,
        matrix.get("context_size", [4096]),
        kv_values,
        batch_size_values,
        ubatch_size_values,
        spec_type_values,
    )
    for profile, n_cpu_moe, ctx, kv, batch, ubatch, spec_type in base_product:
        if spec_type is not None:
            draft_values: list[Any] = matrix.get("spec_draft_n_max", [None]) if flags.get("spec_draft_n_max") else [None]
            p_values: list[Any] = matrix.get("spec_draft_p_min", [None]) if flags.get("spec_draft_p_min") else [None]
        else:
            draft_values = [None]
            p_values = [None]
        for draft_n, p_min in itertools.product(draft_values, p_values):
            jobs.append(
                {
                    "prompt_profile": profile["name"],
                    "prompt_file": profile["file"],
                    "n_cpu_moe": n_cpu_moe,
                    "context_size": ctx,
                    "kv_type": kv,
                    "batch_size": batch,
                    "ubatch_size": ubatch,
                    "spec_type": spec_type,
                    "spec_draft_n_max": draft_n,
                    "spec_draft_p_min": p_min,
                }
            )
    return jobs, skipped


def annotate_jobs(cfg: dict[str, Any], jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for job in jobs:
        item = dict(job)
        item["config_hash"] = config_hash(cfg, item)
        out.append(item)
    return out


def result_indexes(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_hash: dict[str, dict[str, Any]] = {}
    for row in rows:
        h = row_config_hash(cfg, row)
        if not h:
            continue
        current = by_hash.get(h)
        if current is None or row.get("created_at", "") > current.get("created_at", ""):
            by_hash[h] = row
    return by_hash


def is_safe(row: dict[str, Any] | None, min_headroom_gb: float) -> bool:
    if not row or row.get("status") != "success":
        return False
    if benchmark_invalid_reason(row) is not None:
        return False
    if row.get("stability_status") == "stable":
        return True
    free = get_field(row, "monitor.min_vram_free_mib")
    return free is not None and free >= min_headroom_gb * 1024


def job_key(job: dict[str, Any]) -> tuple[Any, ...]:
    return (
        job.get("prompt_profile"),
        job.get("prompt_file"),
        job.get("context_size"),
        job.get("kv_type"),
        job.get("n_cpu_moe"),
        job.get("batch_size"),
        job.get("ubatch_size"),
        job.get("spec_type"),
        job.get("spec_draft_n_max"),
        job.get("spec_draft_p_min"),
    )


def baseline_key(row_or_job: dict[str, Any]) -> tuple[Any, ...]:
    cfg = row_or_job.get("config", row_or_job)
    return (
        cfg.get("context_size"),
        cfg.get("kv_type"),
        cfg.get("n_cpu_moe"),
        cfg.get("generation_tokens"),
        cfg.get("seed"),
        cfg.get("prompt_profile", "default"),
        cfg.get("prompt_file"),
    )


def target_context(cfg: dict[str, Any]) -> int:
    values: list[int] = cfg.get("matrix", {}).get("context_size", [4096])
    return values[0] if values else 4096


def usable_kv_values(cfg: dict[str, Any], features: dict[str, Any]) -> list[str]:
    requested: list[str] = cfg.get("matrix", {}).get("kv_type", [])
    supported: list[str] = features.get("kv", {}).get("supported_values") or []
    return [value for value in requested if not supported or value in supported]


def first_kv(cfg: dict[str, Any], features: dict[str, Any]) -> str | None:
    values = usable_kv_values(cfg, features)
    return values[0] if values else None


def risk_level(job: dict[str, Any], rows: list[dict[str, Any]], min_headroom_gb: float) -> str:
    ctx = job.get("context_size") or 0
    moe = job.get("n_cpu_moe")
    for row in rows:
        rcfg = row.get("config", {})
        if rcfg.get("kv_type") != job.get("kv_type"):
            continue
        rctx = rcfg.get("context_size") or 0
        rmoe = rcfg.get("n_cpu_moe")
        if row.get("status") == "oom" and ctx >= rctx and (moe is None or rmoe is None or moe <= rmoe):
            return "high"
        if (
            not is_safe(row, min_headroom_gb)
            and row.get("status") == "success"
            and ctx >= rctx
            and (moe is None or rmoe is None or moe <= rmoe)
        ):
            return "high"
    return "low"


def select_smoke(cfg: dict[str, Any], candidates: list[dict[str, Any]], features: dict[str, Any]) -> list[dict[str, Any]]:
    ctx = min(cfg.get("matrix", {}).get("context_size", [target_context(cfg)]))
    kv = first_kv(cfg, features)
    n_values = [c.get("n_cpu_moe") for c in candidates if c.get("context_size") == ctx and c.get("kv_type") == kv]
    n_cpu = max([n for n in n_values if n is not None], default=None)
    selected = [c for c in candidates if c.get("context_size") == ctx and c.get("kv_type") == kv and c.get("n_cpu_moe") == n_cpu]
    return selected[:1]


def safest_n_cpu_moe(cfg: dict[str, Any], candidates: list[dict[str, Any]]) -> int | None:
    values: list[int] = [c["n_cpu_moe"] for c in candidates if isinstance(c.get("n_cpu_moe"), int)]
    if values:
        return max(values)
    configured = [n for n in cfg.get("matrix", {}).get("n_cpu_moe", []) if n is not None]
    return max(configured) if configured else None


def select_quick(
    cfg: dict[str, Any],
    candidates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    features: dict[str, Any],
    max_runs: int,
    min_headroom_gb: float,
    candidate_hashes: set[str] | None = None,
) -> list[dict[str, Any]]:
    ctx = target_context(cfg)
    kv = first_kv(cfg, features)
    n_cpu = safest_n_cpu_moe(cfg, candidates)
    selected = [c for c in candidates if c.get("context_size") == ctx and c.get("kv_type") == kv and c.get("n_cpu_moe") == n_cpu]
    return unique_jobs(selected)[:max_runs]


def select_focused(
    cfg: dict[str, Any],
    candidates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    features: dict[str, Any],
    max_runs: int,
    min_headroom_gb: float,
    candidate_hashes: set[str] | None = None,
) -> list[dict[str, Any]]:
    safe_rows = [r for r in successful(rows) if is_safe(r, min_headroom_gb)]
    safe_rows.sort(key=lambda r: get_field(r, "parsed.generation_tok_s") or 0, reverse=True)
    if not safe_rows:
        return select_quick(cfg, candidates, rows, features, min(8, max_runs), min_headroom_gb, candidate_hashes)
    best = safe_rows[0].get("config", {})
    n_values: list[Any] = cfg.get("matrix", {}).get("n_cpu_moe", [best.get("n_cpu_moe")])
    try:
        idx = n_values.index(best.get("n_cpu_moe"))
        neighbor_ns = [n_values[i] for i in range(max(0, idx - 2), min(len(n_values), idx + 3))]
    except ValueError:
        neighbor_ns = [best.get("n_cpu_moe")]
    kv_values = (features["kv"]["usable_values"] or cfg.get("matrix", {}).get("kv_type", []))[:2]
    selected: list[dict[str, Any]] = [
        c
        for c in candidates
        if c.get("context_size") == best.get("context_size") and c.get("kv_type") in kv_values and c.get("n_cpu_moe") in neighbor_ns
    ]
    return unique_jobs(selected)[:max_runs]


def unique_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[Any] = set()
    out: list[dict[str, Any]] = []
    for job in jobs:
        key = job.get("config_hash") or job_key(job)
        if key in seen:
            continue
        seen.add(key)
        out.append(job)
    return out


def action_for_job(
    job: dict[str, Any], by_hash: dict[str, dict[str, Any]], retry_failed: bool, reuse_existing: bool, force: bool = False
) -> tuple[str, str | None]:
    if force:
        return "run", None
    existing = by_hash.get(job["config_hash"])
    if existing and existing.get("status") == "success":
        invalid_reason = benchmark_invalid_reason(existing)
        if invalid_reason is None and reuse_existing:
            return "reuse", f"successful result already exists: {existing.get('run_id')}"
        if invalid_reason is not None:
            return "run", f"previous successful result is invalid: {invalid_reason}"
    if existing and existing.get("status") != "success" and not retry_failed:
        return "skip", f"previous result is {existing.get('status')}; pass --retry-failed to rerun"
    return "run", None


def make_plan(
    cfg: dict[str, Any],
    features: dict[str, Any],
    rows: list[dict[str, Any]],
    mode: str | None = None,
    max_runs: int | None = None,
    retry_failed: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    budget = budget_config(cfg, mode, max_runs)
    mode = budget.get("mode", "quick")
    max_runs = int(budget.get("max_runs", 8))
    min_headroom = float(cfg.get("run", {}).get("min_vram_headroom_gb", 1.5))
    candidates, skipped = candidate_jobs(cfg, features)
    candidates = annotate_jobs(cfg, candidates)
    candidate_hashes = {job["config_hash"] for job in candidates}
    by_hash = result_indexes(cfg, rows)

    if mode == "smoke":
        selected = select_smoke(cfg, candidates, features)
    elif mode == "focused":
        selected = select_focused(cfg, candidates, rows, features, max_runs or 16, min_headroom, candidate_hashes)
    elif mode == "full":
        selected = candidates
    else:
        selected = select_quick(cfg, candidates, rows, features, max_runs or 8, min_headroom, candidate_hashes)

    selected = unique_jobs(selected)
    uncapped_selected_count = len(selected)
    if max_runs > 0:
        selected = selected[:max_runs]

    plan_runs: list[dict[str, Any]] = []
    for idx, job in enumerate(selected, 1):
        action, reason = action_for_job(job, by_hash, retry_failed, bool(budget.get("reuse_existing_results", True)), force)
        risk = risk_level(job, rows, min_headroom)
        if mode != "full" and budget.get("stop_if_all_remaining_are_risky") and risk == "high" and action == "run":
            action, reason = "skip", "risk is high after prior OOM/unsafe result"
        plan_runs.append(
            {
                "run_id": f"plan-{idx:04d}",
                "config_hash": job["config_hash"],
                "reason": selection_reason(mode, job),
                "risk_level": risk,
                "action": action,
                "action_reason": reason,
                "job": job,
            }
        )

    return {
        "mode": mode,
        "max_runs": max_runs,
        "reuse_existing_results": bool(budget.get("reuse_existing_results", True)),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "uncapped_selected_count": uncapped_selected_count,
        "max_runs_capped": max_runs > 0 and uncapped_selected_count > len(selected),
        "extra_args": cfg.get("_resolved_extra_args", []),
        "estimated_runs": len([p for p in plan_runs if p["action"] == "run"]),
        "planned": plan_runs,
        "skipped": skipped,
        "notes": plan_notes(mode, candidates, plan_runs),
        "warning": "full mode may run the full Cartesian matrix and take a long time" if mode == "full" else None,
    }


def selection_reason(mode: str, job: dict[str, Any]) -> str:
    if mode == "smoke":
        return "smoke: conservative single configuration"
    if mode == "focused":
        return "focused: refine around existing safe configuration"
    if mode == "full":
        return "full: explicit Cartesian sweep"
    return "quick: prioritized safe configuration"


def plan_notes(mode: str, candidates: list[dict[str, Any]], plan_runs: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    return notes


def write_plan(plan: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p
