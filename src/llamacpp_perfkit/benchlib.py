#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path("config/benchmark.example.yaml")
SEARCH_TERMS = [
    "n-cpu-moe",
    "cache-type",
    "cache-type-k",
    "cache-type-v",
    "kv",
    "turbo",
    "spec",
    "draft",
    "mtp",
    "ngram",
    "hfd",
    "ctx",
    "n-ctx",
    "seed",
    "predict",
    "n-predict",
    "tokens",
]

TOOLKIT_OWNED_LLAMA_FLAGS = {
    "-hf",
    "--hf-repo",
    "-m",
    "--model",
    "--ctx-size",
    "--n-ctx",
    "--context-size",
    "--predict",
    "--n-predict",
    "--file",
    "--prompt",
    "--seed",
    "--n-cpu-moe",
    "--cache-type",
    "--cache-type-k",
    "--cache-type-v",
    "--spec-type",
    "--spec-draft-n-max",
    "--spec-draft-p-min",
    "--log-file",
    "--no-warmup",
    "--host",
    "--port",
    "--no-webui",
    "--batch-size",
    "--ubatch-size",
}

REQUEST_ARG_MAP = {
    "--temp": "temperature",
    "--temperature": "temperature",
    "--top-p": "top_p",
    "--top-k": "top_k",
    "--presence-penalty": "presence_penalty",
    "--frequency-penalty": "frequency_penalty",
    "--min-p": "min_p",
    "--typical": "typical_p",
    "--typical-p": "typical_p",
    "--repeat-penalty": "repeat_penalty",
    "--repeat-last-n": "repeat_last_n",
    "--seed": "seed",
}

SERVER_EXTRA_DENYLIST = set(REQUEST_ARG_MAP) | {
    "--predict",
    "--n-predict",
    "--file",
    "--prompt",
}


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def relpath(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("models", {})
    cfg.setdefault("llama", {})
    cfg.setdefault("prompt", {})
    cfg.setdefault("run", {})
    cfg.setdefault("budget", {})
    cfg.setdefault("matrix", {})
    cfg.setdefault("output", {})
    cfg["budget"].setdefault("mode", "quick")
    cfg["budget"].setdefault("max_runs", 8)
    cfg["budget"].setdefault("reuse_existing_results", True)
    cfg["budget"].setdefault("stop_if_all_remaining_are_risky", True)
    cfg["llama"].setdefault("extra_args", {})
    cfg["llama"].setdefault("preferred_binary", "llama-server")
    cfg["llama"].setdefault("server", {})
    cfg["llama"]["server"].setdefault("host", "127.0.0.1")
    cfg["llama"]["server"].setdefault("startup_timeout_seconds", 300)
    cfg["llama"]["server"].setdefault("shutdown_timeout_seconds", 15)
    cfg["run"].setdefault("cache_prompt", False)
    cfg["run"].setdefault("endpoint", "chat")
    return cfg


def backend_binary_name(cfg: dict[str, Any]) -> str:
    return "llama-server"


def output_dirs(cfg: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    logs_dir = Path.cwd() / cfg["output"].get("logs_dir", "logs")
    results_dir = Path.cwd() / cfg["output"].get("results_dir", "runs")
    raw_dir = logs_dir / "raw"
    mon_dir = logs_dir / "monitoring"
    for d in (logs_dir, results_dir, raw_dir, mon_dir):
        d.mkdir(parents=True, exist_ok=True)
    return logs_dir, results_dir, raw_dir, mon_dir


def llama_bin_dir(cfg: dict[str, Any]) -> Path:
    value = os.environ.get("LLAMA_BIN_DIR") or cfg["llama"].get("bin_dir", "../llama.cpp/build/bin")
    return (Path.cwd() / value).resolve() if not Path(value).is_absolute() else Path(value)


def model_hf(cfg: dict[str, Any]) -> str | None:
    return os.environ.get("MODEL_HF") or cfg["models"].get("hf")


def prompt_file(cfg: dict[str, Any]) -> Path:
    value = cfg["prompt"].get("file", "prompts/default.txt")
    return (Path.cwd() / value).resolve() if not Path(value).is_absolute() else Path(value)


def resolve_path(value: str | Path) -> Path:
    p = Path(value)
    return (Path.cwd() / p).resolve() if not p.is_absolute() else p


def prompt_profiles(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = cfg.get("prompt", {}).get("profiles")
    if not profiles:
        return [{"name": "default", "file": str(prompt_file(cfg))}]
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(profiles, 1):
        if isinstance(item, str):
            name = Path(item).stem
            value = item
        else:
            value = item.get("file")
            name = item.get("name") or Path(str(value)).stem
        if not value:
            continue
        out.append({"name": str(name), "file": str(resolve_path(value)), "index": idx})
    return out or [{"name": "default", "file": str(prompt_file(cfg))}]


def prompt_file_for_job(cfg: dict[str, Any], job: dict[str, Any]) -> Path:
    value = job.get("prompt_file")
    if value:
        return resolve_path(value)
    return prompt_file(cfg)


def normalize_extra_args(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    raw = cfg.get("llama", {}).get("extra_args", {}) or {}
    if isinstance(raw, list):
        items: list[Any] = raw
    elif isinstance(raw, dict):
        items = [{"flag": flag, "value": value} for flag, value in raw.items()]
    else:
        items = []
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            flag, value = item, True
        else:
            flag = str(item.get("flag", "")).strip()
            value = item.get("value")
        if not flag:
            continue
        out.append({"flag": flag, "value": value})
    return out


def server_args_from_features(features: dict[str, Any]) -> list[dict[str, Any]]:
    extra = features.get("extra_args", {})
    return extra.get("server_usable") or extra.get("usable", [])


def request_args_from_features(features: dict[str, Any]) -> dict[str, Any]:
    return features.get("extra_args", {}).get("request", {})


def append_extra_args(cmd: list[str], extra_args: list[dict[str, Any]]) -> list[str]:
    for item in extra_args:
        flag = item["flag"]
        value = item.get("value")
        if value is False or value is None:
            continue
        cmd.append(flag)
        if value is not True:
            cmd.append(str(value))
    return cmd


def file_identity(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    stat = p.stat()
    identity: dict[str, Any] = {
        "path": str(p.resolve()),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if p.is_file() and stat.st_size <= 1024 * 1024:
        identity["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return identity


def nearest_existing_path(path: str | Path) -> Path:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


def git_metadata(path: str | Path) -> dict[str, Any]:
    requested = Path(path).resolve()
    probe = nearest_existing_path(requested)
    rc, stdout, stderr = run_capture(["git", "-C", str(probe), "rev-parse", "--show-toplevel"], timeout=10)
    if rc != 0:
        return {
            "repo": None,
            "commit": None,
            "commit_short": None,
            "error": (stderr or stdout or "git rev-parse --show-toplevel failed").strip(),
        }
    repo = stdout.strip()
    rc, stdout, stderr = run_capture(["git", "-C", repo, "rev-parse", "HEAD"], timeout=10)
    if rc != 0:
        return {
            "repo": repo,
            "commit": None,
            "commit_short": None,
            "error": (stderr or stdout or "git rev-parse HEAD failed").strip(),
        }
    commit = stdout.strip()
    rc, stdout, stderr = run_capture(["git", "-C", repo, "rev-parse", "--short", "HEAD"], timeout=10)
    if rc != 0:
        return {
            "repo": repo,
            "commit": commit,
            "commit_short": None,
            "error": (stderr or stdout or "git rev-parse --short HEAD failed").strip(),
        }
    commit_short = stdout.strip()
    rc, stdout, stderr = run_capture(["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"], timeout=10)
    if rc != 0:
        branch = None
    else:
        branch = stdout.strip()
        if branch == "HEAD":
            branch = None
    return {
        "repo": repo,
        "commit": commit,
        "commit_short": commit_short,
        "branch": branch,
        "error": None,
    }


def llama_cpp_git_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    if "_llama_cpp_git_metadata" not in cfg:
        cfg["_llama_cpp_git_metadata"] = git_metadata(llama_bin_dir(cfg))
    return cfg["_llama_cpp_git_metadata"]


def llama_cpp_commit_for_hash(cfg: dict[str, Any]) -> str | None:
    return (cfg.get("_llama_cpp") or llama_cpp_git_metadata(cfg)).get("commit")


def config_hash_payload(cfg: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    binary = llama_bin_dir(cfg) / backend_binary_name(cfg)
    return {
        "backend": "server",
        "llama_cpp_commit": llama_cpp_commit_for_hash(cfg),
        "model_hf": model_hf(cfg),
        "context_size": job.get("context_size"),
        "kv_type": job.get("kv_type"),
        "n_cpu_moe": job.get("n_cpu_moe"),
        "spec_type": job.get("spec_type"),
        "spec_draft_n_max": job.get("spec_draft_n_max"),
        "spec_draft_p_min": job.get("spec_draft_p_min"),
        "batch_size": job.get("batch_size"),
        "ubatch_size": job.get("ubatch_size"),
        "generation_tokens": cfg["run"].get("generation_tokens", 512),
        "endpoint": cfg["run"].get("endpoint", "chat"),
        "seed": cfg["run"].get("seed"),
        "prompt_profile": job.get("prompt_profile", "default"),
        "prompt": file_identity(prompt_file_for_job(cfg, job)),
        "binary": file_identity(binary),
        "server_args": cfg.get("_resolved_server_args", normalize_extra_args(cfg)),
        "request_args": cfg.get("_resolved_request_args", {}),
        "cache_prompt": cfg.get("run", {}).get("cache_prompt", False),
    }


def config_hash(cfg: dict[str, Any], job: dict[str, Any]) -> str:
    payload = config_hash_payload(cfg, job)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def row_config_hash(cfg: dict[str, Any], row: dict[str, Any]) -> str | None:
    if row.get("config_hash"):
        return row["config_hash"]
    rcfg = row.get("config", {})
    if not rcfg:
        return None
    job = {
        "n_cpu_moe": rcfg.get("n_cpu_moe"),
        "context_size": rcfg.get("context_size"),
        "kv_type": rcfg.get("kv_type"),
        "spec_type": rcfg.get("spec_type") or rcfg.get("mtp_spec_type"),
        "spec_draft_n_max": rcfg.get("spec_draft_n_max") or rcfg.get("mtp_draft_n_max"),
        "spec_draft_p_min": rcfg.get("spec_draft_p_min") or rcfg.get("mtp_draft_p_min"),
        "batch_size": rcfg.get("batch_size"),
        "ubatch_size": rcfg.get("ubatch_size"),
        "prompt_profile": rcfg.get("prompt_profile", "default"),
        "prompt_file": rcfg.get("prompt_file"),
    }
    return config_hash(cfg, job)


def run_capture(cmd: list[str], timeout: float = 30) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except subprocess.TimeoutExpired as exc:
        return 124, str(exc.stdout or ""), str(exc.stderr or "timeout")


def has_flag(help_text: str, flag: str) -> bool:
    return re.search(r"(^|[\s,])" + re.escape(flag) + r"([\s,]|$)", help_text, re.MULTILINE) is not None


def allowed_values(help_text: str, flag: str) -> list[str]:
    lines = help_text.splitlines()
    values: list[str] = []
    for i, line in enumerate(lines):
        if flag not in line:
            continue
        block = "\n".join(lines[i : min(i + 5, len(lines))])
        match = re.search(r"allowed values:\s*([^\n]+)", block, re.IGNORECASE)
        if match:
            values.extend(v.strip(" ,") for v in match.group(1).split(","))
        inline = re.search(re.escape(flag) + r"\s+([A-Za-z0-9_,\-]+)", line)
        if flag == "--spec-type" and inline:
            values.extend(v.strip(" ,") for v in inline.group(1).split(","))
    return sorted({v for v in values if v and not v.startswith("<")})


def interesting_help(help_text: str) -> str:
    out: list[str] = []
    for line in help_text.splitlines():
        if any(term.lower() in line.lower() for term in SEARCH_TERMS):
            out.append(line)
    return "\n".join(out)


def detect_features(cfg: dict[str, Any]) -> dict[str, Any]:
    bin_dir = llama_bin_dir(cfg)
    backend = "server"
    binaries: dict[str, Any] = {}
    help_texts: dict[str, str] = {}
    for name in ("llama-bench", "llama-server"):
        path = bin_dir / name
        exists = path.exists() and os.access(path, os.X_OK)
        rc, stdout, stderr = run_capture([str(path), "--help"], timeout=20) if exists else (127, "", "missing")
        binaries[name] = {
            "path": str(path),
            "exists": exists,
            "usable": exists and rc == 0,
            "help_returncode": rc,
            "help_error": stderr.strip(),
        }
        help_texts[name] = stdout + "\n" + stderr

    bench = help_texts.get("llama-bench", "")
    server = help_texts.get("llama-server", "")
    active_help = server
    spec_values = allowed_values(active_help, "--spec-type")
    kv_values = allowed_values(active_help, "--cache-type-k") or allowed_values(bench, "--cache-type-k")
    requested_extra_args = normalize_extra_args(cfg)
    usable_extra_args: list[dict[str, Any]] = []
    server_usable_extra_args: list[dict[str, Any]] = []
    request_extra_args: dict[str, Any] = {}
    skipped_extra_args: list[dict[str, Any]] = []
    for item in requested_extra_args:
        flag = item["flag"]
        value = item.get("value")
        if flag in TOOLKIT_OWNED_LLAMA_FLAGS:
            skipped_extra_args.append({**item, "reason": "flag is controlled by the benchmark toolkit"})
        elif flag in REQUEST_ARG_MAP:
            request_extra_args[REQUEST_ARG_MAP[flag]] = value
        elif flag in SERVER_EXTRA_DENYLIST:
            skipped_extra_args.append({**item, "reason": "flag is a request option or controlled by the benchmark toolkit"})
        elif not flag.startswith("-"):
            skipped_extra_args.append({**item, "reason": "extra arg key must be a llama.cpp flag"})
        elif not has_flag(server, flag):
            skipped_extra_args.append({**item, "reason": "local llama-server --help does not list this flag"})
        else:
            usable_extra_args.append(item)
            server_usable_extra_args.append(item)

    features: dict[str, Any] = {
        "created_at": now_iso(),
        "project_root": str(PROJECT_ROOT),
        "bin_dir": str(bin_dir),
        "llama_cpp": llama_cpp_git_metadata(cfg),
        "backend": backend,
        "binaries": binaries,
        "flags": {
            "llama_bench": {
                "hf": has_flag(bench, "-hf") or has_flag(bench, "--hf-repo"),
                "n_prompt": "--n-prompt" if has_flag(bench, "--n-prompt") else None,
                "n_gen": "--n-gen" if has_flag(bench, "--n-gen") else None,
                "n_cpu_moe": "--n-cpu-moe" if has_flag(bench, "--n-cpu-moe") else None,
                "cache_type_k": "--cache-type-k" if has_flag(bench, "--cache-type-k") else None,
                "cache_type_v": "--cache-type-v" if has_flag(bench, "--cache-type-v") else None,
                "spec_type": "--spec-type" if has_flag(bench, "--spec-type") else None,
            },
            "llama_server": {
                "hf": has_flag(server, "-hf") or has_flag(server, "--hf-repo"),
                "context": "--ctx-size" if has_flag(server, "--ctx-size") else None,
                "generation": "--n-predict"
                if has_flag(server, "--n-predict")
                else ("--predict" if has_flag(server, "--predict") else None),
                "seed": "--seed" if has_flag(server, "--seed") else None,
                "n_cpu_moe": "--n-cpu-moe" if has_flag(server, "--n-cpu-moe") else None,
                "cache_type_k": "--cache-type-k" if has_flag(server, "--cache-type-k") else None,
                "cache_type_v": "--cache-type-v" if has_flag(server, "--cache-type-v") else None,
                "spec_type": "--spec-type" if has_flag(server, "--spec-type") else None,
                "spec_draft_n_max": "--spec-draft-n-max" if has_flag(server, "--spec-draft-n-max") else None,
                "spec_draft_p_min": "--spec-draft-p-min" if has_flag(server, "--spec-draft-p-min") else None,
                "no_webui": "--no-webui" if has_flag(server, "--no-webui") else None,
                "host": "--host" if has_flag(server, "--host") else None,
                "port": "--port" if has_flag(server, "--port") else None,
                "metrics": "--metrics" if has_flag(server, "--metrics") else None,
                "slots": "--slots" if has_flag(server, "--slots") else None,
                "batch_size": "--batch-size" if has_flag(server, "--batch-size") else None,
                "ubatch_size": "--ubatch-size" if has_flag(server, "--ubatch-size") else None,
            },
        },
        "kv": {
            "supported_values": kv_values,
            "requested_values": cfg["matrix"].get("kv_type", []),
            "usable_values": [v for v in cfg["matrix"].get("kv_type", []) if not kv_values or v in kv_values],
            "skipped": [
                {"value": v, "reason": "not listed in local cache-type allowed values"}
                for v in cfg["matrix"].get("kv_type", [])
                if kv_values and v not in kv_values
            ],
        },
        "spec": {
            "supported_values": spec_values,
            "requested_values": [v for v in cfg.get("matrix", {}).get("spec_type", []) if v is not None],
            "usable_values": [v for v in cfg.get("matrix", {}).get("spec_type", []) if not spec_values or v in spec_values],
            "skipped": [
                {"value": v, "reason": "not listed in local --spec-type allowed values"}
                for v in cfg.get("matrix", {}).get("spec_type", [])
                if v is not None and spec_values and v not in spec_values
            ],
        },
        "extra_args": {
            "requested": requested_extra_args,
            "usable": usable_extra_args,
            "server_usable": server_usable_extra_args,
            "request": request_extra_args,
            "skipped": skipped_extra_args,
        },
        "help_excerpt": {
            "llama-bench": interesting_help(bench),
            "llama-server": interesting_help(server),
        },
    }
    active = features["flags"]["llama_server"]
    valid = features["binaries"]["llama-server"]["usable"] and active["hf"] and active["context"] and active["port"]
    invalid_reason = "llama-server is missing or lacks -hf/context/port flags"
    features["valid_for_bench"] = bool(valid)
    features["invalid_reason"] = None if valid else invalid_reason
    return features


def write_features(features: dict[str, Any], results_dir: Path) -> tuple[Path, Path]:
    json_path = results_dir / "features.json"
    txt_path = results_dir / "features.txt"
    json_path.write_text(json.dumps(features, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines: list[str] = [
        f"created_at: {features['created_at']}",
        f"bin_dir: {features['bin_dir']}",
        f"llama_cpp_commit: {(features.get('llama_cpp') or {}).get('commit_short') or 'unknown'}",
        f"backend: {features.get('backend', 'cli')}",
        f"valid_for_bench: {features['valid_for_bench']}",
    ]
    if (features.get("llama_cpp") or {}).get("error"):
        lines.append(f"llama_cpp_commit_error: {features['llama_cpp']['error']}")
    if features.get("invalid_reason"):
        lines.append(f"invalid_reason: {features['invalid_reason']}")
    lines.append("\nBinaries:")
    for name, item in features["binaries"].items():
        lines.append(f"  {name}: usable={item['usable']} path={item['path']}")
    lines.append("\nllama-server flags:")
    for key, value in features["flags"]["llama_server"].items():
        lines.append(f"  {key}: {value}")
    lines.append("\nKV:")
    lines.append(f"  supported_values: {', '.join(features['kv']['supported_values']) or 'unknown'}")
    lines.append(f"  usable_values: {', '.join(features['kv']['usable_values']) or 'none'}")
    for item in features["kv"]["skipped"]:
        lines.append(f"  skipped {item['value']}: {item['reason']}")
    lines.append("\nSpec:")
    lines.append(f"  supported_values: {', '.join(features['spec']['supported_values']) or 'unknown'}")
    lines.append(f"  usable_values: {', '.join(str(v) for v in features['spec']['usable_values']) or 'none'}")
    for item in features["spec"]["skipped"]:
        lines.append(f"  skipped {item['value']}: {item['reason']}")
    lines.append("\nExtra server args:")
    if features.get("extra_args", {}).get("server_usable"):
        for item in features["extra_args"]["server_usable"]:
            value = item.get("value")
            suffix = "" if value is True else f" {value}"
            lines.append(f"  usable: {item['flag']}{suffix}")
    else:
        lines.append("  usable: none")
    lines.append("\nRequest args:")
    if features.get("extra_args", {}).get("request"):
        for key, value in sorted(features["extra_args"]["request"].items()):
            lines.append(f"  {key}: {value}")
    else:
        lines.append("  usable: none")
    for item in features.get("extra_args", {}).get("skipped", []):
        value = item.get("value")
        suffix = "" if value is True else f" {value}"
        lines.append(f"  skipped {item['flag']}{suffix}: {item['reason']}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, txt_path


def load_features(results_dir: Path) -> dict[str, Any] | None:
    path = results_dir / "features.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_meminfo() -> dict[str, float | None]:
    values: dict[str, float] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, rest = line.split(":", 1)
            values[key] = int(rest.strip().split()[0]) / 1024.0
    except Exception:
        return {"ram_used_mib": None, "ram_free_mib": None}
    total = values.get("MemTotal")
    available = values.get("MemAvailable", values.get("MemFree"))
    return {
        "ram_used_mib": (total - available) if total is not None and available is not None else None,
        "ram_free_mib": available,
    }


def sample_gpu() -> dict[str, float | None]:
    if not shutil.which("nvidia-smi"):
        return {}
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    rc, stdout, _ = run_capture(cmd, timeout=5)
    if rc != 0 or not stdout.strip():
        return {}
    parts = [p.strip() for p in stdout.splitlines()[0].split(",")]
    keys = ["vram_used_mib", "vram_free_mib", "gpu_util_pct", "gpu_power_w", "gpu_temp_c"]
    out: dict[str, float | None] = {}
    for key, value in zip(keys, parts):
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = None
    return out


class Monitor:
    def __init__(self, path: str | Path, interval: float) -> None:
        self.path = Path(path)
        self.interval = float(interval)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.samples: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.active_path: Path | None = None

    def __enter__(self) -> Monitor:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=max(2.0, self.interval + 1.0))

    def _run(self) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            while not self.stop.is_set():
                sample = self.sample_once()
                f.write(json.dumps(sample, sort_keys=True) + "\n")
                f.flush()
                self._write_active_sample(sample)
                self.stop.wait(self.interval)

    def sample_once(self) -> dict[str, Any]:
        sample: dict[str, Any] = {"time": now_iso()}
        sample.update(sample_gpu())
        sample.update(parse_meminfo())
        with self.lock:
            self.samples.append(sample)
        return sample

    def set_active_path(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=True)
        with self.lock:
            self.active_path = target

    def clear_active_path(self) -> None:
        with self.lock:
            self.active_path = None

    def _write_active_sample(self, sample: dict[str, Any]) -> None:
        with self.lock:
            target = self.active_path
        if not target:
            return
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample, sort_keys=True) + "\n")

    def checkpoint(self) -> int:
        with self.lock:
            return len(self.samples)

    def summary(self, start_index: int = 0) -> dict[str, float | None]:
        with self.lock:
            samples = list(self.samples[start_index:])

        def vals(key: str) -> list[float]:
            return [s[key] for s in samples if isinstance(s.get(key), (int, float))]

        gpu_utils = vals("gpu_util_pct")
        return {
            "peak_vram_mib": max(vals("vram_used_mib"), default=None),
            "min_vram_free_mib": min(vals("vram_free_mib"), default=None),
            "peak_ram_mib": max(vals("ram_used_mib"), default=None),
            "min_ram_free_mib": min(vals("ram_free_mib"), default=None),
            "avg_gpu_util_pct": (sum(gpu_utils) / len(gpu_utils)) if gpu_utils else None,
            "peak_gpu_power_w": max(vals("gpu_power_w"), default=None),
            "peak_gpu_temp_c": max(vals("gpu_temp_c"), default=None),
        }


def parse_llama_output(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "prompt_eval_tok_s": None,
        "generation_tok_s": None,
        "total_time_ms": None,
        "load_time_ms": None,
        "tokens_generated": None,
        "tokens_prompt": None,
        "speculative_stats": None,
    }
    patterns = [
        ("load_time_ms", r"load time\s*=\s*([0-9.]+)\s*ms"),
        ("total_time_ms", r"total time\s*=\s*([0-9.]+)\s*ms"),
    ]
    for key, pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            parsed[key] = float(m.group(1))

    m = re.search(r"prompt eval time\s*=\s*[^\n]*?/\s*(\d+)\s+tokens?.*?\(\s*([0-9.]+)\s+tokens? per second", text, re.IGNORECASE)
    if m:
        parsed["tokens_prompt"] = int(m.group(1))
        parsed["prompt_eval_tok_s"] = float(m.group(2))

    gen_matches = re.findall(
        r"(?<!prompt )eval time\s*=\s*[^\n]*?/\s*(\d+)\s+(?:runs?|tokens?).*?\(\s*([0-9.]+)\s+tokens? per second", text, re.IGNORECASE
    )
    if gen_matches:
        tokens, speed = gen_matches[-1]
        parsed["tokens_generated"] = int(tokens)
        parsed["generation_tok_s"] = float(speed)

    spec_lines = [line.strip() for line in text.splitlines() if re.search(r"spec|draft|accept", line, re.IGNORECASE)]
    if spec_lines:
        parsed["speculative_stats"] = " | ".join(spec_lines[-10:])
    return parsed


def parse_server_completion(data: dict[str, Any]) -> dict[str, Any]:
    timings = data.get("timings") or {}
    prompt_n = timings.get("prompt_n")
    cache_n = timings.get("cache_n") or 0
    predicted_n = timings.get("predicted_n")
    parsed: dict[str, Any] = {
        "prompt_eval_tok_s": timings.get("prompt_per_second"),
        "generation_tok_s": timings.get("predicted_per_second"),
        "total_time_ms": None,
        "load_time_ms": None,
        "tokens_generated": predicted_n,
        "tokens_prompt": (prompt_n + cache_n) if isinstance(prompt_n, (int, float)) else prompt_n,
        "speculative_stats": None,
        "server_timings": timings,
        "tokens_cached": data.get("tokens_cached", cache_n),
        "tokens_evaluated": data.get("tokens_evaluated"),
        "truncated": data.get("truncated"),
        "stop_type": data.get("stop_type"),
    }
    return parsed


def parse_chat_completion(
    data: dict[str, Any], elapsed_seconds: float | None = None, log_text: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    timings = data.get("timings") or {}
    content = message.get("content") or ""
    reasoning_content = message.get("reasoning_content") or message.get("reasoning") or ""
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    log_parsed = parse_llama_output(log_text or "")
    speed = timings.get("predicted_per_second") or log_parsed.get("generation_tok_s")
    if speed is None and isinstance(completion_tokens, (int, float)) and elapsed_seconds and elapsed_seconds > 0:
        speed = completion_tokens / elapsed_seconds
    parsed: dict[str, Any] = {
        "prompt_eval_tok_s": timings.get("prompt_per_second") or log_parsed.get("prompt_eval_tok_s"),
        "generation_tok_s": speed,
        "total_time_ms": log_parsed.get("total_time_ms"),
        "load_time_ms": log_parsed.get("load_time_ms"),
        "tokens_generated": completion_tokens if completion_tokens is not None else log_parsed.get("tokens_generated"),
        "tokens_prompt": prompt_tokens if prompt_tokens is not None else log_parsed.get("tokens_prompt"),
        "speculative_stats": log_parsed.get("speculative_stats"),
        "server_timings": timings or log_parsed,
        "tokens_cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "tokens_evaluated": prompt_tokens,
        "truncated": choice.get("finish_reason") == "length",
        "stop_type": choice.get("finish_reason"),
        "throughput_source": "server_timings"
        if timings.get("predicted_per_second")
        else ("server_log" if log_parsed.get("generation_tok_s") else "wall_clock"),
    }
    return parsed, {
        "content": content,
        "reasoning_content": reasoning_content,
        "stop_type": choice.get("finish_reason"),
        "truncated": choice.get("finish_reason") == "length",
    }


def has_response_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return value not in (None, "", [], {})


def benchmark_invalid_reason(row: dict[str, Any]) -> str | None:
    if row.get("status") != "success":
        return row.get("status") or "not successful"
    parsed = row.get("parsed") or {}
    speed = parsed.get("generation_tok_s")
    tokens = parsed.get("tokens_generated")
    requested = get_field(row, "request.n_predict") or get_field(row, "config.generation_tokens") or 0
    min_tokens = min(8, int(requested)) if requested else 8
    if speed is None:
        return "missing generation throughput"
    if not isinstance(speed, (int, float)) or speed <= 0:
        return "invalid generation throughput"
    if speed > 10000:
        return "implausible generation throughput"
    if not isinstance(tokens, (int, float)):
        return "missing generated token count"
    if tokens < min_tokens:
        return f"too few generated tokens for reliable throughput: {tokens} < {min_tokens}"
    response = row.get("response") or {}
    if get_field(row, "config.endpoint") == "chat" and not (
        has_response_text(response.get("content")) or has_response_text(response.get("reasoning_content"))
    ):
        return "empty chat response content"
    if not has_response_text(response.get("content")) and tokens <= 1:
        return "empty response with one or fewer generated tokens"
    return None


def benchmark_valid(row: dict[str, Any]) -> bool:
    return benchmark_invalid_reason(row) is None


def classify_run(returncode: int, timed_out: bool, text: str) -> str:
    lower = text.lower()
    if timed_out:
        return "timeout"
    if returncode == 0:
        return "success"
    if any(p in lower for p in ["out of memory", "cuda error 2", "cuda_malloc", "cudamalloc", "failed to allocate", "no memory"]):
        return "oom"
    if any(p in lower for p in ["unknown argument", "invalid argument", "invalid value", "error: unrecognized", "invalid choice"]):
        return "unsupported"
    return "failed"


def classify_http_error(status_code: int, text: str) -> str:
    lower = (text or "").lower()
    if status_code in {400, 404, 422} or any(p in lower for p in ["unknown argument", "invalid argument", "invalid value", "unsupported"]):
        return "unsupported"
    if any(p in lower for p in ["out of memory", "cuda error 2", "cuda_malloc", "cudamalloc", "failed to allocate", "no memory"]):
        return "oom"
    return "failed"


def free_tcp_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 30) -> tuple[int, dict[str, Any], str]:
    data: bytes | None = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            text = res.read().decode("utf-8", errors="replace")
            return res.status, json.loads(text) if text else {}, text
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(text) if text else {}
        except json.JSONDecodeError:
            body = {}
        return exc.code, body, text


def build_server_cmd(cfg: dict[str, Any], features: dict[str, Any], job: dict[str, Any], port: int, server_log_path: Path) -> list[str]:
    flags = features["flags"]["llama_server"]
    server_cfg = cfg.get("llama", {}).get("server", {})
    host = server_cfg.get("host", "127.0.0.1")
    model = model_hf(cfg)
    cmd: list[str] = [
        str(llama_bin_dir(cfg) / "llama-server"),
        "-hf",
        str(model) if model else "",
    ]
    if flags.get("context"):
        cmd += [flags["context"], str(job["context_size"])]
    if flags.get("host"):
        cmd += [flags["host"], str(host)]
    if flags.get("port"):
        cmd += [flags["port"], str(port)]
    if flags.get("n_cpu_moe") and job.get("n_cpu_moe") is not None:
        cmd += [flags["n_cpu_moe"], str(job["n_cpu_moe"])]
    if flags.get("cache_type_k"):
        cmd += [flags["cache_type_k"], job["kv_type"]]
    if flags.get("cache_type_v"):
        cmd += [flags["cache_type_v"], job["kv_type"]]
    if flags.get("batch_size") and job.get("batch_size") is not None:
        cmd += [flags["batch_size"], str(job["batch_size"])]
    if flags.get("ubatch_size") and job.get("ubatch_size") is not None:
        cmd += [flags["ubatch_size"], str(job["ubatch_size"])]
    append_extra_args(cmd, server_args_from_features(features))
    if flags.get("no_webui"):
        cmd += [flags["no_webui"]]
    if flags.get("metrics"):
        cmd += [flags["metrics"]]
    if flags.get("spec_type") and job.get("spec_type") is not None:
        cmd += [flags["spec_type"], job["spec_type"]]
        if flags.get("spec_draft_n_max") and job.get("spec_draft_n_max") is not None:
            cmd += [flags["spec_draft_n_max"], str(job["spec_draft_n_max"])]
        if flags.get("spec_draft_p_min") and job.get("spec_draft_p_min") is not None:
            cmd += [flags["spec_draft_p_min"], str(job["spec_draft_p_min"])]
    return cmd


def server_group_key(job: dict[str, Any]) -> tuple[Any, ...]:
    return (
        job.get("context_size"),
        job.get("kv_type"),
        job.get("n_cpu_moe"),
        job.get("spec_type"),
        job.get("spec_draft_n_max"),
        job.get("spec_draft_p_min"),
        job.get("batch_size"),
        job.get("ubatch_size"),
    )


def build_completion_payload(cfg: dict[str, Any], features: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": prompt_file_for_job(cfg, job).read_text(encoding="utf-8"),
        "n_predict": int(cfg["run"].get("generation_tokens", 512)),
        "stream": False,
        "cache_prompt": bool(cfg["run"].get("cache_prompt", False)),
        "id_slot": 0,
    }
    if cfg["run"].get("seed") is not None:
        payload["seed"] = int(cfg["run"].get("seed"))
    payload.update(request_args_from_features(features))
    return payload


def request_endpoint(cfg: dict[str, Any]) -> str:
    value = str(cfg.get("run", {}).get("endpoint", "chat")).strip().lower()
    if value in {"chat", "chat-completions", "chat_completions", "/v1/chat/completions"}:
        return "chat"
    if value in {"completion", "completions", "/completion"}:
        return "completion"
    return value


def build_chat_payload(cfg: dict[str, Any], features: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt_file_for_job(cfg, job).read_text(encoding="utf-8")}],
        "max_tokens": int(cfg["run"].get("generation_tokens", 512)),
        "stream": False,
    }
    if cfg["run"].get("seed") is not None:
        payload["seed"] = int(cfg["run"].get("seed"))
    chat_template_kwargs = cfg["run"].get("chat_template_kwargs")
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    payload.update(request_args_from_features(features))
    return payload


def build_request_payload(cfg: dict[str, Any], features: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    endpoint = request_endpoint(cfg)
    if endpoint == "chat":
        return build_chat_payload(cfg, features, job)
    if endpoint == "completion":
        return build_completion_payload(cfg, features, job)
    raise ValueError(f"Unsupported run.endpoint: {endpoint}")


def stability_status(status: str, min_vram_free_mib: float | None, min_headroom_gb: float) -> str:
    if status != "success":
        return status
    if min_vram_free_mib is None:
        return "unknown"
    if min_vram_free_mib < min_headroom_gb * 1024:
        return "too_close_to_vram_limit"
    return "stable"


def write_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for k2, v2 in value.items():
                flat[f"{key}.{k2}"] = json.dumps(v2) if isinstance(v2, (dict, list)) else v2
        elif isinstance(value, list):
            flat[key] = json.dumps(value)
        else:
            flat[key] = value
    return flat


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    flats = [flatten(r) for r in rows]
    keys = sorted({k for r in flats for k in r.keys()})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flats)


def table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None, gradients: dict[str, bool] | None = None) -> str:
    from .output import gradient_value, style, style_table_value, visible_len

    rows = rows[:limit] if limit else rows
    if not rows:
        return "No rows."
    grad_ranges: dict[str, tuple[float, float, bool]] = {}
    if gradients:
        for col, higher_is_better in gradients.items():
            raw_key = f"_raw_{col}"
            raw_values = [row[raw_key] for row in rows if isinstance(row.get(raw_key), (int, float))]
            if len(raw_values) >= 2:
                grad_ranges[col] = (min(raw_values), max(raw_values), higher_is_better)
    widths = {col: max(len(str(col)), *(visible_len(fmt_value(get_field(r, col), col)) for r in rows)) for col in columns}
    header = "  ".join(style(str(col), "bold") + (" " * (widths[col] - visible_len(str(col)))) for col in columns)
    sep = "  ".join("-" * widths[col] for col in columns)
    body: list[str] = []
    for row in rows:
        parts: list[str] = []
        for col in columns:
            value = fmt_value(get_field(row, col), col)
            grad_range = grad_ranges.get(col)
            if grad_range and isinstance(row.get(f"_raw_{col}"), (int, float)):
                min_val, max_val, higher_is_better = grad_range
                value = gradient_value(value, row[f"_raw_{col}"], min_val, max_val, higher_is_better=higher_is_better)
            styled = style_table_value(col, value)
            padding = " " * (widths[col] - visible_len(value))
            parts.append(styled + padding)
        body.append("  ".join(parts))
    return "\n".join([header, sep, *body])


def get_field(row: dict[str, Any], dotted: str) -> Any:
    cur: Any = row
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def fmt_value(v: Any, column: str | None = None) -> str:
    if column == "duration_seconds":
        from .output import format_duration

        return format_duration(v)
    if v is None:
        return "-"
    if isinstance(v, (int, float)):
        from .output import format_number

        return format_number(v)
    return str(v)


def filter_rows(rows: list[dict[str, Any]], args: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        cfg = row.get("config", {})
        if getattr(args, "model", None):
            model_hf = str(cfg.get("model_hf", "")).lower().replace("-gguf", "")
            filter_model = args.model.lower().replace("-gguf", "")
            if filter_model not in model_hf:
                continue
        if getattr(args, "quant", None) and args.quant not in str(cfg.get("model_hf", "")).split(":")[-1]:
            continue
        if getattr(args, "context_size", None) is not None and int(cfg.get("context_size", -1)) != args.context_size:
            continue
        if getattr(args, "kv_type", None) and cfg.get("kv_type") != args.kv_type:
            continue
        if getattr(args, "prompt_profile", None) and cfg.get("prompt_profile", "default") != args.prompt_profile:
            continue
        if getattr(args, "n_cpu_moe", None) is not None and int(cfg.get("n_cpu_moe", -1)) != args.n_cpu_moe:
            continue
        if getattr(args, "batch_size", None) is not None and int(cfg.get("batch_size", -1)) != args.batch_size:
            continue
        if getattr(args, "ubatch_size", None) is not None and int(cfg.get("ubatch_size", -1)) != args.ubatch_size:
            continue
        if getattr(args, "status", None) and row.get("status") != args.status:
            continue
        out.append(row)
    return out


def sort_rows(rows: list[dict[str, Any]], sort_key: str) -> list[dict[str, Any]]:
    key_map = {
        "latest": "created_at",
        "generation_tok_s": "parsed.generation_tok_s",
        "vram_headroom": "monitor.min_vram_free_mib",
        "peak_vram": "monitor.peak_vram_mib",
        "context_size": "config.context_size",
    }
    field = key_map.get(sort_key, sort_key)
    if sort_key == "latest":
        return sorted(rows, key=lambda r: (get_field(r, field) is not None, get_field(r, field) or ""), reverse=True)
    if sort_key == "generation_tok_s":
        return sorted(
            rows,
            key=lambda r: (
                benchmark_valid(r),
                get_field(r, field) if get_field(r, field) is not None else -1,
            ),
            reverse=True,
        )
    return sorted(rows, key=lambda r: get_field(r, field) if get_field(r, field) is not None else -1, reverse=True)


def successful(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if benchmark_valid(r)]


def command_to_shell(cmd: list[str]) -> str:
    return " ".join(shlex_quote(x) for x in cmd)


def command_template(cmd: list[str], replacements: dict[str, str] | None = None) -> str:
    replacements = replacements or {}
    out: list[str] = []
    skip_next = False
    for i, item in enumerate(cmd):
        if skip_next:
            skip_next = False
            continue
        if item in replacements and i + 1 < len(cmd):
            out.extend([item, replacements[item]])
            skip_next = True
        else:
            out.append(item)
    return command_to_shell(out)


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(str(value))
