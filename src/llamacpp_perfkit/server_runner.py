from __future__ import annotations

import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .benchlib import (
    Monitor,
    benchmark_invalid_reason,
    benchmark_valid,
    build_request_payload,
    build_server_cmd,
    classify_http_error,
    classify_run,
    command_to_shell,
    free_tcp_port,
    http_json,
    model_hf,
    parse_chat_completion,
    parse_llama_output,
    parse_server_completion,
    prompt_file_for_job,
    request_endpoint,
    server_group_key,
    stability_status,
)
from .planner import make_plan, write_plan
from .run_storage import (
    append_llamacpp_metric,
    append_system_metric,
    llamacpp_metrics_path,
    load_run_rows,
    run_dir,
    system_metrics_path,
    terminal_llamacpp_sample,
    write_run_summary,
)
from .tui import BenchmarkTUIState, BuildInfoView, TUIRenderer


class LlamaServerProcess:
    def __init__(
        self,
        cmd: list[str],
        host: str,
        port: int,
        log_path: Path,
        startup_timeout: float,
        shutdown_timeout: float,
    ) -> None:
        self.cmd = cmd
        self.host = host
        self.port = port
        self.log_path = log_path
        self.startup_timeout = float(startup_timeout)
        self.shutdown_timeout = float(shutdown_timeout)
        self.proc: subprocess.Popen[Any] | None = None
        self.log_file: Any = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> LlamaServerProcess:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(self.cmd, stdout=self.log_file, stderr=subprocess.STDOUT, text=True)
        return self

    def __exit__(self, *_: Any) -> None:
        self.terminate()
        if self.log_file:
            self.log_file.close()

    def wait_until_ready(self) -> tuple[bool, int, str]:
        deadline = time.time() + self.startup_timeout
        last_status: int | None = None
        last_text = ""
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                return False, self.proc.returncode, f"server exited before healthy with returncode {self.proc.returncode}"
            try:
                status, body, text = http_json("GET", f"{self.base_url}/health", timeout=5)
                last_status = status
                last_text = text
                if status == 200 and body.get("status") == "ok":
                    return True, 0, "ok"
            except (OSError, urllib.error.URLError) as exc:
                last_text = str(exc)
            time.sleep(1)
        return False, 124, f"server health timeout; last_status={last_status} last_response={last_text}"

    def terminate(self) -> None:
        if not self.proc or self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=self.shutdown_timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)


class LlamaCppMetricsSampler:
    def __init__(self, base_url: str, path: Path, interval: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.path = path
        self.interval = float(interval)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> LlamaCppMetricsSampler:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=max(2.0, self.interval + 1.0))

    def _run(self) -> None:
        while not self.stop.is_set():
            sample = self.sample_once()
            if sample:
                append_llamacpp_metric(self.path.parent.parent, sample)
            self.stop.wait(self.interval)

    def sample_once(self) -> dict[str, Any] | None:
        sample: dict[str, Any] = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            status, body, _ = http_json("GET", f"{self.base_url}/slots", timeout=2)
        except Exception:
            status, body = None, None
        if status == 200:
            slots: Any = body if isinstance(body, list) else (body or {}).get("slots") or (body or {}).get("data") or []
            if isinstance(slots, list):
                processing = 0
                idle = 0
                prompt_tokens: list[float] = []
                generated_tokens: list[float] = []
                for slot in slots:
                    if not isinstance(slot, dict):
                        continue
                    busy = bool(slot.get("is_processing")) or str(slot.get("state", "")).lower() in {"processing", "busy"}
                    processing += 1 if busy else 0
                    idle += 0 if busy else 1
                    for key in ("n_prompt_tokens", "prompt_tokens", "prompt_n"):
                        if isinstance(slot.get(key), (int, float)):
                            prompt_tokens.append(float(slot[key]))
                    for key in ("n_decoded", "generated_tokens", "predicted_n"):
                        if isinstance(slot.get(key), (int, float)):
                            generated_tokens.append(float(slot[key]))
                sample.update(
                    {
                        "slots_idle": idle,
                        "slots_processing": processing,
                        "prompt_tokens": max(prompt_tokens, default=None),
                        "generated_tokens": max(generated_tokens, default=None),
                    }
                )
        try:
            status, text = self._http_text(f"{self.base_url}/metrics", timeout=2)
        except Exception:
            status, text = None, ""
        if status == 200 and text:
            sample.update(self._parse_prometheus(text))
        return sample if len(sample) > 1 else None

    @staticmethod
    def _http_text(url: str, timeout: float = 2) -> tuple[int, str]:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.status, res.read().decode("utf-8", errors="replace")

    @staticmethod
    def _parse_prometheus(text: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                continue
            name, value = parts
            try:
                number = float(value)
            except ValueError:
                continue
            label_free_name = name.split("{", 1)[0].lower()
            if "prompt" in label_free_name and "token" in label_free_name:
                out.setdefault("prompt_tokens", number)
            elif (
                "predicted" in label_free_name or "generation" in label_free_name or "eval" in label_free_name
            ) and "token" in label_free_name:
                out.setdefault("generated_tokens", number)
        return out


class LlamaServerBenchmarkRunner:
    def __init__(
        self,
        cfg: dict[str, Any],
        args: Any,
        features: dict[str, Any],
        results_dir: Path,
        raw_dir: Path,
        mon_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.args = args
        self.features = features
        self.results_dir = results_dir
        self.raw_dir = raw_dir
        self.mon_dir = mon_dir
        self.plan_path = results_dir / "last_plan.json"
        self.timeout = int(cfg["run"].get("timeout_seconds", 900))
        self.interval = float(cfg["run"].get("monitor_interval_seconds", 1))
        self.min_headroom = float(cfg["run"].get("min_vram_headroom_gb", 1.5))
        server_cfg = cfg.get("llama", {}).get("server", {})
        self.host = server_cfg.get("host", "127.0.0.1")
        self.startup_timeout = server_cfg.get("startup_timeout_seconds", 300)
        self.shutdown_timeout = server_cfg.get("shutdown_timeout_seconds", 15)

        llama_cpp_meta = features.get("llama_cpp", {}) or {}
        run_id = f"{int(time.time())}_{llama_cpp_meta.get('commit_short', 'unknown')[:8]}"
        self.tui_state = BenchmarkTUIState(
            run_id=run_id,
            build_info=BuildInfoView(
                commit_short=llama_cpp_meta.get("commit_short", "unknown"),
                branch=llama_cpp_meta.get("branch", "unknown"),
                backend=features.get("backend", "server"),
            ),
            model_name=model_hf(cfg) or "unknown",
        )
        self.tui_renderer = TUIRenderer(self.tui_state)
        self._jobs_completed_total = 0

    def run(self, max_runs: int | None) -> int:
        return self.tui_renderer.run(lambda: self._run_benchmark(max_runs))

    def _run_benchmark(self, max_runs: int | None) -> int:
        max_runs = None if max_runs is None or int(max_runs) <= 0 else int(max_runs)
        run_started = time.time()
        executed = 0
        group_index = 0
        attempted_hashes: set[str] = set()
        self._update_tui(lifecycle_state="planning", status_message="Building benchmark plan.")
        while not self.tui_renderer.stop_requested and (max_runs is None or executed < max_runs):
            rows = load_run_rows(self.results_dir)
            plan_limit = self.args.max_runs if self.args.max_runs is not None else self.args.limit
            plan = make_plan(
                self.cfg,
                self.features,
                rows,
                mode=self.args.mode,
                max_runs=plan_limit,
                retry_failed=self.args.retry_failed,
                force=getattr(self.args, "force", False),
            )
            write_plan(plan, self.plan_path)
            groups = self._pending_groups(plan, attempted_hashes)
            if not groups:
                break
            total_servers = len(groups)
            total_jobs = sum(len(g) for g in groups)
            self._update_tui(
                servers_total=total_servers,
                jobs_total=total_jobs,
                elapsed_seconds=time.time() - run_started,
                lifecycle_state="running",
                status_message="Running benchmark matrix.",
            )
            remaining = None if max_runs is None else max_runs - executed
            group = groups[0] if remaining is None else groups[0][:remaining]
            group_index += 1
            attempted_hashes.update(item["config_hash"] for item in group)
            executed += self._run_group(group, group_index, "unlimited" if max_runs is None else max_runs)
        final_state = "stopped" if self.tui_renderer.stop_requested else "complete"
        final_message = "Stopped after active request finished." if self.tui_renderer.stop_requested else "No runnable jobs remain."
        self._update_tui(lifecycle_state=final_state, status_message=final_message, elapsed_seconds=time.time() - run_started)
        return executed

    def _pending_groups(self, plan: dict[str, Any], attempted_hashes: set[str]) -> list[list[dict[str, Any]]]:
        groups: list[list[dict[str, Any]]] = []
        by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for item in plan["planned"]:
            if item["action"] != "run":
                continue
            if item["config_hash"] in attempted_hashes:
                continue
            key = server_group_key(item["job"])
            if key not in by_key:
                by_key[key] = []
                groups.append(by_key[key])
            by_key[key].append(item)
        return groups

    def _run_group(self, group: list[dict[str, Any]], group_index: int, max_groups: int | str) -> int:
        representative = group[0]["job"]
        prompt_profiles = [item["job"].get("prompt_profile", "default") for item in group]
        port = free_tcp_port(self.host)
        server_run_id = f"{int(time.time())}-server-{group_index:04d}"
        raw_path = self.raw_dir / f"{server_run_id}.log"
        mon_path = self.mon_dir / f"{server_run_id}.jsonl"
        cmd = build_server_cmd(self.cfg, self.features, representative, port, raw_path)
        started = time.time()
        endpoint_kind = request_endpoint(self.cfg)
        endpoint_path = "/v1/chat/completions" if endpoint_kind == "chat" else "/completion"
        server_info: dict[str, Any] = {
            "backend": "server",
            "server_run_id": server_run_id,
            "host": self.host,
            "port": port,
            "endpoint": endpoint_path,
            "endpoint_kind": endpoint_kind,
            "server_command": cmd,
            "server_command_shell": command_to_shell(cmd),
            "server_log_path": str(raw_path),
            "prompt_profiles": prompt_profiles,
        }
        # Update TUI for server start
        self._update_tui(
            server_index=group_index,
            current_server_id=server_run_id,
            current_prompt_total=len(group),
            context_size=representative["context_size"],
            kv_type=representative["kv_type"],
            n_cpu_moe=representative["n_cpu_moe"],
            spec_type=representative.get("spec_type") or "none",
            batch_size=representative.get("batch_size"),
            ubatch_size=representative.get("ubatch_size"),
            lifecycle_state="starting server",
            status_message=f"Launching {server_run_id}.",
        )
        self._update_tui(clear_prompts=True)
        for name in prompt_profiles:
            self._update_tui(prompt=True, prompt_profile=name, status="pending", phase="pending")
        with Monitor(mon_path, self.interval) as monitor:
            with LlamaServerProcess(cmd, self.host, port, raw_path, self.startup_timeout, self.shutdown_timeout) as server:
                healthy, health_rc, health_text = server.wait_until_ready()
                if not healthy:
                    return self._record_startup_failure(group, server_info, raw_path, mon_path, monitor, health_rc, health_text, started)
                executed = 0
                prompts_started = time.time()
                for idx, pending in enumerate(group):
                    if self.tui_renderer.stop_requested:
                        break
                    profile = pending["job"].get("prompt_profile", "default")
                    self._update_tui(
                        prompt=True,
                        prompt_profile=profile,
                        status="running",
                        phase="starting",
                        current_prompt_index=idx + 1,
                        lifecycle_state="running prompt",
                        status_message=f"Running {profile}.",
                    )
                    executed += self._run_request(pending, server_info, raw_path, mon_path, monitor, group_index, executed + 1)
                prompt_seconds = time.time() - prompts_started
                # Update TUI for server completion
                self._update_tui(
                    servers_completed=group_index,
                    elapsed_seconds=prompt_seconds,
                    eta_seconds=prompt_seconds,
                    lifecycle_state="server complete",
                    status_message=f"Finished {server_run_id}.",
                )
                return executed

    def _record_startup_failure(
        self,
        group: list[dict[str, Any]],
        server_info: dict[str, Any],
        raw_path: Path,
        mon_path: Path,
        monitor: Monitor,
        returncode: int,
        health_text: str,
        started: float,
    ) -> int:
        raw_text = raw_path.read_text(encoding="utf-8", errors="replace") + "\n" + health_text
        status = classify_run(returncode, returncode == 124, raw_text)
        mon_summary = monitor.summary()
        for pending in group:
            job = pending["job"]
            run_id = f"{server_info['server_run_id']}-{job.get('prompt_profile', 'default')}"
            row = self._base_row(
                pending,
                job,
                run_id,
                started,
                server_info,
                raw_path,
                mon_path,
            )
            row.update(
                {
                    "status": status,
                    "stability_status": stability_status(status, mon_summary.get("min_vram_free_mib"), self.min_headroom),
                    "returncode": returncode,
                    "parsed": parse_llama_output(raw_text),
                    "monitor": mon_summary,
                    "error": health_text,
                }
            )
            run_path = run_dir(self.results_dir, row["run_id"])
            for sample in monitor.samples:
                append_system_metric(run_path, sample)
            write_run_summary(self.results_dir, self._summary_from_row(row, server_info["server_run_id"]))
            # Update TUI for failed prompt
            self._jobs_completed_total += 1
            self._update_tui(
                prompt=True,
                jobs_completed=self._jobs_completed_total,
                status=status,
                prompt_profile=job.get("prompt_profile", "default"),
                phase=status,
                duration_seconds=row["duration_seconds"],
                min_vram_mib=mon_summary.get("min_vram_free_mib"),
                lifecycle_state=status,
                status_message=health_text,
            )
        return len(group)

    def _run_request(
        self,
        pending: dict[str, Any],
        server_info: dict[str, Any],
        raw_path: Path,
        mon_path: Path,
        monitor: Monitor,
        group_index: int,
        request_index: int,
    ) -> int:
        job = pending["job"]
        run_id = f"{int(time.time())}-{group_index:04d}-{request_index:04d}"
        request_run_dir = run_dir(self.results_dir, run_id)
        request_run_dir.mkdir(parents=True, exist_ok=True)
        (request_run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        started = time.time()
        monitor_start = monitor.checkpoint()
        monitor.set_active_path(system_metrics_path(request_run_dir))
        payload = build_request_payload(self.cfg, self.features, job)
        response_status: int | None = None
        response_body: dict[str, Any] = {}
        response_text = ""
        status = "failed"
        returncode = 1
        prompt_profile = job.get("prompt_profile", "default")
        try:
            with LlamaCppMetricsSampler(
                f"http://{server_info['host']}:{server_info['port']}",
                llamacpp_metrics_path(request_run_dir),
                self.interval,
            ):
                response_status, response_body, response_text = http_json(
                    "POST",
                    f"http://{server_info['host']}:{server_info['port']}{server_info['endpoint']}",
                    payload,
                    timeout=self.timeout,
                )
            if response_status == 200:
                status = "success"
                returncode = 0
            else:
                status = classify_http_error(response_status, response_text)
                returncode = response_status
        except TimeoutError as exc:
            status = "timeout"
            returncode = 124
            response_text = str(exc)
        except (OSError, urllib.error.URLError) as exc:
            status = "failed"
            returncode = 1
            response_text = str(exc)

        final_system_sample = monitor.sample_once()
        append_system_metric(request_run_dir, final_system_sample)
        monitor.clear_active_path()
        mon_summary = monitor.summary(monitor_start)
        if status == "success" and server_info.get("endpoint_kind") == "chat":
            raw_text = raw_path.read_text(encoding="utf-8", errors="replace") if raw_path.exists() else ""
            parsed, response = parse_chat_completion(response_body, time.time() - started, raw_text)
        elif status == "success":
            parsed = parse_server_completion(response_body)
            response = {
                "content": response_body.get("content"),
                "stop_type": response_body.get("stop_type"),
                "truncated": response_body.get("truncated"),
            }
        else:
            parsed = parse_llama_output(response_text)
            response = {
                "content": response_body.get("content"),
                "stop_type": response_body.get("stop_type"),
                "truncated": response_body.get("truncated"),
            }
        row = self._base_row(pending, job, run_id, started, server_info, raw_path, mon_path)
        row.update(
            {
                "status": status,
                "stability_status": stability_status(status, mon_summary.get("min_vram_free_mib"), self.min_headroom),
                "returncode": returncode,
                "http_status": response_status,
                "request": {k: v for k, v in payload.items() if k not in {"prompt", "messages"}},
                "response": response,
                "parsed": parsed,
                "monitor": mon_summary,
            }
        )
        if status == "success":
            parsed["benchmark_valid"] = benchmark_valid(row)
            invalid_reason = benchmark_invalid_reason(row)
            if invalid_reason:
                parsed["benchmark_invalid_reason"] = invalid_reason
        if response_text and status != "success":
            row["error"] = response_text
        append_llamacpp_metric(
            request_run_dir,
            terminal_llamacpp_sample(row["created_at"], parsed, row.get("duration_seconds")),
        )
        write_run_summary(self.results_dir, self._summary_from_row(row, server_info["server_run_id"]))
        # Update TUI for prompt completion
        self._jobs_completed_total += 1
        self._update_tui(
            prompt=True,
            jobs_completed=self._jobs_completed_total,
            status=status,
            prompt_profile=prompt_profile,
            phase="done" if status == "success" else status,
            duration_seconds=row["duration_seconds"],
            gen_tok_s=parsed.get("generation_tok_s"),
            prompt_tok_s=parsed.get("prompt_eval_tok_s"),
            min_vram_mib=mon_summary.get("min_vram_free_mib"),
            lifecycle_state="running" if status == "success" else status,
            status_message=f"{prompt_profile}: {status}",
        )
        return 1

    def _update_tui(self, **updates: Any) -> None:
        """Update TUI state with new values."""
        self.tui_renderer.update(updates)

    def _summary_from_row(self, row: dict[str, Any], batch_id: str) -> dict[str, Any]:
        cfg = row.get("config") or {}
        return {
            **row,
            "batch_id": batch_id,
            "model": cfg.get("model_hf"),
            "prompt_profile": cfg.get("prompt_profile", "default"),
            "server_config": {
                "model": cfg.get("model_hf"),
                "context_size": cfg.get("context_size"),
                "kv_type": cfg.get("kv_type"),
                "n_cpu_moe": cfg.get("n_cpu_moe"),
                "batch_size": cfg.get("batch_size"),
                "ubatch_size": cfg.get("ubatch_size"),
                "spec_type": cfg.get("spec_type"),
                "spec_draft_n_max": cfg.get("spec_draft_n_max"),
                "spec_draft_p_min": cfg.get("spec_draft_p_min"),
            },
        }

    def _base_row(
        self,
        pending: dict[str, Any],
        job: dict[str, Any],
        run_id: str,
        started: float,
        server_info: dict[str, Any],
        raw_path: Path,
        mon_path: Path,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "config_hash": job["config_hash"],
            "plan_reason": pending.get("reason"),
            "risk_level": pending.get("risk_level"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_seconds": round(time.time() - started, 3),
            "command": server_info["server_command"],
            "command_shell": server_info["server_command_shell"],
            "backend": "server",
            "llama_cpp": self.features.get("llama_cpp") or {},
            "server": server_info,
            "config": {
                **job,
                "model_hf": model_hf(self.cfg),
                "generation_tokens": self.cfg["run"].get("generation_tokens", 512),
                "endpoint": request_endpoint(self.cfg),
                "seed": self.cfg["run"].get("seed"),
                "cache_prompt": self.cfg["run"].get("cache_prompt", False),
                "prompt_profile": job.get("prompt_profile", "default"),
                "prompt_file": str(prompt_file_for_job(self.cfg, job)),
            },
            "raw_log_path": str(raw_path),
            "monitoring_log_path": str(mon_path),
        }
