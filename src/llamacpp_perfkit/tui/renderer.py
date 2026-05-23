from __future__ import annotations

import copy
import threading
from collections.abc import Callable
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import DataTable, Footer, Header, Static

from .colors import phase_color, status_color
from .formatters import (
    format_context_size,
    format_duration_header,
    format_duration_row,
    format_gen_tok_s,
    format_gib_from_mib,
    format_progress_bar,
    format_prompt_tok_s,
)
from .state import BenchmarkTUIState, BuildInfoView, CurrentServerView, PromptJobView


class TUIRenderer:
    def __init__(self, state: BenchmarkTUIState) -> None:
        self._state = state
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._app: BenchmarkTUIApp | None = None

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested.is_set()

    def start(self) -> None:
        """Compatibility hook for older tests and callers."""

    def stop_tui(self) -> None:
        """Compatibility hook for older tests and callers."""

    def run(self, benchmark: Callable[[], int]) -> int:
        app = BenchmarkTUIApp(self, benchmark)
        self._app = app
        result = app.run()
        self._app = None
        if app.error is not None:
            raise app.error
        return int(result if result is not None else app.result)

    def request_stop(self) -> None:
        self._stop_requested.set()
        self.update({"lifecycle_state": "stopping", "status_message": "Stop requested; waiting for the active request to finish."})

    def update(self, updates: dict[str, Any]) -> None:
        with self._lock:
            self._apply_updates(updates)
            snapshot = copy.deepcopy(self._state)
        app = self._app
        if app is None:
            return
        try:
            app.call_from_thread(app.apply_state, snapshot)
        except RuntimeError:
            if app.is_running:
                app.apply_state(snapshot)

    def snapshot(self) -> BenchmarkTUIState:
        with self._lock:
            return copy.deepcopy(self._state)

    def _apply_updates(self, updates: dict[str, Any]) -> None:
        s = self._state

        if "run_id" in updates:
            s.run_id = str(updates["run_id"])

        ci = updates.get("build_info")
        if isinstance(ci, dict):
            s.build_info = BuildInfoView(
                commit_short=str(ci.get("commit_short", s.build_info.commit_short)),
                branch=str(ci.get("branch", s.build_info.branch)),
                backend=str(ci.get("backend", s.build_info.backend)),
            )
        else:
            for key in ("llama_cpp_commit", "llama_cpp_branch", "backend"):
                if key in updates:
                    if key == "llama_cpp_commit":
                        s.build_info.commit_short = str(updates[key])
                    elif key == "llama_cpp_branch":
                        s.build_info.branch = str(updates[key])
                    elif key == "backend":
                        s.build_info.backend = str(updates[key])

        if "model_name" in updates:
            s.model_name = str(updates["model_name"])
        if "lifecycle_state" in updates:
            s.lifecycle_state = str(updates["lifecycle_state"])
        if "status_message" in updates:
            s.status_message = str(updates["status_message"])
        if "active_prompt_profile" in updates:
            value = updates["active_prompt_profile"]
            s.active_prompt_profile = None if value is None else str(value)

        p = s.progress
        for attr in (
            "servers_completed",
            "servers_total",
            "jobs_completed",
            "jobs_total",
            "current_prompt_index",
            "current_prompt_total",
        ):
            if attr in updates:
                setattr(p, attr, int(updates[attr]))

        if "server_index" in updates:
            p.servers_completed = int(updates["server_index"])
        if "server_total" in updates:
            p.servers_total = int(updates["server_total"])
        if "job_index" in updates:
            p.jobs_completed = int(updates["job_index"])
        if "job_total" in updates:
            p.jobs_total = int(updates["job_total"])

        if "elapsed_seconds" in updates:
            s.elapsed_seconds = float(updates["elapsed_seconds"])
        if "eta_seconds" in updates:
            s.eta_seconds = float(updates["eta_seconds"])

        cs_data = updates.get("current_server")
        if isinstance(cs_data, dict):
            s.current_server = CurrentServerView(
                id=str(cs_data.get("id", "")),
                context_size=int(cs_data.get("context_size", 0)),
                kv_type=str(cs_data.get("kv_type", "")),
                n_cpu_moe=int(cs_data.get("n_cpu_moe", 0)),
                spec_type=str(cs_data.get("spec_type", "none")),
                batch_size=cs_data.get("batch_size"),
                ubatch_size=cs_data.get("ubatch_size"),
            )
        elif all(k in updates for k in ("context_size", "kv_type", "n_cpu_moe")):
            s.current_server = CurrentServerView(
                id=str(updates.get("server_run_id") or updates.get("current_server_id") or ""),
                context_size=int(updates["context_size"]),
                kv_type=str(updates["kv_type"]),
                n_cpu_moe=int(updates["n_cpu_moe"]),
                spec_type=str(updates.get("spec_type", "none")),
                batch_size=updates.get("batch_size"),
                ubatch_size=updates.get("ubatch_size"),
            )

        if "clear_prompts" in updates:
            s.prompt_jobs.clear()
            s.active_prompt_profile = None

        if "prompt" in updates:
            profile = str(updates.get("prompt_profile", "default"))
            if updates.get("status") == "running":
                s.active_prompt_profile = profile
            found = False
            for i, job in enumerate(s.prompt_jobs):
                if job.profile == profile:
                    s.prompt_jobs[i] = PromptJobView(
                        profile=profile,
                        status=str(updates.get("status", job.status)),
                        phase=str(updates.get("phase", job.phase)),
                        duration_seconds=updates.get("duration_seconds", job.duration_seconds),
                        gen_tok_s=updates.get("gen_tok_s", job.gen_tok_s),
                        prompt_tok_s=updates.get("prompt_tok_s", job.prompt_tok_s),
                        min_vram_mib=updates.get("min_vram_mib", job.min_vram_mib),
                    )
                    found = True
                    break
            if not found:
                s.prompt_jobs.append(
                    PromptJobView(
                        profile=profile,
                        status=str(updates.get("status", "pending")),
                        phase=str(updates.get("phase", "pending")),
                        duration_seconds=updates.get("duration_seconds"),
                        gen_tok_s=updates.get("gen_tok_s"),
                        prompt_tok_s=updates.get("prompt_tok_s"),
                        min_vram_mib=updates.get("min_vram_mib"),
                    )
                )


class BenchmarkTUIApp(App[int]):
    CSS = """
    Screen {
        background: #080d12;
        color: #d8e2ea;
    }

    Header, Footer {
        background: #0d151c;
        color: #d8e2ea;
    }

    #dashboard {
        height: 100%;
        layout: vertical;
        padding: 1 2;
    }

    .panel {
        border: round #294256;
        background: #0d151c;
        padding: 0 1;
    }

    #title {
        height: 5;
        margin-bottom: 1;
    }

    #top-row {
        height: 12;
        margin-bottom: 1;
    }

    #progress-panel {
        width: 42%;
        margin-right: 1;
    }

    #server-panel {
        width: 58%;
    }

    #message-panel {
        height: 3;
        margin-bottom: 1;
    }

    #prompt-table {
        height: 1fr;
        border: round #294256;
        background: #0d151c;
    }
    """

    BINDINGS = [
        ("q", "request_stop", "Stop"),
        ("ctrl+c", "request_stop", "Stop"),
    ]

    def __init__(self, renderer: TUIRenderer, benchmark: Callable[[], int]) -> None:
        super().__init__()
        self._renderer = renderer
        self._benchmark = benchmark
        self._state = renderer.snapshot()
        self.result = 0
        self.error: Exception | None = None
        self._table_ready = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="dashboard"):
            yield Static(id="title", classes="panel")
            with Horizontal(id="top-row"):
                yield Static(id="progress-panel", classes="panel")
                yield Static(id="server-panel", classes="panel")
            yield Static(id="message-panel", classes="panel")
            yield DataTable(id="prompt-table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#prompt-table", DataTable)
        table.cursor_type = "none"
        table.zebra_stripes = True
        table.add_columns("profile", "status", "phase", "time", "gen tok/s", "prompt tok/s", "min vram")
        self._table_ready = True
        self.apply_state(self._state)
        self.run_worker(self._benchmark_worker, thread=True, exit_on_error=False)

    def action_request_stop(self) -> None:
        self._renderer.request_stop()

    def apply_state(self, state: BenchmarkTUIState) -> None:
        self._state = state
        if not self._table_ready:
            return
        self._refresh_title()
        self._refresh_progress()
        self._refresh_server()
        self._refresh_message()
        self._refresh_prompts()

    def _benchmark_worker(self) -> None:
        try:
            self.result = self._benchmark()
            self._renderer.update({"lifecycle_state": "complete", "status_message": "Benchmark run complete."})
        except Exception as exc:
            self.error = exc
            self.result = 1
            self._renderer.update({"lifecycle_state": "failed", "status_message": str(exc)})
        finally:
            self.call_from_thread(self.exit, self.result)

    def _refresh_title(self) -> None:
        state = self._state
        bi = state.build_info
        text = Text()
        text.append("llama-cpp-perfkit", style="bold cyan")
        text.append(f"  run {state.run_id}", style="dim")
        text.append(f"  {state.lifecycle_state}", style=_state_style(state.lifecycle_state))
        text.append("\n")
        text.append("model ", style="dim")
        text.append(state.model_name or "-", style="bold")
        text.append("\n")
        text.append("llama.cpp ", style="dim")
        text.append(bi.commit_short or "unknown", style="cyan")
        if bi.branch and bi.branch != "unknown":
            text.append(f"  {bi.branch}", style="blue")
        text.append("  backend ", style="dim")
        text.append(bi.backend or "unknown", style="cyan")
        self.query_one("#title", Static).update(text)

    def _refresh_progress(self) -> None:
        state = self._state
        p = state.progress
        text = Text("Progress\n", style="bold cyan")
        _append_bar(text, "Servers", p.servers_completed, p.servers_total)
        _append_bar(text, "Jobs", p.jobs_completed, p.jobs_total)
        _append_bar(text, "Current", p.current_prompt_index, p.current_prompt_total)
        text.append("\n")
        _append_kv(text, "elapsed", format_duration_header(state.elapsed_seconds), "green")
        _append_kv(text, "eta", format_duration_header(state.eta_seconds), "yellow")
        if state.active_prompt_profile:
            text.append("\nactive ", style="dim")
            text.append(state.active_prompt_profile, style="bold cyan")
        self.query_one("#progress-panel", Static).update(text)

    def _refresh_server(self) -> None:
        state = self._state
        text = Text("Current Server\n", style="bold cyan")
        cs = state.current_server
        if cs is None:
            text.append("waiting for runnable server group", style="dim")
        else:
            _append_kv(text, "id", cs.id or "-", "dim")
            text.append("\n")
            _append_kv(text, "ctx", format_context_size(cs.context_size), "blue")
            _append_kv(text, "kv", cs.kv_type or "-", "cyan")
            _append_kv(text, "moe", str(cs.n_cpu_moe), "blue")
            _append_kv(text, "spec", cs.spec_type or "none", "yellow" if cs.spec_type and cs.spec_type != "none" else "dim")
            text.append("\n")
            _append_kv(text, "batch", _optional_int(cs.batch_size), "green")
            _append_kv(text, "ubatch", _optional_int(cs.ubatch_size), "green")
        self.query_one("#server-panel", Static).update(text)

    def _refresh_message(self) -> None:
        state = self._state
        text = Text()
        text.append("Status ", style="bold cyan")
        text.append(state.lifecycle_state, style=_state_style(state.lifecycle_state))
        if state.status_message:
            text.append("  ")
            text.append(state.status_message, style="dim")
        self.query_one("#message-panel", Static).update(text)

    def _refresh_prompts(self) -> None:
        if not self._table_ready:
            return
        table = self.query_one("#prompt-table", DataTable)
        table.clear(columns=False)
        for job in self._state.prompt_jobs:
            table.add_row(
                Text(job.profile, style="bold" if job.profile == self._state.active_prompt_profile else ""),
                Text(job.status, style=_rich_color(status_color(job.status))),
                Text(job.phase, style=_rich_color(phase_color(job.phase))),
                format_duration_row(job.duration_seconds),
                format_gen_tok_s(job.gen_tok_s),
                format_prompt_tok_s(job.prompt_tok_s),
                format_gib_from_mib(job.min_vram_mib),
            )


def _append_bar(text: Text, label: str, done: int, total: int) -> None:
    display_total = total if total > 0 else 1
    display_done = min(done, display_total)
    text.append(f"{label:<8}", style="dim")
    text.append(format_progress_bar(done, total, width=24), style="cyan")
    text.append(f" {display_done}/{display_total}\n", style="white")


def _append_kv(text: Text, key: str, value: str, style: str) -> None:
    text.append(f"{key} ", style="dim")
    text.append(value, style=style)
    text.append("  ")


def _optional_int(value: int | None) -> str:
    return str(value) if value is not None else "-"


def _rich_color(color: str) -> str:
    return "bright_black" if color == "dim" else color


def _state_style(value: str) -> str:
    lower = value.lower()
    if "complete" in lower:
        return "green"
    if "stop" in lower or "timeout" in lower:
        return "yellow"
    if "fail" in lower or "oom" in lower:
        return "red"
    return "cyan"
