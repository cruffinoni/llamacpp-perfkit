from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BuildInfoView:
    commit_short: str
    branch: str
    backend: str


@dataclass
class ProgressState:
    servers_completed: int = 0
    servers_total: int = 0
    jobs_completed: int = 0
    jobs_total: int = 0
    current_prompt_index: int = 0
    current_prompt_total: int = 0


@dataclass
class CurrentServerView:
    id: str
    context_size: int
    kv_type: str
    n_cpu_moe: int
    spec_type: str
    batch_size: int | None = None
    ubatch_size: int | None = None


@dataclass
class PromptJobView:
    profile: str
    status: str
    phase: str
    duration_seconds: float | None = None
    gen_tok_s: float | None = None
    prompt_tok_s: float | None = None
    min_vram_mib: float | None = None


@dataclass
class BenchmarkTUIState:
    run_id: str
    build_info: BuildInfoView
    model_name: str
    progress: ProgressState = field(default_factory=ProgressState)
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    lifecycle_state: str = "planning"
    status_message: str = ""
    active_prompt_profile: str | None = None
    current_server: CurrentServerView | None = None
    prompt_jobs: list[PromptJobView] = field(default_factory=list)
