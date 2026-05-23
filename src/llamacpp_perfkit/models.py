from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Mode(StrEnum):
    smoke = "smoke"
    quick = "quick"
    focused = "focused"
    full = "full"


class Status(StrEnum):
    success = "success"
    failed = "failed"
    oom = "oom"
    timeout = "timeout"
    unsupported = "unsupported"


class SortKey(StrEnum):
    balanced = "balanced"
    latest = "latest"
    generation_tok_s = "generation_tok_s"
    vram_headroom = "vram_headroom"
    peak_vram = "peak_vram"
    context_size = "context_size"


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    host: str = "127.0.0.1"
    startup_timeout_seconds: int = 300
    shutdown_timeout_seconds: int = 15


class LlamaConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    bin_dir: str = "../llama.cpp/build/bin"
    preferred_binary: str = "llama-server"
    server: ServerConfig = Field(default_factory=ServerConfig)
    extra_args: dict[str, Any] | list[Any] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    hf: str | None = None


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    file: str = "prompts/default.txt"
    profiles: list[Any] | None = None


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    generation_tokens: int = 512
    seed: int | None = None
    min_vram_headroom_gb: float = 1.5
    monitor_interval_seconds: float = 1
    timeout_seconds: int = 900
    cache_prompt: bool = False


class BudgetConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: Mode = Mode.quick
    max_runs: int = 8
    reuse_existing_results: bool = True
    stop_if_all_remaining_are_risky: bool = True


class MatrixConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    n_cpu_moe: list[int | None] = Field(default_factory=lambda: [0])  # type: ignore[arg-type]
    context_size: list[int] = Field(default_factory=lambda: [4096])
    kv_type: list[str] = Field(default_factory=list)
    batch_size: list[int] = Field(default_factory=lambda: [1024])
    ubatch_size: list[int] = Field(default_factory=lambda: [1024])
    spec_type: list[str | None] = Field(default_factory=list)
    spec_draft_n_max: list[int | None] = Field(default_factory=list)
    spec_draft_p_min: list[float | None] = Field(default_factory=list)


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    logs_dir: str = "logs"
    results_dir: str = "runs"


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    llama: LlamaConfig = Field(default_factory=LlamaConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    matrix: MatrixConfig = Field(default_factory=MatrixConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


class BenchOptions(BaseModel):
    mode: Mode | None = None
    max_runs: int | None = None
    limit: int | None = None
    retry_failed: bool = False
    force: bool = False
    dry_run: bool = False


class ReportFilters(BaseModel):
    runs: Path = Path("runs")
    results: Path | None = None
    model: str | None = None
    quant: str | None = None
    context_size: int | None = None
    kv_type: str | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    prompt_profile: str | None = None
    n_cpu_moe: int | None = None
    status: Status | None = None
    sort: SortKey = SortKey.generation_tok_s
    limit: int = 20

    def as_report_namespace(self) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            runs=str(self.results or self.runs),
            results=str(self.results or self.runs),
            model=self.model,
            quant=self.quant,
            context_size=self.context_size,
            kv_type=self.kv_type,
            batch_size=self.batch_size,
            ubatch_size=self.ubatch_size,
            prompt_profile=self.prompt_profile,
            n_cpu_moe=self.n_cpu_moe,
            status=self.status.value if self.status else None,
            sort=self.sort.value,
            limit=self.limit,
        )


class GenerateOptions(BaseModel):
    output_dir: Path = Path("config")
    model: str
    llama_bin: Path
    name: str | None = None
    temp: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 0.0
    min_p: float = 0.0
    n_gpu_layers: int = 99
    split_mode: str = "none"
    parallel: int = 1
