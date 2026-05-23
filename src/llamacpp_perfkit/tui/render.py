from __future__ import annotations

from .colors import phase_color, status_color, styled
from .formatters import (
    format_context_size,
    format_duration_header,
    format_duration_row,
    format_gen_tok_s,
    format_gib_from_mib,
    format_progress_bar,
    format_prompt_tok_s,
)
from .state import BenchmarkTUIState


def render(state: BenchmarkTUIState) -> str:
    lines: list[str] = [
        _render_header(state),
        "",
        _render_summary(state),
        "",
        _render_progress(state),
        "",
        _render_current_server(state),
        "",
        _render_prompt_table(state),
    ]
    return "\n".join(lines)


def _render_header(state: BenchmarkTUIState) -> str:
    bi = state.build_info
    line1 = styled("llama-cpp-perfkit", "bold") + f"  run {state.run_id}"
    parts = [styled("llama.cpp: ", "bold") + styled(bi.commit_short, "dim")]
    if bi.branch and bi.branch != "unknown":
        parts.append(styled(bi.branch, "cyan"))
    parts.append(styled(bi.backend, "cyan"))
    line2 = "  ".join(parts)
    line3 = f"Model: {state.model_name}"
    return "\n".join([line1, line2, line3])


def _render_summary(state: BenchmarkTUIState) -> str:
    p = state.progress
    elapsed = format_duration_header(state.elapsed_seconds)
    eta = format_duration_header(state.eta_seconds)
    return f"Matrix: {p.servers_completed}/{p.servers_total} servers   Jobs: {p.jobs_completed}/{p.jobs_total}   Elapsed: {elapsed}   ETA: {eta}"


def _render_progress(state: BenchmarkTUIState) -> str:
    p = state.progress
    bar_width = 30
    lines: list[str] = []
    for label, done, total in [
        ("Servers ", p.servers_completed, p.servers_total),
        ("Jobs    ", p.jobs_completed, p.jobs_total),
        ("Current ", p.current_prompt_index, p.current_prompt_total),
    ]:
        bar = format_progress_bar(done, total, bar_width)
        display_total = total if total > 0 else 1
        display_done = min(done, display_total)
        suffix = f" {display_done}/{display_total}"
        if label.startswith("Current"):
            suffix += " prompts"
        lines.append(f"{label} {styled(bar, 'cyan')}{suffix}")
    return "\n".join(lines)


def _render_current_server(state: BenchmarkTUIState) -> str:
    if state.current_server is None:
        return "Current server\n(none)"
    cs = state.current_server
    ctx_str = styled(format_context_size(cs.context_size), "blue")
    moe_str = styled(str(cs.n_cpu_moe), "blue")
    spec_str = styled(cs.spec_type, "yellow")
    kv_str = styled(cs.kv_type, "cyan")
    bs = cs.batch_size
    ub = cs.ubatch_size
    lines = [
        styled("Current server", "bold"),
        f"id: {cs.id}",
        f"  ctx={ctx_str}  kv={kv_str}  moe={moe_str}  spec={spec_str}  batch={bs}  ubatch={ub}",
    ]
    return "\n".join(lines)


def _render_prompt_table(state: BenchmarkTUIState) -> str:
    if not state.prompt_jobs:
        return "No prompts."

    header_fmt = "{:<18} {:^9} {:^10} {:>6} {:>10} {:>13} {:>10}"
    header = header_fmt.format("profile", "status", "phase", "time", "gen tok/s", "prompt tok/s", "min vram")
    sep = "-" * len(header)

    rows: list[str] = []
    for job in state.prompt_jobs:
        s_color = status_color(job.status)
        p_color = phase_color(job.phase)
        status_col = styled(f"{job.status:^9}", s_color)
        phase_col = styled(f"{job.phase:^10}", p_color)
        time_col = format_duration_row(job.duration_seconds)
        gen_col = format_gen_tok_s(job.gen_tok_s)
        prompt_col = format_prompt_tok_s(job.prompt_tok_s)
        vram_col = format_gib_from_mib(job.min_vram_mib)
        row = header_fmt.format(
            job.profile,
            status_col,
            phase_col,
            f"{time_col:>6}",
            f"{gen_col:>10}",
            f"{prompt_col:>13}",
            f"{vram_col:>10}",
        )
        rows.append(row)

    return "\n".join([header, sep] + rows)
