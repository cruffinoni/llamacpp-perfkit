from __future__ import annotations

from ..output import style as _style


def status_color(status: str) -> str:
    m: dict[str, str] = {
        "success": "green",
        "running": "cyan",
        "pending": "dim",
        "timeout": "yellow",
        "oom": "red",
        "failed": "red",
    }
    return m.get(status.lower(), "dim")


def phase_color(phase: str) -> str:
    m: dict[str, str] = {
        "prefill": "cyan",
        "generating": "cyan",
        "done": "dim",
        "starting": "cyan",
        "pending": "dim",
        "timeout": "yellow",
        "oom": "red",
        "failed": "red",
        "-": "dim",
    }
    return m.get(phase.lower(), "dim")


def styled(text: str, *names: str) -> str:
    return _style(text, *names)
