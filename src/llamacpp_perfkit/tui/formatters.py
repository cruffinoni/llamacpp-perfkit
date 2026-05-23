from __future__ import annotations


def format_context_size(tokens: int) -> str:
    if tokens < 1000:
        return str(tokens)
    rounded = round(tokens / 1000)
    return f"{rounded}k"


def format_gib_from_mib(mib: float | None) -> str:
    if mib is None:
        return "-"
    return f"{mib / 1024.0:.2f} GiB"


def format_duration_row(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds:.2f}s"


def format_duration_header(seconds: float) -> str:
    whole = int(seconds)
    minutes, secs = divmod(whole, 60)
    hours, mins = divmod(minutes, 60)
    if hours:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def format_gen_tok_s(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def format_prompt_tok_s(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 100 or value == 0:
        return str(int(value))
    return f"{value:.1f}"


def format_progress_bar(done: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "[" + "\u2591" * width + "]"
    ratio = min(done, total) / total
    filled = round(ratio * width)
    empty = width - filled
    block = "\u2588"
    shade = "\u2591"
    return f"[{block * filled}{shade * empty}]"
