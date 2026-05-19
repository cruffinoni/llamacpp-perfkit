from __future__ import annotations

import os
import re
from enum import Enum


class ColorLevel(Enum):
    NONE = 0
    SIMPLE = 1
    FULL = 2


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
DEFAULT_RICH_COLOR_SYSTEM = "auto"

STYLES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
}

STATUS_STYLES = {
    "success": "green",
    "stable": "green",
    "ok": "green",
    "failed": "red",
    "oom": "red",
    "timeout": "yellow",
    "unsupported": "magenta",
    "too_close_to_vram_limit": "yellow",
    "low_vram": "yellow",
    "invalid": "yellow",
    "tight": "yellow",
    "unknown": "dim",
}


class TermColor:
    def __init__(self, color: str = "auto", no_color: bool = False) -> None:
        self._level = self._resolve(color, no_color)

    @staticmethod
    def _resolve(color: str, no_color: bool) -> ColorLevel:
        if no_color or color == "never" or "NO_COLOR" in os.environ:
            return ColorLevel.NONE
        if color == "always":
            return ColorLevel.FULL
        if os.environ.get("TERM") == "dumb":
            return ColorLevel.NONE
        if _detect_true_color():
            return ColorLevel.FULL
        return ColorLevel.SIMPLE

    @property
    def enabled(self) -> bool:
        return self._level != ColorLevel.NONE

    @property
    def level(self) -> ColorLevel:
        return self._level

    def configure(self, color: str = "auto", no_color: bool = False) -> None:
        self._level = self._resolve(color, no_color)
        try:
            import typer.rich_utils

            typer.rich_utils.COLOR_SYSTEM = DEFAULT_RICH_COLOR_SYSTEM if self._level != ColorLevel.NONE else None  # type: ignore[assignment]
        except Exception:
            pass

    def strip(self, value: str) -> str:
        return ANSI_RE.sub("", value)

    def visible_len(self, value: str) -> int:
        return len(self.strip(value))

    def sgr(self, value: object, *names: str) -> str:
        text = str(value)
        if self._level == ColorLevel.NONE or not names:
            return text
        codes = [STYLES[name] for name in names if name in STYLES]
        if not codes:
            return text
        return f"\x1b[{';'.join(codes)}m{text}\x1b[0m"

    def error(self, value: object) -> str:
        return self.sgr(value, "red", "bold")

    def warning(self, value: object) -> str:
        return self.sgr(value, "yellow", "bold")

    def note(self, value: object) -> str:
        return self.sgr(value, "cyan")

    def skip(self, value: object) -> str:
        return self.sgr(value, "magenta")

    def heading(self, value: object) -> str:
        return self.sgr(value, "cyan", "bold")

    def command(self, value: object) -> str:
        return self.sgr(value, "green")

    def status(self, value: object) -> str:
        text = str(value)
        style_name = STATUS_STYLES.get(text)
        return self.sgr(text, style_name) if style_name else text

    def colorize_evidence(self, value: object) -> str:
        text = str(value)
        status_word = text.split()[0] if text else text
        if status_word == "stable":
            return self.sgr(text, "green")
        if status_word in {"tight", "timeout"}:
            return self.sgr(text, "yellow")
        if status_word in {"oom", "failed"}:
            return self.sgr(text, "red")
        if status_word == "unsupported":
            return self.sgr(text, "magenta")
        return text

    def colorize_delta(
        self,
        text: str,
        value: float,
        *,
        higher_is_better: bool = True,
        mild_threshold: float | None = None,
    ) -> str:
        good = value >= 0 if higher_is_better else value <= 0
        if good:
            return self.sgr(text, "green")
        if mild_threshold is not None and abs(value) < mild_threshold:
            return self.sgr(text, "yellow")
        return self.sgr(text, "red")

    def gradient(
        self,
        text: str,
        value: float,
        min_val: float,
        max_val: float,
        *,
        higher_is_better: bool = True,
    ) -> str:
        if self._level != ColorLevel.FULL:
            return text
        if max_val == min_val:
            return text
        pos = (value - min_val) / (max_val - min_val)
        if higher_is_better:
            pos = 1.0 - pos
        pos = max(0.0, min(1.0, pos))
        if pos <= 0.5:
            r = int(pos * 2 * 255)
            g = 255
        else:
            r = 255
            g = int((1.0 - pos) * 2 * 255)
        return f"\x1b[38;2;{r};{g};0m{text}\x1b[0m"

    def table_value(self, column: str, value: str) -> str:
        if column in {"status", "stability_status", "stability", "risk"}:
            return self.status(value)
        if column == "parsed.benchmark_valid":
            if value == "True":
                return self.sgr(value, "green")
            if value == "False":
                return self.sgr(value, "yellow")
            return value
        if column in {"parsed.benchmark_invalid_reason"} and value != "-":
            return self.warning(value)
        if column in {"speedup_pct", "gain_tok_s"}:
            try:
                number = float(value)
            except ValueError:
                return value
            return self.sgr(value, "green") if number >= 0 else self.sgr(value, "red")
        if column == "kv":
            return self._colorize_config_token(value)
        if column == "moe":
            if value != "-" and value.isdigit():
                return self.sgr(value, "blue")
            return value
        if column == "spec":
            if value == "base":
                return self.sgr(value, "green")
            if value == "mtp":
                return self.sgr(value, "magenta")
            if value not in {"-", "base", "mtp"}:
                return self.sgr(value, "yellow")
            return value
        if column == "ctx":
            if value != "-":
                return self.sgr(value, "blue")
            return value
        return value

    def colorize_config_label(self, label: str) -> str:
        tokens = label.split(" ")
        colored = []
        for token in tokens:
            colored.append(self._colorize_config_token(token))
        return " ".join(colored)

    def _colorize_config_token(self, token: str) -> str:
        if token in {
            "f16",
            "f32",
            "q8_0",
            "q4_0",
            "q4_1",
            "q5_0",
            "q5_1",
            "q2_k",
            "q3_k",
            "q4_k",
            "q5_k",
            "q6_k",
            "q8_k",
            "iq1_s",
            "iq1_m",
            "iq2_s",
            "iq2_m",
            "iq3_s",
            "iq3_m",
            "iq4_nl",
            "iq4_xs",
        }:
            return self.sgr(token, "cyan")
        if token == "base":
            return self.sgr(token, "green")
        if token == "mtp":
            return self.sgr(token, "magenta")
        if token in {"draft-mtp", "ngram", "hfd", "speculative"}:
            return self.sgr(token, "yellow")
        if token.endswith(("-it", "-instruct", "-chat")):
            return self.sgr(token, "dim")
        if token.lower().startswith("moe") and token[3:].isdigit():
            return self.sgr(token, "blue")
        if token.startswith("ctx") and (token[3:].isdigit() or re.match(r"^\d+\.\d{2}k$", token[3:])):
            return self.sgr(token, "blue")
        return token


def _detect_true_color() -> bool:
    if os.environ.get("COLORTERM") in {"truecolor", "24bit"}:
        return True
    term = os.environ.get("TERM", "")
    return "256color" in term or "truecolor" in term or "24bit" in term


_term_color = TermColor()


def configure_color(color: str = "auto", no_color: bool = False) -> None:
    _term_color.configure(color, no_color)


def color_enabled() -> bool:
    return _term_color.enabled


def strip_style(value: str) -> str:
    return _term_color.strip(value)


def visible_len(value: str) -> int:
    return _term_color.visible_len(value)


def style(value: object, *names: str) -> str:
    return _term_color.sgr(value, *names)


def error(value: object) -> str:
    return _term_color.error(value)


def warning(value: object) -> str:
    return _term_color.warning(value)


def note(value: object) -> str:
    return _term_color.note(value)


def skip(value: object) -> str:
    return _term_color.skip(value)


def heading(value: object) -> str:
    return _term_color.heading(value)


def command(value: object) -> str:
    return _term_color.command(value)


def status(value: object) -> str:
    return _term_color.status(value)


def style_table_value(column: str, value: str) -> str:
    return _term_color.table_value(column, value)


def colorize_evidence(value: object) -> str:
    return _term_color.colorize_evidence(value)


def colorize_delta(
    text: str,
    value: float,
    *,
    higher_is_better: bool = True,
    mild_threshold: float | None = None,
) -> str:
    return _term_color.colorize_delta(text, value, higher_is_better=higher_is_better, mild_threshold=mild_threshold)


def colorize_config_label(label: str) -> str:
    return _term_color.colorize_config_label(label)


def gradient_value(
    text: str,
    value: float,
    min_val: float,
    max_val: float,
    *,
    higher_is_better: bool = True,
) -> str:
    return _term_color.gradient(text, value, min_val, max_val, higher_is_better=higher_is_better)


def format_duration(seconds: object) -> str:
    if isinstance(seconds, str):
        try:
            seconds = float(seconds)
        except ValueError:
            return "-"
    if not isinstance(seconds, (int, float)):
        return "-"
    if seconds < 0:
        return "-"
    if seconds < 60:
        return f"{seconds:.2f}s"
    whole = int(round(seconds))
    minutes, secs = divmod(whole, 60)
    hours, mins = divmod(minutes, 60)
    if hours:
        return f"{hours}h {mins:02d}m {secs:02d}s"
    return f"{mins}m {secs:02d}s"


def format_number(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    if value > 1000:
        return f"{value / 1000:.2f}k"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
