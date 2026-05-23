from __future__ import annotations

import os
import unittest
from typing import Any

from llamacpp_perfkit.output import (
    ColorLevel,
    TermColor,
    color_enabled,
    colorize_config_label,
    colorize_delta,
    colorize_evidence,
    command,
    configure_color,
    error,
    format_number,
    gradient_value,
    heading,
    note,
    skip,
    status,
    strip_style,
    style,
    style_table_value,
    visible_len,
    warning,
)


class ColorEnvTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_no_color = os.environ.pop("NO_COLOR", None)

    def tearDown(self) -> None:
        if self._saved_no_color is not None:
            os.environ["NO_COLOR"] = self._saved_no_color


class TestTermColorConfiguration(ColorEnvTestCase):
    def test_auto_enabled_by_default(self) -> None:
        self.assertTrue(TermColor().enabled)

    def test_never_disabled(self) -> None:
        self.assertFalse(TermColor(color="never").enabled)

    def test_always_enabled(self) -> None:
        self.assertTrue(TermColor(color="always").enabled)

    def test_auto_disabled_when_term_dumb(self) -> None:
        with _env("TERM", "dumb"):
            self.assertFalse(TermColor().enabled)

    def test_auto_disabled_when_no_color_env(self) -> None:
        with _env("NO_COLOR", "1"):
            self.assertFalse(TermColor().enabled)

    def test_no_color_overrides_always(self) -> None:
        tc = TermColor(color="always", no_color=True)
        self.assertFalse(tc.enabled)
        self.assertEqual(tc.level, ColorLevel.NONE)

    def test_auto_with_true_color_gives_full(self) -> None:
        with _env("COLORTERM", "truecolor"):
            self.assertEqual(TermColor().level, ColorLevel.FULL)

    def test_auto_without_true_color_gives_simple(self) -> None:
        with _env("COLORTERM", None), _env("TERM", "xterm"):
            self.assertEqual(TermColor().level, ColorLevel.SIMPLE)

    def test_configure_toggle(self) -> None:
        tc = TermColor()
        self.assertTrue(tc.enabled)
        tc.configure(no_color=True)
        self.assertFalse(tc.enabled)
        tc.configure(color="always")
        self.assertTrue(tc.enabled)
        tc.configure(color="never")
        self.assertFalse(tc.enabled)


class TestTermColorBehavior(ColorEnvTestCase):
    def test_sgr_preserves_visible_text_when_enabled(self) -> None:
        tc = TermColor(color="always")
        result = tc.sgr("hello", "red")

        self.assertNotEqual(result, "hello")
        self.assertEqual(tc.strip(result), "hello")
        self.assertEqual(tc.visible_len(result), 5)

    def test_sgr_plain_when_disabled(self) -> None:
        self.assertEqual(TermColor(no_color=True).sgr("hello", "red"), "hello")

    def test_sgr_unknown_name_ignored(self) -> None:
        self.assertEqual(TermColor().sgr("x", "neon"), "x")

    def test_shortcuts_preserve_visible_text(self) -> None:
        tc = TermColor(color="always")
        for rendered, plain in (
            (tc.error("fail"), "fail"),
            (tc.warning("careful"), "careful"),
            (tc.note("info"), "info"),
            (tc.skip("ignored"), "ignored"),
            (tc.heading("Title"), "Title"),
            (tc.command("run"), "run"),
            (tc.status("stable"), "stable"),
        ):
            with self.subTest(plain=plain):
                self.assertNotEqual(rendered, plain)
                self.assertEqual(tc.strip(rendered), plain)

    def test_unknown_status_returns_plain(self) -> None:
        self.assertEqual(TermColor().status("bogus"), "bogus")

    def test_colorize_evidence_preserves_known_status_text(self) -> None:
        tc = TermColor(color="always")
        for value in ("stable (2/2)", "tight (1/3)", "timeout (0/3)", "oom (0/3)", "failed (0/3)", "unsupported"):
            with self.subTest(value=value):
                result = tc.colorize_evidence(value)
                self.assertNotEqual(result, value)
                self.assertEqual(tc.strip(result), value)

    def test_colorize_evidence_unknown_returns_plain(self) -> None:
        self.assertEqual(TermColor().colorize_evidence("bogus status"), "bogus status")

    def test_colorize_delta_preserves_text_for_threshold_cases(self) -> None:
        tc = TermColor(color="always")
        cases = (
            ("+10%", 10.0, True, None),
            ("-10%", -10.0, True, None),
            ("-5s", -5.0, False, None),
            ("+5s", 5.0, False, None),
            ("-0.5G", -0.5, True, 1.0),
            ("-2.0G", -2.0, True, 1.0),
        )
        for text, value, higher_is_better, threshold in cases:
            with self.subTest(text=text):
                result = tc.colorize_delta(text, value, higher_is_better=higher_is_better, mild_threshold=threshold)
                self.assertNotEqual(result, text)
                self.assertEqual(tc.strip(result), text)

    def test_colorize_config_label_preserves_visible_tokens(self) -> None:
        tc = TermColor(color="always")
        label = "model ctx4096 q8_0 moe16 spec draft-mtp"
        result = tc.colorize_config_label(label)

        self.assertNotEqual(result, label)
        self.assertEqual(tc.strip(result), label)

    def test_colorize_config_label_plain_tokens_unchanged(self) -> None:
        tc = TermColor(color="always")
        label = "model b1024/u1024"
        self.assertEqual(tc.colorize_config_label(label), label)

    def test_colorize_config_label_disabled(self) -> None:
        label = "model ctx4096 q8_0 moe16 spec draft-mtp"
        self.assertEqual(TermColor(no_color=True).colorize_config_label(label), label)

    def test_table_value_preserves_visible_text_for_styled_columns(self) -> None:
        tc = TermColor(color="always")
        for column, value in (
            ("status", "success"),
            ("parsed.benchmark_valid", "True"),
            ("parsed.benchmark_valid", "False"),
            ("parsed.benchmark_invalid_reason", "bad input"),
            ("speedup_pct", "5.0"),
            ("speedup_pct", "-3.0"),
            ("gain_tok_s", "10.0"),
            ("kv", "q8_0"),
            ("moe", "16"),
            ("spec", "draft-mtp"),
            ("ctx", "4096"),
        ):
            with self.subTest(column=column, value=value):
                result = tc.table_value(column, value)
                self.assertNotEqual(result, value)
                self.assertEqual(tc.strip(result), value)

    def test_table_value_passthrough_cases(self) -> None:
        tc = TermColor(color="always")
        self.assertEqual(tc.table_value("some_column", "hello"), "hello")
        self.assertEqual(tc.table_value("parsed.benchmark_invalid_reason", "-"), "-")
        self.assertEqual(tc.table_value("kv", "-"), "-")
        self.assertEqual(tc.table_value("moe", "-"), "-")
        self.assertEqual(tc.table_value("spec", "none"), "none")
        self.assertEqual(tc.table_value("spec", "-"), "-")
        self.assertEqual(tc.table_value("ctx", "-"), "-")

    def test_gradient_preserves_visible_text_when_full_color_applies(self) -> None:
        tc = TermColor(color="always")
        result = tc.gradient("50", 50.0, 0.0, 100.0)

        self.assertNotEqual(result, "50")
        self.assertEqual(tc.strip(result), "50")

    def test_gradient_plain_when_disabled_or_no_range(self) -> None:
        self.assertEqual(TermColor(no_color=True).gradient("50", 50.0, 0.0, 100.0), "50")
        self.assertEqual(TermColor(color="always").gradient("50", 50.0, 50.0, 50.0), "50")


class TestModuleFunctions(ColorEnvTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._saved_enabled = color_enabled()
        configure_color(color="always")

    def tearDown(self) -> None:
        configure_color(color="always" if self._saved_enabled else "never")
        super().tearDown()

    def test_configure_color_no_color_disables(self) -> None:
        configure_color(no_color=True)
        self.assertFalse(color_enabled())

    def test_configure_color_always_enables(self) -> None:
        configure_color(color="always")
        self.assertTrue(color_enabled())
        self.assertEqual(strip_style(style("x", "red")), "x")

    def test_module_shortcuts_preserve_visible_text(self) -> None:
        for rendered, plain in (
            (error("fail"), "fail"),
            (warning("careful"), "careful"),
            (heading("Title"), "Title"),
            (command("run"), "run"),
            (note("info"), "info"),
            (skip("skip"), "skip"),
            (status("stable"), "stable"),
            (style_table_value("speedup_pct", "5.0"), "5.0"),
            (colorize_evidence("stable (2/2)"), "stable (2/2)"),
            (colorize_delta("+10%", 10.0, higher_is_better=True), "+10%"),
            (colorize_config_label("model ctx4096 q8_0 moe0 spec -"), "model ctx4096 q8_0 moe0 spec -"),
            (gradient_value("100", 100.0, 0.0, 100.0, higher_is_better=True), "100"),
        ):
            with self.subTest(plain=plain):
                self.assertEqual(strip_style(rendered), plain)

    def test_style_disabled_returns_plain(self) -> None:
        configure_color(no_color=True)
        self.assertEqual(style("x", "red"), "x")

    def test_strip_style_and_visible_len(self) -> None:
        rendered = TermColor(color="always").sgr("hi", "red")

        self.assertEqual(strip_style(rendered), "hi")
        self.assertEqual(visible_len(rendered), 2)

    def test_gradient_value_module_level_none_plain(self) -> None:
        configure_color(no_color=True)
        self.assertEqual(gradient_value("50", 50.0, 0.0, 100.0), "50")


def _env(key: str, value: str | None) -> _EnvOverride:
    return _EnvOverride(key, value)


class _EnvOverride:
    def __init__(self, key: str, value: str | None) -> None:
        self._key = key
        self._value = value
        self._saved: str | None = None

    def __enter__(self) -> _EnvOverride:
        self._saved = os.environ.get(self._key)
        if self._value is None:
            os.environ.pop(self._key, None)
        else:
            os.environ[self._key] = self._value
        return self

    def __exit__(self, *args: Any) -> None:
        if self._saved is None:
            os.environ.pop(self._key, None)
        else:
            os.environ[self._key] = self._saved


class TestFormatNumber(unittest.TestCase):
    def test_large_int_k(self) -> None:
        self.assertEqual(format_number(65536), "65.54k")

    def test_4096_k(self) -> None:
        self.assertEqual(format_number(4096), "4.10k")

    def test_1024_k(self) -> None:
        self.assertEqual(format_number(1024), "1.02k")

    def test_exactly_1000_no_k(self) -> None:
        self.assertEqual(format_number(1000), "1000")

    def test_small_int_no_k(self) -> None:
        self.assertEqual(format_number(512), "512")

    def test_zero_no_k(self) -> None:
        self.assertEqual(format_number(0), "0")

    def test_large_float_k(self) -> None:
        self.assertEqual(format_number(24576.0), "24.58k")

    def test_small_float_no_k(self) -> None:
        self.assertEqual(format_number(512.34), "512.34")

    def test_none_returns_dash(self) -> None:
        self.assertEqual(format_number(None), "-")

    def test_string_returns_dash(self) -> None:
        self.assertEqual(format_number("not_a_number"), "-")


if __name__ == "__main__":
    unittest.main()
