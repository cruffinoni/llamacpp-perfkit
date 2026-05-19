from __future__ import annotations

import os
import unittest
from typing import Any

from llamacpp_perfkit.output import (
    ANSI_RE,
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


class TestTermColor(unittest.TestCase):
    def test_auto_enabled_by_default(self) -> None:
        tc = TermColor()
        self.assertTrue(tc.enabled)

    def test_never_disabled(self) -> None:
        tc = TermColor(color="never")
        self.assertFalse(tc.enabled)

    def test_always_enabled(self) -> None:
        tc = TermColor(color="always")
        self.assertTrue(tc.enabled)

    def test_auto_disabled_when_term_dumb(self) -> None:
        with _env("TERM", "dumb"):
            tc = TermColor()
            self.assertFalse(tc.enabled)

    def test_auto_disabled_when_no_color_env(self) -> None:
        with _env("NO_COLOR", "1"):
            tc = TermColor()
            self.assertFalse(tc.enabled)

    def test_no_color_overrides_always(self) -> None:
        tc = TermColor(color="always", no_color=True)
        self.assertFalse(tc.enabled)
        self.assertEqual(tc.level, ColorLevel.NONE)

    def test_always_level_is_full(self) -> None:
        tc = TermColor(color="always")
        self.assertEqual(tc.level, ColorLevel.FULL)

    def test_never_level_is_none(self) -> None:
        tc = TermColor(color="never")
        self.assertEqual(tc.level, ColorLevel.NONE)

    def test_auto_with_true_color_gives_full(self) -> None:
        with _env("COLORTERM", "truecolor"):
            tc = TermColor()
            self.assertEqual(tc.level, ColorLevel.FULL)

    def test_auto_without_true_color_gives_simple(self) -> None:
        with _env("COLORTERM", None), _env("TERM", "xterm"):
            tc = TermColor()
            self.assertEqual(tc.level, ColorLevel.SIMPLE)

    def test_auto_with_term_dumb_gives_none(self) -> None:
        with _env("TERM", "dumb"):
            tc = TermColor()
            self.assertEqual(tc.level, ColorLevel.NONE)

    def test_sgr_works_at_simple_level(self) -> None:
        tc = TermColor()
        tc._level = ColorLevel.SIMPLE
        result = tc.sgr("hello", "red")
        self.assertIn("\x1b[31m", result)

    def test_sgr_plain_at_none_level(self) -> None:
        tc = TermColor(color="never")
        result = tc.sgr("hello", "red")
        self.assertEqual(result, "hello")

    def test_configure_toggle(self) -> None:
        tc = TermColor()
        self.assertTrue(tc.enabled)
        tc.configure(no_color=True)
        self.assertFalse(tc.enabled)
        tc.configure(color="always")
        self.assertTrue(tc.enabled)
        tc.configure(color="never")
        self.assertFalse(tc.enabled)

    def test_sgr_wraps_when_enabled(self) -> None:
        tc = TermColor()
        result = tc.sgr("hello", "red")
        self.assertIn("\x1b[31m", result)
        self.assertIn("hello", result)
        self.assertIn("\x1b[0m", result)

    def test_sgr_plain_when_disabled(self) -> None:
        tc = TermColor(no_color=True)
        self.assertEqual(tc.sgr("hello", "red"), "hello")

    def test_sgr_multiple_codes(self) -> None:
        tc = TermColor()
        result = tc.sgr("x", "bold", "red")
        self.assertIn("\x1b[1;31m", result)

    def test_sgr_unknown_name_ignored(self) -> None:
        tc = TermColor()
        self.assertEqual(tc.sgr("x", "neon"), "x")

    def test_sgr_no_names_returns_plain(self) -> None:
        tc = TermColor()
        self.assertEqual(tc.sgr("plain"), "plain")

    def test_error_shortcut(self) -> None:
        tc = TermColor()
        result = tc.error("fail")
        self.assertIn("31", result)
        self.assertIn("1", result)

    def test_warning_shortcut(self) -> None:
        tc = TermColor()
        result = tc.warning("careful")
        self.assertIn("33", result)
        self.assertIn("1", result)

    def test_note_shortcut(self) -> None:
        tc = TermColor()
        result = tc.note("info")
        self.assertIn("36", result)

    def test_skip_shortcut(self) -> None:
        tc = TermColor()
        result = tc.skip("ignored")
        self.assertIn("35", result)

    def test_heading_shortcut(self) -> None:
        tc = TermColor()
        result = tc.heading("Title")
        self.assertIn("36", result)
        self.assertIn("1", result)

    def test_command_shortcut(self) -> None:
        tc = TermColor()
        result = tc.command("run")
        self.assertIn("32", result)

    def test_status_stable(self) -> None:
        tc = TermColor()
        result = tc.status("stable")
        self.assertIn("32", result)

    def test_status_oom(self) -> None:
        tc = TermColor()
        result = tc.status("oom")
        self.assertIn("31", result)

    def test_status_timeout(self) -> None:
        tc = TermColor()
        result = tc.status("timeout")
        self.assertIn("33", result)

    def test_status_unsupported(self) -> None:
        tc = TermColor()
        result = tc.status("unsupported")
        self.assertIn("35", result)

    def test_status_unknown_returns_plain(self) -> None:
        tc = TermColor()
        self.assertEqual(tc.status("bogus"), "bogus")

    def test_colorize_evidence_stable(self) -> None:
        tc = TermColor()
        result = tc.colorize_evidence("stable (2/2)")
        self.assertIn("32", result)
        self.assertIn("stable (2/2)", result)

    def test_colorize_evidence_tight(self) -> None:
        tc = TermColor()
        result = tc.colorize_evidence("tight (1/3)")
        self.assertIn("33", result)

    def test_colorize_evidence_timeout(self) -> None:
        tc = TermColor()
        result = tc.colorize_evidence("timeout (0/3)")
        self.assertIn("33", result)

    def test_colorize_evidence_oom(self) -> None:
        tc = TermColor()
        result = tc.colorize_evidence("oom (0/3)")
        self.assertIn("31", result)

    def test_colorize_evidence_failed(self) -> None:
        tc = TermColor()
        result = tc.colorize_evidence("failed (0/3)")
        self.assertIn("31", result)

    def test_colorize_evidence_unsupported(self) -> None:
        tc = TermColor()
        result = tc.colorize_evidence("unsupported")
        self.assertIn("35", result)

    def test_colorize_evidence_unknown_returns_plain(self) -> None:
        tc = TermColor()
        self.assertEqual(tc.colorize_evidence("bogus status"), "bogus status")

    def test_colorize_delta_positive_green_higher_is_better(self) -> None:
        tc = TermColor()
        result = tc.colorize_delta("+10%", 10.0, higher_is_better=True)
        self.assertIn("32", result)

    def test_colorize_delta_negative_red_higher_is_better(self) -> None:
        tc = TermColor()
        result = tc.colorize_delta("-10%", -10.0, higher_is_better=True)
        self.assertIn("31", result)

    def test_colorize_delta_negative_green_lower_is_better(self) -> None:
        tc = TermColor()
        result = tc.colorize_delta("-5s", -5.0, higher_is_better=False)
        self.assertIn("32", result)

    def test_colorize_delta_positive_red_lower_is_better(self) -> None:
        tc = TermColor()
        result = tc.colorize_delta("+5s", 5.0, higher_is_better=False)
        self.assertIn("31", result)

    def test_colorize_delta_zero_green(self) -> None:
        tc = TermColor()
        result = tc.colorize_delta("0%", 0.0, higher_is_better=True)
        self.assertIn("32", result)

    def test_colorize_delta_mild_threshold_yellow(self) -> None:
        tc = TermColor()
        result = tc.colorize_delta("-0.5G", -0.5, higher_is_better=True, mild_threshold=1.0)
        self.assertIn("33", result)

    def test_colorize_delta_severe_threshold_red(self) -> None:
        tc = TermColor()
        result = tc.colorize_delta("-2.0G", -2.0, higher_is_better=True, mild_threshold=1.0)
        self.assertIn("31", result)

    def test_colorize_delta_threshold_exact_boundary_red(self) -> None:
        tc = TermColor()
        result = tc.colorize_delta("-1.0G", -1.0, higher_is_better=True, mild_threshold=1.0)
        self.assertIn("31", result)

    def test_colorize_config_label_kv_token(self) -> None:
        tc = TermColor()
        result = tc.colorize_config_label("model ctx4096 q8_0 moe0 base")
        self.assertIn("\x1b[36mq8_0\x1b[0m", result)

    def test_colorize_config_label_moe_token(self) -> None:
        tc = TermColor()
        result = tc.colorize_config_label("model ctx4096 q8_0 moe16 base")
        self.assertIn("\x1b[34mmoe16\x1b[0m", result)

    def test_colorize_config_label_base_green(self) -> None:
        tc = TermColor()
        result = tc.colorize_config_label("model ctx4096 q8_0 moe0 base")
        self.assertIn("\x1b[32mbase\x1b[0m", result)

    def test_colorize_config_label_mtp_magenta(self) -> None:
        tc = TermColor()
        result = tc.colorize_config_label("model ctx4096 q8_0 moe0 mtp")
        self.assertIn("\x1b[35mmtp\x1b[0m", result)

    def test_colorize_config_label_spec_token_yellow(self) -> None:
        tc = TermColor()
        result = tc.colorize_config_label("model ctx4096 q8_0 moe0 mtp draft-mtp")
        self.assertIn("\x1b[33mdraft-mtp\x1b[0m", result)

    def test_colorize_config_label_ctx_token_blue(self) -> None:
        tc = TermColor()
        result = tc.colorize_config_label("model ctx4096 q8_0 moe0 base")
        self.assertIn("\x1b[34mctx4096\x1b[0m", result)

    def test_colorize_config_label_plain_tokens_unchanged(self) -> None:
        tc = TermColor()
        result = tc.colorize_config_label("model b1024/u1024")
        self.assertIn("model", result)
        self.assertIn("b1024/u1024", result)
        stripped = ANSI_RE.sub("", result)
        self.assertEqual(stripped, "model b1024/u1024")

    def test_colorize_config_label_disabled(self) -> None:
        tc = TermColor(no_color=True)
        result = tc.colorize_config_label("model ctx4096 q8_0 moe16 base mtp draft-mtp")
        self.assertNotIn("\x1b[", result)

    def test_table_value_status_column(self) -> None:
        tc = TermColor()
        result = tc.table_value("status", "success")
        self.assertIn("32", result)

    def test_table_value_benchmark_valid_true(self) -> None:
        tc = TermColor()
        result = tc.table_value("parsed.benchmark_valid", "True")
        self.assertIn("32", result)

    def test_table_value_benchmark_valid_false(self) -> None:
        tc = TermColor()
        result = tc.table_value("parsed.benchmark_valid", "False")
        self.assertIn("33", result)

    def test_table_value_invalid_reason(self) -> None:
        tc = TermColor()
        result = tc.table_value("parsed.benchmark_invalid_reason", "bad input")
        self.assertIn("33", result)
        self.assertIn("1", result)

    def test_table_value_invalid_reason_dash(self) -> None:
        tc = TermColor()
        result = tc.table_value("parsed.benchmark_invalid_reason", "-")
        self.assertEqual(result, "-")

    def test_table_value_speedup_positive(self) -> None:
        tc = TermColor()
        result = tc.table_value("speedup_pct", "5.0")
        self.assertIn("32", result)

    def test_table_value_speedup_negative(self) -> None:
        tc = TermColor()
        result = tc.table_value("speedup_pct", "-3.0")
        self.assertIn("31", result)

    def test_table_value_gain_tok_s_positive(self) -> None:
        tc = TermColor()
        result = tc.table_value("gain_tok_s", "10.0")
        self.assertIn("32", result)

    def test_table_value_unknown_column_passthrough(self) -> None:
        tc = TermColor()
        self.assertEqual(tc.table_value("some_column", "hello"), "hello")

    def test_table_value_kv_cyan(self) -> None:
        tc = TermColor()
        result = tc.table_value("kv", "q8_0")
        self.assertIn("36", result)

    def test_table_value_kv_dash_passthrough(self) -> None:
        tc = TermColor()
        result = tc.table_value("kv", "-")
        self.assertEqual(result, "-")

    def test_table_value_moe_numeric_blue(self) -> None:
        tc = TermColor()
        result = tc.table_value("moe", "16")
        self.assertIn("34", result)

    def test_table_value_moe_dash_passthrough(self) -> None:
        tc = TermColor()
        result = tc.table_value("moe", "-")
        self.assertEqual(result, "-")

    def test_table_value_spec_base_green(self) -> None:
        tc = TermColor()
        result = tc.table_value("spec", "base")
        self.assertIn("32", result)

    def test_table_value_spec_mtp_magenta(self) -> None:
        tc = TermColor()
        result = tc.table_value("spec", "mtp")
        self.assertIn("35", result)

    def test_table_value_spec_draft_yellow(self) -> None:
        tc = TermColor()
        result = tc.table_value("spec", "draft-mtp")
        self.assertIn("33", result)

    def test_table_value_ctx_blue(self) -> None:
        tc = TermColor()
        result = tc.table_value("ctx", "4096")
        self.assertIn("34", result)

    def test_table_value_ctx_dash_passthrough(self) -> None:
        tc = TermColor()
        result = tc.table_value("ctx", "-")
        self.assertEqual(result, "-")

    def test_gradient_best_green(self) -> None:
        tc = TermColor(color="always")
        result = tc.gradient("100", 100.0, 0.0, 100.0, higher_is_better=True)
        self.assertIn("0;255;0", result)

    def test_gradient_worst_red(self) -> None:
        tc = TermColor(color="always")
        result = tc.gradient("0", 0.0, 0.0, 100.0, higher_is_better=True)
        self.assertIn("255;0;0", result)

    def test_gradient_mid_yellowish(self) -> None:
        tc = TermColor(color="always")
        result = tc.gradient("50", 50.0, 0.0, 100.0, higher_is_better=True)
        self.assertIn("255;255;0", result)

    def test_gradient_lower_is_better(self) -> None:
        tc = TermColor(color="always")
        result = tc.gradient("0", 0.0, 0.0, 100.0, higher_is_better=False)
        self.assertIn("0;255;0", result)

    def test_gradient_disabled_returns_plain(self) -> None:
        tc = TermColor(no_color=True)
        result = tc.gradient("50", 50.0, 0.0, 100.0)
        self.assertEqual(result, "50")

    def test_gradient_min_equals_max_returns_plain(self) -> None:
        tc = TermColor(color="always")
        result = tc.gradient("50", 50.0, 50.0, 50.0)
        self.assertEqual(result, "50")

    def test_gradient_clamped_below_zero(self) -> None:
        tc = TermColor(color="always")
        result = tc.gradient("-10", -10.0, 0.0, 100.0, higher_is_better=True)
        self.assertIn("255;0;0", result)

    def test_gradient_skipped_when_simple_level(self) -> None:
        tc = TermColor(no_color=False)
        tc.configure(color="auto")
        tc._level = ColorLevel.SIMPLE
        result = tc.gradient("50", 50.0, 0.0, 100.0)
        self.assertEqual(result, "50")

    def test_gradient_skipped_when_none_level(self) -> None:
        tc = TermColor(no_color=True)
        result = tc.gradient("50", 50.0, 0.0, 100.0)
        self.assertEqual(result, "50")

    def test_strip_removes_ansi(self) -> None:
        tc = TermColor()
        self.assertEqual(tc.strip("\x1b[31mhi\x1b[0m"), "hi")

    def test_visible_len_ignores_ansi(self) -> None:
        tc = TermColor()
        self.assertEqual(tc.visible_len("\x1b[31mhi\x1b[0m"), 2)

    def test_enabled_property(self) -> None:
        tc = TermColor()
        self.assertTrue(tc.enabled)
        tc2 = TermColor(color="never")
        self.assertFalse(tc2.enabled)


class TestModuleFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_enabled = color_enabled()
        configure_color(color="always")

    def tearDown(self) -> None:
        configure_color(color="always" if self._saved_enabled else "never")

    def test_configure_color_no_color_disables(self) -> None:
        configure_color(no_color=True)
        self.assertFalse(color_enabled())

    def test_configure_color_always_enables(self) -> None:
        configure_color(color="always")
        self.assertTrue(color_enabled())
        result = style("x", "red")
        self.assertIn("\x1b[31m", result)

    def test_style_disabled_returns_plain(self) -> None:
        configure_color(no_color=True)
        self.assertEqual(style("x", "red"), "x")

    def test_module_error_function(self) -> None:
        result = error("fail")
        self.assertIn("31", result)

    def test_module_warning_function(self) -> None:
        result = warning("careful")
        self.assertIn("33", result)

    def test_module_heading_function(self) -> None:
        result = heading("Title")
        self.assertIn("36", result)

    def test_module_command_function(self) -> None:
        result = command("run")
        self.assertIn("32", result)

    def test_module_note_function(self) -> None:
        result = note("info")
        self.assertIn("36", result)

    def test_module_skip_function(self) -> None:
        result = skip("skip")
        self.assertIn("35", result)

    def test_module_status_function(self) -> None:
        result = status("stable")
        self.assertIn("32", result)

    def test_module_style_table_value(self) -> None:
        result = style_table_value("speedup_pct", "5.0")
        self.assertIn("32", result)

    def test_module_strip_style(self) -> None:
        self.assertEqual(strip_style("\x1b[31mhi\x1b[0m"), "hi")

    def test_module_visible_len(self) -> None:
        self.assertEqual(visible_len("\x1b[31mhi\x1b[0m"), 2)

    def test_colorize_evidence_module_level(self) -> None:
        result = colorize_evidence("stable (2/2)")
        self.assertIn("32", result)
        self.assertIn("stable (2/2)", result)

    def test_colorize_delta_module_level(self) -> None:
        result = colorize_delta("+10%", 10.0, higher_is_better=True)
        self.assertIn("32", result)

    def test_colorize_config_label_module_level(self) -> None:
        result = colorize_config_label("model ctx4096 q8_0 moe0 base")
        self.assertIn("\x1b[36mq8_0\x1b[0m", result)

    def test_gradient_value_module_level_full(self) -> None:
        from llamacpp_perfkit.output import gradient_value

        configure_color(color="always")
        result = gradient_value("100", 100.0, 0.0, 100.0, higher_is_better=True)
        self.assertIn("0;255;0", result)

    def test_gradient_value_module_level_none_plain(self) -> None:
        from llamacpp_perfkit.output import gradient_value

        configure_color(no_color=True)
        result = gradient_value("50", 50.0, 0.0, 100.0)
        self.assertEqual(result, "50")


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
