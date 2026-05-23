from __future__ import annotations

import unittest

from llamacpp_perfkit.tui.formatters import (
    format_context_size,
    format_duration_header,
    format_duration_row,
    format_gen_tok_s,
    format_gib_from_mib,
    format_progress_bar,
    format_prompt_tok_s,
)


class TestFormatContextSize(unittest.TestCase):
    def test_below_1000_returns_raw(self) -> None:
        self.assertEqual(format_context_size(512), "512")

    def test_exactly_1000_rounds_to_1k(self) -> None:
        self.assertEqual(format_context_size(1000), "1k")

    def test_4096_rounds_to_4k(self) -> None:
        self.assertEqual(format_context_size(4096), "4k")

    def test_8192_rounds_to_8k(self) -> None:
        self.assertEqual(format_context_size(8192), "8k")

    def test_16384_rounds_to_16k(self) -> None:
        self.assertEqual(format_context_size(16384), "16k")

    def test_32768_rounds_to_33k(self) -> None:
        self.assertEqual(format_context_size(32768), "33k")

    def test_65536_rounds_to_66k(self) -> None:
        self.assertEqual(format_context_size(65536), "66k")

    def test_131072_rounds_to_131k(self) -> None:
        self.assertEqual(format_context_size(131072), "131k")

    def test_zero_returns_string(self) -> None:
        self.assertEqual(format_context_size(0), "0")

    def test_999_returns_raw(self) -> None:
        self.assertEqual(format_context_size(999), "999")


class TestFormatGibFromMib(unittest.TestCase):
    def test_5040_mib(self) -> None:
        self.assertEqual(format_gib_from_mib(5040.0), "4.92 GiB")

    def test_5190_mib(self) -> None:
        self.assertEqual(format_gib_from_mib(5190.0), "5.07 GiB")

    def test_14115_mib(self) -> None:
        self.assertEqual(format_gib_from_mib(14115.0), "13.78 GiB")

    def test_zero_mib(self) -> None:
        self.assertEqual(format_gib_from_mib(0.0), "0.00 GiB")

    def test_none_returns_dash(self) -> None:
        self.assertEqual(format_gib_from_mib(None), "-")


class TestFormatDurationRow(unittest.TestCase):
    def test_two_point_seven(self) -> None:
        self.assertEqual(format_duration_row(2.7), "2.70s")

    def test_twelve_point_four_one(self) -> None:
        self.assertEqual(format_duration_row(12.41), "12.41s")

    def test_zero(self) -> None:
        self.assertEqual(format_duration_row(0.0), "0.00s")

    def test_none_returns_dash(self) -> None:
        self.assertEqual(format_duration_row(None), "-")

    def test_large_duration(self) -> None:
        self.assertEqual(format_duration_row(71.2), "71.20s")


class TestFormatDurationHeader(unittest.TestCase):
    def test_522_seconds_gives_08_42(self) -> None:
        self.assertEqual(format_duration_header(522), "08:42")

    def test_2350_seconds_gives_39_10(self) -> None:
        self.assertEqual(format_duration_header(2350), "39:10")

    def test_3822_seconds_gives_1_03_42(self) -> None:
        self.assertEqual(format_duration_header(3822), "1:03:42")

    def test_zero(self) -> None:
        self.assertEqual(format_duration_header(0), "00:00")

    def test_under_one_minute(self) -> None:
        self.assertEqual(format_duration_header(42), "00:42")


class TestFormatGenTokS(unittest.TestCase):
    def test_78_point_0(self) -> None:
        self.assertEqual(format_gen_tok_s(78.0), "78.0")

    def test_63_point_2(self) -> None:
        self.assertEqual(format_gen_tok_s(63.2), "63.2")

    def test_zero(self) -> None:
        self.assertEqual(format_gen_tok_s(0.0), "0.0")

    def test_none_returns_dash(self) -> None:
        self.assertEqual(format_gen_tok_s(None), "-")


class TestFormatPromptTokS(unittest.TestCase):
    def test_812_returns_integer(self) -> None:
        self.assertEqual(format_prompt_tok_s(812.0), "812")

    def test_100_returns_integer(self) -> None:
        self.assertEqual(format_prompt_tok_s(100.0), "100")

    def test_87_point_4_returns_one_decimal(self) -> None:
        self.assertEqual(format_prompt_tok_s(87.4), "87.4")

    def test_99_point_1_returns_one_decimal(self) -> None:
        self.assertEqual(format_prompt_tok_s(99.1), "99.1")

    def test_zero(self) -> None:
        self.assertEqual(format_prompt_tok_s(0.0), "0")

    def test_none_returns_dash(self) -> None:
        self.assertEqual(format_prompt_tok_s(None), "-")


class TestFormatProgressBar(unittest.TestCase):
    def test_zero_done(self) -> None:
        result = format_progress_bar(0, 10, width=10)
        self.assertEqual(result, "[\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591]")

    def test_half_done(self) -> None:
        result = format_progress_bar(5, 10, width=10)
        self.assertEqual(result, "[\u2588\u2588\u2588\u2588\u2588\u2591\u2591\u2591\u2591\u2591]")

    def test_all_done(self) -> None:
        result = format_progress_bar(10, 10, width=10)
        self.assertEqual(result, "[\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588]")

    def test_total_zero(self) -> None:
        result = format_progress_bar(0, 0, width=10)
        self.assertEqual(result, "[\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591]")

    def test_done_exceeds_total_clamps(self) -> None:
        result = format_progress_bar(15, 10, width=10)
        self.assertEqual(result, "[\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588]")

    def test_custom_width(self) -> None:
        result = format_progress_bar(1, 4, width=4)
        self.assertEqual(result, "[\u2588\u2591\u2591\u2591]")

    def test_rounding_up_small_fraction(self) -> None:
        result = format_progress_bar(1, 7, width=7)
        self.assertEqual(result, "[\u2588\u2591\u2591\u2591\u2591\u2591\u2591]")


if __name__ == "__main__":
    unittest.main()
