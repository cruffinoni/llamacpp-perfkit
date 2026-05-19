from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cmd(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )


class ReportingCliTest(unittest.TestCase):
    def test_main_help_only_lists_remaining_commands(self) -> None:
        result = run_cmd("-m", "llamacpp_perfkit.cli", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout
        self.assertIn("detect", output)
        self.assertIn("bench", output)
        self.assertIn("report", output)

    def test_report_help_only_lists_remaining_commands(self) -> None:
        result = run_cmd("-m", "llamacpp_perfkit.cli", "report", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout
        self.assertIn("summary", output)
        self.assertIn("by-profile", output)
        self.assertIn("compare", output)
        self.assertNotIn("export-csv", output)

    def test_legacy_reporting_help_only_lists_remaining_commands(self) -> None:
        result = run_cmd("-m", "llamacpp_perfkit.reporting", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout
        self.assertIn("summary", output)
        self.assertIn("by-profile", output)
        self.assertIn("compare", output)
        self.assertNotIn("export-csv", output)
        self.assertNotIn("compare-kv", output)
        self.assertNotIn("compare-moe", output)
        self.assertNotIn("compare-context", output)

    def test_removed_compare_commands_fail(self) -> None:
        for args, expected in (
            (("-m", "llamacpp_perfkit.cli", "report", "export-csv", "--help"), "No such command 'export-csv'"),
            (("-m", "llamacpp_perfkit.reporting", "compare-kv", "--help"), "invalid choice"),
            (("-m", "llamacpp_perfkit.reporting", "compare-moe", "--help"), "invalid choice"),
            (("-m", "llamacpp_perfkit.reporting", "compare-context", "--help"), "invalid choice"),
        ):
            with self.subTest(args=args):
                result = run_cmd(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr + result.stdout)

    def test_no_color_disables_ansi(self) -> None:
        result = run_cmd("-m", "llamacpp_perfkit.reporting", "--no-color", "summary", "--runs", "/tmp/does-not-exist")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("\x1b[", result.stdout)

    def test_no_color_env_disables_ansi(self) -> None:
        result = run_cmd(
            "-m", "llamacpp_perfkit.reporting", "--color", "always", "summary", "--runs", "/tmp/does-not-exist", env={"NO_COLOR": "1"}
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("\x1b[", result.stdout)

    def test_color_always_produces_ansi(self) -> None:
        result = run_cmd("-m", "llamacpp_perfkit.reporting", "--color", "always", "summary", "--runs", "/tmp/does-not-exist")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("\x1b[", result.stdout)


if __name__ == "__main__":
    unittest.main()
