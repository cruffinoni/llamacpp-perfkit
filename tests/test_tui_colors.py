from __future__ import annotations

import unittest

from llamacpp_perfkit.tui.colors import phase_color, status_color


class TestStatusColors(unittest.TestCase):
    def test_success(self) -> None:
        self.assertEqual(status_color("success"), "green")

    def test_running(self) -> None:
        self.assertEqual(status_color("running"), "cyan")

    def test_pending(self) -> None:
        self.assertEqual(status_color("pending"), "dim")

    def test_timeout(self) -> None:
        self.assertEqual(status_color("timeout"), "yellow")

    def test_oom(self) -> None:
        self.assertEqual(status_color("oom"), "red")

    def test_failed(self) -> None:
        self.assertEqual(status_color("failed"), "red")

    def test_case_insensitive(self) -> None:
        self.assertEqual(status_color("SUCCESS"), "green")

    def test_unknown_defaults_to_dim(self) -> None:
        self.assertEqual(status_color("bogus"), "dim")


class TestPhaseColors(unittest.TestCase):
    def test_prefill(self) -> None:
        self.assertEqual(phase_color("prefill"), "cyan")

    def test_generating(self) -> None:
        self.assertEqual(phase_color("generating"), "cyan")

    def test_done(self) -> None:
        self.assertEqual(phase_color("done"), "dim")

    def test_starting(self) -> None:
        self.assertEqual(phase_color("starting"), "cyan")

    def test_pending(self) -> None:
        self.assertEqual(phase_color("pending"), "dim")

    def test_timeout(self) -> None:
        self.assertEqual(phase_color("timeout"), "yellow")

    def test_oom(self) -> None:
        self.assertEqual(phase_color("oom"), "red")

    def test_failed(self) -> None:
        self.assertEqual(phase_color("failed"), "red")

    def test_dash(self) -> None:
        self.assertEqual(phase_color("-"), "dim")

    def test_unknown_defaults_to_dim(self) -> None:
        self.assertEqual(phase_color("bogus"), "dim")


if __name__ == "__main__":
    unittest.main()
