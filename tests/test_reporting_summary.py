from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from llamacpp_perfkit.output import configure_color
from llamacpp_perfkit.reporting import load_rows, print_compare, print_summary
from llamacpp_perfkit.run_storage import append_llamacpp_metric, append_system_metric, write_run_summary


def args(root: Path, **overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "runs": str(root),
        "results": None,
        "model": None,
        "quant": None,
        "context_size": None,
        "kv_type": None,
        "prompt_profile": None,
        "mtp_mode": None,
        "n_cpu_moe": None,
        "status": None,
        "sort": "balanced",
        "limit": 20,
        "details": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def write_run(root: Path, run_id: str, *, ctx: int, profile: str, gen: float, prompt: float, vram: float, status: str = "success") -> None:
    write_run_summary(
        root,
        {
            "run_id": run_id,
            "batch_id": "batch",
            "created_at": f"2026-01-01T00:00:0{len(run_id)}Z",
            "model": "model:A",
            "prompt_profile": profile,
            "server_config": {"context_size": ctx, "kv_type": "q8_0", "n_cpu_moe": 0, "mtp_enabled": False},
            "status": status,
            "config_hash": run_id,
            "config": {
                "model_hf": "model:A",
                "prompt_profile": profile,
                "context_size": ctx,
                "kv_type": "q8_0",
                "n_cpu_moe": 0,
                "mtp_enabled": False,
            },
            "duration_seconds": 10.0,
            "response": {"content": "ok"},
        },
    )
    append_llamacpp_metric(root / run_id, {"time": "t1", "generation_tok_s": gen, "prompt_eval_tok_s": prompt, "total_time_seconds": 10.0})
    append_system_metric(root / run_id, {"time": "t1", "vram_free_mib": vram, "vram_used_mib": 1024.0})


class SummaryRankingTest(unittest.TestCase):
    def setUp(self) -> None:
        configure_color(no_color=True)

    def test_summary_groups_and_uses_metric_notation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for ctx, gen in ((2048, 100.0), (4096, 90.0)):
                write_run(root, f"code-{ctx}", ctx=ctx, profile="code", gen=gen, prompt=50.0, vram=4096.0)
                write_run(root, f"qa-{ctx}", ctx=ctx, profile="qa", gen=gen - 10.0, prompt=45.0, vram=4096.0)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                print_summary(load_rows(args(root)), args(root))
            output = out.getvalue()

            self.assertIn("Groups: 2 from 4 runs", output)
            self.assertIn("gen tok/s", output)
            self.assertIn("g", output)
            self.assertIn("p10:", output)
            self.assertIn("stable (2/2)", output)

    def test_summary_allows_different_prompt_profile_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_run(root, "code-2048", ctx=2048, profile="code", gen=100.0, prompt=50.0, vram=4096.0)
            write_run(root, "code-4096", ctx=4096, profile="code", gen=90.0, prompt=45.0, vram=4096.0)
            write_run(root, "qa-4096", ctx=4096, profile="qa", gen=80.0, prompt=40.0, vram=4096.0)

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                print_summary(load_rows(args(root)), args(root))
            output = out.getvalue()
            self.assertIn("Groups:", output)
            self.assertIn("gen tok/s", output)

    def test_compare_refuses_different_prompt_profile_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_run(root, "base", ctx=2048, profile="code", gen=100.0, prompt=50.0, vram=4096.0)
            write_run(root, "extra", ctx=4096, profile="qa", gen=90.0, prompt=45.0, vram=4096.0)

            with self.assertRaisesRegex(ValueError, "cannot compare configs: prompt profile sets differ"):
                print_compare(SimpleNamespace(baseline=str(root / "base"), candidates=[str(root / "extra")], limit=20))

    def test_compare_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_run(root, "base", ctx=2048, profile="code", gen=100.0, prompt=50.0, vram=4096.0)
            write_run(root, "candidate", ctx=4096, profile="code", gen=110.0, prompt=55.0, vram=5120.0)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                print_compare(SimpleNamespace(baseline=str(root / "base"), candidates=[str(root / "candidate")], limit=20))
            output = out.getvalue()

            self.assertIn("base", output)
            self.assertIn("+10.0%", output)
            self.assertIn("+1.0G", output)
            self.assertIn("g110", output)


if __name__ == "__main__":
    unittest.main()
