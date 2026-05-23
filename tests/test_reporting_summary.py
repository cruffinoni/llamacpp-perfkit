from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from llamacpp_perfkit.reporting import (
    aggregate_server_config_reports,
    enforce_prompt_profile_comparability,
    load_rows,
)
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
            "server_config": {"context_size": ctx, "kv_type": "q8_0", "n_cpu_moe": 0},
            "status": status,
            "config_hash": run_id,
            "config": {
                "model_hf": "model:A",
                "prompt_profile": profile,
                "context_size": ctx,
                "kv_type": "q8_0",
                "n_cpu_moe": 0,
            },
            "duration_seconds": 10.0,
            "response": {"content": "ok"},
        },
    )
    append_llamacpp_metric(root / run_id, {"time": "t1", "generation_tok_s": gen, "prompt_eval_tok_s": prompt, "total_time_seconds": 10.0})
    append_system_metric(root / run_id, {"time": "t1", "vram_free_mib": vram, "vram_used_mib": 1024.0})


class SummaryRankingTest(unittest.TestCase):
    def test_summary_groups_runs_by_server_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for ctx, gen in ((2048, 100.0), (4096, 90.0)):
                write_run(root, f"code-{ctx}", ctx=ctx, profile="code", gen=gen, prompt=50.0, vram=4096.0)
                write_run(root, f"qa-{ctx}", ctx=ctx, profile="qa", gen=gen - 10.0, prompt=45.0, vram=4096.0)
            reports, _ = aggregate_server_config_reports(load_rows(args(root)))

            self.assertEqual(len(reports), 2)
            self.assertEqual(sum(report.total_runs for report in reports), 4)
            self.assertEqual({report.profiles_seen for report in reports}, {("code", "qa")})
            self.assertTrue(all(report.status == "stable" for report in reports))
            self.assertTrue(all(report.generation_tok_s.count == 2 for report in reports))
            self.assertTrue(all(report.generation_tok_s.p10 is not None for report in reports))

    def test_summary_allows_different_prompt_profile_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_run(root, "code-2048", ctx=2048, profile="code", gen=100.0, prompt=50.0, vram=4096.0)
            write_run(root, "code-4096", ctx=4096, profile="code", gen=90.0, prompt=45.0, vram=4096.0)
            write_run(root, "qa-4096", ctx=4096, profile="qa", gen=80.0, prompt=40.0, vram=4096.0)

            reports, _ = aggregate_server_config_reports(load_rows(args(root)))

            self.assertEqual(len(reports), 2)
            self.assertEqual({report.key.context_size for report in reports}, {2048, 4096})
            self.assertEqual(
                {report.key.context_size: report.profiles_seen for report in reports},
                {2048: ("code",), 4096: ("code", "qa")},
            )

    def test_compare_refuses_different_prompt_profile_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_run(root, "base", ctx=2048, profile="code", gen=100.0, prompt=50.0, vram=4096.0)
            write_run(root, "extra", ctx=4096, profile="qa", gen=90.0, prompt=45.0, vram=4096.0)

            reports, _ = aggregate_server_config_reports(load_rows(args(root)))
            with self.assertRaises(ValueError):
                enforce_prompt_profile_comparability(reports)

    def test_compare_candidates_keep_metric_values_for_delta_calculation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_run(root, "base", ctx=2048, profile="code", gen=100.0, prompt=50.0, vram=4096.0)
            write_run(root, "candidate", ctx=4096, profile="code", gen=110.0, prompt=55.0, vram=5120.0)
            reports, _ = aggregate_server_config_reports(load_rows(args(root)))
            enforce_prompt_profile_comparability(reports)
            by_context = {report.key.context_size: report for report in reports}
            base = by_context[2048]
            candidate = by_context[4096]

            base_gen = base.generation_tok_s.geometric_mean
            candidate_gen = candidate.generation_tok_s.geometric_mean
            candidate_prompt = candidate.prompt_tok_s.geometric_mean
            base_vram = base.min_free_vram
            candidate_vram = candidate.min_free_vram
            assert base_gen is not None
            assert candidate_gen is not None
            assert candidate_prompt is not None
            assert base_vram is not None
            assert candidate_vram is not None

            self.assertAlmostEqual(candidate_gen, 110.0)
            self.assertAlmostEqual(candidate_prompt, 55.0)
            self.assertAlmostEqual((candidate_gen - base_gen) / base_gen, 0.1)
            self.assertEqual(candidate_vram - base_vram, 1024.0)


if __name__ == "__main__":
    unittest.main()
