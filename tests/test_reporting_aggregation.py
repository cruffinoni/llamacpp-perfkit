from __future__ import annotations

import unittest
from typing import Any

from llamacpp_perfkit.reporting import aggregate_server_config_reports


def row(
    run_id: str,
    *,
    prompt_profile: str,
    status: str = "success",
    created_at: str = "2026-01-01T00:00:00Z",
    generation_tok_s: float | None = 100.0,
    prompt_eval_tok_s: float | None = 50.0,
    min_vram_free_mib: float | None = 1024.0,
    command: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": created_at,
        "status": status,
        "stability_status": "stable" if status == "success" else "unknown",
        "command": command
        or [
            "/tmp/llama-server",
            "--ctx-size",
            "2048",
            "--batch-size",
            "128",
            "--ubatch-size",
            "32",
        ],
        "config": {
            "model_hf": "model:A",
            "prompt_profile": prompt_profile,
            "context_size": 2048,
            "kv_type": "q8_0",
            "n_cpu_moe": 0,
            "batch_size": 128,
            "ubatch_size": 32,
            "mtp_enabled": False,
            "generation_tokens": 16,
            "seed": 42,
        },
        "parsed": {
            "generation_tok_s": generation_tok_s,
            "prompt_eval_tok_s": prompt_eval_tok_s,
            "benchmark_valid": status == "success" and generation_tok_s is not None,
        },
        "duration_seconds": 10.0,
        "monitor": {"min_vram_free_mib": min_vram_free_mib} if min_vram_free_mib is not None else {},
    }


class ReportingAggregationTest(unittest.TestCase):
    def test_grouping_by_server_config_ignores_prompt_profile(self) -> None:
        reports, _ = aggregate_server_config_reports(
            [
                row("run-a", prompt_profile="code"),
                row("run-b", prompt_profile="qa", created_at="2026-01-01T00:01:00Z"),
            ]
        )

        self.assertEqual(len(reports), 1)
        report = reports[0]
        self.assertEqual(report.total_runs, 2)
        self.assertEqual(report.success_count, 2)
        self.assertEqual(report.profiles_seen, ("code", "qa"))
        self.assertEqual(report.key.batch_size, 128)
        self.assertEqual(report.key.ubatch_size, 32)

    def test_evidence_and_missing_metrics(self) -> None:
        reports, _ = aggregate_server_config_reports(
            [
                row("run-a", prompt_profile="code", min_vram_free_mib=1024.0, generation_tok_s=100.0),
                row("run-b", prompt_profile="code", status="oom", generation_tok_s=None, prompt_eval_tok_s=None, min_vram_free_mib=None),
                row(
                    "run-c", prompt_profile="code", status="timeout", generation_tok_s=None, prompt_eval_tok_s=None, min_vram_free_mib=512.0
                ),
            ]
        )

        report = reports[0]
        self.assertEqual(report.total_runs, 3)
        self.assertEqual(report.success_count, 1)
        self.assertEqual(report.failure_count, 1)
        self.assertEqual(report.timeout_count, 1)
        self.assertEqual(report.evidence_display, "timeout (1/3)")
        self.assertEqual(report.min_free_vram, 512.0)
        self.assertEqual(report.mean_free_vram, 768.0)
        self.assertEqual(report.free_vram_mib.count, 2)
        self.assertEqual(report.decode_tok_s.count, 1)


if __name__ == "__main__":
    unittest.main()
