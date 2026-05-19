from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llamacpp_perfkit.run_storage import (
    append_llamacpp_metric,
    append_system_metric,
    load_run_rows,
    read_run_metrics,
    write_run_summary,
)


class RunStorageTest(unittest.TestCase):
    def test_writes_summary_and_metrics_and_reads_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_run_summary(
                root,
                {
                    "run_id": "run-a",
                    "batch_id": "batch-a",
                    "created_at": "2026-01-01T00:00:00Z",
                    "model": "model:A",
                    "prompt_profile": "code",
                    "server_config": {"context_size": 2048, "kv_type": "q8_0"},
                    "status": "success",
                    "config_hash": "hash-a",
                    "config": {"model_hf": "model:A", "prompt_profile": "code", "context_size": 2048, "kv_type": "q8_0"},
                    "duration_seconds": 10.0,
                    "response": {"content": "ok"},
                },
            )
            append_system_metric(root / "run-a", {"time": "t1", "vram_free_mib": 4096.0, "vram_used_mib": 1024.0})
            append_llamacpp_metric(root / "run-a", {"time": "t1", "generation_tok_s": 44.0, "prompt_eval_tok_s": 120.0})

            system, llamacpp = read_run_metrics(root / "run-a")
            self.assertEqual(system[0]["vram_free_mib"], 4096.0)
            self.assertEqual(llamacpp[0]["generation_tok_s"], 44.0)

            rows = load_run_rows(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "run-a")
            self.assertEqual(rows[0]["monitor"]["min_vram_free_mib"], 4096.0)
            self.assertEqual(rows[0]["parsed"]["generation_tok_s"], 44.0)


if __name__ == "__main__":
    unittest.main()
