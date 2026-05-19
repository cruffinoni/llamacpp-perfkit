from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from typing import Any

from llamacpp_perfkit.output import configure_color
from llamacpp_perfkit.services import BenchmarkService


def job(profile: str, context_size: int = 2048) -> dict[str, Any]:
    return {
        "prompt_profile": profile,
        "prompt_file": f"/tmp/{profile}.txt",
        "n_cpu_moe": 4,
        "context_size": context_size,
        "kv_type": "q8_0",
        "batch_size": 1024,
        "ubatch_size": 1024,
        "mtp_enabled": False,
        "mtp_spec_type": None,
        "mtp_draft_n_max": None,
        "mtp_draft_p_min": None,
    }


def plan_entry(
    run_id: str,
    action: str,
    profile: str,
    *,
    context_size: int = 2048,
    mtp_enabled: bool = False,
    action_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "config_hash": f"hash-{run_id}",
        "kind": "mtp" if mtp_enabled else "baseline",
        "risk_level": "low",
        "action": action,
        "action_reason": action_reason,
        "job": {
            **job(profile, context_size),
            "mtp_enabled": mtp_enabled,
            "mtp_spec_type": "draft-mtp" if mtp_enabled else None,
            "mtp_draft_n_max": 2 if mtp_enabled else None,
            "mtp_draft_p_min": 0.75 if mtp_enabled else None,
        },
    }


class DryRunPlanPrintingTest(unittest.TestCase):
    def setUp(self) -> None:
        configure_color(no_color=True)

    def render(self) -> str:
        cfg = {
            "models": {"baseline_hf": "org/model:Q4", "mtp_hf": "org/model-mtp:Q4"},
            "llama": {
                "bin_dir": "/tmp/llama-bin",
                "server": {"host": "127.0.0.1"},
            },
        }
        features = {
            "flags": {
                "llama_server": {
                    "context": "--ctx-size",
                    "host": "--host",
                    "port": "--port",
                    "n_cpu_moe": "--n-cpu-moe",
                    "cache_type_k": "--cache-type-k",
                    "cache_type_v": "--cache-type-v",
                    "no_webui": "--no-webui",
                    "spec_type": "--spec-type",
                    "spec_draft_n_max": "--spec-draft-n-max",
                    "spec_draft_p_min": "--spec-draft-p-min",
                }
            },
            "extra_args": {"server": [{"flag": "--parallel", "value": 1}], "request": {"temperature": 0.6}},
            "llama_cpp": {"commit_short": "abc1234"},
        }
        plan = {
            "mode": "full",
            "max_runs": 9,
            "reuse_existing_results": True,
            "candidate_count": 8,
            "selected_count": 6,
            "estimated_runs": 2,
            "max_runs_capped": False,
            "warning": "full mode may run the full Cartesian matrix and take a long time",
            "notes": ["metadata note"],
            "skipped": [{"dimension": "kv_type", "value": "q4_0", "reason": "unsupported"}],
            "planned": [
                plan_entry("plan-0001", "run", "alpha"),
                plan_entry("plan-0002", "reuse", "beta", action_reason="successful result already exists: old-run"),
                plan_entry("plan-0003", "skip", "gamma", action_reason="previous result is oom"),
                plan_entry("plan-0004", "run", "delta", mtp_enabled=True),
                plan_entry("plan-0005", "reuse", "epsilon"),
                plan_entry("plan-0006", "skip", "zeta", mtp_enabled=True, action_reason="risk is high after prior OOM/unsafe result"),
            ],
        }
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir, contextlib.redirect_stdout(out):
            BenchmarkService().print_plan(plan, cfg, features, Path(tmpdir))
        return out.getvalue()

    def test_dry_run_groups_all_actions_by_server_key(self) -> None:
        output = self.render()

        self.assertIn("Budget mode: full", output)
        self.assertIn("Max new requests: 9", output)
        self.assertIn("Reuse existing results: True", output)
        self.assertIn("Candidate combinations: 8", output)
        self.assertIn("Selected plan entries: 6", output)
        self.assertIn("Estimated new requests now: 2", output)
        self.assertIn("WARNING: full mode may run the full Cartesian matrix and take a long time", output)
        self.assertIn("note: metadata note", output)
        self.assertIn("skip kv_type=q4_0: unsupported", output)

        self.assertIn(
            "[server 1/2] dry-run-server-0001 entries=4 run=1 reuse=2 skip=1 blocked=0 "
            "ctx=2.05k kv=q8_0 n_cpu_moe=4 mtp=False batch=1.02k ubatch=1.02k",
            output,
        )
        self.assertIn(
            "[server 2/2] dry-run-server-0002 entries=2 run=1 reuse=0 skip=1 blocked=0 "
            "ctx=2.05k kv=q8_0 n_cpu_moe=4 mtp=True batch=1.02k ubatch=1.02k",
            output,
        )
        self.assertIn("request plan_id=plan-0001 action=run profile=alpha hash=hash-plan-0001 kind=baseline risk=low", output)
        self.assertIn("request plan_id=plan-0002 action=reuse profile=beta", output)
        self.assertIn("reason=successful result already exists: old-run", output)
        self.assertIn("request plan_id=plan-0003 action=skip profile=gamma", output)
        self.assertIn("request plan_id=plan-0004 action=run profile=delta hash=hash-plan-0004 kind=mtp risk=low", output)
        self.assertIn("request plan_id=plan-0006 action=skip profile=zeta", output)
        self.assertNotIn("baseline=", output)
        self.assertNotIn("action=blocked", output)

        first_group = output.split("[server 1/2] dry-run-server-0001", 1)[1].split("[server 2/2] dry-run-server-0002", 1)[0]
        second_group = output.split("[server 2/2] dry-run-server-0002", 1)[1]
        self.assertEqual(first_group.count("/tmp/llama-bin/llama-server"), 1)
        self.assertEqual(second_group.count("/tmp/llama-bin/llama-server"), 1)


if __name__ == "__main__":
    unittest.main()
