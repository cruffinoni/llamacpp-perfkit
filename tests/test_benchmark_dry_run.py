from __future__ import annotations

import unittest
from typing import Any

from llamacpp_perfkit.benchlib import server_group_key


def job(profile: str, context_size: int = 2048) -> dict[str, Any]:
    return {
        "prompt_profile": profile,
        "prompt_file": f"/tmp/{profile}.txt",
        "n_cpu_moe": 4,
        "context_size": context_size,
        "kv_type": "q8_0",
        "batch_size": 1024,
        "ubatch_size": 1024,
        "spec_type": None,
        "spec_draft_n_max": None,
        "spec_draft_p_min": None,
    }


def plan_entry(
    run_id: str,
    action: str,
    profile: str,
    *,
    context_size: int = 2048,
    spec_type: str | None = None,
    action_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "config_hash": f"hash-{run_id}",
        "risk_level": "low",
        "action": action,
        "action_reason": action_reason,
        "job": {
            **job(profile, context_size),
            "spec_type": spec_type,
            "spec_draft_n_max": 2 if spec_type else None,
            "spec_draft_p_min": 0.75 if spec_type else None,
        },
    }


class DryRunPlanGroupingTest(unittest.TestCase):
    def test_server_group_key_groups_all_actions_by_server_config(self) -> None:
        planned = [
            plan_entry("plan-0001", "run", "alpha"),
            plan_entry("plan-0002", "reuse", "beta", action_reason="successful result already exists: old-run"),
            plan_entry("plan-0003", "skip", "gamma", action_reason="previous result is oom"),
            plan_entry("plan-0004", "run", "delta", spec_type="draft-mtp"),
            plan_entry("plan-0005", "reuse", "epsilon"),
            plan_entry("plan-0006", "skip", "zeta", spec_type="draft-mtp", action_reason="risk is high after prior OOM/unsafe result"),
        ]
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for item in planned:
            groups.setdefault(server_group_key(item["job"]), []).append(item)

        group_sizes = sorted(len(group) for group in groups.values())
        action_counts = [
            {action: sum(1 for item in group if item["action"] == action) for action in ("run", "reuse", "skip", "blocked")}
            for group in groups.values()
        ]

        self.assertEqual(group_sizes, [2, 4])
        self.assertIn({"run": 1, "reuse": 2, "skip": 1, "blocked": 0}, action_counts)
        self.assertIn({"run": 1, "reuse": 0, "skip": 1, "blocked": 0}, action_counts)


if __name__ == "__main__":
    unittest.main()
