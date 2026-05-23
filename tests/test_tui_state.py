from __future__ import annotations

import unittest

from llamacpp_perfkit.tui.state import (
    BenchmarkTUIState,
    BuildInfoView,
    CurrentServerView,
    ProgressState,
    PromptJobView,
)


class TestProgressState(unittest.TestCase):
    def test_defaults(self) -> None:
        ps = ProgressState()
        self.assertEqual(ps.servers_completed, 0)
        self.assertEqual(ps.servers_total, 0)
        self.assertEqual(ps.jobs_completed, 0)
        self.assertEqual(ps.jobs_total, 0)
        self.assertEqual(ps.current_prompt_index, 0)
        self.assertEqual(ps.current_prompt_total, 0)


class TestPromptJobView(unittest.TestCase):
    def test_minimal_job(self) -> None:
        job = PromptJobView(profile="code_python", status="pending", phase="pending")
        self.assertEqual(job.profile, "code_python")
        self.assertEqual(job.status, "pending")
        self.assertEqual(job.phase, "pending")
        self.assertIsNone(job.duration_seconds)
        self.assertIsNone(job.gen_tok_s)
        self.assertIsNone(job.prompt_tok_s)
        self.assertIsNone(job.min_vram_mib)

    def test_full_job(self) -> None:
        job = PromptJobView(
            profile="long_prefill_8k",
            status="success",
            phase="done",
            duration_seconds=2.70,
            gen_tok_s=78.0,
            prompt_tok_s=812.0,
            min_vram_mib=5039.0,
        )
        self.assertEqual(job.profile, "long_prefill_8k")
        self.assertEqual(job.status, "success")
        self.assertEqual(job.phase, "done")
        self.assertEqual(job.duration_seconds, 2.70)
        self.assertEqual(job.gen_tok_s, 78.0)
        self.assertEqual(job.prompt_tok_s, 812.0)
        self.assertEqual(job.min_vram_mib, 5039.0)

    def test_failed_job(self) -> None:
        job = PromptJobView(profile="code_cpp", status="oom", phase="failed")
        self.assertEqual(job.status, "oom")
        self.assertEqual(job.phase, "failed")

    def test_running_job(self) -> None:
        job = PromptJobView(
            profile="code_python",
            status="running",
            phase="generating",
            duration_seconds=1.5,
        )
        self.assertEqual(job.status, "running")
        self.assertEqual(job.phase, "generating")
        self.assertEqual(job.duration_seconds, 1.5)
        self.assertIsNone(job.gen_tok_s)


class TestCurrentServerView(unittest.TestCase):
    def test_minimal_server(self) -> None:
        svr = CurrentServerView(
            id="1779537195-server-0014",
            context_size=8192,
            kv_type="q8_0",
            n_cpu_moe=18,
            spec_type="draft-mtp",
        )
        self.assertEqual(svr.id, "1779537195-server-0014")
        self.assertEqual(svr.context_size, 8192)
        self.assertEqual(svr.kv_type, "q8_0")
        self.assertEqual(svr.n_cpu_moe, 18)
        self.assertEqual(svr.spec_type, "draft-mtp")
        self.assertIsNone(svr.batch_size)
        self.assertIsNone(svr.ubatch_size)

    def test_full_server(self) -> None:
        svr = CurrentServerView(
            id="server-01",
            context_size=16384,
            kv_type="q4_0",
            n_cpu_moe=12,
            spec_type="none",
            batch_size=512,
            ubatch_size=512,
        )
        self.assertEqual(svr.batch_size, 512)
        self.assertEqual(svr.ubatch_size, 512)


class TestBenchmarkTUIState(unittest.TestCase):
    def test_minimal_state(self) -> None:
        build = BuildInfoView(commit_short="b64739ea", branch="master", backend="cuda")
        state = BenchmarkTUIState(
            run_id="1779537110",
            build_info=build,
            model_name="Qwen3.6-35B-A3B-MTP UD-Q4_K_M",
        )
        self.assertEqual(state.run_id, "1779537110")
        self.assertEqual(state.build_info.commit_short, "b64739ea")
        self.assertEqual(state.model_name, "Qwen3.6-35B-A3B-MTP UD-Q4_K_M")
        self.assertEqual(state.elapsed_seconds, 0.0)
        self.assertEqual(state.eta_seconds, 0.0)
        self.assertIsNone(state.current_server)
        self.assertEqual(state.prompt_jobs, [])

    def test_initial_progress_defaults(self) -> None:
        build = BuildInfoView(commit_short="abc", branch="main", backend="cpu")
        state = BenchmarkTUIState(run_id="1", build_info=build, model_name="test")
        self.assertEqual(state.progress.servers_completed, 0)
        self.assertEqual(state.progress.jobs_completed, 0)

    def test_with_current_server(self) -> None:
        build = BuildInfoView(commit_short="abc", branch="main", backend="cpu")
        state = BenchmarkTUIState(run_id="1", build_info=build, model_name="test")
        state.current_server = CurrentServerView(
            id="srv-01",
            context_size=4096,
            kv_type="f16",
            n_cpu_moe=0,
            spec_type="none",
        )
        self.assertIsNotNone(state.current_server)
        assert state.current_server is not None
        self.assertEqual(state.current_server.id, "srv-01")

    def test_prompt_jobs_accumulation(self) -> None:
        build = BuildInfoView(commit_short="abc", branch="main", backend="cpu")
        state = BenchmarkTUIState(run_id="1", build_info=build, model_name="test")
        state.prompt_jobs.append(PromptJobView(profile="code_python", status="success", phase="done"))
        state.prompt_jobs.append(PromptJobView(profile="code_cpp", status="running", phase="generating"))
        self.assertEqual(len(state.prompt_jobs), 2)
        self.assertEqual(state.prompt_jobs[0].profile, "code_python")
        self.assertEqual(state.prompt_jobs[1].profile, "code_cpp")


if __name__ == "__main__":
    unittest.main()
