from __future__ import annotations

import asyncio
import threading
import unittest

from textual.widgets import Static

from llamacpp_perfkit.tui.renderer import BenchmarkTUIApp, TUIRenderer
from llamacpp_perfkit.tui.state import BenchmarkTUIState, BuildInfoView


class TestTUIRenderer(unittest.TestCase):
    def state(self) -> BenchmarkTUIState:
        return BenchmarkTUIState(
            run_id="run-1",
            build_info=BuildInfoView(commit_short="abc123", branch="main", backend="server"),
            model_name="test-model",
        )

    def test_prompt_updates_preserve_inserted_order(self) -> None:
        state = self.state()
        renderer = TUIRenderer(state)

        renderer.update({"prompt": True, "prompt_profile": "qa_factual", "status": "pending", "phase": "pending"})
        renderer.update({"prompt": True, "prompt_profile": "creative_short", "status": "pending", "phase": "pending"})
        renderer.update({"prompt": True, "prompt_profile": "qa_factual", "status": "running", "phase": "starting"})

        self.assertEqual([job.profile for job in state.prompt_jobs], ["qa_factual", "creative_short"])
        self.assertEqual(state.prompt_jobs[0].status, "running")
        self.assertEqual(state.prompt_jobs[1].status, "pending")

    def test_updates_build_info_without_replacing_unspecified_fields(self) -> None:
        state = self.state()
        renderer = TUIRenderer(state)

        renderer.update({"llama_cpp_commit": "def456", "backend": "cuda"})

        self.assertEqual(state.build_info.commit_short, "def456")
        self.assertEqual(state.build_info.branch, "main")
        self.assertEqual(state.build_info.backend, "cuda")

    def test_updates_progress_aliases(self) -> None:
        state = self.state()
        renderer = TUIRenderer(state)

        renderer.update({"server_index": 2, "server_total": 5, "job_index": 3, "job_total": 8})

        self.assertEqual(state.progress.servers_completed, 2)
        self.assertEqual(state.progress.servers_total, 5)
        self.assertEqual(state.progress.jobs_completed, 3)
        self.assertEqual(state.progress.jobs_total, 8)

    def test_request_stop_updates_lifecycle_state(self) -> None:
        state = self.state()
        renderer = TUIRenderer(state)

        renderer.request_stop()

        self.assertTrue(renderer.stop_requested)
        self.assertEqual(state.lifecycle_state, "stopping")
        self.assertIn("Stop requested", state.status_message)

    def test_textual_app_mounts_and_reflects_updates(self) -> None:
        async def run_app() -> None:
            state = self.state()
            renderer = TUIRenderer(state)
            release = threading.Event()

            def benchmark() -> int:
                release.wait(timeout=2)
                return 0

            app = BenchmarkTUIApp(renderer, benchmark)
            async with app.run_test() as pilot:
                renderer.update(
                    {
                        "lifecycle_state": "running",
                        "status_message": "sample update",
                        "server_total": 2,
                        "job_total": 4,
                        "prompt": True,
                        "prompt_profile": "qa_factual",
                        "status": "running",
                        "phase": "starting",
                    }
                )
                app.apply_state(renderer.snapshot())
                await pilot.pause()
                title = app.query_one("#title", Static)
                message = app.query_one("#message-panel", Static)
                self.assertIn("llama-cpp-perfkit", str(title.render()))
                self.assertIn("sample update", str(message.render()))
                release.set()

        asyncio.run(run_app())


if __name__ == "__main__":
    unittest.main()
