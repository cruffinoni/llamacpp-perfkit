package fixtures

import "github.com/cruffinoni/llamacpp-perfkit/internal/tui/viewmodel"

// StaticState returns a pre-populated BenchmarkTUIState for development and
// preview purposes.
func StaticState() viewmodel.BenchmarkTUIState {
	v1, v2, v3, v4, v6 := 78.0, 63.2, 68.2, 65.3, 79.3
	p1, p2, p3, p4, p6 := 812.0, 691.0, 604.0, 755.0, 902.0
	d1, d2, d3, d4, d6 := 2.70, 3.21, 3.25, 3.35, 1.91
	m1, m2, m3, m4, m6 := 5520.0, 5417.0, 5448.0, 5489.0, 5509.0
	dFail := 0.42
	dTimeout := 30.0
	dOOM := 0.15
	return viewmodel.BenchmarkTUIState{
		RunID:          "1779537110",
		BuildInfo:      viewmodel.BuildInfoView{CommitShort: "b64739ea", Branch: "master", Backend: "cuda"},
		ModelName:      "Qwen3.6-35B-A3B-MTP UD-Q4_K_M",
		Progress:       viewmodel.ProgressState{ServersCompleted: 14, ServersTotal: 96, JobsCompleted: 104, JobsTotal: 768, CurrentPrompt: 5, CurrentPromptTotal: 8},
		ElapsedSeconds: 522,
		ETASeconds:     2350,
		CurrentServer: &viewmodel.CurrentServerView{
			ID: "1779537195-server-0014", ContextSize: 8192, KVType: "q8_0", NCPUMOE: 18, SpecType: "draft-mtp", BatchSize: 512, UBatchSize: 512,
		},
		PromptJobs: []viewmodel.PromptJobView{
			{Profile: "code_python", Status: "success", Phase: "done", DurationSeconds: &d1, GenTokS: &v1, PromptTokS: &p1, MinVRAMMiB: &m1},
			{Profile: "code_cpp", Status: "success", Phase: "done", DurationSeconds: &d2, GenTokS: &v2, PromptTokS: &p2, MinVRAMMiB: &m2},
			{Profile: "long_code_review", Status: "success", Phase: "done", DurationSeconds: &d3, GenTokS: &v3, PromptTokS: &p3, MinVRAMMiB: &m3},
			{Profile: "long_prefill_8k", Status: "success", Phase: "done", DurationSeconds: &d4, GenTokS: &v4, PromptTokS: &p4, MinVRAMMiB: &m4},
			{Profile: "long_prefill_16k", Status: "timeout", Phase: "timeout", DurationSeconds: &dTimeout},
			{Profile: "long_prefill_32k", Status: "running", Phase: "generating", DurationSeconds: &d6, GenTokS: &v6, PromptTokS: &p6, MinVRAMMiB: &m6},
			{Profile: "long_prefill_48k", Status: "failed", Phase: "failed", DurationSeconds: &dFail},
			{Profile: "long_prefill_60k", Status: "pending", Phase: "-"},
			{Profile: "chat_translate", Status: "oom", Phase: "oom", DurationSeconds: &dOOM},
			{Profile: "summarize_doc", Status: "pending", Phase: "-"},
		},
		StatusMessage: "dev tui static benchmark state",
	}
}
