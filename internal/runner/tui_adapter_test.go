package runner

import (
	"testing"
	"time"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/viewmodel"
)

func TestTUIAdapterStateTransitions(t *testing.T) {
	updates := make(chan viewmodel.StateUpdate, 8)
	adapter := newTUIAdapter(updates, nowTimeForTest())
	item := domain.PlannedRun{Job: domain.BenchmarkJob{PromptProfile: domain.PromptProfile{Name: "code"}, ServerConfig: domain.ServerConfig{ContextSize: 2048, KVType: "q8_0"}}}
	server := serverExecution{ID: "server-a"}

	adapter.BeginGroup(server, []domain.PlannedRun{item}, 1, 1)
	adapter.BeginPrompt(item, 0)
	result := requestResult{Status: domain.StatusSuccess, Duration: 1.5, Loaded: domain.LoadedRun{
		LlamaSummary:  domain.LlamaSummary{GenerationTokS: domain.Ptr(42.0), PromptEvalTokS: domain.Ptr(100.0)},
		SystemSummary: domain.SystemSummary{MinVRAMFreeMiB: domain.Ptr(2048.0)},
	}}
	adapter.CompletePrompt(item, result, 1)
	adapter.CompleteGroup(1, 1)
	adapter.CompleteBenchmark(1)

	state := viewmodel.BenchmarkTUIState{}
	for len(updates) > 0 {
		update := <-updates
		update.Apply(&state)
	}
	if state.CurrentServer == nil || state.CurrentServer.ID != "server-a" {
		t.Fatalf("current server = %+v", state.CurrentServer)
	}
	if len(state.PromptJobs) != 1 || state.PromptJobs[0].Status != "success" || *state.PromptJobs[0].GenTokS != 42 {
		t.Fatalf("prompt jobs = %+v", state.PromptJobs)
	}
	if state.LifecycleState != "complete" || state.Progress.JobsCompleted != 1 || state.Progress.ServersCompleted != 1 {
		t.Fatalf("state = %+v", state)
	}
}

func nowTimeForTest() time.Time {
	return time.Unix(1, 0)
}
