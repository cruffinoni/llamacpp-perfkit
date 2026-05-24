package runner

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/viewmodel"
)

func TestTUIAdapterStateTransitions(t *testing.T) {
	updates := make(chan viewmodel.StateUpdate, 16)
	adapter := newTUIAdapter(updates, nowTimeForTest())
	item := domain.PlannedRun{
		Job: domain.BenchmarkJob{
			PromptProfile: domain.PromptProfile{Name: "code"},
			ServerConfig:  domain.ServerConfig{ContextSize: 2048, KVType: "q8_0"},
		},
	}
	server := serverExecution{ID: "server-a"}

	adapter.BeginGroup(server, []domain.PlannedRun{item}, 1, 1)
	adapter.BeginPrompt(item, 0)
	result := requestResult{
		Status: domain.StatusSuccess, Duration: 1.5,
		Loaded: domain.LoadedRun{
			LlamaSummary:  domain.LlamaSummary{GenerationTokS: new(42.0), PromptEvalTokS: new(100.0)},
			SystemSummary: domain.SystemSummary{MinVRAMFreeMiB: new(2048.0)},
		},
	}
	adapter.CompletePrompt(item, result, 1)
	adapter.CompleteGroup(1, 1)
	adapter.CompleteBenchmark(1)

	state := applyAll(updates)

	require.NotNil(t, state.CurrentServer)
	assert.Equal(t, "server-a", state.CurrentServer.ID)
	require.Len(t, state.PromptJobs, 1)
	assert.Equal(t, domain.StatusSuccess, state.PromptJobs[0].Status)
	require.NotNil(t, state.PromptJobs[0].GenTokS)
	assert.InDelta(t, 42.0, *state.PromptJobs[0].GenTokS, 0.0001)
	assert.Equal(t, "complete", state.LifecycleState)
	assert.Equal(t, 1, state.Progress.JobsCompleted)
	assert.Equal(t, 1, state.Progress.ServersCompleted)
}

func TestTUIAdapterStartupFailedPrompt(t *testing.T) {
	updates := make(chan viewmodel.StateUpdate, 8)
	adapter := newTUIAdapter(updates, nowTimeForTest())
	item := domain.PlannedRun{
		Job: domain.BenchmarkJob{
			PromptProfile: domain.PromptProfile{Name: "oom_profile"},
			ServerConfig:  domain.ServerConfig{ContextSize: 8192},
		},
	}

	adapter.StartupFailedPrompt(item, "process crashed", 2.0, 5)

	state := applyAll(updates)
	require.Len(t, state.PromptJobs, 1)
	assert.Equal(t, domain.StatusFailed, state.PromptJobs[0].Status)
	assert.Equal(t, domain.PhaseFailed, state.PromptJobs[0].Phase)
	assert.Equal(t, 5, state.Progress.JobsCompleted)
}

func TestTUIAdapterNilChannelDoesNotPanic(t *testing.T) {
	adapter := newTUIAdapter(nil, nowTimeForTest())
	item := domain.PlannedRun{
		Job: domain.BenchmarkJob{
			PromptProfile: domain.PromptProfile{Name: "code"},
		},
	}

	assert.NotPanics(t, func() {
		adapter.BeginPrompt(item, 0)
	})
}

func TestTUIAdapterMultiplePrompts(t *testing.T) {
	updates := make(chan viewmodel.StateUpdate, 32)
	adapter := newTUIAdapter(updates, nowTimeForTest())

	items := []domain.PlannedRun{
		{Job: domain.BenchmarkJob{PromptProfile: domain.PromptProfile{Name: "code"}, ServerConfig: domain.ServerConfig{ContextSize: 2048}}},
		{Job: domain.BenchmarkJob{PromptProfile: domain.PromptProfile{Name: "qa"}, ServerConfig: domain.ServerConfig{ContextSize: 2048}}},
		{Job: domain.BenchmarkJob{PromptProfile: domain.PromptProfile{Name: "chat"}, ServerConfig: domain.ServerConfig{ContextSize: 2048}}},
	}
	server := serverExecution{ID: "server-b"}

	adapter.BeginGroup(server, items, 1, 3)

	for i, item := range items {
		adapter.BeginPrompt(item, i)
		result := requestResult{
			Status: domain.StatusSuccess, Duration: 1.0,
			Loaded: domain.LoadedRun{
				LlamaSummary:  domain.LlamaSummary{GenerationTokS: new(float64(50 + i))},
				SystemSummary: domain.SystemSummary{},
			},
		}
		adapter.CompletePrompt(item, result, i+1)
	}

	state := applyAll(updates)
	assert.Equal(t, 3, state.Progress.JobsCompleted)
	require.Len(t, state.PromptJobs, 3)
	assert.Equal(t, "code", state.PromptJobs[0].Profile)
	assert.Equal(t, "qa", state.PromptJobs[1].Profile)
	assert.Equal(t, "chat", state.PromptJobs[2].Profile)
}

func TestTUIAdapterCompleteBenchmark(t *testing.T) {
	updates := make(chan viewmodel.StateUpdate, 8)
	adapter := newTUIAdapter(updates, nowTimeForTest())

	adapter.CompleteBenchmark(5)

	state := applyAll(updates)
	assert.Equal(t, "complete", state.LifecycleState)
	assert.Equal(t, 5, state.Progress.ServersCompleted)
}

func applyAll(updates chan viewmodel.StateUpdate) *viewmodel.BenchmarkTUIState {
	state := &viewmodel.BenchmarkTUIState{}
	close(updates)
	for update := range updates {
		update.Apply(state)
	}
	return state
}

func nowTimeForTest() time.Time {
	return time.Unix(1, 0)
}
