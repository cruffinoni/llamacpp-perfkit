package viewmodel

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
)

func TestUpsertPrompt(t *testing.T) {
	tests := map[string]struct {
		setup  func(*BenchmarkTUIState)
		update PromptJobView
		verify func(*testing.T, *BenchmarkTUIState)
	}{
		"inserts new prompt": {
			setup:  nil,
			update: PromptJobView{Profile: "code", Status: domain.StatusPending},
			verify: func(t *testing.T, s *BenchmarkTUIState) {
				assert.Len(t, s.PromptJobs, 1)
				assert.Equal(t, "code", s.PromptJobs[0].Profile)
				assert.Equal(t, domain.StatusPending, s.PromptJobs[0].Status)
			},
		},
		"preserves order when inserting multiple": {
			setup: func(s *BenchmarkTUIState) {
				s.UpsertPrompt(PromptJobView{Profile: "qa", Status: domain.StatusPending, Phase: "-"})
			},
			update: PromptJobView{Profile: "code", Status: domain.StatusPending, Phase: "-"},
			verify: func(t *testing.T, s *BenchmarkTUIState) {
				assert.Equal(t, []string{"qa", "code"}, []string{s.PromptJobs[0].Profile, s.PromptJobs[1].Profile})
			},
		},
		"updates existing prompt status and phase": {
			setup: func(s *BenchmarkTUIState) {
				s.UpsertPrompt(PromptJobView{Profile: "qa", Status: domain.StatusPending, Phase: "-"})
			},
			update: PromptJobView{Profile: "qa", Status: domain.StatusRunning, Phase: domain.PhaseGenerating},
			verify: func(t *testing.T, s *BenchmarkTUIState) {
				assert.Equal(t, domain.StatusRunning, s.PromptJobs[0].Status)
				assert.Equal(t, domain.PhaseGenerating, s.PromptJobs[0].Phase)
			},
		},
		"does not overwrite non-nil phase with empty string": {
			setup: func(s *BenchmarkTUIState) {
				s.UpsertPrompt(PromptJobView{Profile: "code", Status: domain.StatusPending, Phase: domain.PhaseStarting})
			},
			update: PromptJobView{Profile: "code", Status: domain.StatusRunning},
			verify: func(t *testing.T, s *BenchmarkTUIState) {
				assert.Equal(t, domain.StatusRunning, s.PromptJobs[0].Status)
				assert.Equal(t, domain.PhaseStarting, s.PromptJobs[0].Phase)
			},
		},
		"updates all metrics on completion": {
			setup: func(s *BenchmarkTUIState) {
				s.UpsertPrompt(PromptJobView{Profile: "code", Status: domain.StatusRunning})
			},
			update: PromptJobView{Profile: "code", Status: domain.StatusSuccess, Phase: domain.PhaseDone,
				DurationSeconds: new(2.5), GenTokS: new(78.0), PromptTokS: new(812.0), MinVRAMMiB: new(5520.0)},
			verify: func(t *testing.T, s *BenchmarkTUIState) {
				assert.Equal(t, domain.StatusSuccess, s.PromptJobs[0].Status)
				assert.Equal(t, domain.PhaseDone, s.PromptJobs[0].Phase)
				assert.InDelta(t, 2.5, *s.PromptJobs[0].DurationSeconds, 0.0001)
				assert.InDelta(t, 78.0, *s.PromptJobs[0].GenTokS, 0.0001)
				assert.InDelta(t, 812.0, *s.PromptJobs[0].PromptTokS, 0.0001)
				assert.InDelta(t, 5520.0, *s.PromptJobs[0].MinVRAMMiB, 0.0001)
			},
		},
		"does not overwrite metrics with nil": {
			setup: func(s *BenchmarkTUIState) {
				s.UpsertPrompt(PromptJobView{Profile: "code", Status: domain.StatusSuccess, DurationSeconds: new(2.5)})
			},
			update: PromptJobView{Profile: "code", Status: domain.StatusRunning},
			verify: func(t *testing.T, s *BenchmarkTUIState) {
				assert.NotNil(t, s.PromptJobs[0].DurationSeconds)
				assert.InDelta(t, 2.5, *s.PromptJobs[0].DurationSeconds, 0.0001)
			},
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			state := &BenchmarkTUIState{}
			if tc.setup != nil {
				tc.setup(state)
			}
			state.UpsertPrompt(tc.update)
			tc.verify(t, state)
		})
	}
}

func TestPromptJobView(t *testing.T) {
	tests := map[string]struct {
		job    PromptJobView
		verify func(*testing.T, PromptJobView)
	}{
		"running prefill has prompt tok s and no gen tok s": {
			job: PromptJobView{Profile: "code", Status: domain.StatusRunning, Phase: domain.PhasePrefill,
				DurationSeconds: new(0.75), PromptTokS: new(812.0), GenTokS: nil},
			verify: func(t *testing.T, j PromptJobView) {
				assert.Equal(t, domain.StatusRunning, j.Status)
				assert.Equal(t, domain.PhasePrefill, j.Phase)
				assert.NotNil(t, j.PromptTokS)
				assert.InDelta(t, 812.0, *j.PromptTokS, 0.0001)
				assert.Nil(t, j.GenTokS)
			},
		},
		"running generating has gen tok s": {
			job: PromptJobView{Profile: "code", Status: domain.StatusRunning, Phase: domain.PhaseGenerating,
				DurationSeconds: new(2.1), GenTokS: new(63.2)},
			verify: func(t *testing.T, j PromptJobView) {
				assert.Equal(t, domain.PhaseGenerating, j.Phase)
				assert.NotNil(t, j.GenTokS)
				assert.InDelta(t, 63.2, *j.GenTokS, 0.0001)
			},
		},
		"pending uses unavailable values": {
			job: PromptJobView{Profile: "qa", Status: domain.StatusPending, Phase: "-"},
			verify: func(t *testing.T, j PromptJobView) {
				assert.Equal(t, domain.StatusPending, j.Status)
				assert.Equal(t, domain.Phase("-"), j.Phase)
				assert.Nil(t, j.DurationSeconds)
				assert.Nil(t, j.GenTokS)
				assert.Nil(t, j.PromptTokS)
				assert.Nil(t, j.MinVRAMMiB)
			},
		},
		"timeout maps correctly": {
			job: PromptJobView{Profile: "long", Status: domain.StatusTimeout, Phase: domain.PhaseTimeout, DurationSeconds: new(30.0)},
			verify: func(t *testing.T, j PromptJobView) {
				assert.Equal(t, domain.StatusTimeout, j.Status)
				assert.Equal(t, domain.PhaseTimeout, j.Phase)
			},
		},
		"oom maps correctly": {
			job: PromptJobView{Profile: "big", Status: domain.StatusOOM, Phase: domain.PhaseOOM, DurationSeconds: new(5.1)},
			verify: func(t *testing.T, j PromptJobView) {
				assert.Equal(t, domain.StatusOOM, j.Status)
				assert.Equal(t, domain.PhaseOOM, j.Phase)
			},
		},
		"failed maps correctly": {
			job: PromptJobView{Profile: "doc", Status: domain.StatusFailed, Phase: domain.PhaseFailed, DurationSeconds: new(0.42)},
			verify: func(t *testing.T, j PromptJobView) {
				assert.Equal(t, domain.StatusFailed, j.Status)
				assert.Equal(t, domain.PhaseFailed, j.Phase)
			},
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			tc.verify(t, tc.job)
		})
	}
}

func TestBenchmarkTUIState(t *testing.T) {
	t.Run("defaults are zero valued", func(t *testing.T) {
		s := BenchmarkTUIState{}
		assert.Equal(t, "", s.RunID)
		assert.Equal(t, "", s.ModelName)
		assert.Equal(t, 0.0, s.ElapsedSeconds)
		assert.Nil(t, s.CurrentServer)
		assert.Nil(t, s.PromptJobs)
	})

	t.Run("state update applies transformation", func(t *testing.T) {
		s := BenchmarkTUIState{RunID: "before"}
		StateUpdate{Apply: func(s *BenchmarkTUIState) { s.RunID = "after" }}.Apply(&s)
		assert.Equal(t, "after", s.RunID)
	})

	t.Run("progress counts terminal jobs and completed servers", func(t *testing.T) {
		s := BenchmarkTUIState{Progress: ProgressState{
			ServersCompleted: 3, ServersTotal: 5,
			JobsCompleted: 42, JobsTotal: 50,
		}}
		assert.Equal(t, 3, s.Progress.ServersCompleted)
		assert.Equal(t, 5, s.Progress.ServersTotal)
		assert.Equal(t, 42, s.Progress.JobsCompleted)
		assert.Equal(t, 50, s.Progress.JobsTotal)
	})
}
