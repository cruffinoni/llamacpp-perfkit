package viewmodel

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUpsertPrompt(t *testing.T) {
	tests := map[string]struct {
		setup  func(*BenchmarkTUIState)
		update PromptJobView
		verify func(*testing.T, *BenchmarkTUIState)
	}{
		"inserts new prompt": {
			setup:  nil,
			update: PromptJobView{Profile: "code", Status: "pending"},
			verify: func(t *testing.T, s *BenchmarkTUIState) {
				assert.Len(t, s.PromptJobs, 1)
				assert.Equal(t, "code", s.PromptJobs[0].Profile)
				assert.Equal(t, "pending", s.PromptJobs[0].Status)
			},
		},
		"preserves order when inserting multiple": {
			setup: func(s *BenchmarkTUIState) {
				s.UpsertPrompt(PromptJobView{Profile: "qa", Status: "pending", Phase: "-"})
			},
			update: PromptJobView{Profile: "code", Status: "pending", Phase: "-"},
			verify: func(t *testing.T, s *BenchmarkTUIState) {
				assert.Equal(t, []string{"qa", "code"}, []string{s.PromptJobs[0].Profile, s.PromptJobs[1].Profile})
			},
		},
		"updates existing prompt status and phase": {
			setup: func(s *BenchmarkTUIState) {
				s.UpsertPrompt(PromptJobView{Profile: "qa", Status: "pending", Phase: "-"})
			},
			update: PromptJobView{Profile: "qa", Status: "running", Phase: "generating"},
			verify: func(t *testing.T, s *BenchmarkTUIState) {
				assert.Equal(t, "running", s.PromptJobs[0].Status)
				assert.Equal(t, "generating", s.PromptJobs[0].Phase)
			},
		},
		"does not overwrite non-nil phase with empty string": {
			setup: func(s *BenchmarkTUIState) {
				s.UpsertPrompt(PromptJobView{Profile: "code", Status: "pending", Phase: "starting"})
			},
			update: PromptJobView{Profile: "code", Status: "running"},
			verify: func(t *testing.T, s *BenchmarkTUIState) {
				assert.Equal(t, "running", s.PromptJobs[0].Status)
				assert.Equal(t, "starting", s.PromptJobs[0].Phase)
			},
		},
		"updates all metrics on completion": {
			setup: func(s *BenchmarkTUIState) {
				s.UpsertPrompt(PromptJobView{Profile: "code", Status: "running"})
			},
			update: PromptJobView{Profile: "code", Status: "success", Phase: "done",
				DurationSeconds: new(2.5), GenTokS: new(78.0), PromptTokS: new(812.0), MinVRAMMiB: new(5520.0)},
			verify: func(t *testing.T, s *BenchmarkTUIState) {
				assert.Equal(t, "success", s.PromptJobs[0].Status)
				assert.Equal(t, "done", s.PromptJobs[0].Phase)
				assert.InDelta(t, 2.5, *s.PromptJobs[0].DurationSeconds, 0.0001)
				assert.InDelta(t, 78.0, *s.PromptJobs[0].GenTokS, 0.0001)
				assert.InDelta(t, 812.0, *s.PromptJobs[0].PromptTokS, 0.0001)
				assert.InDelta(t, 5520.0, *s.PromptJobs[0].MinVRAMMiB, 0.0001)
			},
		},
		"does not overwrite metrics with nil": {
			setup: func(s *BenchmarkTUIState) {
				s.UpsertPrompt(PromptJobView{Profile: "code", Status: "success", DurationSeconds: new(2.5)})
			},
			update: PromptJobView{Profile: "code", Status: "running"},
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
			job: PromptJobView{Profile: "code", Status: "running", Phase: "prefill",
				DurationSeconds: new(0.75), PromptTokS: new(812.0), GenTokS: nil},
			verify: func(t *testing.T, j PromptJobView) {
				assert.Equal(t, "running", j.Status)
				assert.Equal(t, "prefill", j.Phase)
				assert.NotNil(t, j.PromptTokS)
				assert.InDelta(t, 812.0, *j.PromptTokS, 0.0001)
				assert.Nil(t, j.GenTokS)
			},
		},
		"running generating has gen tok s": {
			job: PromptJobView{Profile: "code", Status: "running", Phase: "generating",
				DurationSeconds: new(2.1), GenTokS: new(63.2)},
			verify: func(t *testing.T, j PromptJobView) {
				assert.Equal(t, "generating", j.Phase)
				assert.NotNil(t, j.GenTokS)
				assert.InDelta(t, 63.2, *j.GenTokS, 0.0001)
			},
		},
		"pending uses unavailable values": {
			job: PromptJobView{Profile: "qa", Status: "pending", Phase: "-"},
			verify: func(t *testing.T, j PromptJobView) {
				assert.Equal(t, "pending", j.Status)
				assert.Equal(t, "-", j.Phase)
				assert.Nil(t, j.DurationSeconds)
				assert.Nil(t, j.GenTokS)
				assert.Nil(t, j.PromptTokS)
				assert.Nil(t, j.MinVRAMMiB)
			},
		},
		"timeout maps correctly": {
			job: PromptJobView{Profile: "long", Status: "timeout", Phase: "timeout", DurationSeconds: new(30.0)},
			verify: func(t *testing.T, j PromptJobView) {
				assert.Equal(t, "timeout", j.Status)
				assert.Equal(t, "timeout", j.Phase)
			},
		},
		"oom maps correctly": {
			job: PromptJobView{Profile: "big", Status: "oom", Phase: "oom", DurationSeconds: new(5.1)},
			verify: func(t *testing.T, j PromptJobView) {
				assert.Equal(t, "oom", j.Status)
				assert.Equal(t, "oom", j.Phase)
			},
		},
		"failed maps correctly": {
			job: PromptJobView{Profile: "doc", Status: "failed", Phase: "failed", DurationSeconds: new(0.42)},
			verify: func(t *testing.T, j PromptJobView) {
				assert.Equal(t, "failed", j.Status)
				assert.Equal(t, "failed", j.Phase)
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
