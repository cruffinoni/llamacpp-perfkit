package sim

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/viewmodel"
)

func TestAdvance(t *testing.T) {
	tests := map[string]struct {
		scenario   Scenario
		loop       bool
		totalTicks int
		verify     func(*testing.T, *Simulator, int)
	}{
		"through prompt phases ending in done state": {
			scenario: Scenario{
				ModelName: "test",
				Configs: []Server{{
					ID: "srv-1", Prompts: []Prompt{{
						Profile: "test_prompt",
						Steps: []Step{
							{Phase: PhaseStarting, TickCount: 2},
							{Phase: PhasePrefill, TickCount: 3},
							{Phase: PhaseGenerating, TickCount: 5},
							{Phase: PhaseDone, TickCount: 1},
						},
					}},
				}},
			},
			totalTicks: 11,
			verify: func(t *testing.T, s *Simulator, _ int) {
				assert.Equal(t, 1, s.jobsDone)
				assert.Equal(t, 1, s.serversDone)
				assert.Equal(t, stateDone, s.state)
			},
		},
		"through multiple servers": {
			scenario: Scenario{
				ModelName: "test",
				Configs: []Server{
					{ID: "srv-1", Prompts: []Prompt{
						{Profile: "p1", Steps: []Step{{Phase: PhaseDone, TickCount: 1}}},
						{Profile: "p2", Steps: []Step{{Phase: PhaseDone, TickCount: 1}}},
					}},
					{ID: "srv-2", Prompts: []Prompt{
						{Profile: "p3", Steps: []Step{{Phase: PhaseDone, TickCount: 1}}},
					}},
				},
			},
			totalTicks: 3,
			verify: func(t *testing.T, s *Simulator, _ int) {
				assert.Equal(t, 3, s.jobsDone)
				assert.Equal(t, 2, s.serversDone)
				assert.Equal(t, stateDone, s.state)
			},
		},
		"loop restarts from beginning": {
			scenario: Scenario{
				ModelName: "test",
				Configs: []Server{{
					ID: "srv-1", Prompts: []Prompt{
						{Profile: "p1", Steps: []Step{{Phase: PhaseDone, TickCount: 1}}},
					},
				}},
			},
			loop:       true,
			totalTicks: 1,
			verify: func(t *testing.T, s *Simulator, _ int) {
				assert.Equal(t, stateRunning, s.state)
				assert.Equal(t, 0, s.serverIdx)
				assert.Equal(t, 0, s.jobsDone)
			},
		},
		"mixed scenario completes within max ticks": {
			scenario:   MixedScenario(),
			totalTicks: 1000,
			verify: func(t *testing.T, s *Simulator, totalTicks int) {
				assert.Equal(t, stateDone, s.state)
				assert.Equal(t, 3, s.serversDone)
				assert.Equal(t, 13, s.jobsDone)
				assert.GreaterOrEqual(t, totalTicks, 50)
			},
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			s := New(tc.scenario, tc.loop)
			actualTicks := 0
			for s.Advance() {
				actualTicks++
				if actualTicks > tc.totalTicks {
					break
				}
			}
			tc.verify(t, s, actualTicks)
		})
	}
}

func TestPauseResume(t *testing.T) {
	s := New(Scenario{
		ModelName: "test",
		Configs: []Server{{
			ID: "srv-1", Prompts: []Prompt{{
				Profile: "p1", Steps: []Step{{Phase: PhaseStarting, TickCount: 10}},
			}},
		}},
	}, false)

	s.Advance()
	s.Advance()
	elapsedBefore := s.elapsed
	tickCountBefore := s.tickCount

	s.Pause()
	assert.Equal(t, statePaused, s.state)

	s.Advance()
	s.Advance()
	assert.Equal(t, elapsedBefore, s.elapsed, "elapsed should not change while paused")
	assert.Equal(t, tickCountBefore, s.tickCount, "tickCount should not change while paused")

	s.Pause()
	assert.Equal(t, stateRunning, s.state, "should resume after second pause")

	s.Advance()
	assert.Greater(t, s.elapsed, elapsedBefore, "elapsed should advance after resume")
}

func TestReset(t *testing.T) {
	s := New(Scenario{
		ModelName: "test",
		Configs: []Server{{
			ID: "srv-1", Prompts: []Prompt{
				{Profile: "p1", Steps: []Step{{Phase: PhaseDone, TickCount: 1}}},
				{Profile: "p2", Steps: []Step{{Phase: PhaseDone, TickCount: 1}}},
			},
		}},
	}, false)

	s.Advance()
	s.Advance()
	assert.Equal(t, 2, s.jobsDone)

	s.Reset()

	tests := map[string]struct {
		got  any
		want any
	}{
		"state":       {got: s.state, want: stateRunning},
		"serverIdx":   {got: s.serverIdx, want: 0},
		"promptIdx":   {got: s.promptIdx, want: 0},
		"stepIdx":     {got: s.stepIdx, want: 0},
		"stepTicks":   {got: s.stepTicks, want: 0},
		"jobsDone":    {got: s.jobsDone, want: 0},
		"serversDone": {got: s.serversDone, want: 0},
		"elapsed":     {got: s.elapsed, want: float64(0)},
		"tickCount":   {got: s.tickCount, want: 0},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.want, tc.got)
		})
	}
}

func TestBuildUpdate(t *testing.T) {
	tests := map[string]struct {
		scenario Scenario
		advance  int
		verify   func(*testing.T, viewmodel.BenchmarkTUIState)
	}{
		"progress counts from multi server scenario": {
			scenario: Scenario{
				ModelName: "test-model",
				Configs: []Server{
					{ID: "srv-1", Prompts: []Prompt{
						{Profile: "p1", Steps: []Step{{Phase: PhaseDone, TickCount: 1}}},
						{Profile: "p2", Steps: []Step{{Phase: PhaseDone, TickCount: 1}}},
					}},
					{ID: "srv-2", Prompts: []Prompt{
						{Profile: "p3", Steps: []Step{{Phase: PhaseDone, TickCount: 1}}},
					}},
				},
			},
			advance: 1,
			verify: func(t *testing.T, s viewmodel.BenchmarkTUIState) {
				require.NotNil(t, s.CurrentServer)
				assert.Equal(t, "srv-1", s.CurrentServer.ID)
				assert.Equal(t, 2, s.Progress.ServersTotal)
				assert.Equal(t, 3, s.Progress.JobsTotal)
				assert.Equal(t, 1, s.Progress.JobsCompleted)
				assert.Equal(t, 2, s.Progress.CurrentPrompt)
				assert.Len(t, s.PromptJobs, 2)
			},
		},
		"running prompt has correct phase values": {
			scenario: Scenario{
				ModelName: "test-model",
				Configs: []Server{{
					ID: "srv-1", ContextSize: 4096, KVType: "q8_0",
					NCPUMOE: 18, SpecType: "none", BatchSize: 512, UBatchSize: 512,
					Prompts: []Prompt{{
						Profile: "code_python",
						Steps: []Step{
							{Phase: PhasePrefill, DurationSec: 0.75, PromptTokS: new(812.0), TickCount: 3},
							{Phase: PhaseGenerating, DurationSec: 2.1, GenTokS: new(78.0), TickCount: 5},
							{Phase: PhaseDone, DurationSec: 2.7, TickCount: 1},
						},
					}},
				}},
			},
			advance: 1,
			verify: func(t *testing.T, s viewmodel.BenchmarkTUIState) {
				require.Len(t, s.PromptJobs, 1)
				job := s.PromptJobs[0]
				assert.Equal(t, "running", job.Status)
				assert.Equal(t, "prefill", job.Phase)
				assert.NotNil(t, job.PromptTokS)
				assert.InDelta(t, 812.0, *job.PromptTokS, 0.0001)
			},
		},
		"done state shows complete lifecycle": {
			scenario: Scenario{
				ModelName: "test",
				Configs: []Server{{
					ID: "srv-1", Prompts: []Prompt{
						{Profile: "p1", Steps: []Step{{Phase: PhaseDone, TickCount: 1}}},
					},
				}},
			},
			advance: 1,
			verify: func(t *testing.T, s viewmodel.BenchmarkTUIState) {
				assert.Equal(t, "complete", s.LifecycleState)
				assert.Equal(t, 1, s.Progress.ServersCompleted)
				assert.Equal(t, 1, s.Progress.JobsCompleted)
			},
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			sim := New(tc.scenario, false)
			for i := 0; i < tc.advance; i++ {
				sim.Advance()
			}
			update := sim.BuildUpdate()
			var state viewmodel.BenchmarkTUIState
			update.Apply(&state)
			tc.verify(t, state)
		})
	}
}

func TestLifecycleStates(t *testing.T) {
	s := Scenario{
		ModelName: "test",
		Configs: []Server{{
			ID: "srv-1", Prompts: []Prompt{
				{Profile: "p1", Steps: []Step{{Phase: PhaseDone, TickCount: 1}}},
			},
		}},
	}

	tests := map[string]struct {
		prepare func(*Simulator)
		want    string
	}{
		"running state": {
			prepare: func(sim *Simulator) {},
			want:    "running",
		},
		"paused state": {
			prepare: func(sim *Simulator) { sim.Pause() },
			want:    "paused",
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			sim := New(s, false)
			tc.prepare(sim)
			update := sim.BuildUpdate()
			var state viewmodel.BenchmarkTUIState
			update.Apply(&state)
			assert.Contains(t, state.LifecycleState, tc.want)
		})
	}
}

func TestInitialState(t *testing.T) {
	s := New(MixedScenario(), false)
	state := s.InitialState()

	assert.Equal(t, 3, state.Progress.ServersTotal)
	assert.Equal(t, 13, state.Progress.JobsTotal)
	assert.Equal(t, 0, state.Progress.ServersCompleted)
	assert.Equal(t, 0, state.Progress.JobsCompleted)
	assert.Equal(t, "simulation", state.RunID)
	assert.Equal(t, "Simulation starting...", state.StatusMessage)
}

func TestBenchmarkFunc(t *testing.T) {
	t.Run("exits on done", func(t *testing.T) {
		scenario := Scenario{
			ModelName: "test",
			Configs: []Server{{
				ID: "srv-1", Prompts: []Prompt{
					{Profile: "p1", Steps: []Step{{Phase: PhaseDone, TickCount: 1}}},
				},
			}},
		}
		s := New(scenario, false)
		ctrl := make(Controller, 8)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		updates := make(chan viewmodel.StateUpdate, 128)

		errCh := make(chan error, 1)
		go func() { errCh <- s.BenchmarkFunc(ctrl)(ctx, updates) }()
		go func() {
			for range updates {
			}
		}()

		select {
		case err := <-errCh:
			assert.NoError(t, err)
		case <-time.After(5 * time.Second):
			t.Fatal("BenchmarkFunc did not exit within 5 seconds")
		}
	})

	t.Run("responds to control actions", func(t *testing.T) {
		scenario := Scenario{
			ModelName: "test",
			Configs: []Server{{
				ID: "srv-1", Prompts: []Prompt{
					{Profile: "p1", Steps: []Step{{Phase: PhaseDone, TickCount: 2}}},
				},
			}},
		}
		sim := New(scenario, false)
		ctrl := make(Controller, 4)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		updates := make(chan viewmodel.StateUpdate, 32)

		go sim.BenchmarkFunc(ctrl)(ctx, updates)
		<-updates // drain initial
		ctrl <- TogglePause
		<-updates
		ctrl <- Reset
		<-updates
		cancel()
	})
}
