package sim

import (
	"context"
	"fmt"
	"time"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/viewmodel"
)

type simState int

const (
	stateRunning simState = iota
	statePaused
	stateDone
)

const defaultTickInterval = 250 * time.Millisecond

// Simulator drives a Scenario deterministically, sending StateUpdate closures
// through a channel at each tick.
type Simulator struct {
	scenario Scenario
	loop     bool

	state       simState
	tickCount   int
	elapsed     float64
	serverIdx   int
	promptIdx   int
	stepIdx     int
	stepTicks   int
	jobsDone    int
	serversDone int
}

// New creates a Simulator for the given scenario. If loop is true, the
// simulation restarts from the beginning when it completes.
func New(scenario Scenario, loop bool) *Simulator {
	return &Simulator{
		scenario: scenario,
		loop:     loop,
		state:    stateRunning,
	}
}

// Advance advances the simulation by one tick. It returns false when the
// simulation has finished and looping is disabled.
func (s *Simulator) Advance() bool {
	if s.state != stateRunning {
		return s.state != stateDone || s.loop
	}

	s.tickCount++
	s.elapsed += defaultTickInterval.Seconds()

	step := s.currentStep()
	s.stepTicks++

	if s.stepTicks < step.TickCount {
		return true
	}

	s.stepTicks = 0
	s.stepIdx++

	if s.stepIdx >= len(s.currentPrompt().Steps) {
		s.jobsDone++
		s.promptIdx++
		s.stepIdx = 0

		if s.promptIdx >= len(s.currentServer().Prompts) {
			s.serversDone++
			s.serverIdx++
			s.promptIdx = 0

			if s.serverIdx >= len(s.scenario.Configs) {
				if s.loop {
					s.reset()
				} else {
					s.state = stateDone
				}
				return s.loop
			}
		}
	}

	return true
}

// BuildUpdate produces a StateUpdate closure that fully populates a
// BenchmarkTUIState from the current simulator position.
func (s *Simulator) BuildUpdate() viewmodel.StateUpdate {
	return viewmodel.StateUpdate{
		Apply: func(vs *viewmodel.BenchmarkTUIState) {
			s.applyTo(vs)
		},
	}
}

func (s *Simulator) applyTo(vs *viewmodel.BenchmarkTUIState) {
	totalServers := len(s.scenario.Configs)
	totalJobs := 0
	for _, sv := range s.scenario.Configs {
		totalJobs += len(sv.Prompts)
	}

	vs.RunID = "simulation"
	vs.BuildInfo = viewmodel.BuildInfoView{
		CommitShort: "sim",
		Branch:      "dev",
		Backend:     "simulated",
	}
	vs.ModelName = s.scenario.ModelName

	serverIdx := s.serverIdx
	if serverIdx >= len(s.scenario.Configs) {
		serverIdx = len(s.scenario.Configs) - 1
	}
	promptIdx := s.promptIdx
	if s.state == stateDone {
		sv := s.scenario.Configs[serverIdx]
		promptIdx = len(sv.Prompts)
	}
	server := &s.scenario.Configs[serverIdx]
	currentPromptTotal := len(server.Prompts)

	vs.Progress = viewmodel.ProgressState{
		ServersCompleted:   s.serversDone,
		ServersTotal:       totalServers,
		JobsCompleted:      s.jobsDone,
		JobsTotal:          totalJobs,
		CurrentPrompt:      min(promptIdx+1, currentPromptTotal),
		CurrentPromptTotal: currentPromptTotal,
	}

	vs.ElapsedSeconds = s.elapsed
	vs.ETASeconds = s.computeETA()
	vs.LifecycleState = s.lifecycleString()
	vs.StatusMessage = s.statusMessage()
	vs.CurrentServer = &viewmodel.CurrentServerView{
		ID:          server.ID,
		ContextSize: server.ContextSize,
		KVType:      server.KVType,
		NCPUMOE:     server.NCPUMOE,
		SpecType:    server.SpecType,
		BatchSize:   server.BatchSize,
		UBatchSize:  server.UBatchSize,
	}

	vs.PromptJobs = make([]viewmodel.PromptJobView, 0, len(server.Prompts))
	stepIdx := s.stepIdx
	if s.state == stateDone {
		stepIdx = 0
	}
	for i, sp := range server.Prompts {
		pj := viewmodel.PromptJobView{Profile: sp.Profile}
		if i < promptIdx {
			last := sp.Steps[len(sp.Steps)-1]
			pj.Status = domain.RunStatus(last.Phase)
			pj.Phase = last.Phase
			pj.DurationSeconds = &last.DurationSec
			pj.GenTokS = last.GenTokS
			pj.PromptTokS = last.PromptTokS
			pj.MinVRAMMiB = last.MinVRAMMiB
		} else if i == promptIdx && s.state != stateDone {
			cs := sp.Steps[stepIdx]
			pj.Status = domain.StatusRunning
			pj.Phase = cs.Phase
			pj.DurationSeconds = &cs.DurationSec
			pj.GenTokS = cs.GenTokS
			pj.PromptTokS = cs.PromptTokS
			pj.MinVRAMMiB = cs.MinVRAMMiB
		} else {
			pj.Status = domain.StatusPending
			pj.Phase = "-"
		}
		vs.PromptJobs = append(vs.PromptJobs, pj)
	}
}

// InitialState returns a BenchmarkTUIState with scenario metadata populated
// for the first TUI frame before any ticks.
func (s *Simulator) InitialState() viewmodel.BenchmarkTUIState {
	totalJobs := 0
	for _, sv := range s.scenario.Configs {
		totalJobs += len(sv.Prompts)
	}
	return viewmodel.BenchmarkTUIState{
		RunID:     "simulation",
		BuildInfo: viewmodel.BuildInfoView{CommitShort: "sim", Branch: "dev", Backend: "simulated"},
		ModelName: s.scenario.ModelName,
		Progress: viewmodel.ProgressState{
			ServersTotal:       len(s.scenario.Configs),
			JobsTotal:          totalJobs,
			CurrentPromptTotal: len(s.scenario.Configs[0].Prompts),
		},
		StatusMessage: "Simulation starting...",
	}
}

// BenchmarkFunc returns a function matching the app.Run benchmark signature.
// It owns a ticker that advances the simulation and sends StateUpdate values.
func (s *Simulator) BenchmarkFunc(ctrl Controller) func(context.Context, chan<- viewmodel.StateUpdate) error {
	return func(ctx context.Context, updates chan<- viewmodel.StateUpdate) error {
		ticker := time.NewTicker(defaultTickInterval)
		defer ticker.Stop()

		updates <- s.BuildUpdate()

		for {
			select {
			case <-ctx.Done():
				return nil

			case action := <-ctrl:
				switch action {
				case TogglePause:
					s.Pause()
					updates <- s.BuildUpdate()
				case Reset:
					s.Reset()
					updates <- s.BuildUpdate()
				}

			case <-ticker.C:
				if s.state == statePaused {
					continue
				}
				if s.state == stateDone {
					return nil
				}
				if !s.Advance() {
					return nil
				}
				updates <- s.BuildUpdate()
			}
		}
	}
}

// Pause toggles the simulation between running and paused states.
func (s *Simulator) Pause() {
	if s.state == stateRunning {
		s.state = statePaused
	} else if s.state == statePaused {
		s.state = stateRunning
	}
}

// Reset restarts the simulation from the beginning.
func (s *Simulator) Reset() {
	s.reset()
}

func (s *Simulator) reset() {
	s.state = stateRunning
	s.tickCount = 0
	s.elapsed = 0
	s.serverIdx = 0
	s.promptIdx = 0
	s.stepIdx = 0
	s.stepTicks = 0
	s.jobsDone = 0
	s.serversDone = 0
}

func (s *Simulator) currentServer() *Server {
	return &s.scenario.Configs[s.serverIdx]
}

func (s *Simulator) currentPrompt() *Prompt {
	return &s.scenario.Configs[s.serverIdx].Prompts[s.promptIdx]
}

func (s *Simulator) currentStep() Step {
	return s.scenario.Configs[s.serverIdx].Prompts[s.promptIdx].Steps[s.stepIdx]
}

func (s *Simulator) computeETA() float64 {
	if s.serversDone <= 0 || s.serversDone >= len(s.scenario.Configs) {
		return 0
	}
	perServer := s.elapsed / float64(s.serversDone)
	remaining := len(s.scenario.Configs) - s.serversDone
	return perServer * float64(remaining)
}

func (s *Simulator) lifecycleString() string {
	switch s.state {
	case statePaused:
		return "paused"
	case stateDone:
		return "complete"
	default:
		return fmt.Sprintf("running server %d/%d prompt %d/%d",
			s.serverIdx+1, len(s.scenario.Configs),
			s.promptIdx+1, len(s.currentServer().Prompts))
	}
}

func (s *Simulator) statusMessage() string {
	switch s.state {
	case statePaused:
		return "SIMULATION PAUSED — press space to resume"
	case stateDone:
		return "Simulation complete"
	default:
		cv := s.currentServer()
		cp := cv.Prompts[s.promptIdx]
		cs := cp.Steps[s.stepIdx]
		return fmt.Sprintf("SIM: %s | %s | %s", cv.ID, cp.Profile, cs.Phase)
	}
}

func (s *Simulator) currentServerView() *viewmodel.CurrentServerView {
	sv := s.currentServer()
	return &viewmodel.CurrentServerView{
		ID:          sv.ID,
		ContextSize: sv.ContextSize,
		KVType:      sv.KVType,
		NCPUMOE:     sv.NCPUMOE,
		SpecType:    sv.SpecType,
		BatchSize:   sv.BatchSize,
		UBatchSize:  sv.UBatchSize,
	}
}
