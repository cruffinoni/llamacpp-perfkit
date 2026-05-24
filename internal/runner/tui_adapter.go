package runner

import (
	"time"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/viewmodel"
)

type tuiAdapter struct {
	updates chan<- viewmodel.StateUpdate
	started time.Time
}

func newTUIAdapter(updates chan<- viewmodel.StateUpdate, started time.Time) tuiAdapter {
	return tuiAdapter{updates: updates, started: started}
}

// BeginGroup initialises the TUI state for a new server group.
func (a tuiAdapter) BeginGroup(server serverExecution, group []domain.PlannedRun, groupIndex int, totalGroups int) {
	job := group[0].Job
	a.send(func(s *viewmodel.BenchmarkTUIState) {
		s.Progress.ServersTotal = totalGroups
		s.Progress.CurrentPromptTotal = len(group)
		s.CurrentServer = &viewmodel.CurrentServerView{
			ID:          server.ID,
			ContextSize: job.ServerConfig.ContextSize,
			KVType:      job.ServerConfig.KVType,
			NCPUMOE:     intValue(job.ServerConfig.NCPUMOE),
			SpecType:    or(job.ServerConfig.SpecType, "none"),
			BatchSize:   intValue(job.ServerConfig.BatchSize),
			UBatchSize:  intValue(job.ServerConfig.UBatchSize),
		}
		s.PromptJobs = nil
		for _, item := range group {
			s.UpsertPrompt(viewmodel.PromptJobView{Profile: item.Job.PromptProfile.Name, Status: domain.StatusPending, Phase: "-"})
		}
		s.LifecycleState = "starting server"
		s.StatusMessage = "Launching " + server.ID + "."
	})
	_ = groupIndex
}

// BeginPrompt updates the TUI state when a prompt evaluation begins.
func (a tuiAdapter) BeginPrompt(item domain.PlannedRun, promptIndex int) {
	a.send(func(s *viewmodel.BenchmarkTUIState) {
		s.Progress.CurrentPrompt = promptIndex + 1
		s.LifecycleState = "running prompt"
		s.StatusMessage = "Running " + item.Job.PromptProfile.Name + "."
		s.UpsertPrompt(viewmodel.PromptJobView{Profile: item.Job.PromptProfile.Name, Status: domain.StatusRunning, Phase: domain.PhaseStarting})
	})
}

// CompletePrompt updates the TUI state when a prompt evaluation completes.
func (a tuiAdapter) CompletePrompt(item domain.PlannedRun, result requestResult, jobsCompleted int) {
	a.send(func(s *viewmodel.BenchmarkTUIState) {
		s.Progress.JobsCompleted = jobsCompleted
		s.ElapsedSeconds = time.Since(a.started).Seconds()
		s.UpsertPrompt(viewmodel.PromptJobView{
			Profile:         item.Job.PromptProfile.Name,
			Status:          result.Status,
			Phase:           phaseForStatus(result.Status),
			DurationSeconds: &result.Duration,
			GenTokS:         result.Loaded.LlamaSummary.GenerationTokS,
			PromptTokS:      result.Loaded.LlamaSummary.PromptEvalTokS,
			MinVRAMMiB:      result.Loaded.SystemSummary.MinVRAMFreeMiB,
		})
		s.StatusMessage = item.Job.PromptProfile.Name + ": " + string(result.Status)
	})
}

// StartupFailedPrompt records a server startup failure in the TUI state.
func (a tuiAdapter) StartupFailedPrompt(item domain.PlannedRun, errText string, duration float64, jobsCompleted int) {
	a.send(func(s *viewmodel.BenchmarkTUIState) {
		s.Progress.JobsCompleted = jobsCompleted
		s.ElapsedSeconds = time.Since(a.started).Seconds()
		s.UpsertPrompt(viewmodel.PromptJobView{Profile: item.Job.PromptProfile.Name, Status: domain.StatusFailed, Phase: domain.PhaseFailed, DurationSeconds: &duration})
		s.StatusMessage = errText
	})
}

// CompleteGroup updates the TUI state when a server group finishes.
func (a tuiAdapter) CompleteGroup(groupIndex int, totalGroups int) {
	a.send(func(s *viewmodel.BenchmarkTUIState) {
		s.Progress.ServersCompleted = groupIndex
		s.ElapsedSeconds = time.Since(a.started).Seconds()
		s.ETASeconds = estimateETA(s.Progress.ServersCompleted, totalGroups, s.ElapsedSeconds)
	})
}

// CompleteBenchmark marks the benchmark as complete in the TUI state.
func (a tuiAdapter) CompleteBenchmark(totalGroups int) {
	a.send(func(s *viewmodel.BenchmarkTUIState) {
		s.LifecycleState = "complete"
		s.StatusMessage = "No runnable jobs remain."
		s.ElapsedSeconds = time.Since(a.started).Seconds()
		s.Progress.ServersCompleted = totalGroups
	})
}

func (a tuiAdapter) send(apply func(*viewmodel.BenchmarkTUIState)) {
	if a.updates == nil {
		return
	}
	select {
	case a.updates <- viewmodel.StateUpdate{Apply: apply}:
	default:
		a.updates <- viewmodel.StateUpdate{Apply: apply}
	}
}
