package viewmodel

// BuildInfoView holds build metadata displayed in the benchmark header.
type BuildInfoView struct {
	CommitShort string
	Branch      string
	Backend     string
}

// ProgressState tracks benchmark completion counts across servers and jobs.
type ProgressState struct {
	ServersCompleted   int
	ServersTotal       int
	JobsCompleted      int
	JobsTotal          int
	CurrentPrompt      int
	CurrentPromptTotal int
}

// CurrentServerView describes the currently running server configuration.
type CurrentServerView struct {
	ID          string
	ContextSize int
	KVType      string
	NCPUMOE     int
	SpecType    string
	BatchSize   int
	UBatchSize  int
}

// PromptJobView represents a single prompt profile's execution status.
type PromptJobView struct {
	Profile         string
	Status          string
	Phase           string
	DurationSeconds *float64
	GenTokS         *float64
	PromptTokS      *float64
	MinVRAMMiB      *float64
}

// BenchmarkTUIState is the top-level application state for the TUI.
type BenchmarkTUIState struct {
	RunID               string
	BuildInfo           BuildInfoView
	ModelName           string
	Progress            ProgressState
	ElapsedSeconds      float64
	ETASeconds          float64
	LifecycleState      string
	StatusMessage       string
	ActivePromptProfile string
	CurrentServer       *CurrentServerView
	PromptJobs          []PromptJobView
}

// StateUpdate carries a transformation to apply to the TUI state.
type StateUpdate struct {
	Apply func(*BenchmarkTUIState)
}

// UpsertPrompt inserts or updates a prompt job entry in the state slice.
func (s *BenchmarkTUIState) UpsertPrompt(job PromptJobView) {
	for i := range s.PromptJobs {
		if s.PromptJobs[i].Profile == job.Profile {
			if job.Status != "" {
				s.PromptJobs[i].Status = job.Status
			}
			if job.Phase != "" {
				s.PromptJobs[i].Phase = job.Phase
			}
			if job.DurationSeconds != nil {
				s.PromptJobs[i].DurationSeconds = job.DurationSeconds
			}
			if job.GenTokS != nil {
				s.PromptJobs[i].GenTokS = job.GenTokS
			}
			if job.PromptTokS != nil {
				s.PromptJobs[i].PromptTokS = job.PromptTokS
			}
			if job.MinVRAMMiB != nil {
				s.PromptJobs[i].MinVRAMMiB = job.MinVRAMMiB
			}
			return
		}
	}
	s.PromptJobs = append(s.PromptJobs, job)
}
