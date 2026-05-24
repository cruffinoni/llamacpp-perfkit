package sim

// Phase represents a stage in a prompt's lifecycle for display.
type Phase string

const (
	PhasePending    Phase = "pending"
	PhaseStarting   Phase = "starting"
	PhasePrefill    Phase = "prefill"
	PhaseGenerating Phase = "generating"
	PhaseDone       Phase = "done"
	PhaseTimeout    Phase = "timeout"
	PhaseOOM        Phase = "oom"
	PhaseFailed     Phase = "failed"
)

// Step describes one advancement step for a scripted prompt. Each step
// specifies how many sim-ticks it lasts, which phase/status to display, and
// what metrics to show.
type Step struct {
	Phase       Phase
	DurationSec float64
	GenTokS     *float64
	PromptTokS  *float64
	MinVRAMMiB  *float64
	TickCount   int
}

// Prompt defines one scripted prompt profile.
type Prompt struct {
	Profile string
	Steps   []Step
}

// Server defines one server configuration with its prompts.
type Server struct {
	ID          string
	ContextSize int
	KVType      string
	NCPUMOE     int
	SpecType    string
	BatchSize   int
	UBatchSize  int
	Prompts     []Prompt
}

// Scenario is a complete scripted benchmark scenario.
type Scenario struct {
	ModelName string
	Configs   []Server
}
