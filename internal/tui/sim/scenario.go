package sim

import "github.com/cruffinoni/llamacpp-perfkit/internal/domain"

// Step describes one advancement step for a scripted prompt. Each step
// specifies how many sim-ticks it lasts, which phase/status to display, and
// what metrics to show.
type Step struct {
	Phase       domain.Phase
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
