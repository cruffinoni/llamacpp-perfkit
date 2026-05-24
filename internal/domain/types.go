package domain

import (
	"encoding/json"
	"fmt"
)

// RunStatus represents the status of a benchmark run.
type RunStatus string

const (
	StatusSuccess     RunStatus = "success"
	StatusFailed      RunStatus = "failed"
	StatusOOM         RunStatus = "oom"
	StatusTimeout     RunStatus = "timeout"
	StatusUnsupported RunStatus = "unsupported"
	StatusUnknown     RunStatus = "unknown"
)

// RunStatusInfo holds the detailed status flags for a benchmark run.
type RunStatusInfo struct {
	Success     bool    `json:"success"`
	Failed      bool    `json:"failed,omitempty"`
	Timeout     bool    `json:"timeout"`
	OOM         bool    `json:"oom"`
	Unsupported bool    `json:"unsupported,omitempty"`
	Error       *string `json:"error"`
}

// NewRunStatusInfo creates a RunStatusInfo from a status string and optional error text.
func NewRunStatusInfo(status RunStatus, errText string) RunStatusInfo {
	var errPtr *string
	if errText != "" {
		errPtr = &errText
	}
	return RunStatusInfo{
		Success:     status == StatusSuccess,
		Failed:      status == StatusFailed,
		Timeout:     status == StatusTimeout,
		OOM:         status == StatusOOM,
		Unsupported: status == StatusUnsupported,
		Error:       errPtr,
	}
}

// Kind returns the dominant RunStatus from the status flags.
func (s RunStatusInfo) Kind() RunStatus {
	switch {
	case s.Success:
		return StatusSuccess
	case s.Failed:
		return StatusFailed
	case s.Timeout:
		return StatusTimeout
	case s.OOM:
		return StatusOOM
	case s.Unsupported:
		return StatusUnsupported
	case s.Error != nil && *s.Error != "":
		return StatusFailed
	default:
		return StatusUnknown
	}
}

// UnmarshalJSON implements the json.Unmarshaler interface for RunStatusInfo.
func (s *RunStatusInfo) UnmarshalJSON(data []byte) error {
	var status string
	if err := json.Unmarshal(data, &status); err == nil {
		*s = NewRunStatusInfo(RunStatus(status), "")
		return nil
	}
	type alias RunStatusInfo
	var out alias
	if err := json.Unmarshal(data, &out); err != nil {
		return err
	}
	*s = RunStatusInfo(out)
	return nil
}

// ServerConfig holds the llama.cpp server configuration parameters.
type ServerConfig struct {
	Model           string   `json:"model,omitempty"`
	ContextSize     int      `json:"ctx,omitempty"`
	KVType          string   `json:"kv_type,omitempty"`
	NCPUMOE         *int     `json:"n_cpu_moe,omitempty"`
	MTP             *bool    `json:"mtp,omitempty"`
	SpecType        string   `json:"spec_type,omitempty"`
	SpecDraftNMax   *int     `json:"spec_draft_n_max,omitempty"`
	SpecDraftPMin   *float64 `json:"spec_draft_p_min,omitempty"`
	BatchSize       *int     `json:"batch_size,omitempty"`
	UBatchSize      *int     `json:"ubatch_size,omitempty"`
	Parallel        *int     `json:"parallel,omitempty"`
	NGPULayers      *int     `json:"n_gpu_layers,omitempty"`
	SplitMode       string   `json:"split_mode,omitempty"`
	Host            string   `json:"-"`
	Port            int      `json:"-"`
	StartupTimeout  int      `json:"-"`
	ShutdownTimeout int      `json:"-"`
}

// UnmarshalJSON implements the json.Unmarshaler interface for ServerConfig,
// supporting the legacy context_size field.
func (s *ServerConfig) UnmarshalJSON(data []byte) error {
	type serverConfig ServerConfig
	var raw struct {
		serverConfig
		ContextSizeLegacy *int `json:"context_size"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*s = ServerConfig(raw.serverConfig)
	if s.ContextSize == 0 && raw.ContextSizeLegacy != nil {
		s.ContextSize = *raw.ContextSizeLegacy
	}
	return nil
}

// PromptProfile defines a named prompt profile with its file path and index.
type PromptProfile struct {
	Name  string `json:"name"`
	File  string `json:"file"`
	Index int    `json:"index,omitempty"`
}

// BenchmarkJob describes a single benchmark job to execute.
type BenchmarkJob struct {
	PromptProfile PromptProfile `json:"prompt_profile"`
	PromptFile    string        `json:"prompt_file"`
	ServerConfig  ServerConfig  `json:"server_config"`
	GenerationTok int           `json:"generation_tokens"`
	Seed          *int          `json:"seed,omitempty"`
	CachePrompt   bool          `json:"cache_prompt"`
	Endpoint      string        `json:"endpoint"`
	ConfigHash    string        `json:"config_hash"`
}

// PlanAction represents the planned action for a run (run, reuse, or skip).
type PlanAction string

const (
	ActionRun   PlanAction = "run"
	ActionReuse PlanAction = "reuse"
	ActionSkip  PlanAction = "skip"
)

// PlannedRun describes a single planned run within a benchmark plan.
type PlannedRun struct {
	RunID       string       `json:"run_id"`
	ConfigHash  string       `json:"config_hash"`
	Action      PlanAction   `json:"action"`
	ActionNote  string       `json:"action_reason,omitempty"`
	RiskLevel   string       `json:"risk_level"`
	Job         BenchmarkJob `json:"job"`
	ServerIndex int          `json:"server_index,omitempty"`
}

// BenchmarkPlan holds the complete benchmark plan with all planned runs.
type BenchmarkPlan struct {
	ReuseExistingResults bool                   `json:"reuse_existing_results"`
	CandidateCount       int                    `json:"candidate_count"`
	SelectedCount        int                    `json:"selected_count"`
	EstimatedRuns        int                    `json:"estimated_runs"`
	Skipped              []SkipReason           `json:"skipped,omitempty"`
	Notes                []string               `json:"notes,omitempty"`
	Planned              []PlannedRun           `json:"planned,omitempty"`
	Metadata             map[string]interface{} `json:"metadata,omitempty"`
}

// SkipReason describes why a particular configuration was skipped.
type SkipReason struct {
	Dimension string `json:"dimension,omitempty"`
	Flag      string `json:"flag,omitempty"`
	Value     any    `json:"value,omitempty"`
	Reason    string `json:"reason"`
}

// BuildInfo contains build metadata for a llama.cpp binary.
type BuildInfo struct {
	CommitShort string `json:"commit_short,omitempty"`
	Commit      string `json:"commit,omitempty"`
	Branch      string `json:"branch,omitempty"`
	Backend     string `json:"backend,omitempty"`
	Repo        string `json:"repo,omitempty"`
	Error       string `json:"error,omitempty"`
}

// RunSummary summarizes a completed benchmark run.
type RunSummary struct {
	RunID          string         `json:"run_id"`
	BatchID        string         `json:"batch_id,omitempty"`
	CreatedAt      string         `json:"created_at"`
	Model          string         `json:"model,omitempty"`
	PromptProfile  string         `json:"prompt_profile"`
	ServerConfig   ServerConfig   `json:"server_config"`
	Status         RunStatusInfo  `json:"status"`
	ConfigHash     string         `json:"config_hash,omitempty"`
	DurationSec    float64        `json:"duration_seconds,omitempty"`
	Request        map[string]any `json:"request,omitempty"`
	Response       map[string]any `json:"response,omitempty"`
	Parsed         map[string]any `json:"parsed,omitempty"`
	Command        []string       `json:"command,omitempty"`
	CommandShell   string         `json:"command_shell,omitempty"`
	LlamaCpp       BuildInfo      `json:"llama_cpp,omitempty"`
	Backend        string         `json:"backend,omitempty"`
	RawLogPath     string         `json:"raw_log_path,omitempty"`
	ServerLogPath  string         `json:"server_log_path,omitempty"`
	PromptFile     string         `json:"prompt_file,omitempty"`
	GenerationToks int            `json:"generation_tokens,omitempty"`
	Seed           *int           `json:"seed,omitempty"`
}

// SystemMetricSample represents a single system metric measurement.
type SystemMetricSample struct {
	Time       string   `json:"time"`
	GPUPowerW  *float64 `json:"gpu_power_w,omitempty"`
	GPUTempC   *float64 `json:"gpu_temp_c,omitempty"`
	GPUUtilPct *float64 `json:"gpu_util_pct,omitempty"`
	VRAMFree   *float64 `json:"vram_free_mib,omitempty"`
	VRAMUsed   *float64 `json:"vram_used_mib,omitempty"`
	RAMFree    *float64 `json:"ram_free_mib,omitempty"`
	RAMUsed    *float64 `json:"ram_used_mib,omitempty"`
}

// LlamaCppMetricSample represents a single llama.cpp performance metric measurement.
type LlamaCppMetricSample struct {
	Time             string   `json:"time"`
	PromptTokens     *float64 `json:"prompt_tokens,omitempty"`
	GeneratedTokens  *float64 `json:"generated_tokens,omitempty"`
	PromptEvalTokens *float64 `json:"prompt_eval_tokens,omitempty"`
	EvalTokens       *float64 `json:"eval_tokens,omitempty"`
	PromptEvalTokS   *float64 `json:"prompt_eval_tok_s,omitempty"`
	GenerationTokS   *float64 `json:"generation_tok_s,omitempty"`
	TotalTokens      *float64 `json:"total_tokens,omitempty"`
	SlotsIdle        *int     `json:"slots_idle,omitempty"`
	SlotsProcessing  *int     `json:"slots_processing,omitempty"`
	TTFTSeconds      *float64 `json:"ttft_seconds,omitempty"`
	TotalTimeSeconds *float64 `json:"total_time_seconds,omitempty"`
}

// LoadedRun holds the full data for a loaded run, including metrics and summaries.
type LoadedRun struct {
	Summary       RunSummary
	SystemMetrics []SystemMetricSample
	LlamaMetrics  []LlamaCppMetricSample
	SystemSummary SystemSummary
	LlamaSummary  LlamaSummary
	Directory     string
}

// SystemSummary summarizes system-level metrics collected during a run.
type SystemSummary struct {
	PeakVRAMMiB     *float64
	MinVRAMFreeMiB  *float64
	MeanVRAMFreeMiB *float64
	PeakRAMMiB      *float64
	MinRAMFreeMiB   *float64
	AvgGPUUtilPct   *float64
	PeakGPUPowerW   *float64
	PeakGPUTempC    *float64
}

// LlamaSummary summarizes llama.cpp performance metrics collected during a run.
type LlamaSummary struct {
	GenerationTokS  *float64
	PromptEvalTokS  *float64
	TokensGenerated *float64
	TokensPrompt    *float64
	TotalTimeSec    *float64
	TTFTSeconds     *float64
}

// DerefString safely dereferences a string pointer, returning an empty string for nil.
func DerefString(v *string) string {
	if v == nil {
		return ""
	}
	return *v
}

// IntValue returns the string representation of an integer pointer, or "-" if nil.
func IntValue(v *int) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%d", *v)
}
