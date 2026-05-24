package domain

import (
	"encoding/json"
	"fmt"
)

type RunStatus string

const (
	StatusSuccess     RunStatus = "success"
	StatusFailed      RunStatus = "failed"
	StatusOOM         RunStatus = "oom"
	StatusTimeout     RunStatus = "timeout"
	StatusUnsupported RunStatus = "unsupported"
	StatusUnknown     RunStatus = "unknown"
)

type RunStatusInfo struct {
	Success     bool    `json:"success"`
	Failed      bool    `json:"failed,omitempty"`
	Timeout     bool    `json:"timeout"`
	OOM         bool    `json:"oom"`
	Unsupported bool    `json:"unsupported,omitempty"`
	Error       *string `json:"error"`
}

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

type PromptProfile struct {
	Name  string `json:"name"`
	File  string `json:"file"`
	Index int    `json:"index,omitempty"`
}

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

type PlanAction string

const (
	ActionRun   PlanAction = "run"
	ActionReuse PlanAction = "reuse"
	ActionSkip  PlanAction = "skip"
)

type PlannedRun struct {
	RunID       string       `json:"run_id"`
	ConfigHash  string       `json:"config_hash"`
	Action      PlanAction   `json:"action"`
	ActionNote  string       `json:"action_reason,omitempty"`
	RiskLevel   string       `json:"risk_level"`
	Job         BenchmarkJob `json:"job"`
	ServerIndex int          `json:"server_index,omitempty"`
}

type BenchmarkPlan struct {
	Mode                 string                 `json:"mode"`
	MaxRuns              int                    `json:"max_runs"`
	ReuseExistingResults bool                   `json:"reuse_existing_results"`
	CandidateCount       int                    `json:"candidate_count"`
	SelectedCount        int                    `json:"selected_count"`
	EstimatedRuns        int                    `json:"estimated_runs"`
	MaxRunsCapped        bool                   `json:"max_runs_capped"`
	Skipped              []SkipReason           `json:"skipped,omitempty"`
	Notes                []string               `json:"notes,omitempty"`
	Planned              []PlannedRun           `json:"planned"`
	Metadata             map[string]interface{} `json:"metadata,omitempty"`
}

type SkipReason struct {
	Dimension string `json:"dimension,omitempty"`
	Flag      string `json:"flag,omitempty"`
	Value     any    `json:"value,omitempty"`
	Reason    string `json:"reason"`
}

type BuildInfo struct {
	CommitShort string `json:"commit_short,omitempty"`
	Commit      string `json:"commit,omitempty"`
	Branch      string `json:"branch,omitempty"`
	Backend     string `json:"backend,omitempty"`
	Repo        string `json:"repo,omitempty"`
	Error       string `json:"error,omitempty"`
}

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

type LoadedRun struct {
	Summary       RunSummary
	SystemMetrics []SystemMetricSample
	LlamaMetrics  []LlamaCppMetricSample
	SystemSummary SystemSummary
	LlamaSummary  LlamaSummary
	Directory     string
}

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

type LlamaSummary struct {
	GenerationTokS  *float64
	PromptEvalTokS  *float64
	TokensGenerated *float64
	TokensPrompt    *float64
	TotalTimeSec    *float64
	TTFTSeconds     *float64
}

func Ptr[T any](v T) *T {
	return &v
}

func DerefString(v *string) string {
	if v == nil {
		return ""
	}
	return *v
}

func IntValue(v *int) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%d", *v)
}
