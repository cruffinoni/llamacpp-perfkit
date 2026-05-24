package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
	"gopkg.in/yaml.v3"
)

type Config struct {
	Models ModelsConfig `yaml:"models"`
	Llama  LlamaConfig  `yaml:"llama"`
	Prompt PromptConfig `yaml:"prompt"`
	Run    RunConfig    `yaml:"run"`
	Budget BudgetConfig `yaml:"budget"`
	Matrix MatrixConfig `yaml:"matrix"`
	Output OutputConfig `yaml:"output"`
}

type ModelsConfig struct {
	HF string `yaml:"hf"`
}

type ServerConfig struct {
	Host                   string `yaml:"host"`
	StartupTimeoutSeconds  int    `yaml:"startup_timeout_seconds"`
	ShutdownTimeoutSeconds int    `yaml:"shutdown_timeout_seconds"`
}

type LlamaConfig struct {
	BinDir          string       `yaml:"bin_dir"`
	PreferredBinary string       `yaml:"preferred_binary"`
	Server          ServerConfig `yaml:"server"`
	ExtraArgs       any          `yaml:"extra_args"`
}

type PromptConfig struct {
	File     string       `yaml:"file"`
	Profiles []ProfileRef `yaml:"profiles"`
}

type ProfileRef struct {
	Name string
	File string
}

func (p *ProfileRef) UnmarshalYAML(value *yaml.Node) error {
	if value.Kind == yaml.ScalarNode {
		var file string
		if err := value.Decode(&file); err != nil {
			return err
		}
		p.File = file
		p.Name = strings.TrimSuffix(filepath.Base(file), filepath.Ext(file))
		return nil
	}
	var raw struct {
		Name string `yaml:"name"`
		File string `yaml:"file"`
	}
	if err := value.Decode(&raw); err != nil {
		return err
	}
	p.Name = raw.Name
	p.File = raw.File
	if p.Name == "" && p.File != "" {
		p.Name = strings.TrimSuffix(filepath.Base(p.File), filepath.Ext(p.File))
	}
	return nil
}

type RunConfig struct {
	Endpoint               string         `yaml:"endpoint"`
	GenerationTokens       int            `yaml:"generation_tokens"`
	Seed                   *int           `yaml:"seed"`
	MinVRAMHeadroomGB      float64        `yaml:"min_vram_headroom_gb"`
	MonitorIntervalSeconds float64        `yaml:"monitor_interval_seconds"`
	TimeoutSeconds         int            `yaml:"timeout_seconds"`
	CachePrompt            bool           `yaml:"cache_prompt"`
	ChatTemplateKwargs     map[string]any `yaml:"chat_template_kwargs"`
}

type BudgetConfig struct {
	Mode                       string `yaml:"mode"`
	MaxRuns                    int    `yaml:"max_runs"`
	ReuseExistingResults       bool   `yaml:"reuse_existing_results"`
	StopIfAllRemainingAreRisky bool   `yaml:"stop_if_all_remaining_are_risky"`
}

type MatrixConfig struct {
	NCPUMOE       []*int     `yaml:"n_cpu_moe"`
	ContextSize   []int      `yaml:"context_size"`
	KVType        []string   `yaml:"kv_type"`
	BatchSize     []int      `yaml:"batch_size"`
	UBatchSize    []int      `yaml:"ubatch_size"`
	SpecType      []*string  `yaml:"spec_type"`
	SpecDraftNMax []*int     `yaml:"spec_draft_n_max"`
	SpecDraftPMin []*float64 `yaml:"spec_draft_p_min"`
}

type OutputConfig struct {
	LogsDir    string `yaml:"logs_dir"`
	ResultsDir string `yaml:"results_dir"`
}

type ExtraArg struct {
	Flag  string
	Value any
}

func Load(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("read config %s: %w", path, err)
	}
	cfg := Defaults()
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse config %s: %w", path, err)
	}
	cfg.ApplyDefaults()
	return cfg, nil
}

func Defaults() Config {
	return Config{
		Llama: LlamaConfig{
			BinDir:          "../llama.cpp/build/bin",
			PreferredBinary: "llama-server",
			Server: ServerConfig{
				Host:                   "127.0.0.1",
				StartupTimeoutSeconds:  300,
				ShutdownTimeoutSeconds: 15,
			},
			ExtraArgs: map[string]any{},
		},
		Prompt: PromptConfig{File: "prompts/default.txt"},
		Run: RunConfig{
			Endpoint:               "chat",
			GenerationTokens:       512,
			MinVRAMHeadroomGB:      1.5,
			MonitorIntervalSeconds: 1,
			TimeoutSeconds:         900,
		},
		Budget: BudgetConfig{
			Mode:                       "quick",
			MaxRuns:                    8,
			ReuseExistingResults:       true,
			StopIfAllRemainingAreRisky: true,
		},
		Matrix: MatrixConfig{
			NCPUMOE:     []*int{domain.Ptr(0)},
			ContextSize: []int{4096},
			BatchSize:   []int{1024},
			UBatchSize:  []int{1024},
		},
		Output: OutputConfig{LogsDir: "logs", ResultsDir: "runs"},
	}
}

func (c *Config) ApplyDefaults() {
	if c.Llama.BinDir == "" {
		c.Llama.BinDir = "../llama.cpp/build/bin"
	}
	if c.Llama.PreferredBinary == "" {
		c.Llama.PreferredBinary = "llama-server"
	}
	if c.Llama.Server.Host == "" {
		c.Llama.Server.Host = "127.0.0.1"
	}
	if c.Llama.Server.StartupTimeoutSeconds == 0 {
		c.Llama.Server.StartupTimeoutSeconds = 300
	}
	if c.Llama.Server.ShutdownTimeoutSeconds == 0 {
		c.Llama.Server.ShutdownTimeoutSeconds = 15
	}
	if c.Prompt.File == "" {
		c.Prompt.File = "prompts/default.txt"
	}
	if c.Run.Endpoint == "" {
		c.Run.Endpoint = "chat"
	}
	if c.Run.GenerationTokens == 0 {
		c.Run.GenerationTokens = 512
	}
	if c.Run.MinVRAMHeadroomGB == 0 {
		c.Run.MinVRAMHeadroomGB = 1.5
	}
	if c.Run.MonitorIntervalSeconds == 0 {
		c.Run.MonitorIntervalSeconds = 1
	}
	if c.Run.TimeoutSeconds == 0 {
		c.Run.TimeoutSeconds = 900
	}
	if c.Budget.Mode == "" {
		c.Budget.Mode = "quick"
	}
	if c.Budget.MaxRuns == 0 && c.Budget.Mode != "full" {
		c.Budget.MaxRuns = modeDefault(c.Budget.Mode)
	}
	if c.Matrix.NCPUMOE == nil {
		c.Matrix.NCPUMOE = []*int{domain.Ptr(0)}
	}
	if len(c.Matrix.ContextSize) == 0 {
		c.Matrix.ContextSize = []int{4096}
	}
	if len(c.Matrix.BatchSize) == 0 {
		c.Matrix.BatchSize = []int{1024}
	}
	if len(c.Matrix.UBatchSize) == 0 {
		c.Matrix.UBatchSize = []int{1024}
	}
	if c.Output.LogsDir == "" {
		c.Output.LogsDir = "logs"
	}
	if c.Output.ResultsDir == "" {
		c.Output.ResultsDir = "runs"
	}
}

func modeDefault(mode string) int {
	switch mode {
	case "smoke":
		return 2
	case "focused":
		return 16
	case "full":
		return 0
	default:
		return 8
	}
}

func (c Config) ModelHF() string {
	if value := os.Getenv("MODEL_HF"); value != "" {
		return value
	}
	return c.Models.HF
}

func (c Config) LlamaBinDir() string {
	if value := os.Getenv("LLAMA_BIN_DIR"); value != "" {
		return abs(value)
	}
	return abs(c.Llama.BinDir)
}

func (c Config) LlamaServerPath() string {
	return filepath.Join(c.LlamaBinDir(), "llama-server")
}

func (c Config) OutputDirs() (logsDir string, runsDir string, rawDir string, monitoringDir string, err error) {
	logsDir = abs(c.Output.LogsDir)
	runsDir = abs(c.Output.ResultsDir)
	rawDir = filepath.Join(logsDir, "raw")
	monitoringDir = filepath.Join(logsDir, "monitoring")
	for _, dir := range []string{logsDir, runsDir, rawDir, monitoringDir} {
		if mkErr := os.MkdirAll(dir, 0o755); mkErr != nil {
			err = fmt.Errorf("create output directory %s: %w", dir, mkErr)
			return
		}
	}
	return
}

func (c Config) PromptProfiles() []domain.PromptProfile {
	if len(c.Prompt.Profiles) == 0 {
		return []domain.PromptProfile{{Name: "default", File: abs(c.Prompt.File), Index: 1}}
	}
	out := make([]domain.PromptProfile, 0, len(c.Prompt.Profiles))
	for i, profile := range c.Prompt.Profiles {
		if profile.File == "" {
			continue
		}
		name := profile.Name
		if name == "" {
			name = strings.TrimSuffix(filepath.Base(profile.File), filepath.Ext(profile.File))
		}
		out = append(out, domain.PromptProfile{Name: name, File: abs(profile.File), Index: i + 1})
	}
	if len(out) == 0 {
		return []domain.PromptProfile{{Name: "default", File: abs(c.Prompt.File), Index: 1}}
	}
	return out
}

func NormalizeExtraArgs(raw any) []ExtraArg {
	switch value := raw.(type) {
	case nil:
		return nil
	case map[string]any:
		out := make([]ExtraArg, 0, len(value))
		for flag, argValue := range value {
			out = append(out, ExtraArg{Flag: flag, Value: argValue})
		}
		return out
	case map[any]any:
		out := make([]ExtraArg, 0, len(value))
		for flag, argValue := range value {
			out = append(out, ExtraArg{Flag: fmt.Sprint(flag), Value: argValue})
		}
		return out
	case []any:
		out := make([]ExtraArg, 0, len(value))
		for _, item := range value {
			switch typed := item.(type) {
			case string:
				out = append(out, ExtraArg{Flag: typed, Value: true})
			case map[string]any:
				out = append(out, ExtraArg{Flag: fmt.Sprint(typed["flag"]), Value: typed["value"]})
			case map[any]any:
				out = append(out, ExtraArg{Flag: fmt.Sprint(typed["flag"]), Value: typed["value"]})
			}
		}
		return out
	default:
		return nil
	}
}

func ExtraArgValue(args []ExtraArg, flags ...string) any {
	for _, target := range flags {
		for _, arg := range args {
			if arg.Flag == target {
				return arg.Value
			}
		}
	}
	return nil
}

func ExtraArgInt(args []ExtraArg, flags ...string) *int {
	value := ExtraArgValue(args, flags...)
	switch typed := value.(type) {
	case int:
		return &typed
	case int64:
		out := int(typed)
		return &out
	case float64:
		out := int(typed)
		return &out
	case string:
		parsed, err := strconv.Atoi(typed)
		if err == nil {
			return &parsed
		}
	}
	return nil
}

func ExtraArgString(args []ExtraArg, flags ...string) string {
	value := ExtraArgValue(args, flags...)
	if value == nil {
		return ""
	}
	return fmt.Sprint(value)
}

func abs(path string) string {
	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}
	wd, err := os.Getwd()
	if err != nil {
		return filepath.Clean(path)
	}
	return filepath.Clean(filepath.Join(wd, path))
}
