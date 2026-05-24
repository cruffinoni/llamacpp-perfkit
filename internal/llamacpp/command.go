package llamacpp

import (
	"fmt"
	"os"
	"strings"

	"github.com/cruffinoni/llamacpp-perfkit/internal/config"
	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
)

func shellQuote(value string) string {
	if value == "" {
		return "''"
	}
	if strings.IndexFunc(value, func(r rune) bool {
		return !(r == '-' || r == '_' || r == '/' || r == '.' || r == ':' || r == '=' || r == '+' || r == ',' || r == '%' || r >= '0' && r <= '9' || r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z')
	}) == -1 {
		return value
	}
	return "'" + strings.ReplaceAll(value, "'", "'\\''") + "'"
}

func appendExtraArgs(cmd []string, args []config.ExtraArg) []string {
	for _, item := range args {
		if item.Value == nil || item.Value == false {
			continue
		}
		cmd = append(cmd, item.Flag)
		if item.Value != true {
			cmd = append(cmd, fmt.Sprint(item.Value))
		}
	}
	return cmd
}

// EndpointPath returns the API path for the given endpoint kind.
func EndpointPath(kind string) string {
	if kind == "chat" {
		return "/v1/chat/completions"
	}
	return "/completion"
}

// EndpointKind returns the normalized endpoint type from the configuration.
func EndpointKind(cfg config.Config) string {
	value := strings.ToLower(strings.TrimSpace(cfg.Run.Endpoint))
	switch value {
	case "", "chat", "chat-completions", "chat_completions", "/v1/chat/completions":
		return "chat"
	case "completion", "completions", "/completion":
		return "completion"
	default:
		return value
	}
}

// CommandToShell converts a command slice to a shell-quoted string.
func CommandToShell(cmd []string) string {
	parts := make([]string, len(cmd))
	for i, part := range cmd {
		parts[i] = shellQuote(part)
	}
	return strings.Join(parts, " ")
}

// BuildRequestPayload constructs the JSON request payload for a benchmark job.
func BuildRequestPayload(cfg config.Config, features Features, job domain.BenchmarkJob) (map[string]any, error) {
	data, err := os.ReadFile(job.PromptFile)
	if err != nil {
		return nil, fmt.Errorf("read prompt %s: %w", job.PromptFile, err)
	}
	prompt := string(data)
	payload := map[string]any{}
	switch EndpointKind(cfg) {
	case "chat":
		payload["messages"] = []map[string]string{{"role": "user", "content": prompt}}
		payload["max_tokens"] = cfg.Run.GenerationTokens
		payload["stream"] = false
		if len(cfg.Run.ChatTemplateKwargs) > 0 {
			payload["chat_template_kwargs"] = cfg.Run.ChatTemplateKwargs
		}
	case "completion":
		payload["prompt"] = prompt
		payload["n_predict"] = cfg.Run.GenerationTokens
		payload["stream"] = false
		payload["cache_prompt"] = cfg.Run.CachePrompt
		payload["id_slot"] = 0
	default:
		return nil, fmt.Errorf("unsupported run.endpoint: %s", cfg.Run.Endpoint)
	}
	if cfg.Run.Seed != nil {
		payload["seed"] = *cfg.Run.Seed
	}
	for key, value := range RequestArgs(features) {
		payload[key] = value
	}
	return payload, nil
}

// BuildServerCommand constructs the command-line arguments for llama-server.
func BuildServerCommand(cfg config.Config, features Features, job domain.BenchmarkJob, port int, _ string) []string {
	flags := features.Flags.LlamaServer
	model := cfg.ModelHF()
	cmd := []string{cfg.LlamaServerPath(), "-hf", model}
	if flags.Context != "" {
		cmd = append(cmd, flags.Context, fmt.Sprintf("%d", job.ServerConfig.ContextSize))
	}
	if flags.Host != "" {
		cmd = append(cmd, flags.Host, cfg.Llama.Server.Host)
	}
	if flags.Port != "" {
		cmd = append(cmd, flags.Port, fmt.Sprintf("%d", port))
	}
	if flags.NCPUMOE != "" && job.ServerConfig.NCPUMOE != nil {
		cmd = append(cmd, flags.NCPUMOE, fmt.Sprintf("%d", *job.ServerConfig.NCPUMOE))
	}
	if flags.CacheTypeK != "" && job.ServerConfig.KVType != "" {
		cmd = append(cmd, flags.CacheTypeK, job.ServerConfig.KVType)
	}
	if flags.CacheTypeV != "" && job.ServerConfig.KVType != "" {
		cmd = append(cmd, flags.CacheTypeV, job.ServerConfig.KVType)
	}
	if flags.BatchSize != "" && job.ServerConfig.BatchSize != nil {
		cmd = append(cmd, flags.BatchSize, fmt.Sprintf("%d", *job.ServerConfig.BatchSize))
	}
	if flags.UBatchSize != "" && job.ServerConfig.UBatchSize != nil {
		cmd = append(cmd, flags.UBatchSize, fmt.Sprintf("%d", *job.ServerConfig.UBatchSize))
	}
	cmd = appendExtraArgs(cmd, ServerExtraArgs(features))
	if flags.NoWebUI != "" {
		cmd = append(cmd, flags.NoWebUI)
	}
	if flags.Metrics != "" {
		cmd = append(cmd, flags.Metrics)
	}
	if flags.SpecType != "" && job.ServerConfig.SpecType != "" {
		cmd = append(cmd, flags.SpecType, job.ServerConfig.SpecType)
		if flags.SpecDraftNMax != "" && job.ServerConfig.SpecDraftNMax != nil {
			cmd = append(cmd, flags.SpecDraftNMax, fmt.Sprintf("%d", *job.ServerConfig.SpecDraftNMax))
		}
		if flags.SpecDraftPMin != "" && job.ServerConfig.SpecDraftPMin != nil {
			cmd = append(cmd, flags.SpecDraftPMin, fmt.Sprintf("%g", *job.ServerConfig.SpecDraftPMin))
		}
	}
	return cmd
}
