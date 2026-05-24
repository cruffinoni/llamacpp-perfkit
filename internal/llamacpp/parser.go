package llamacpp

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
)

type ParsedMetrics struct {
	PromptEvalTokS  any
	GenerationTokS  any
	TotalTimeMS     any
	LoadTimeMS      any
	TokensGenerated any
	TokensPrompt    any
	Fields          map[string]any
}

func (m ParsedMetrics) Map() map[string]any {
	out := map[string]any{
		"prompt_eval_tok_s": m.PromptEvalTokS,
		"generation_tok_s":  m.GenerationTokS,
		"total_time_ms":     m.TotalTimeMS,
		"load_time_ms":      m.LoadTimeMS,
		"tokens_generated":  m.TokensGenerated,
		"tokens_prompt":     m.TokensPrompt,
	}
	for key, value := range m.Fields {
		out[key] = value
	}
	return out
}

func ParseCompletion(data map[string]any) (map[string]any, map[string]any) {
	timings := mapValue(data["timings"])
	promptN := number(timings["prompt_n"])
	cacheN := number(timings["cache_n"])
	predictedN := number(timings["predicted_n"])
	tokensPrompt := promptN
	if tokensPrompt != nil && cacheN != nil {
		value := *tokensPrompt + *cacheN
		tokensPrompt = &value
	}
	parsed := ParsedMetrics{
		PromptEvalTokS:  numberValue(timings["prompt_per_second"]),
		GenerationTokS:  numberValue(timings["predicted_per_second"]),
		TotalTimeMS:     nil,
		LoadTimeMS:      nil,
		TokensGenerated: numberValue(predictedN),
		TokensPrompt:    numberValue(tokensPrompt),
		Fields: map[string]any{
			"server_timings":   timings,
			"tokens_cached":    data["tokens_cached"],
			"tokens_evaluated": data["tokens_evaluated"],
			"truncated":        data["truncated"],
			"stop_type":        data["stop_type"],
		},
	}.Map()
	response := map[string]any{"content": data["content"], "stop_type": data["stop_type"], "truncated": data["truncated"]}
	return parsed, response
}

func ParseChatCompletion(data map[string]any, elapsedSeconds float64, logText string) (map[string]any, map[string]any) {
	choices := sliceValue(data["choices"])
	choice := map[string]any{}
	if len(choices) > 0 {
		choice = mapValue(choices[0])
	}
	message := mapValue(choice["message"])
	usage := mapValue(data["usage"])
	timings := mapValue(data["timings"])
	logParsed := ParseLlamaOutput(logText)
	completionTokens := number(usage["completion_tokens"])
	promptTokens := number(usage["prompt_tokens"])
	speed := number(timings["predicted_per_second"])
	if speed == nil {
		speed = number(logParsed["generation_tok_s"])
	}
	if speed == nil && completionTokens != nil && elapsedSeconds > 0 {
		value := *completionTokens / elapsedSeconds
		speed = &value
	}
	promptSpeed := number(timings["prompt_per_second"])
	if promptSpeed == nil {
		promptSpeed = number(logParsed["prompt_eval_tok_s"])
	}
	parsed := ParsedMetrics{
		PromptEvalTokS:  numberValue(promptSpeed),
		GenerationTokS:  numberValue(speed),
		TotalTimeMS:     logParsed["total_time_ms"],
		LoadTimeMS:      logParsed["load_time_ms"],
		TokensGenerated: numberValue(orNumber(completionTokens, number(logParsed["tokens_generated"]))),
		TokensPrompt:    numberValue(orNumber(promptTokens, number(logParsed["tokens_prompt"]))),
		Fields: map[string]any{
			"speculative_stats": logParsed["speculative_stats"],
			"server_timings":    timings,
			"tokens_cached":     mapValue(usage["prompt_tokens_details"])["cached_tokens"],
			"tokens_evaluated":  numberValue(promptTokens),
			"truncated":         choice["finish_reason"] == "length",
			"stop_type":         choice["finish_reason"],
			"throughput_source": throughputSource(timings, logParsed),
		},
	}.Map()
	response := map[string]any{
		"content":           message["content"],
		"reasoning_content": firstAny(message["reasoning_content"], message["reasoning"]),
		"stop_type":         choice["finish_reason"],
		"truncated":         choice["finish_reason"] == "length",
	}
	return parsed, response
}

func ParseLlamaOutput(text string) map[string]any {
	parsed := ParsedMetrics{
		PromptEvalTokS:  nil,
		GenerationTokS:  nil,
		TotalTimeMS:     nil,
		LoadTimeMS:      nil,
		TokensGenerated: nil,
		TokensPrompt:    nil,
		Fields:          map[string]any{"speculative_stats": nil},
	}.Map()
	for key, pattern := range map[string]string{
		"load_time_ms":  `(?i)load time\s*=\s*([0-9.]+)\s*ms`,
		"total_time_ms": `(?i)total time\s*=\s*([0-9.]+)\s*ms`,
	} {
		match := regexp.MustCompile(pattern).FindStringSubmatch(text)
		if len(match) == 2 {
			parsed[key] = parseFloat(match[1])
		}
	}
	promptMatch := regexp.MustCompile(`(?i)prompt eval time\s*=\s*[^\n]*?/\s*(\d+)\s+tokens?.*?\(\s*([0-9.]+)\s+tokens? per second`).FindStringSubmatch(text)
	if len(promptMatch) == 3 {
		parsed["tokens_prompt"] = parseFloat(promptMatch[1])
		parsed["prompt_eval_tok_s"] = parseFloat(promptMatch[2])
	}
	genMatches := regexp.MustCompile(`(?i)(?m)(?:(?:^|[^\w])eval time)\s*=\s*[^\n]*?/\s*(\d+)\s+(?:runs?|tokens?).*?\(\s*([0-9.]+)\s+tokens? per second`).FindAllStringSubmatch(text, -1)
	if len(genMatches) > 0 {
		last := genMatches[len(genMatches)-1]
		parsed["tokens_generated"] = parseFloat(last[1])
		parsed["generation_tok_s"] = parseFloat(last[2])
	}
	var specLines []string
	for _, line := range strings.Split(text, "\n") {
		lower := strings.ToLower(line)
		if strings.Contains(lower, "spec") || strings.Contains(lower, "draft") || strings.Contains(lower, "accept") {
			specLines = append(specLines, strings.TrimSpace(line))
		}
	}
	if len(specLines) > 0 {
		if len(specLines) > 10 {
			specLines = specLines[len(specLines)-10:]
		}
		parsed["speculative_stats"] = strings.Join(specLines, " | ")
	}
	return parsed
}

func BenchmarkInvalidReason(summary domain.RunSummary, parsed map[string]any) string {
	if summary.Status.Kind() != domain.StatusSuccess {
		return string(summary.Status.Kind())
	}
	speed := number(parsed["generation_tok_s"])
	tokens := number(parsed["tokens_generated"])
	minTokens := 8
	if summary.GenerationToks > 0 && summary.GenerationToks < minTokens {
		minTokens = summary.GenerationToks
	}
	if speed == nil {
		return "missing generation throughput"
	}
	if *speed <= 0 {
		return "invalid generation throughput"
	}
	if *speed > 10000 {
		return "implausible generation throughput"
	}
	if tokens == nil {
		return "missing generated token count"
	}
	if *tokens < float64(minTokens) {
		return fmt.Sprintf("too few generated tokens for reliable throughput: %.0f < %d", *tokens, minTokens)
	}
	content := fmt.Sprint(summary.Response["content"])
	reasoning := fmt.Sprint(summary.Response["reasoning_content"])
	if EndpointKindFromSummary(summary) == "chat" && strings.TrimSpace(content) == "" && strings.TrimSpace(reasoning) == "" {
		return "empty chat response content"
	}
	return ""
}

func EndpointKindFromSummary(summary domain.RunSummary) string {
	if _, ok := summary.Request["messages"]; ok {
		return "chat"
	}
	if _, ok := summary.Request["prompt"]; ok {
		return "completion"
	}
	return "chat"
}

func ClassifyRun(returnCode int, timedOut bool, text string) domain.RunStatus {
	lower := strings.ToLower(text)
	if timedOut {
		return domain.StatusTimeout
	}
	if returnCode == 0 {
		return domain.StatusSuccess
	}
	if containsAny(lower, "out of memory", "cuda error 2", "cuda_malloc", "cudamalloc", "failed to allocate", "no memory") {
		return domain.StatusOOM
	}
	if containsAny(lower, "unknown argument", "invalid argument", "invalid value", "error: unrecognized", "invalid choice") {
		return domain.StatusUnsupported
	}
	return domain.StatusFailed
}

func ClassifyHTTPError(statusCode int, text string) domain.RunStatus {
	lower := strings.ToLower(text)
	if statusCode == 400 || statusCode == 404 || statusCode == 422 || containsAny(lower, "unknown argument", "invalid argument", "invalid value", "unsupported") {
		return domain.StatusUnsupported
	}
	if containsAny(lower, "out of memory", "cuda error 2", "cuda_malloc", "cudamalloc", "failed to allocate", "no memory") {
		return domain.StatusOOM
	}
	return domain.StatusFailed
}

func LogText(path string) string {
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return ""
	}
	return string(data)
}

func mapValue(value any) map[string]any {
	if value == nil {
		return map[string]any{}
	}
	if typed, ok := value.(map[string]any); ok {
		return typed
	}
	return map[string]any{}
}

func sliceValue(value any) []any {
	if typed, ok := value.([]any); ok {
		return typed
	}
	return nil
}

func number(value any) *float64 {
	switch typed := value.(type) {
	case nil:
		return nil
	case *float64:
		return typed
	case float64:
		return &typed
	case float32:
		out := float64(typed)
		return &out
	case int:
		out := float64(typed)
		return &out
	case int64:
		out := float64(typed)
		return &out
	case json.Number:
		if parsed, err := typed.Float64(); err == nil {
			return &parsed
		}
	}
	return nil
}

func numberValue(value any) any {
	if ptr, ok := value.(*float64); ok {
		if ptr == nil {
			return nil
		}
		return *ptr
	}
	if ptr := number(value); ptr != nil {
		return *ptr
	}
	return nil
}

func orNumber(values ...*float64) *float64 {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func firstAny(values ...any) any {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func throughputSource(timings, logParsed map[string]any) string {
	if timings["predicted_per_second"] != nil {
		return "server_timings"
	}
	if logParsed["generation_tok_s"] != nil {
		return "server_log"
	}
	return "wall_clock"
}

func parseFloat(value string) float64 {
	var out float64
	fmt.Sscanf(value, "%f", &out)
	return out
}

func containsAny(value string, patterns ...string) bool {
	for _, pattern := range patterns {
		if strings.Contains(value, pattern) {
			return true
		}
	}
	return false
}
