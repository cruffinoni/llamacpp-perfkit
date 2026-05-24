package summary

import (
	"encoding/json"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
)

func values[T any](samples []T, pick func(T) *float64) []float64 {
	out := make([]float64, 0, len(samples))
	for _, sample := range samples {
		if value := pick(sample); value != nil {
			out = append(out, *value)
		}
	}
	return out
}

func minPtr(values []float64) *float64 {
	if len(values) == 0 {
		return nil
	}
	out := values[0]
	for _, value := range values[1:] {
		if value < out {
			out = value
		}
	}
	return &out
}

func maxPtr(values []float64) *float64 {
	if len(values) == 0 {
		return nil
	}
	out := values[0]
	for _, value := range values[1:] {
		if value > out {
			out = value
		}
	}
	return &out
}

func meanPtr(values []float64) *float64 {
	if len(values) == 0 {
		return nil
	}
	var sum float64
	for _, value := range values {
		sum += value
	}
	out := sum / float64(len(values))
	return &out
}

func lastPtr(values []float64) *float64 {
	if len(values) == 0 {
		return nil
	}
	out := values[len(values)-1]
	return &out
}

func numberPtr(value any) *float64 {
	switch typed := value.(type) {
	case int:
		out := float64(typed)
		return &out
	case int64:
		out := float64(typed)
		return &out
	case float64:
		return &typed
	case float32:
		out := float64(typed)
		return &out
	case json.Number:
		if parsed, err := typed.Float64(); err == nil {
			return &parsed
		}
	}
	return nil
}

// TerminalLlamaSample creates a terminal llama.cpp metric sample from parsed output.
func TerminalLlamaSample(timeValue string, parsed map[string]any, durationSeconds float64) domain.LlamaCppMetricSample {
	promptTokens := numberPtr(parsed["tokens_prompt"])
	generatedTokens := numberPtr(parsed["tokens_generated"])
	promptTokS := numberPtr(parsed["prompt_eval_tok_s"])
	generationTokS := numberPtr(parsed["generation_tok_s"])
	var totalTokens *float64
	if promptTokens != nil && generatedTokens != nil {
		out := *promptTokens + *generatedTokens
		totalTokens = &out
	}
	return domain.LlamaCppMetricSample{
		Time:             timeValue,
		PromptTokens:     promptTokens,
		GeneratedTokens:  generatedTokens,
		PromptEvalTokens: promptTokens,
		EvalTokens:       generatedTokens,
		PromptEvalTokS:   promptTokS,
		GenerationTokS:   generationTokS,
		TotalTokens:      totalTokens,
		TotalTimeSeconds: &durationSeconds,
	}
}

// SummarizeLlama computes aggregate llama.cpp metrics from metric samples.
func SummarizeLlama(samples []domain.LlamaCppMetricSample) domain.LlamaSummary {
	generation := values(samples, func(s domain.LlamaCppMetricSample) *float64 { return s.GenerationTokS })
	prompt := values(samples, func(s domain.LlamaCppMetricSample) *float64 { return s.PromptEvalTokS })
	generatedTokens := values(samples, func(s domain.LlamaCppMetricSample) *float64 {
		if s.GeneratedTokens != nil {
			return s.GeneratedTokens
		}
		return s.EvalTokens
	})
	promptTokens := values(samples, func(s domain.LlamaCppMetricSample) *float64 {
		if s.PromptTokens != nil {
			return s.PromptTokens
		}
		return s.PromptEvalTokens
	})
	totalTime := values(samples, func(s domain.LlamaCppMetricSample) *float64 { return s.TotalTimeSeconds })
	ttft := values(samples, func(s domain.LlamaCppMetricSample) *float64 { return s.TTFTSeconds })
	return domain.LlamaSummary{
		GenerationTokS:  lastPtr(generation),
		PromptEvalTokS:  lastPtr(prompt),
		TokensGenerated: maxPtr(generatedTokens),
		TokensPrompt:    maxPtr(promptTokens),
		TotalTimeSec:    lastPtr(totalTime),
		TTFTSeconds:     lastPtr(ttft),
	}
}

// SummarizeSystem computes aggregate system metrics from metric samples.
func SummarizeSystem(samples []domain.SystemMetricSample) domain.SystemSummary {
	vramFree := values(samples, func(s domain.SystemMetricSample) *float64 { return s.VRAMFree })
	vramUsed := values(samples, func(s domain.SystemMetricSample) *float64 { return s.VRAMUsed })
	ramFree := values(samples, func(s domain.SystemMetricSample) *float64 { return s.RAMFree })
	ramUsed := values(samples, func(s domain.SystemMetricSample) *float64 { return s.RAMUsed })
	gpuUtil := values(samples, func(s domain.SystemMetricSample) *float64 { return s.GPUUtilPct })
	gpuPower := values(samples, func(s domain.SystemMetricSample) *float64 { return s.GPUPowerW })
	gpuTemp := values(samples, func(s domain.SystemMetricSample) *float64 { return s.GPUTempC })
	return domain.SystemSummary{
		PeakVRAMMiB:     maxPtr(vramUsed),
		MinVRAMFreeMiB:  minPtr(vramFree),
		MeanVRAMFreeMiB: meanPtr(vramFree),
		PeakRAMMiB:      maxPtr(ramUsed),
		MinRAMFreeMiB:   minPtr(ramFree),
		AvgGPUUtilPct:   meanPtr(gpuUtil),
		PeakGPUPowerW:   maxPtr(gpuPower),
		PeakGPUTempC:    maxPtr(gpuTemp),
	}
}
