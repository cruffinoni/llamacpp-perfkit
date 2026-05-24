package metrics

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
	"github.com/cruffinoni/llamacpp-perfkit/internal/storage"
)

type SystemCollector struct {
	RunDir   string
	Interval time.Duration
}

func (c SystemCollector) Run(ctx context.Context) error {
	interval := c.Interval
	if interval <= 0 {
		interval = time.Second
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	if err := storage.AppendSystemMetric(c.RunDir, SampleSystem()); err != nil {
		return err
	}
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			if err := storage.AppendSystemMetric(c.RunDir, SampleSystem()); err != nil {
				return err
			}
		}
	}
}

func SampleSystem() domain.SystemMetricSample {
	sample := domain.SystemMetricSample{Time: nowISO()}
	if gpu := sampleGPU(); gpu != nil {
		sample.VRAMUsed = gpu["vram_used_mib"]
		sample.VRAMFree = gpu["vram_free_mib"]
		sample.GPUUtilPct = gpu["gpu_util_pct"]
		sample.GPUPowerW = gpu["gpu_power_w"]
		sample.GPUTempC = gpu["gpu_temp_c"]
	}
	ramUsed, ramFree := sampleRAM()
	sample.RAMUsed = ramUsed
	sample.RAMFree = ramFree
	return sample
}

type LlamaCollector struct {
	BaseURL  string
	RunDir   string
	Interval time.Duration
}

func (c LlamaCollector) Run(ctx context.Context) error {
	interval := c.Interval
	if interval <= 0 {
		interval = time.Second
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	if sample, ok := c.SampleOnce(ctx); ok {
		if err := storage.AppendLlamaMetric(c.RunDir, sample); err != nil {
			return err
		}
	}
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			if sample, ok := c.SampleOnce(ctx); ok {
				if err := storage.AppendLlamaMetric(c.RunDir, sample); err != nil {
					return err
				}
			}
		}
	}
}

func (c LlamaCollector) SampleOnce(parent context.Context) (domain.LlamaCppMetricSample, bool) {
	sample := domain.LlamaCppMetricSample{Time: nowISO()}
	found := false
	ctx, cancel := context.WithTimeout(parent, 2*time.Second)
	body, status, err := getJSON(ctx, strings.TrimRight(c.BaseURL, "/")+"/slots")
	cancel()
	if err == nil && status == http.StatusOK {
		if applySlots(&sample, body) {
			found = true
		}
	}

	ctx, cancel = context.WithTimeout(parent, 2*time.Second)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(c.BaseURL, "/")+"/metrics", nil)
	if err == nil {
		resp, err := http.DefaultClient.Do(req)
		if err == nil {
			data, _ := ioReadAllAndClose(resp.Body)
			if resp.StatusCode == http.StatusOK && len(data) > 0 {
				if applyPrometheus(&sample, string(data)) {
					found = true
				}
			}
		}
	}
	cancel()
	return sample, found
}

func sampleGPU() map[string]*float64 {
	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu",
		"--format=csv,noheader,nounits",
	)
	data, err := cmd.Output()
	if err != nil {
		return nil
	}
	line := strings.TrimSpace(strings.Split(string(data), "\n")[0])
	if line == "" {
		return nil
	}
	parts := strings.Split(line, ",")
	keys := []string{"vram_used_mib", "vram_free_mib", "gpu_util_pct", "gpu_power_w", "gpu_temp_c"}
	out := map[string]*float64{}
	for i, key := range keys {
		if i >= len(parts) {
			continue
		}
		value, err := strconv.ParseFloat(strings.TrimSpace(parts[i]), 64)
		if err != nil {
			continue
		}
		out[key] = &value
	}
	return out
}

func sampleRAM() (*float64, *float64) {
	file, err := os.Open("/proc/meminfo")
	if err != nil {
		return nil, nil
	}
	defer file.Close()
	values := map[string]float64{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		key, rest, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		fields := strings.Fields(rest)
		if len(fields) == 0 {
			continue
		}
		value, err := strconv.ParseFloat(fields[0], 64)
		if err == nil {
			values[key] = value / 1024.0
		}
	}
	total, totalOK := values["MemTotal"]
	free, freeOK := values["MemAvailable"]
	if !freeOK {
		free, freeOK = values["MemFree"]
	}
	if !totalOK || !freeOK {
		return nil, nil
	}
	used := total - free
	return &used, &free
}

func applySlots(sample *domain.LlamaCppMetricSample, body any) bool {
	var slots []any
	if raw, ok := body.([]any); ok {
		slots = raw
	} else if obj, ok := body.(map[string]any); ok {
		if raw, ok := obj["slots"].([]any); ok {
			slots = raw
		} else if raw, ok := obj["data"].([]any); ok {
			slots = raw
		}
	}
	if len(slots) == 0 {
		return false
	}
	idle, processing := 0, 0
	var promptTokens, generatedTokens []float64
	for _, raw := range slots {
		slot, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		state := strings.ToLower(fmt.Sprint(slot["state"]))
		busy := slot["is_processing"] == true || state == "processing" || state == "busy"
		if busy {
			processing++
		} else {
			idle++
		}
		for _, key := range []string{"n_prompt_tokens", "prompt_tokens", "prompt_n"} {
			if value := numeric(slot[key]); value != nil {
				promptTokens = append(promptTokens, *value)
			}
		}
		for _, key := range []string{"n_decoded", "generated_tokens", "predicted_n"} {
			if value := numeric(slot[key]); value != nil {
				generatedTokens = append(generatedTokens, *value)
			}
		}
	}
	sample.SlotsIdle = &idle
	sample.SlotsProcessing = &processing
	sample.PromptTokens = maxFloat(promptTokens)
	sample.GeneratedTokens = maxFloat(generatedTokens)
	return true
}

func applyPrometheus(sample *domain.LlamaCppMetricSample, text string) bool {
	found := false
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		name := strings.ToLower(strings.Split(parts[0], "{")[0])
		value, err := strconv.ParseFloat(parts[len(parts)-1], 64)
		if err != nil {
			continue
		}
		v := value
		switch {
		case strings.Contains(name, "prompt") && strings.Contains(name, "per_second"):
			if sample.PromptEvalTokS == nil {
				sample.PromptEvalTokS = &v
				found = true
			}
		case (strings.Contains(name, "predicted") || strings.Contains(name, "generation") || strings.Contains(name, "eval")) && strings.Contains(name, "per_second"):
			if sample.GenerationTokS == nil {
				sample.GenerationTokS = &v
				found = true
			}
		case strings.Contains(name, "prompt") && strings.Contains(name, "token"):
			if sample.PromptTokens == nil {
				sample.PromptTokens = &v
				found = true
			}
		case (strings.Contains(name, "predicted") || strings.Contains(name, "generation") || strings.Contains(name, "eval")) && strings.Contains(name, "token"):
			if sample.GeneratedTokens == nil {
				sample.GeneratedTokens = &v
				found = true
			}
		}
	}
	return found
}

func numeric(value any) *float64 {
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
	}
	return nil
}

func maxFloat(values []float64) *float64 {
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

func nowISO() string {
	return time.Now().UTC().Format(time.RFC3339Nano)
}

func getJSON(ctx context.Context, url string) (any, int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, 0, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, 0, err
	}
	data, err := ioReadAllAndClose(resp.Body)
	if err != nil {
		return nil, resp.StatusCode, err
	}
	var decoded any
	if err := json.Unmarshal(data, &decoded); err != nil {
		return nil, resp.StatusCode, err
	}
	return decoded, resp.StatusCode, nil
}
