package storage

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
	metricsummary "github.com/cruffinoni/llamacpp-perfkit/internal/summary"
)

func RunDir(root, runID string) string {
	return filepath.Join(root, runID)
}

func MetricsDir(runDir string) string {
	return filepath.Join(runDir, "metrics")
}

func SummaryPath(runDir string) string {
	return filepath.Join(runDir, "summary.json")
}

func SystemMetricsPath(runDir string) string {
	return filepath.Join(MetricsDir(runDir), "system.jsonl")
}

func LlamaMetricsPath(runDir string) string {
	return filepath.Join(MetricsDir(runDir), "llamacpp.jsonl")
}

func WriteRunSummary(root string, summary domain.RunSummary) (string, error) {
	runDir := RunDir(root, summary.RunID)
	if err := os.MkdirAll(MetricsDir(runDir), 0o755); err != nil {
		return "", fmt.Errorf("create metrics directory %s: %w", MetricsDir(runDir), err)
	}
	if err := touch(SystemMetricsPath(runDir)); err != nil {
		return "", fmt.Errorf("initialize system metrics %s: %w", SystemMetricsPath(runDir), err)
	}
	if err := touch(LlamaMetricsPath(runDir)); err != nil {
		return "", fmt.Errorf("initialize llama metrics %s: %w", LlamaMetricsPath(runDir), err)
	}
	target := SummaryPath(runDir)
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return "", fmt.Errorf("marshal run summary %s: %w", summary.RunID, err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(target, data, 0o644); err != nil {
		return "", fmt.Errorf("write run summary %s: %w", target, err)
	}
	return target, nil
}

func AppendSystemMetric(runDir string, sample domain.SystemMetricSample) error {
	return appendJSONL(SystemMetricsPath(runDir), sample)
}

func AppendLlamaMetric(runDir string, sample domain.LlamaCppMetricSample) error {
	return appendJSONL(LlamaMetricsPath(runDir), sample)
}

func ReadRun(rootOrRunDir string) (domain.LoadedRun, error) {
	runDir := rootOrRunDir
	if filepath.Base(rootOrRunDir) == "summary.json" {
		runDir = filepath.Dir(rootOrRunDir)
	}
	summary, err := ReadSummary(runDir)
	if err != nil {
		return domain.LoadedRun{}, err
	}
	systemMetrics, err := ReadJSONL[domain.SystemMetricSample](SystemMetricsPath(runDir))
	if err != nil {
		return domain.LoadedRun{}, err
	}
	llamaMetrics, err := ReadJSONL[domain.LlamaCppMetricSample](LlamaMetricsPath(runDir))
	if err != nil {
		return domain.LoadedRun{}, err
	}
	return domain.LoadedRun{
		Summary:       summary,
		SystemMetrics: systemMetrics,
		LlamaMetrics:  llamaMetrics,
		SystemSummary: metricsummary.SummarizeSystem(systemMetrics),
		LlamaSummary:  metricsummary.SummarizeLlama(llamaMetrics),
		Directory:     runDir,
	}, nil
}

func ReadSummary(runDir string) (domain.RunSummary, error) {
	data, err := os.ReadFile(SummaryPath(runDir))
	if err != nil {
		return domain.RunSummary{}, fmt.Errorf("read run summary %s: %w", SummaryPath(runDir), err)
	}
	var summary domain.RunSummary
	if err := json.Unmarshal(data, &summary); err != nil {
		return domain.RunSummary{}, fmt.Errorf("parse run summary %s: %w", SummaryPath(runDir), err)
	}
	return summary, nil
}

func LoadRuns(paths ...string) ([]domain.LoadedRun, error) {
	runDirs, err := DiscoverRunDirs(paths...)
	if err != nil {
		return nil, err
	}
	out := make([]domain.LoadedRun, 0, len(runDirs))
	for _, dir := range runDirs {
		run, err := ReadRun(dir)
		if err != nil {
			return nil, err
		}
		out = append(out, run)
	}
	return out, nil
}

func DiscoverRunDirs(paths ...string) ([]string, error) {
	seen := map[string]bool{}
	var out []string
	for _, raw := range paths {
		if raw == "" {
			continue
		}
		path := raw
		if !filepath.IsAbs(path) {
			wd, err := os.Getwd()
			if err != nil {
				return nil, fmt.Errorf("resolve runs path %s: %w", raw, err)
			}
			path = filepath.Join(wd, path)
		}
		info, err := os.Stat(path)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return nil, fmt.Errorf("stat runs path %s: %w", path, err)
		}
		if !info.IsDir() && filepath.Base(path) == "summary.json" {
			dir := filepath.Dir(path)
			addRunDir(&out, seen, dir)
			continue
		}
		if !info.IsDir() {
			path = filepath.Dir(path)
		}
		if _, err := os.Stat(SummaryPath(path)); err == nil {
			addRunDir(&out, seen, path)
			continue
		}
		entries, err := os.ReadDir(path)
		if err != nil {
			return nil, fmt.Errorf("read runs directory %s: %w", path, err)
		}
		sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}
			dir := filepath.Join(path, entry.Name())
			if _, err := os.Stat(SummaryPath(dir)); err == nil {
				addRunDir(&out, seen, dir)
			}
		}
	}
	return out, nil
}

func ReadJSONL[T any](path string) ([]T, error) {
	file, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("open JSONL %s: %w", path, err)
	}
	defer file.Close()

	var out []T
	scanner := bufio.NewScanner(file)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var row T
		if err := json.Unmarshal(line, &row); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
		out = append(out, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan JSONL %s: %w", path, err)
	}
	return out, nil
}

func appendJSONL(path string, row any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create JSONL directory %s: %w", filepath.Dir(path), err)
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("open JSONL %s: %w", path, err)
	}
	defer file.Close()
	data, err := json.Marshal(row)
	if err != nil {
		return fmt.Errorf("marshal JSONL row for %s: %w", path, err)
	}
	if _, err := file.Write(append(data, '\n')); err != nil {
		return fmt.Errorf("append JSONL row to %s: %w", path, err)
	}
	return nil
}

func touch(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create directory %s: %w", filepath.Dir(path), err)
	}
	file, err := os.OpenFile(path, os.O_CREATE, 0o644)
	if err != nil {
		return fmt.Errorf("touch %s: %w", path, err)
	}
	return file.Close()
}

func addRunDir(out *[]string, seen map[string]bool, dir string) {
	clean := filepath.Clean(dir)
	if seen[clean] {
		return
	}
	seen[clean] = true
	*out = append(*out, clean)
}
