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

func addRunDir(out *[]string, seen map[string]bool, dir string) {
	clean := filepath.Clean(dir)
	if seen[clean] {
		return
	}
	seen[clean] = true
	*out = append(*out, clean)
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

// RunDir returns the path to a run directory within the given root.
func RunDir(root, runID string) string {
	return filepath.Join(root, runID)
}

// MetricsDir returns the metrics subdirectory within a run directory.
func MetricsDir(runDir string) string {
	return filepath.Join(runDir, "metrics")
}

// SummaryPath returns the path to the summary.json file in a run directory.
func SummaryPath(runDir string) string {
	return filepath.Join(runDir, "summary.json")
}

// ReadJSONL reads a JSONL file and returns a slice of decoded values.
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

// SystemMetricsPath returns the system metrics JSONL path.
func SystemMetricsPath(runDir string) string {
	return filepath.Join(MetricsDir(runDir), "system.jsonl")
}

// LlamaMetricsPath returns the llama.cpp metrics JSONL path.
func LlamaMetricsPath(runDir string) string {
	return filepath.Join(MetricsDir(runDir), "llamacpp.jsonl")
}

// ReadSummary reads and unmarshals the summary.json for a run directory.
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

// DiscoverRunDirs walks the given paths and returns all discovered run directories.
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

// WriteRunSummary creates the run directory structure and writes summary.json.
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

// AppendSystemMetric appends a system metric sample to the run's system JSONL.
func AppendSystemMetric(runDir string, sample domain.SystemMetricSample) error {
	return appendJSONL(SystemMetricsPath(runDir), sample)
}

// AppendLlamaMetric appends a llama.cpp metric sample to the run's llama JSONL.
func AppendLlamaMetric(runDir string, sample domain.LlamaCppMetricSample) error {
	return appendJSONL(LlamaMetricsPath(runDir), sample)
}

// ReadRun reads a full run from disk including summary, system metrics, llama
// metrics, and their computed summaries.
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

// LoadRuns discovers and loads all runs from the given paths.
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
