package storage

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
)

func TestWriteSummaryMetricsAndReadRun(t *testing.T) {
	root := t.TempDir()
	summary := domain.RunSummary{
		RunID:         "run-a",
		BatchID:       "batch-a",
		CreatedAt:     "2026-01-01T00:00:00Z",
		Model:         "model:A",
		PromptProfile: "code",
		ServerConfig:  domain.ServerConfig{ContextSize: 2048, KVType: "q8_0"},
		Status:        domain.NewRunStatusInfo(domain.StatusSuccess, ""),
		ConfigHash:    "hash-a",
		DurationSec:   10,
		Response:      map[string]any{"content": "ok"},
	}
	if _, err := WriteRunSummary(root, summary); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(SummaryPath(RunDir(root, "run-a"))); err != nil {
		t.Fatal(err)
	}
	if err := AppendSystemMetric(RunDir(root, "run-a"), domain.SystemMetricSample{Time: "t1", VRAMFree: domain.Ptr(4096.0), VRAMUsed: domain.Ptr(1024.0)}); err != nil {
		t.Fatal(err)
	}
	if err := AppendLlamaMetric(RunDir(root, "run-a"), domain.LlamaCppMetricSample{Time: "t1", GenerationTokS: domain.Ptr(44.0), PromptEvalTokS: domain.Ptr(120.0)}); err != nil {
		t.Fatal(err)
	}
	rows, err := LoadRuns(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("rows = %d", len(rows))
	}
	if *rows[0].SystemSummary.MinVRAMFreeMiB != 4096 {
		t.Fatalf("min vram = %v", *rows[0].SystemSummary.MinVRAMFreeMiB)
	}
	if *rows[0].LlamaSummary.GenerationTokS != 44 {
		t.Fatalf("gen = %v", *rows[0].LlamaSummary.GenerationTokS)
	}
}

func TestReadJSONLParseErrorIncludesPathAndLine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rows.jsonl")
	if err := os.WriteFile(path, []byte("{\"ok\":true}\nnot-json\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := ReadJSONL[map[string]any](path)
	if err == nil {
		t.Fatal("expected parse error")
	}
	text := err.Error()
	if !strings.Contains(text, path) || !strings.Contains(text, ":2:") {
		t.Fatalf("missing path/line in error: %v", err)
	}
}
