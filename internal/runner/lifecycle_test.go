package runner

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cruffinoni/llamacpp-perfkit/internal/config"
	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
	"github.com/cruffinoni/llamacpp-perfkit/internal/llamacpp"
	"github.com/cruffinoni/llamacpp-perfkit/internal/storage"
)

func TestRecordStartupFailureWritesSummariesWithoutLaunchingServer(t *testing.T) {
	root := t.TempDir()
	r := &Runner{
		Config:     testConfig(root),
		Features:   testFeatures(),
		ResultsDir: filepath.Join(root, "results"),
		RawDir:     filepath.Join(root, "logs", "raw"),
	}
	server := serverExecution{
		ID:      "server-a",
		RawPath: filepath.Join(root, "logs", "raw", "server-a.log"),
		CmdArgs: []string{"/missing/llama-server", "-hf", "model"},
	}
	item := domain.PlannedRun{
		ConfigHash: "hash-a",
		Job: domain.BenchmarkJob{
			PromptProfile: domain.PromptProfile{Name: "code"},
			PromptFile:    "prompt.txt",
			ServerConfig:  domain.ServerConfig{ContextSize: 2048, KVType: "q8_0"},
		},
	}
	completed := 0
	if err := r.recordStartupFailure([]domain.PlannedRun{item}, server, "start failed", &completed, newTUIAdapter(nil, nowTimeForTest())); err != nil {
		t.Fatal(err)
	}
	if completed != 1 {
		t.Fatalf("completed = %d", completed)
	}
	run, err := storage.ReadRun(storage.RunDir(r.ResultsDir, "server-a-code"))
	if err != nil {
		t.Fatal(err)
	}
	if run.Summary.Status.Kind() != domain.StatusFailed || *run.Summary.Status.Error != "start failed" {
		t.Fatalf("status = %+v", run.Summary.Status)
	}
	if !strings.Contains(run.Summary.CommandShell, "llama-server") {
		t.Fatalf("command shell = %q", run.Summary.CommandShell)
	}
	if _, err := os.Stat(storage.SummaryPath(storage.RunDir(r.ResultsDir, "server-a-code"))); err != nil {
		t.Fatal(err)
	}
}

func TestPrepareServerBuildsCommandWithoutLaunchingServer(t *testing.T) {
	root := t.TempDir()
	r := &Runner{
		Config:        testConfig(root),
		Features:      testFeatures(),
		RawDir:        filepath.Join(root, "logs", "raw"),
		MonitoringDir: filepath.Join(root, "logs", "monitoring"),
	}
	item := domain.PlannedRun{Job: domain.BenchmarkJob{
		ServerConfig: domain.ServerConfig{ContextSize: 2048, KVType: "q8_0"},
	}}
	server, err := r.prepareServer([]domain.PlannedRun{item}, 7)
	if err != nil {
		t.Fatal(err)
	}
	shell := llamacpp.CommandToShell(server.CmdArgs)
	for _, want := range []string{"llama-server", "-hf model", "--ctx-size 2048", "--host 127.0.0.1", "--port"} {
		if !strings.Contains(shell, want) {
			t.Fatalf("command %q missing %q", shell, want)
		}
	}
	if !strings.Contains(server.RawPath, "server-0007.log") || !strings.Contains(server.MonitorPath, "server-0007.jsonl") {
		t.Fatalf("paths = raw %q monitor %q", server.RawPath, server.MonitorPath)
	}
	if !strings.HasPrefix(server.BaseURL, "http://127.0.0.1:") {
		t.Fatalf("base URL = %q", server.BaseURL)
	}
}

func testConfig(root string) config.Config {
	cfg := config.Defaults()
	cfg.Models.HF = "model"
	cfg.Llama.BinDir = filepath.Join(root, "bin")
	cfg.Output.ResultsDir = filepath.Join(root, "results")
	cfg.Output.LogsDir = filepath.Join(root, "logs")
	return cfg
}

func testFeatures() llamacpp.Features {
	return llamacpp.Features{
		Flags: llamacpp.FeatureFlags{
			LlamaServer: llamacpp.ServerFlags{HF: "-hf", Context: "--ctx-size", Host: "--host", Port: "--port"},
		},
		LlamaCpp: domain.BuildInfo{CommitShort: "test"},
	}
}
