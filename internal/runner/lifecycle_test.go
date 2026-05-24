package runner

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

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
	err := r.recordStartupFailure(
		[]domain.PlannedRun{item}, server, "start failed",
		&completed, newTUIAdapter(nil, nowTimeForTest()),
	)
	require.NoError(t, err)
	assert.Equal(t, 1, completed)

	run, err := storage.ReadRun(storage.RunDir(r.ResultsDir, "server-a-code"))
	require.NoError(t, err)
	assert.Equal(t, domain.StatusFailed, run.Summary.Status.Kind())
	require.NotNil(t, run.Summary.Status.Error)
	assert.Equal(t, "start failed", *run.Summary.Status.Error)
	assert.True(t, strings.Contains(run.Summary.CommandShell, "llama-server"))

	_, err = os.Stat(storage.SummaryPath(storage.RunDir(r.ResultsDir, "server-a-code")))
	require.NoError(t, err)
}

func TestPrepareServerBuildsCommandWithoutLaunchingServer(t *testing.T) {
	root := t.TempDir()
	r := &Runner{
		Config:        testConfig(root),
		Features:      testFeatures(),
		RawDir:        filepath.Join(root, "logs", "raw"),
		MonitoringDir: filepath.Join(root, "logs", "monitoring"),
	}
	item := domain.PlannedRun{
		Job: domain.BenchmarkJob{
			ServerConfig: domain.ServerConfig{ContextSize: 2048, KVType: "q8_0"},
		},
	}
	server, err := r.prepareServer([]domain.PlannedRun{item}, 7)
	require.NoError(t, err)

	shell := llamacpp.CommandToShell(server.CmdArgs)
	for _, want := range []string{"llama-server", "-hf model", "--ctx-size 2048", "--host 127.0.0.1", "--port"} {
		assert.Contains(t, shell, want, "command %q missing %q", shell, want)
	}
	assert.Contains(t, server.RawPath, "server-0007.log")
	assert.Contains(t, server.MonitorPath, "server-0007.jsonl")
	assert.True(t, strings.HasPrefix(server.BaseURL, "http://127.0.0.1:"))
}

func TestRecordStartupFailureMultipleItems(t *testing.T) {
	root := t.TempDir()
	r := &Runner{
		Config:     testConfig(root),
		Features:   testFeatures(),
		ResultsDir: filepath.Join(root, "results"),
		RawDir:     filepath.Join(root, "logs", "raw"),
	}
	server := serverExecution{
		ID: "server-x", RawPath: filepath.Join(root, "logs", "raw", "server-x.log"),
		CmdArgs: []string{"llama-server"},
	}
	items := []domain.PlannedRun{
		{ConfigHash: "h1", Job: domain.BenchmarkJob{
			PromptProfile: domain.PromptProfile{Name: "a"}, PromptFile: "a.txt",
			ServerConfig: domain.ServerConfig{ContextSize: 2048},
		}},
		{ConfigHash: "h2", Job: domain.BenchmarkJob{
			PromptProfile: domain.PromptProfile{Name: "b"}, PromptFile: "b.txt",
			ServerConfig: domain.ServerConfig{ContextSize: 4096},
		}},
	}

	completed := 0
	err := r.recordStartupFailure(items, server, "crashed", &completed, newTUIAdapter(nil, nowTimeForTest()))
	require.NoError(t, err)
	assert.Equal(t, 2, completed)

	for _, id := range []string{"server-x-a", "server-x-b"} {
		run, err := storage.ReadRun(storage.RunDir(r.ResultsDir, id))
		require.NoError(t, err, "failed to read run %s", id)
		assert.Equal(t, domain.StatusFailed, run.Summary.Status.Kind())
	}
}

func TestPrepareServerUsesFreeTCPPort(t *testing.T) {
	root := t.TempDir()
	r := &Runner{
		Config:        testConfig(root),
		Features:      testFeatures(),
		RawDir:        filepath.Join(root, "logs", "raw"),
		MonitoringDir: filepath.Join(root, "logs", "monitoring"),
	}
	item := domain.PlannedRun{
		Job: domain.BenchmarkJob{
			ServerConfig: domain.ServerConfig{ContextSize: 2048, KVType: "q4_0"},
		},
	}
	server, err := r.prepareServer([]domain.PlannedRun{item}, 0)
	require.NoError(t, err)
	assert.NotEmpty(t, server.BaseURL)
	assert.NotZero(t, server.ID)
}

func TestTerminateProcessNilDoesNotPanic(t *testing.T) {
	var p *serverProcess
	assert.NotPanics(t, func() {
		p.Terminate()
	})
}

func TestTerminateProcessNilCmd(t *testing.T) {
	p := &serverProcess{}
	assert.NotPanics(t, func() {
		p.Terminate()
	})
}

// --- helpers ---

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
