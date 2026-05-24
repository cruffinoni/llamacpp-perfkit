package storage

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
)

func TestWriteRunSummary(t *testing.T) {
	tests := map[string]struct {
		summary domain.RunSummary
		verify  func(*testing.T, string, domain.RunSummary)
	}{
		"full roundtrip with metrics and readback": {
			summary: domain.RunSummary{
				RunID: "run-a", BatchID: "batch-a", CreatedAt: "2026-01-01T00:00:00Z",
				Model: "model:A", PromptProfile: "code",
				ServerConfig: domain.ServerConfig{ContextSize: 2048, KVType: "q8_0"},
				Status:       domain.NewRunStatusInfo(domain.StatusSuccess, ""),
				ConfigHash:   "hash-a", DurationSec: 10,
				Response: map[string]any{"content": "ok"},
			},
			verify: func(t *testing.T, root string, summary domain.RunSummary) {
				writtenPath, err := WriteRunSummary(root, summary)
				require.NoError(t, err)
				assert.NotEmpty(t, writtenPath)

				_, err = os.Stat(SummaryPath(RunDir(root, "run-a")))
				require.NoError(t, err)

				err = AppendSystemMetric(RunDir(root, "run-a"), domain.SystemMetricSample{
					Time: "t1", VRAMFree: new(4096.0), VRAMUsed: new(1024.0),
				})
				require.NoError(t, err)

				err = AppendLlamaMetric(RunDir(root, "run-a"), domain.LlamaCppMetricSample{
					Time: "t1", GenerationTokS: new(44.0), PromptEvalTokS: new(120.0),
				})
				require.NoError(t, err)

				rows, err := LoadRuns(root)
				require.NoError(t, err)
				require.Len(t, rows, 1)

				loaded := rows[0]
				assert.Equal(t, "run-a", loaded.Summary.RunID)
				require.NotNil(t, loaded.SystemSummary.MinVRAMFreeMiB)
				assert.InDelta(t, 4096.0, *loaded.SystemSummary.MinVRAMFreeMiB, 0.0001)
				require.NotNil(t, loaded.LlamaSummary.GenerationTokS)
				assert.InDelta(t, 44.0, *loaded.LlamaSummary.GenerationTokS, 0.0001)
				assert.NotEmpty(t, loaded.Directory)
			},
		},
		"initializes system and llama metric files": {
			summary: domain.RunSummary{
				RunID: "init-test", CreatedAt: "2026-01-01T00:00:00Z", Model: "test",
				Status: domain.NewRunStatusInfo(domain.StatusSuccess, ""),
			},
			verify: func(t *testing.T, root string, _ domain.RunSummary) {
				_, err := WriteRunSummary(root, domain.RunSummary{
					RunID: "init-test", CreatedAt: "2026-01-01T00:00:00Z", Model: "test",
					Status: domain.NewRunStatusInfo(domain.StatusSuccess, ""),
				})
				require.NoError(t, err)
				runDir := RunDir(root, "init-test")
				_, err = os.Stat(SystemMetricsPath(runDir))
				require.NoError(t, err, "system.jsonl should be initialized")
				_, err = os.Stat(LlamaMetricsPath(runDir))
				require.NoError(t, err, "llamacpp.jsonl should be initialized")
			},
		},
		"marshal indent produces readable JSON with trailing newline": {
			summary: domain.RunSummary{
				RunID: "indent-test", CreatedAt: "2026-01-01T00:00:00Z",
				Status: domain.NewRunStatusInfo(domain.StatusSuccess, ""),
			},
			verify: func(t *testing.T, root string, _ domain.RunSummary) {
				summary := domain.RunSummary{
					RunID: "indent-test", CreatedAt: "2026-01-01T00:00:00Z",
					Status: domain.NewRunStatusInfo(domain.StatusSuccess, ""),
				}
				_, err := WriteRunSummary(root, summary)
				require.NoError(t, err)
				data, err := os.ReadFile(SummaryPath(RunDir(root, "indent-test")))
				require.NoError(t, err)
				assert.True(t, strings.Contains(string(data), "  "), "should be indented JSON")
				assert.True(t, strings.HasSuffix(string(data), "\n"), "should end with newline")
			},
		},
		"roundtrip preserves server config": {
			summary: domain.RunSummary{
				RunID: "roundtrip", CreatedAt: "2026-05-01T00:00:00Z", Model: "model",
				PromptProfile: "test",
				ServerConfig:  domain.ServerConfig{ContextSize: 4096, KVType: "q4_0"},
				Status:        domain.NewRunStatusInfo(domain.StatusSuccess, ""),
				ConfigHash:    "hash123", DurationSec: 5.5,
				Command: []string{"llama-server", "--port", "8080"},
			},
			verify: func(t *testing.T, root string, summary domain.RunSummary) {
				_, err := WriteRunSummary(root, summary)
				require.NoError(t, err)
				runDir := RunDir(root, "roundtrip")
				data, err := os.ReadFile(SummaryPath(runDir))
				require.NoError(t, err)
				var parsed domain.RunSummary
				err = json.Unmarshal(data, &parsed)
				require.NoError(t, err)
				assert.Equal(t, "roundtrip", parsed.RunID)
				assert.Equal(t, 4096, parsed.ServerConfig.ContextSize)
				assert.Equal(t, "q4_0", parsed.ServerConfig.KVType)
			},
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			tc.verify(t, t.TempDir(), tc.summary)
		})
	}
}

func TestReadJSONL(t *testing.T) {
	tests := map[string]struct {
		setup   func(string) error
		want    []map[string]any
		wantErr string
	}{
		"valid JSONL returns parsed rows": {
			setup: func(path string) error { return os.WriteFile(path, []byte("{\"a\":1}\n{\"b\":2}\n"), 0o644) },
			want:  []map[string]any{{"a": float64(1)}, {"b": float64(2)}},
		},
		"empty file returns nil": {
			setup: func(path string) error { return os.WriteFile(path, []byte{}, 0o644) },
			want:  nil,
		},
		"blank lines are skipped": {
			setup: func(path string) error { return os.WriteFile(path, []byte("{\"a\":1}\n\n{\"b\":2}\n"), 0o644) },
			want:  []map[string]any{{"a": float64(1)}, {"b": float64(2)}},
		},
		"missing file returns nil": {
			setup: nil,
			want:  nil,
		},
		"parse error includes path and line number": {
			setup:   func(path string) error { return os.WriteFile(path, []byte("{\"ok\":true}\nnot-json\n"), 0o644) },
			wantErr: ":2:",
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "rows.jsonl")
			if tc.setup != nil {
				require.NoError(t, tc.setup(path))
			}
			rows, err := ReadJSONL[map[string]any](path)
			if tc.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), path)
				assert.Contains(t, err.Error(), tc.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.want, rows)
		})
	}
}

func TestReadSummary(t *testing.T) {
	tests := map[string]struct {
		setup   func(string) error
		wantErr string
	}{
		"missing file returns clear error": {
			setup:   nil,
			wantErr: "read run summary",
		},
		"invalid JSON returns parse error": {
			setup: func(dir string) error {
				path := SummaryPath(dir)
				if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
					return err
				}
				return os.WriteFile(path, []byte("not-json"), 0o644)
			},
			wantErr: "parse run summary",
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			if tc.setup != nil {
				require.NoError(t, tc.setup(dir))
			}
			_, err := ReadSummary(dir)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestAppendMetric(t *testing.T) {
	t.Run("system metric creates and appends to file", func(t *testing.T) {
		dir := t.TempDir()
		err := AppendSystemMetric(dir, domain.SystemMetricSample{Time: "t1"})
		require.NoError(t, err)
		rows, err := ReadJSONL[domain.SystemMetricSample](SystemMetricsPath(dir))
		require.NoError(t, err)
		require.Len(t, rows, 1)
		assert.Equal(t, "t1", rows[0].Time)
	})

	t.Run("llama metric creates and appends to file", func(t *testing.T) {
		dir := t.TempDir()
		err := AppendLlamaMetric(dir, domain.LlamaCppMetricSample{Time: "t1"})
		require.NoError(t, err)
		rows, err := ReadJSONL[domain.LlamaCppMetricSample](LlamaMetricsPath(dir))
		require.NoError(t, err)
		require.Len(t, rows, 1)
		assert.Equal(t, "t1", rows[0].Time)
	})
}

func TestLoadRuns(t *testing.T) {
	t.Run("multiple runs are loaded", func(t *testing.T) {
		root := t.TempDir()
		for _, runID := range []string{"run-1", "run-2", "run-3"} {
			_, err := WriteRunSummary(root, domain.RunSummary{
				RunID: runID, CreatedAt: "2026-01-01T00:00:00Z", Model: "model",
				Status: domain.NewRunStatusInfo(domain.StatusSuccess, ""),
			})
			require.NoError(t, err)
		}
		rows, err := LoadRuns(root)
		require.NoError(t, err)
		assert.Len(t, rows, 3)
	})

	t.Run("empty directory returns empty slice", func(t *testing.T) {
		rows, err := LoadRuns(t.TempDir())
		require.NoError(t, err)
		assert.Empty(t, rows)
	})
}

func TestDiscoverRunDirs(t *testing.T) {
	t.Run("empty input returns empty slice", func(t *testing.T) {
		dirs, err := DiscoverRunDirs()
		require.NoError(t, err)
		assert.Empty(t, dirs)
	})

	t.Run("discovers runs in directory", func(t *testing.T) {
		root := t.TempDir()
		_, err := WriteRunSummary(root, domain.RunSummary{
			RunID: "discover-me", CreatedAt: "2026-01-01T00:00:00Z", Model: "model",
			Status: domain.NewRunStatusInfo(domain.StatusSuccess, ""),
		})
		require.NoError(t, err)
		dirs, err := DiscoverRunDirs(root)
		require.NoError(t, err)
		assert.Len(t, dirs, 1)
		assert.Contains(t, dirs[0], "discover-me")
	})
}

func TestReadRun(t *testing.T) {
	t.Run("accepts summary.json path", func(t *testing.T) {
		root := t.TempDir()
		summary := domain.RunSummary{
			RunID: "run-x", CreatedAt: "2026-01-01T00:00:00Z", Model: "model",
			Status: domain.NewRunStatusInfo(domain.StatusSuccess, ""),
		}
		_, err := WriteRunSummary(root, summary)
		require.NoError(t, err)
		loaded, err := ReadRun(SummaryPath(RunDir(root, "run-x")))
		require.NoError(t, err)
		assert.Equal(t, "run-x", loaded.Summary.RunID)
	})
}

func TestServerConfigUnmarshalJSON(t *testing.T) {
	tests := map[string]struct {
		raw  string
		want int
	}{
		"ctx field is used":                      {raw: `{"ctx": 4096}`, want: 4096},
		"legacy context_size field is supported": {raw: `{"context_size": 8192}`, want: 8192},
		"ctx takes precedence over legacy field": {raw: `{"ctx": 4096, "context_size": 8192}`, want: 4096},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			var cfg domain.ServerConfig
			err := json.Unmarshal([]byte(tc.raw), &cfg)
			require.NoError(t, err)
			assert.Equal(t, tc.want, cfg.ContextSize)
		})
	}
}

func TestRunStatusInfo(t *testing.T) {
	tests := map[string]struct {
		raw  string
		want domain.RunStatus
	}{
		"string success unmarshals correctly":    {raw: `"success"`, want: domain.StatusSuccess},
		"string timeout unmarshals correctly":    {raw: `"timeout"`, want: domain.StatusTimeout},
		"object with error preserves error text": {raw: `{"success": true, "error": "boom"}`, want: domain.StatusSuccess},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			var info domain.RunStatusInfo
			err := json.Unmarshal([]byte(tc.raw), &info)
			require.NoError(t, err)
			assert.Equal(t, tc.want, info.Kind())
			if strings.Contains(tc.raw, "boom") {
				assert.Equal(t, "boom", *info.Error)
			}
		})
	}

	t.Run("kind variants", func(t *testing.T) {
		kinds := map[string]domain.RunStatus{
			"success":     domain.StatusSuccess,
			"failed":      domain.StatusFailed,
			"timeout":     domain.StatusTimeout,
			"oom":         domain.StatusOOM,
			"unsupported": domain.StatusUnsupported,
		}
		for label, status := range kinds {
			t.Run(label, func(t *testing.T) {
				info := domain.NewRunStatusInfo(status, "")
				assert.Equal(t, status, info.Kind())
			})
		}
	})

	t.Run("error with no explicit status becomes failed", func(t *testing.T) {
		var info domain.RunStatusInfo
		errText := "something went wrong"
		info.Error = &errText
		assert.Equal(t, domain.StatusFailed, info.Kind())
	})

	t.Run("default is unknown", func(t *testing.T) {
		var info domain.RunStatusInfo
		assert.Equal(t, domain.StatusUnknown, info.Kind())
	})
}

func TestDomainHelpers(t *testing.T) {
	t.Run("Ptr returns pointer to value", func(t *testing.T) {
		p := new(42)
		require.NotNil(t, p)
		assert.Equal(t, 42, *p)
	})

	t.Run("DerefString nil returns empty", func(t *testing.T) {
		assert.Equal(t, "", domain.DerefString(nil))
		assert.Equal(t, "hello", domain.DerefString(new("hello")))
	})

	t.Run("IntValue nil returns dash", func(t *testing.T) {
		assert.Equal(t, "-", domain.IntValue(nil))
		assert.Equal(t, "5", domain.IntValue(new(5)))
	})
}

func TestPaths(t *testing.T) {
	root := t.TempDir()
	assert.Equal(t, filepath.Join(root, "myrun"), RunDir(root, "myrun"))
	assert.Equal(t, filepath.Join(root, "myrun", "metrics"), MetricsDir(RunDir(root, "myrun")))
	assert.Equal(t, filepath.Join(root, "myrun", "metrics", "system.jsonl"), SystemMetricsPath(RunDir(root, "myrun")))
	assert.Equal(t, filepath.Join(root, "myrun", "metrics", "llamacpp.jsonl"), LlamaMetricsPath(RunDir(root, "myrun")))
	assert.Equal(t, filepath.Join(root, "myrun", "summary.json"), SummaryPath(RunDir(root, "myrun")))
}

func TestTouchCreatesIntermediateDirs(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "a", "b", "c.jsonl")
	err := touch(path)
	require.NoError(t, err)
	_, err = os.Stat(path)
	require.NoError(t, err)
}
