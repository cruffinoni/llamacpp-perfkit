package views

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/components"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/theme"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/viewmodel"
)

func TestBenchmarkHeaderRendering(t *testing.T) {
	state := viewmodel.BenchmarkTUIState{
		RunID:     "1779537110",
		BuildInfo: viewmodel.BuildInfoView{CommitShort: "b64739ea", Branch: "master", Backend: "cuda"},
		ModelName: "Qwen3.6-35B-A3B-MTP UD-Q4_K_M",
	}
	s := theme.NewStyles(theme.SolarizedDark)
	rendered := BenchmarkHeader(state, s)
	assert.NotEmpty(t, rendered)
	assert.Contains(t, rendered, "llama-cpp-perfkit")
	assert.Contains(t, rendered, "b64739ea")
}

func TestProgressBlockRendering(t *testing.T) {
	state := viewmodel.BenchmarkTUIState{
		Progress: viewmodel.ProgressState{
			ServersCompleted: 14, ServersTotal: 96,
			JobsCompleted: 104, JobsTotal: 768,
			CurrentPrompt: 5, CurrentPromptTotal: 8,
		},
		ElapsedSeconds: 522, ETASeconds: 2350,
	}
	s := theme.NewStyles(theme.SolarizedDark)

	tests := map[string]components.ProgressBarStyle{
		"block style": components.ProgressBarStyleBlock,
		"line style":  components.ProgressBarStyleLine,
	}
	for name, style := range tests {
		t.Run(name, func(t *testing.T) {
			rendered := ProgressBlock(state, s, style)
			assert.NotEmpty(t, rendered)
			assert.Contains(t, rendered, "14/96")
			assert.Contains(t, rendered, "104/768")
			assert.Contains(t, rendered, "5/8 prompts")
			assert.True(t, strings.Contains(rendered, "["), "should contain progress bars")
		})
	}
}

func TestCurrentServerBlockRendering(t *testing.T) {
	tests := map[string]struct {
		server *viewmodel.CurrentServerView
	}{
		"with server": {server: &viewmodel.CurrentServerView{
			ID: "1779537195-server-0014", ContextSize: 8192, KVType: "q8_0",
			NCPUMOE: 18, SpecType: "draft-mtp", BatchSize: 512, UBatchSize: 512,
		}},
		"nil server": {server: nil},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			state := viewmodel.BenchmarkTUIState{CurrentServer: tc.server}
			s := theme.NewStyles(theme.SolarizedDark)
			rendered := CurrentServerBlock(state, s)
			assert.NotEmpty(t, rendered)
			assert.Contains(t, rendered, "Current server")
			if tc.server != nil {
				assert.Contains(t, rendered, "8k")
			}
		})
	}
}

func TestPromptTableRendering(t *testing.T) {
	tests := map[string]struct {
		jobs   []viewmodel.PromptJobView
		verify func(*testing.T, string)
	}{
		"with jobs shows header and data": {
			jobs: []viewmodel.PromptJobView{
				{Profile: "code_python", Status: "success", Phase: "done",
					DurationSeconds: new(2.70), GenTokS: new(78.0), PromptTokS: new(812.0), MinVRAMMiB: new(5520.0)},
				{Profile: "long_prefill_32k", Status: "running", Phase: "generating",
					DurationSeconds: new(1.91), GenTokS: new(79.3), PromptTokS: new(902.0), MinVRAMMiB: new(5509.0)},
				{Profile: "long_prefill_48k", Status: "pending", Phase: "-"},
			},
			verify: func(t *testing.T, r string) {
				assert.Contains(t, r, "profile")
				assert.Contains(t, r, "status")
				assert.GreaterOrEqual(t, len(r), 100)
			},
		},
		"empty jobs shows no prompts message": {
			jobs: nil,
			verify: func(t *testing.T, r string) {
				assert.Contains(t, r, "No prompts")
			},
		},
		"nil columns format as hyphens": {
			jobs: []viewmodel.PromptJobView{
				{Profile: "test", Status: "pending", Phase: "-"},
			},
			verify: func(t *testing.T, r string) {
				assert.NotEmpty(t, r)
			},
		},
		"all nil metrics show enough hyphens": {
			jobs: []viewmodel.PromptJobView{
				{Profile: "test", Status: "pending", Phase: "-",
					DurationSeconds: nil, GenTokS: nil, PromptTokS: nil, MinVRAMMiB: nil},
			},
			verify: func(t *testing.T, r string) {
				assert.GreaterOrEqual(t, strings.Count(r, "-"), 5)
			},
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			state := viewmodel.BenchmarkTUIState{PromptJobs: tc.jobs}
			s := theme.NewStyles(theme.SolarizedDark)
			rendered := PromptTable(state, s)
			assert.NotEmpty(t, rendered)
			tc.verify(t, rendered)
		})
	}
}

func TestLayoutRendering(t *testing.T) {
	state := viewmodel.BenchmarkTUIState{
		RunID:          "1779537110",
		BuildInfo:      viewmodel.BuildInfoView{CommitShort: "b64739ea", Branch: "master", Backend: "cuda"},
		ModelName:      "Qwen3.6-35B-A3B-MTP UD-Q4_K_M",
		Progress:       viewmodel.ProgressState{ServersCompleted: 14, ServersTotal: 96},
		ElapsedSeconds: 522, ETASeconds: 2350,
		PromptJobs: []viewmodel.PromptJobView{
			{Profile: "test", Status: "success", Phase: "done"},
		},
	}
	s := theme.NewStyles(theme.SolarizedDark)
	rendered := Layout(state, s, 0, components.ProgressBarStyleBlock)
	assert.NotEmpty(t, rendered)
	for _, elem := range []string{"llama-cpp-perfkit", "1779537110", "b64739ea", "14/96", "test", "success"} {
		assert.Contains(t, rendered, elem, "Layout should contain %q", elem)
	}
}

func TestDisplayHelpers(t *testing.T) {
	t.Run("formatDuration", func(t *testing.T) {
		tests := map[string]struct {
			input *float64
			want  string
		}{
			"nil returns dash":  {input: nil, want: "-"},
			"2.7 returns 2.70s": {input: new(2.7), want: "2.70s"},
		}
		for name, tc := range tests {
			t.Run(name, func(t *testing.T) {
				assert.Equal(t, tc.want, formatDuration(tc.input))
			})
		}
	})

	t.Run("formatTokS", func(t *testing.T) {
		tests := map[string]struct {
			value *float64
			kind  string
			want  string
		}{
			"nil returns dash":           {value: nil, kind: "gen", want: "-"},
			"gen 78 returns 78.0":        {value: new(78.0), kind: "gen", want: "78.0"},
			"prompt 812 returns integer": {value: new(812.0), kind: "prompt", want: "812"},
		}
		for name, tc := range tests {
			t.Run(name, func(t *testing.T) {
				assert.Equal(t, tc.want, formatTokS(tc.value, tc.kind))
			})
		}
	})

	t.Run("formatVRAM", func(t *testing.T) {
		tests := map[string]struct {
			value *float64
			want  string
		}{
			"nil returns dash":     {value: nil, want: "-"},
			"5520 MiB returns GiB": {value: new(5520.0), want: "5.39 GiB"},
		}
		for name, tc := range tests {
			t.Run(name, func(t *testing.T) {
				assert.Equal(t, tc.want, formatVRAM(tc.value))
			})
		}
	})
}
