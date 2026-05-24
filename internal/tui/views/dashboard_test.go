package views

import (
	"testing"

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
	if rendered == "" {
		t.Fatal("Header should not return empty string")
	}
	if !containsSubstring(rendered, "llama-cpp-perfkit") {
		t.Error("Header should contain project name")
	}
	if !containsSubstring(rendered, "b64739ea") {
		t.Error("Header should contain commit hash")
	}
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
	rendered := ProgressBlock(state, s)
	if rendered == "" {
		t.Fatal("ProgressBlock should not return empty string")
	}
	if !containsSubstring(rendered, "14/96") || !containsSubstring(rendered, "104/768") ||
		!containsSubstring(rendered, "5/8 prompts") {
		t.Error("ProgressBlock should show progress ratios")
	}
	if !checkProgressBarPresence(rendered) {
		t.Error("ProgressBlock should contain progress bars")
	}
}

func TestCurrentServerBlockRendering(t *testing.T) {
	tests := []struct {
		name   string
		server *viewmodel.CurrentServerView
	}{
		{
			name: "with server",
			server: &viewmodel.CurrentServerView{
				ID:          "1779537195-server-0014",
				ContextSize: 8192, KVType: "q8_0", NCPUMOE: 18,
				SpecType: "draft-mtp", BatchSize: 512, UBatchSize: 512,
			},
		},
		{
			name:   "nil server",
			server: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state := viewmodel.BenchmarkTUIState{CurrentServer: tt.server}
			s := theme.NewStyles(theme.SolarizedDark)
			rendered := CurrentServerBlock(state, s)
			if rendered == "" {
				t.Fatal("CurrentServerBlock should not return empty string")
			}
			if !containsSubstring(rendered, "Current server") {
				t.Error("Should contain 'Current server' header")
			}
			if tt.server != nil && !containsSubstring(rendered, "8k") {
				t.Error("With server, should contain context size in k format")
			}
		})
	}
}

func TestPromptTableRendering(t *testing.T) {
	state := viewmodel.BenchmarkTUIState{
		PromptJobs: []viewmodel.PromptJobView{
			{Profile: "code_python", Status: "success", Phase: "done",
				DurationSeconds: func() *float64 { v := 2.70; return &v }(),
				GenTokS:         func() *float64 { v := 78.0; return &v }(),
				PromptTokS:      func() *float64 { v := 812.0; return &v }(),
				MinVRAMMiB:      func() *float64 { v := 5520.0; return &v }()},
			{Profile: "long_prefill_32k", Status: "running", Phase: "generating",
				DurationSeconds: func() *float64 { v := 1.91; return &v }(),
				GenTokS:         func() *float64 { v := 79.3; return &v }(),
				PromptTokS:      func() *float64 { v := 902.0; return &v }(),
				MinVRAMMiB:      func() *float64 { v := 5509.0; return &v }()},
			{Profile: "long_prefill_48k", Status: "pending", Phase: "-"},
		},
	}
	s := theme.NewStyles(theme.SolarizedDark)
	rendered := PromptTable(state, s)
	if rendered == "" {
		t.Fatal("PromptTable should not return empty string")
	}
	if !containsSubstring(rendered, "profile") || !containsSubstring(rendered, "status") {
		t.Error("Should contain table header")
	}
	if rendered == "" || len(rendered) < 100 {
		t.Skip("Render too small to analyze colors")
	}
}

func TestPromptTableNilColumns(t *testing.T) {
	tests := []struct {
		name          string
		duration      *float64
		genTokS       *float64
		promptTokS    *float64
		minVRAMMiB    *float64
		expectedMinus int
	}{
		{"all nil", nil, nil, nil, nil, 92},
		{"some nil", func() *float64 { v := 2.0; return &v }(), nil, nil, nil, 91},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			job := viewmodel.PromptJobView{
				Profile: "test", Status: "pending", Phase: "-",
				DurationSeconds: tt.duration, GenTokS: tt.genTokS,
				PromptTokS: tt.promptTokS, MinVRAMMiB: tt.minVRAMMiB,
			}
			state := viewmodel.BenchmarkTUIState{PromptJobs: []viewmodel.PromptJobView{job}}
			s := theme.NewStyles(theme.SolarizedDark)
			rendered := PromptTable(state, s)
			minusCount := countSubstring(rendered, "-")
			if minusCount != tt.expectedMinus {
				t.Errorf("Expected %d hyphens for missing values, got %d", tt.expectedMinus, minusCount)
			}
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
	rendered := Layout(state, s, 0)
	if rendered == "" {
		t.Fatal("Layout should not return empty string")
	}
	expectedElements := []string{"llama-cpp-perfkit", "1779537110", "b64739ea",
		"14/96", "test", "success"}
	for _, elem := range expectedElements {
		if !containsSubstring(rendered, elem) {
			t.Errorf("Layout should contain %q", elem)
		}
	}
}

func TestNilSafeFormatting(t *testing.T) {
	state := viewmodel.BenchmarkTUIState{
		PromptJobs: []viewmodel.PromptJobView{
			{Profile: "test", Status: "pending", Phase: "-",
				DurationSeconds: nil, GenTokS: nil,
				PromptTokS: nil, MinVRAMMiB: nil},
		},
	}
	s := theme.NewStyles(theme.SolarizedDark)
	rendered := PromptTable(state, s)
	if countSubstring(rendered, "-") < 5 {
		t.Error("Should have hyphens for missing column values")
	}
}

func containsSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func checkProgressBarPresence(s string) bool {
	return containsSubstring(s, "[") && containsSubstring(s, "█")
}

func countSubstring(s, substr string) int {
	count := 0
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			count++
		}
	}
	return count
}
