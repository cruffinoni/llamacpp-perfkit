package views

import (
	"strings"
	"testing"

	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/components"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/theme"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/viewmodel"
)

func testState() viewmodel.BenchmarkTUIState {
	return viewmodel.BenchmarkTUIState{
		RunID:     "run-test",
		ModelName: "test-model.gguf",
		BuildInfo: viewmodel.BuildInfoView{
			CommitShort: "abc123",
			Branch:      "main",
			Backend:     "server",
		},
		Progress: viewmodel.ProgressState{
			ServersTotal:       1,
			JobsTotal:          1,
			CurrentPromptTotal: 1,
		},
		StatusMessage: "ready",
	}
}

func TestLayoutRejectsUnsupportedTerminalSizes(t *testing.T) {
	s := theme.NewStyles(theme.SolarizedDark)
	tests := map[string]struct {
		width  int
		height int
	}{
		"too narrow": {width: 30, height: MinTerminalHeight},
		"too short":  {width: MinTerminalWidth, height: 10},
		"zero size":  {width: 0, height: 0},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			size := TerminalSize{Width: tc.width, Height: tc.height}
			rendered := Layout(testState(), LayoutOptions{
				Styles:   s,
				BarStyle: components.ProgressBarStyleDot,
				Size:     size,
			})

			if !strings.Contains(rendered, "Terminal size unsupported.") {
				t.Fatalf("expected unsupported-size message, got %q", rendered)
			}
			if strings.Contains(rendered, "llama-cpp-perfkit") {
				t.Fatalf("dashboard should not render when terminal is unsupported: %q", rendered)
			}
		})
	}
}

func TestLayoutRendersDashboardAtMinimumSupportedSize(t *testing.T) {
	s := theme.NewStyles(theme.SolarizedDark)
	size := TerminalSize{Width: MinTerminalWidth, Height: MinTerminalHeight}
	rendered := Layout(testState(), LayoutOptions{
		Styles:   s,
		BarStyle: components.ProgressBarStyleDot,
		Size:     size,
	})

	if strings.Contains(rendered, "Terminal size unsupported.") {
		t.Fatalf("minimum supported size should render dashboard, got %q", rendered)
	}
	if !strings.Contains(rendered, "llama-cpp-perfkit") {
		t.Fatalf("dashboard header missing from render: %q", rendered)
	}
	if !strings.Contains(rendered, "No prompts.") {
		t.Fatalf("prompt panel missing from render: %q", rendered)
	}
}

func TestCenterComponentPadsWithinAvailableWidth(t *testing.T) {
	rendered := centerComponent(5, "x")

	if rendered != "  x  " {
		t.Fatalf("centerComponent(5, x) = %q, want %q", rendered, "  x  ")
	}
}
