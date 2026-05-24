package tui

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"

	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/components"
)

func TestModelStoresWindowSize(t *testing.T) {
	m := model{barStyle: components.ProgressBarStyleDot}

	updated, cmd := m.Update(tea.WindowSizeMsg{Width: 92, Height: 20})
	if cmd != nil {
		t.Fatal("window size update should not return a command")
	}

	got, ok := updated.(model)
	if !ok {
		t.Fatalf("updated model type = %T, want tui.model", updated)
	}
	if got.width != 92 || got.height != 20 {
		t.Fatalf("stored size = %dx%d, want 92x20", got.width, got.height)
	}
}

func TestModelViewUsesStoredWindowSize(t *testing.T) {
	m := model{barStyle: components.ProgressBarStyleDot}

	if got := m.View().Content; !strings.Contains(got, "Terminal size unsupported.") {
		t.Fatalf("zero-size model should render unsupported message, got %q", got)
	}

	m.width = 92
	m.height = 20
	if got := m.View().Content; strings.Contains(got, "Terminal size unsupported.") {
		t.Fatalf("supported-size model should render dashboard, got %q", got)
	}
}
