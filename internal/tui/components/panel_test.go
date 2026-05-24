package components

import (
	"testing"

	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/theme"
)

func TestPanelWithContent(t *testing.T) {
	s := theme.NewStyles(theme.SolarizedDark)
	rendered := Panel(s, "hello")
	if rendered == "" {
		t.Fatal("Panel should not return empty string")
	}
	if !containsSub(rendered, "hello") {
		t.Error("Panel should contain content")
	}
}

func TestPanelWithEmptyContent(t *testing.T) {
	s := theme.NewStyles(theme.SolarizedDark)
	rendered := Panel(s, "")
	if rendered == "" {
		t.Fatal("Panel with empty content should still render borders")
	}
}
