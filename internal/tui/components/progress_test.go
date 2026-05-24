package components

import (
	"testing"

	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/theme"
)

func TestProgressBarEmpty(t *testing.T) {
	s := theme.NewStyles(theme.SolarizedDark)
	rendered := ProgressBar(s, 0, 10, 10)
	if rendered == "" {
		t.Fatal("ProgressBar should not return empty string")
	}
	if !containsSub(rendered, "[") {
		t.Error("ProgressBar should start with [")
	}
}

func TestProgressBarFull(t *testing.T) {
	s := theme.NewStyles(theme.SolarizedDark)
	rendered := ProgressBar(s, 10, 10, 10)
	if rendered == "" {
		t.Fatal("ProgressBar should not return empty string")
	}
	if !containsSub(rendered, "]") {
		t.Error("ProgressBar should end with ]")
	}
}

func TestProgressBarZeroTotal(t *testing.T) {
	s := theme.NewStyles(theme.SolarizedDark)
	rendered := ProgressBar(s, 0, 0, 10)
	if rendered == "" {
		t.Fatal("ProgressBar with zero total should not return empty string")
	}
	if !containsSub(rendered, "[") {
		t.Error("ProgressBar should start with [")
	}
}

func TestProgressBarZeroWidth(t *testing.T) {
	s := theme.NewStyles(theme.SolarizedDark)
	rendered := ProgressBar(s, 5, 10, 0)
	// Width 0 should still produce brackets
	if !containsSub(rendered, "[") || !containsSub(rendered, "]") {
		t.Error("ProgressBar with zero width should produce brackets")
	}
}

func TestProgressBarHalfway(t *testing.T) {
	s := theme.NewStyles(theme.SolarizedDark)
	rendered := ProgressBar(s, 5, 10, 10)
	if rendered == "" {
		t.Fatal("ProgressBar should not return empty string")
	}
}
