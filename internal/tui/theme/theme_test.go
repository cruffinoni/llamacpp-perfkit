package theme

import (
	"testing"

	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/theme"
)

func TestStatusAndPhaseStyles(t *testing.T) {
	s := NewStyles()
	cases := map[string]theme.SolarizedDark{
		"success":    SolarizedDark.Green,
		"running":    SolarizedDark.Cyan,
		"pending":    SolarizedDark.Muted,
		"timeout":    SolarizedDark.Yellow,
		"oom":        SolarizedDark.Red,
		"failed":     SolarizedDark.Red,
		"unknown":    SolarizedDark.Muted,
	}
	for status, want := range cases {
		if got := StatusStyle(s, status).GetForeground(); got != want {
			t.Errorf("StatusStyle(%q) = %v, want %v", status, got, want)
		}
	}
	for phase, want := range cases {
		if got := PhaseStyle(s, phase).GetForeground(); got != want {
			t.Errorf("PhaseStyle(%q) = %v, want %v", phase, got, want)
		}
	}
}

func TestStatusStyleMappings(t *testing.T) {
	cases := map[string]theme.SolarizedDark{
		"success": theme.SolarizedDark.Green,
		"running": theme.SolarizedDark.Cyan,
		"pending": theme.SolarizedDark.Muted,
		"timeout": theme.SolarizedDark.Yellow,
		"oom":     theme.SolarizedDark.Red,
		"failed":  theme.SolarizedDark.Red,
		"unknown": theme.SolarizedDark.Muted,
	}
	for status, want := range cases {
		s := NewStyles()
		if got := StatusStyle(s, status).GetForeground(); got != want {
			t.Errorf("StatusStyle(%q) = %v, want %v", status, got, want)
		}
	}
}

func TestPhaseStyleMappings(t *testing.T) {
	cases := map[string]theme.SolarizedDark{
		"prefill":    theme.SolarizedDark.Cyan,
		"generating": theme.SolarizedDark.Cyan,
		"starting":   theme.SolarizedDark.Cyan,
		"done":       theme.SolarizedDark.Muted,
		"pending":    theme.SolarizedDark.Muted,
		"-":          theme.SolarizedDark.Muted,
		"timeout":    theme.SolarizedDark.Yellow,
		"oom":        theme.SolarizedDark.Red,
		"failed":     theme.SolarizedDark.Red,
		"unknown":    theme.SolarizedDark.Muted,
	}
	for phase, want := range cases {
		s := NewStyles()
		if got := PhaseStyle(s, phase).GetForeground(); got != want {
			t.Errorf("PhaseStyle(%q) = %v, want %v", phase, got, want)
		}
	}
}

func TestProgressColors(t *testing.T) {
	s := NewStyles()
	if s.progressBarFilled.GetForeground() != theme.SolarizedDark.Green {
		t.Error("ProgressBarFilled should use green")
	}
	if s.progressBarEmpty.GetForeground() != theme.SolarizedDark.Muted {
		t.Error("ProgressBarEmpty should use muted")
	}
}

func TestPaletteConstants(t *testing.T) {
	expected := map[theme.Token]theme.SolarizedDark{
		theme.TokenBg:      theme.SolarizedDark.BG,
		theme.TokenPanel:   theme.SolarizedDark.Panel,
		theme.TokenBorder:  theme.SolarizedDark.Border,
		theme.TokenText:    theme.SolarizedDark.Text,
		theme.TokenMuted:   theme.SolarizedDark.Muted,
		theme.TokenTitle:   theme.SolarizedDark.Title,
		theme.TokenAccent:  theme.SolarizedDark.Accent,
		theme.TokenCyan:    theme.SolarizedDark.Cyan,
		theme.TokenGreen:   theme.SolarizedDark.Green,
		theme.TokenYellow:  theme.SolarizedDark.Yellow,
		theme.TokenOrange:  theme.SolarizedDark.Orange,
		theme.TokenRed:     theme.SolarizedDark.Red,
		theme.TokenMagenta: theme.SolarizedDark.Magenta,
		theme.TokenBlue:    theme.SolarizedDark.Blue,
	}
	for token, want := range expected {
		if got := token; got != theme.Token(0) && (got == theme.TokenBg || got == theme.TokenPanel) {
			t.Errorf("Token %d has unknown color mapping", got)
		}
	}
}

func TestStyleDefaults(t *testing.T) {
	s := NewStyles()
	if s.Base.GetBackground() != theme.SolarizedDark.BG {
		t.Error("Base should have default background")
	}
	if s.Panel.GetBorder() == nil {
		t.Error("Panel should have rounded border")
	}
	if !s.Title.Bold() {
		t.Error("Title should be bold")
	}
	if !s.TextBold.Bold() {
		t.Error("TextBold should be bold")
	}
}
	if StatusStyle(styles, "oom").GetForeground() != SolarizedDark.Red {
		t.Fatal("oom should use red")
	}
	if PhaseStyle(styles, "generating").GetForeground() != SolarizedDark.Cyan {
		t.Fatal("generating should use cyan")
	}
	if PhaseStyle(styles, "pending").GetForeground() != SolarizedDark.Muted {
		t.Fatal("pending should use muted")
	}
}

func TestStatusStyleMappings(t *testing.T) {
	cases := map[string]theme.SolarizedDark{
		"success": theme.SolarizedDark.Green,
		"running": theme.SolarizedDark.Cyan,
		"pending": theme.SolarizedDark.Muted,
		"timeout": theme.SolarizedDark.Yellow,
		"oom":     theme.SolarizedDark.Red,
		"failed":  theme.SolarizedDark.Red,
		"unknown": theme.SolarizedDark.Muted,
	}
	for status, want := range cases {
		s := NewStyles()
		if got := StatusStyle(s, status).GetForeground(); got != want {
			t.Errorf("StatusStyle(%q) = %v, want %v", status, got, want)
		}
	}
}

func TestPhaseStyleMappings(t *testing.T) {
	cases := map[string]theme.SolarizedDark{
		"prefill":    theme.SolarizedDark.Cyan,
		"generating": theme.SolarizedDark.Cyan,
		"starting":   theme.SolarizedDark.Cyan,
		"done":       theme.SolarizedDark.Muted,
		"pending":    theme.SolarizedDark.Muted,
		"-":          theme.SolarizedDark.Muted,
		"timeout":    theme.SolarizedDark.Yellow,
		"oom":        theme.SolarizedDark.Red,
		"failed":     theme.SolarizedDark.Red,
		"unknown":    theme.SolarizedDark.Muted,
	}
	for phase, want := range cases {
		s := NewStyles()
		if got := PhaseStyle(s, phase).GetForeground(); got != want {
			t.Errorf("PhaseStyle(%q) = %v, want %v", phase, got, want)
		}
	}
}

func TestProgressColors(t *testing.T) {
	s := NewStyles()
	if s.progressBarFilled.GetForeground() != theme.SolarizedDark.Green {
		t.Error("ProgressBarFilled should use green")
	}
	if s.progressBarEmpty.GetForeground() != theme.SolarizedDark.Muted {
		t.Error("ProgressBarEmpty should use muted")
	}
}

func TestPaletteConstants(t *testing.T) {
	expected := map[theme.Token]theme.SolarizedDark{
		theme.TokenBg:      theme.SolarizedDark.BG,
		theme.TokenPanel:   theme.SolarizedDark.Panel,
		theme.TokenBorder:  theme.SolarizedDark.Border,
		theme.TokenText:    theme.SolarizedDark.Text,
		theme.TokenMuted:   theme.SolarizedDark.Muted,
		theme.TokenTitle:   theme.SolarizedDark.Title,
		theme.TokenAccent:  theme.SolarizedDark.Accent,
		theme.TokenCyan:    theme.SolarizedDark.Cyan,
		theme.TokenGreen:   theme.SolarizedDark.Green,
		theme.TokenYellow:  theme.SolarizedDark.Yellow,
		theme.TokenOrange:  theme.SolarizedDark.Orange,
		theme.TokenRed:     theme.SolarizedDark.Red,
		theme.TokenMagenta: theme.SolarizedDark.Magenta,
		theme.TokenBlue:    theme.SolarizedDark.Blue,
	}
	for token, want := range expected {
		if got := token; got != theme.Token(0) && (got == theme.TokenBg || got == theme.TokenPanel) {
			t.Errorf("Token %d has unknown color mapping", got)
		}
	}
}

func TestStyleDefaults(t *testing.T) {
	s := NewStyles()
	if s.Base.GetBackground() != theme.SolarizedDark.BG {
		t.Error("Base should have default background")
	}
	if s.Panel.GetBorder() == nil {
		t.Error("Panel should have rounded border")
	}
	if !s.Title.Bold() {
		t.Error("Title should be bold")
	}
	if !s.TextBold.Bold() {
		t.Error("TextBold should be bold")
	}
}

func TestPhaseStyleMappings(t *testing.T) {
	s := NewStyles()
	cases := map[string]theme.SolarizedDark{
		"prefill":    SolarizedDark.Cyan,
		"generating": SolarizedDark.Cyan,
		"starting":   SolarizedDark.Cyan,
		"done":       SolarizedDark.Muted,
		"pending":    SolarizedDark.Muted,
		"-":          SolarizedDark.Muted,
		"timeout":    SolarizedDark.Yellow,
		"oom":        SolarizedDark.Red,
		"failed":     SolarizedDark.Red,
		"unknown":    SolarizedDark.Muted,
	}
	for phase, want := range cases {
		if got := PhaseStyle(s, phase).GetForeground(); got != want {
			t.Errorf("PhaseStyle(%q) = %v, want %v", phase, got, want)
		}
	}
}

func TestProgressColors(t *testing.T) {
	s := NewStyles()
	if s.progressBarFilled.GetForeground() != theme.SolarizedDark.Green {
		t.Error("ProgressBarFilled should use green")
	}
	if s.progressBarEmpty.GetForeground() != theme.SolarizedDark.Muted {
		t.Error("ProgressBarEmpty should use muted")
	}
}

func TestPaletteConstants(t *testing.T) {
	expected := map[theme.Token]theme.SolarizedDark{
		theme.TokenBg:      theme.SolarizedDark.BG,
		theme.TokenPanel:   theme.SolarizedDark.Panel,
		theme.TokenBorder:  theme.SolarizedDark.Border,
		theme.TokenText:    theme.SolarizedDark.Text,
		theme.TokenMuted:   theme.SolarizedDark.Muted,
		theme.TokenTitle:   theme.SolarizedDark.Title,
		theme.TokenAccent:  theme.SolarizedDark.Accent,
		theme.TokenCyan:    theme.SolarizedDark.Cyan,
		theme.TokenGreen:   theme.SolarizedDark.Green,
		theme.TokenYellow:  theme.SolarizedDark.Yellow,
		theme.TokenOrange:  theme.SolarizedDark.Orange,
		theme.TokenRed:     theme.SolarizedDark.Red,
		theme.TokenMagenta: theme.SolarizedDark.Magenta,
		theme.TokenBlue:    theme.SolarizedDark.Blue,
	}
	for token, want := range expected {
		if got := token; got != theme.Token(0) && (got == theme.TokenBg || got == theme.TokenPanel) {
			t.Errorf("Token %d has unknown color mapping", got)
		}
	}
}

func TestStyleDefaults(t *testing.T) {
	s := NewStyles()
	if s.Base.GetBackground() != theme.SolarizedDark.BG {
		t.Error("Base should have default background")
	}
	if s.Panel.GetBorder() == nil {
		t.Error("Panel should have rounded border")
	}
	if !s.Title.Bold() {
		t.Error("Title should be bold")
	}
	if !s.TextBold.Bold() {
		t.Error("TextBold should be bold")
	}
}
