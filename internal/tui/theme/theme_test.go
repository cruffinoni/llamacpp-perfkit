package theme

import (
	"sort"
	"testing"

	"github.com/charmbracelet/lipgloss"
)

func TestStatusStyleMappings(t *testing.T) {
	s := NewStyles(SolarizedDark)
	cases := []struct {
		status string
		want   lipgloss.Color
	}{
		{"success", lipgloss.Color(SolarizedDark.Success)},
		{"running", lipgloss.Color(SolarizedDark.Running)},
		{"pending", lipgloss.Color(SolarizedDark.Muted)},
		{"timeout", lipgloss.Color(SolarizedDark.Warning)},
		{"oom", lipgloss.Color(SolarizedDark.Error)},
		{"failed", lipgloss.Color(SolarizedDark.Error)},
		{"unknown", lipgloss.Color(SolarizedDark.Muted)},
	}
	// Sort by status for deterministic ordering.
	sort.Slice(cases, func(i, j int) bool { return cases[i].status < cases[j].status })
	for _, tc := range cases {
		if got := StatusStyle(s, tc.status).GetForeground(); got != tc.want {
			t.Errorf("StatusStyle(%q) = %v, want %v", tc.status, got, tc.want)
		}
	}
}

func TestPhaseStyleMappings(t *testing.T) {
	s := NewStyles(SolarizedDark)
	cases := []struct {
		phase string
		want  lipgloss.Color
	}{
		{"prefill", lipgloss.Color(SolarizedDark.Info)},
		{"generating", lipgloss.Color(SolarizedDark.Info)},
		{"starting", lipgloss.Color(SolarizedDark.Info)},
		{"done", lipgloss.Color(SolarizedDark.Muted)},
		{"pending", lipgloss.Color(SolarizedDark.Muted)},
		{"-", lipgloss.Color(SolarizedDark.Muted)},
		{"timeout", lipgloss.Color(SolarizedDark.Warning)},
		{"oom", lipgloss.Color(SolarizedDark.Error)},
		{"failed", lipgloss.Color(SolarizedDark.Error)},
		{"unknown", lipgloss.Color(SolarizedDark.Muted)},
	}
	// Sort by phase for deterministic ordering.
	sort.Slice(cases, func(i, j int) bool { return cases[i].phase < cases[j].phase })
	for _, tc := range cases {
		if got := PhaseStyle(s, tc.phase).GetForeground(); got != tc.want {
			t.Errorf("PhaseStyle(%q) = %v, want %v", tc.phase, got, tc.want)
		}
	}
}

func TestStyleDefaults(t *testing.T) {
	s := NewStyles(SolarizedDark)
	if s.Base.GetBackground() != lipgloss.Color(SolarizedDark.Background) {
		t.Error("Base should have theme background")
	}
	if s.Base.GetForeground() != lipgloss.Color(SolarizedDark.Text) {
		t.Error("Base should have theme text color")
	}
	_, top, _, _, _ := s.Panel.GetBorder()
	if !top {
		t.Error("Panel should have a border set")
	}
	if !s.Title.GetBold() {
		t.Error("Title should be bold")
	}
	if !s.TextBold.GetBold() {
		t.Error("TextBold should be bold")
	}
}

func TestSolarizedDarkConstants(t *testing.T) {
	if SolarizedDark.Background != "#002b36" {
		t.Error("Background mismatch")
	}
	if SolarizedDark.Success != "#bad600" {
		t.Error("Success mismatch")
	}
	if SolarizedDark.Error != "#e02f30" {
		t.Error("Error mismatch")
	}
	if SolarizedDark.Info != "#268bd2" {
		t.Error("Info mismatch")
	}
}

func TestProgressStyles(t *testing.T) {
	s := NewStyles(SolarizedDark)
	if s.ProgressFilled.GetForeground() != lipgloss.Color(SolarizedDark.Info) {
		t.Error("ProgressFilled should use Info theme color")
	}
	if s.ProgressEmpty.GetForeground() != lipgloss.Color(SolarizedDark.Muted) {
		t.Error("ProgressEmpty should use Muted theme color")
	}
}

func TestNewStylesWithCustomTheme(t *testing.T) {
	custom := Theme{
		Background: "#000000",
		Text:       "#ffffff",
		Panel:      "#111111",
		Border:     "#333333",
		Title:      "#ffffff",
		Muted:      "#666666",
		Accent:     "#ff0000",
		Success:    "#00ff00",
		Running:    "#0000ff",
		Warning:    "#ffff00",
		Error:      "#ff0000",
		Info:       "#00ffff",
	}
	s := NewStyles(custom)
	if s.Base.GetBackground() != lipgloss.Color("#000000") {
		t.Error("Custom theme background should be applied")
	}
	if s.Success.GetForeground() != lipgloss.Color("#00ff00") {
		t.Error("Custom theme success should be applied")
	}
}
