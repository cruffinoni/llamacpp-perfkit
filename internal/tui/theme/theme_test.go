package theme

import (
	"sort"
	"testing"

	"github.com/charmbracelet/lipgloss"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
)

func TestStatusStyleMappings(t *testing.T) {
	s := NewStyles(SolarizedDark)
	cases := []struct {
		status domain.RunStatus
		want   lipgloss.Color
	}{
		{domain.StatusSuccess, lipgloss.Color(SolarizedDark.Success)},
		{domain.StatusRunning, lipgloss.Color(SolarizedDark.Running)},
		{domain.StatusPending, lipgloss.Color(SolarizedDark.Muted)},
		{domain.StatusTimeout, lipgloss.Color(SolarizedDark.Warning)},
		{domain.StatusOOM, lipgloss.Color(SolarizedDark.Error)},
		{domain.StatusFailed, lipgloss.Color(SolarizedDark.Error)},
		{domain.StatusUnknown, lipgloss.Color(SolarizedDark.Muted)},
	}
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
		phase domain.Phase
		want  lipgloss.Color
	}{
		{domain.PhasePrefill, lipgloss.Color(SolarizedDark.Info)},
		{domain.PhaseGenerating, lipgloss.Color(SolarizedDark.Info)},
		{domain.PhaseStarting, lipgloss.Color(SolarizedDark.Info)},
		{domain.PhaseDone, lipgloss.Color(SolarizedDark.Muted)},
		{domain.PhasePending, lipgloss.Color(SolarizedDark.Muted)},
		{"-", lipgloss.Color(SolarizedDark.Muted)},
		{domain.PhaseTimeout, lipgloss.Color(SolarizedDark.Warning)},
		{domain.PhaseOOM, lipgloss.Color(SolarizedDark.Error)},
		{domain.PhaseFailed, lipgloss.Color(SolarizedDark.Error)},
		{"unknown", lipgloss.Color(SolarizedDark.Muted)},
	}
	sort.Slice(cases, func(i, j int) bool { return cases[i].phase < cases[j].phase })
	for _, tc := range cases {
		if got := PhaseStyle(s, tc.phase).GetForeground(); got != tc.want {
			t.Errorf("PhaseStyle(%q) = %v, want %v", tc.phase, got, tc.want)
		}
	}
}
