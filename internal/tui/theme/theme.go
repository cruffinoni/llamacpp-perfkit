package theme

import (
	"github.com/charmbracelet/lipgloss"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
)

// Theme defines the complete color scheme for the TUI.
type Theme struct {
	Background string
	Panel      string
	Border     string
	Text       string
	Title      string
	Muted      string
	Accent     string

	Success string
	Running string
	Warning string
	Error   string
	Info    string

	ProgressLow  string
	ProgressMid  string
	ProgressHigh string

	VramFreeLow  string
	VramFreeHigh string
	VramUsedLow  string
	VramUsedHigh string
}

// Styles encapsulates all lipgloss styling primitives.
type Styles struct {
	Base           lipgloss.Style
	Panel          lipgloss.Style
	Title          lipgloss.Style
	Label          lipgloss.Style
	Muted          lipgloss.Style
	Accent         lipgloss.Style
	Success        lipgloss.Style
	Running        lipgloss.Style
	Warning        lipgloss.Style
	Error          lipgloss.Style
	Info           lipgloss.Style
	TextBold       lipgloss.Style
	StatusLine     lipgloss.Style
	ProgressFilled lipgloss.Style
	ProgressEmpty  lipgloss.Style

	PanelBg lipgloss.Color
	TextFg  lipgloss.Color
	MutedFg lipgloss.Color
}

// NewStyles builds the complete style set from a theme.
// Text styles do not set a background — the rendering primitives (LineBuilder,
// Table, Panel) handle background continuity.
func NewStyles(t Theme) Styles {
	return Styles{
		Base:       lipgloss.NewStyle().Foreground(lipgloss.Color(t.Text)).Background(lipgloss.Color(t.Background)),
		Panel:      lipgloss.NewStyle().Foreground(lipgloss.Color(t.Text)).Background(lipgloss.Color(t.Panel)).Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color(t.Border)).Padding(0, 1),
		Title:      lipgloss.NewStyle().Foreground(lipgloss.Color(t.Title)).Bold(true),
		Label:      lipgloss.NewStyle().Foreground(lipgloss.Color(t.Muted)),
		Muted:      lipgloss.NewStyle().Foreground(lipgloss.Color(t.Muted)),
		Accent:     lipgloss.NewStyle().Foreground(lipgloss.Color(t.Accent)).Bold(true),
		Success:    lipgloss.NewStyle().Foreground(lipgloss.Color(t.Success)),
		Running:    lipgloss.NewStyle().Foreground(lipgloss.Color(t.Running)),
		Warning:    lipgloss.NewStyle().Foreground(lipgloss.Color(t.Warning)),
		Error:      lipgloss.NewStyle().Foreground(lipgloss.Color(t.Error)),
		Info:       lipgloss.NewStyle().Foreground(lipgloss.Color(t.Info)),
		TextBold:   lipgloss.NewStyle().Foreground(lipgloss.Color(t.Text)).Bold(true),
		StatusLine: lipgloss.NewStyle().Foreground(lipgloss.Color(t.Muted)).Background(lipgloss.Color(t.Background)),

		ProgressFilled: lipgloss.NewStyle().Foreground(lipgloss.Color(t.Info)).Background(lipgloss.Color(t.Background)),
		ProgressEmpty:  lipgloss.NewStyle().Foreground(lipgloss.Color(t.Muted)).Background(lipgloss.Color(t.Background)),

		PanelBg: lipgloss.Color(t.Panel),
		TextFg:  lipgloss.Color(t.Text),
		MutedFg: lipgloss.Color(t.Muted),
	}
}

// StatusStyle returns the appropriate color style for a job status.
func StatusStyle(s Styles, status domain.RunStatus) lipgloss.Style {
	switch status {
	case domain.StatusSuccess:
		return s.Success
	case domain.StatusRunning:
		return s.Running
	case domain.StatusPending:
		return s.Muted
	case domain.StatusTimeout:
		return s.Warning
	case domain.StatusOOM, domain.StatusFailed:
		return s.Error
	default:
		return s.Muted
	}
}

// PhaseStyle returns the appropriate color style for a job phase.
func PhaseStyle(s Styles, phase domain.Phase) lipgloss.Style {
	switch phase {
	case domain.PhasePrefill, domain.PhaseGenerating, domain.PhaseStarting:
		return s.Info
	case domain.PhaseDone, domain.PhasePending, "-":
		return s.Muted
	case domain.PhaseTimeout:
		return s.Warning
	case domain.PhaseOOM, domain.PhaseFailed:
		return s.Error
	default:
		return s.Muted
	}
}
