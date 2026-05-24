package theme

import "github.com/charmbracelet/lipgloss"

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

var SolarizedDark = Theme{
	Background: "#002b36",
	Panel:      "#073642",
	Border:     "#586e75",
	Text:       "#eee8d5",
	Title:      "#fdf6e3",
	Muted:      "#586e75",
	Accent:     "#b58900",

	Success: "#bad600",
	Running: "#268bd2",
	Warning: "#d6a200",
	Error:   "#e02f30",
	Info:    "#268bd2",

	ProgressLow:  "#4e5900",
	ProgressMid:  "#d6a200",
	ProgressHigh: "#bad600",

	VramFreeLow:  "#4e5900",
	VramFreeHigh: "#bad600",
	VramUsedLow:  "#6e1718",
	VramUsedHigh: "#e02f30",
}

// Styles encapsulates all lipgloss styling primitives.
type Styles struct {
	Base       lipgloss.Style
	Panel      lipgloss.Style
	Title      lipgloss.Style
	Label      lipgloss.Style
	Muted      lipgloss.Style
	Accent     lipgloss.Style
	Success    lipgloss.Style
	Running    lipgloss.Style
	Warning    lipgloss.Style
	Error      lipgloss.Style
	Info       lipgloss.Style
	TextBold   lipgloss.Style
	StatusLine lipgloss.Style

	PanelBg lipgloss.Color
	BaseBg  lipgloss.Color
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
		PanelBg:    lipgloss.Color(t.Panel),
		BaseBg:     lipgloss.Color(t.Background),
		TextFg:     lipgloss.Color(t.Text),
		MutedFg:    lipgloss.Color(t.Muted),
	}
}

// StatusStyle returns the appropriate color style for a job status.
func StatusStyle(s Styles, status string) lipgloss.Style {
	switch status {
	case "success":
		return s.Success
	case "running":
		return s.Running
	case "pending":
		return s.Muted
	case "timeout":
		return s.Warning
	case "oom", "failed":
		return s.Error
	default:
		return s.Muted
	}
}

// PhaseStyle returns the appropriate color style for a job phase.
func PhaseStyle(s Styles, phase string) lipgloss.Style {
	switch phase {
	case "prefill", "generating", "starting":
		return s.Info
	case "done", "pending", "-":
		return s.Muted
	case "timeout":
		return s.Warning
	case "oom", "failed":
		return s.Error
	default:
		return s.Muted
	}
}
