package theme

import "github.com/charmbracelet/lipgloss"

// Palette defines the 16-color Solarized Dark palette.
type Palette struct {
	BG      lipgloss.Color
	Panel   lipgloss.Color
	Border  lipgloss.Color
	Text    lipgloss.Color
	Muted   lipgloss.Color
	Title   lipgloss.Color
	Accent  lipgloss.Color
	Cyan    lipgloss.Color
	Green   lipgloss.Color
	Yellow  lipgloss.Color
	Orange  lipgloss.Color
	Red     lipgloss.Color
	Magenta lipgloss.Color
	Blue    lipgloss.Color
}

var SolarizedDark = Palette{
	BG:      lipgloss.Color("#002b36"),
	Panel:   lipgloss.Color("#073642"),
	Border:  lipgloss.Color("#586e75"),
	Text:    lipgloss.Color("#839496"),
	Muted:   lipgloss.Color("#657b83"),
	Title:   lipgloss.Color("#eee8d5"),
	Accent:  lipgloss.Color("#b58900"),
	Cyan:    lipgloss.Color("#2aa198"),
	Green:   lipgloss.Color("#859900"),
	Yellow:  lipgloss.Color("#b58900"),
	Orange:  lipgloss.Color("#cb4b16"),
	Red:     lipgloss.Color("#dc322f"),
	Magenta: lipgloss.Color("#d33682"),
	Blue:    lipgloss.Color("#268bd2"),
}

// Styles encapsulates all lipgloss styling primitives.
type Styles struct {
	Base     lipgloss.Style
	Panel    lipgloss.Style
	Title    lipgloss.Style
	Label    lipgloss.Style
	Muted    lipgloss.Style
	Accent   lipgloss.Style
	Cyan     lipgloss.Style
	Green    lipgloss.Style
	Yellow   lipgloss.Style
	Orange   lipgloss.Style
	Red      lipgloss.Style
	Blue     lipgloss.Style
	TextBold lipgloss.Style

	progressBarFilled lipgloss.Style
	progressBarEmpty  lipgloss.Style
}

// NewStyles builds the complete style set from the palette.
func NewStyles() Styles {
	p := SolarizedDark
	return Styles{
		Base:     lipgloss.NewStyle().Foreground(p.Text).Background(p.BG),
		Panel:    lipgloss.NewStyle().Foreground(p.Text).Background(p.Panel).Border(lipgloss.RoundedBorder()).BorderForeground(p.Border).Padding(0, 1),
		Title:    lipgloss.NewStyle().Foreground(p.Title).Bold(true),
		Label:    lipgloss.NewStyle().Foreground(p.Muted),
		Muted:    lipgloss.NewStyle().Foreground(p.Muted),
		Accent:   lipgloss.NewStyle().Foreground(p.Accent).Bold(true),
		Cyan:     lipgloss.NewStyle().Foreground(p.Cyan),
		Green:    lipgloss.NewStyle().Foreground(p.Green),
		Yellow:   lipgloss.NewStyle().Foreground(p.Yellow),
		Orange:   lipgloss.NewStyle().Foreground(p.Orange),
		Red:      lipgloss.NewStyle().Foreground(p.Red),
		Blue:     lipgloss.NewStyle().Foreground(p.Blue),
		TextBold: lipgloss.NewStyle().Foreground(p.Text).Bold(true),

		// Progress bar colors — filled uses green for completed work, empty is muted
		progressBarFilled: lipgloss.NewStyle().Foreground(p.Green),
		progressBarEmpty:  lipgloss.NewStyle().Foreground(p.Muted),
	}
}

// StatusStyle returns the appropriate color style for a job status.
func StatusStyle(s Styles, status string) lipgloss.Style {
	switch status {
	case "success":
		return s.Green
	case "running":
		return s.Cyan
	case "pending":
		return s.Muted
	case "timeout":
		return s.Yellow
	case "oom", "failed":
		return s.Red
	default:
		return s.Muted
	}
}

// PhaseStyle returns the appropriate color style for a job phase.
func PhaseStyle(s Styles, phase string) lipgloss.Style {
	switch phase {
	case "prefill", "generating", "starting":
		return s.Cyan
	case "done", "pending", "-":
		return s.Muted
	case "timeout":
		return s.Yellow
	case "oom", "failed":
		return s.Red
	default:
		return s.Muted
	}
}

// ProgressBarFilled returns the style for filled progress bar segments.
func ProgressBarFilled() lipgloss.Style {
	return NewStyles().progressBarFilled
}

// ProgressBarEmpty returns the style for empty progress bar segments.
func ProgressBarEmpty() lipgloss.Style {
	return NewStyles().progressBarEmpty
}
