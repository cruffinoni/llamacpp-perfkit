package components

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/theme"
)

// CompactHeader renders a compact header with label, optional underline, and value.
func CompactHeader(s theme.Styles, label string, underline bool, value string) string {
	var lines []string
	lines = append(lines, s.Title.Render(label))
	if underline && value != "" {
		lines = append(lines, s.Muted.Render(strings.Repeat("-", len(label))))
	}
	if value != "" {
		lines = append(lines, value)
	}
	return s.Panel.Render(strings.Join(lines, "\n"))
}

// TableSeparator renders a dashed separator line.
func TableSeparator(width int, s theme.Styles) string {
	return s.Muted.Render(strings.Repeat("-", width))
}

// StatusBadge renders a small status indicator with its color.
func StatusBadge(s theme.Styles, status string, padding int) string {
	var pad strings.Builder
	for i := 0; i < 2*padding; i++ {
		pad.WriteByte(' ')
	}
	paddingStr := pad.String()
	color := theme.StatusStyle(s, status)
	return color.Render(paddingStr + status + paddingStr)
}

// BorderedBox renders a bordered panel with title and content.
func BorderedBox(s theme.Styles, title string, underline bool, content ...string) string {
	var lines []string
	lines = append(lines, s.Title.Render(title))
	if underline && len(content) > 0 {
		lines = append(lines, s.Muted.Render(strings.Repeat("-", len(title))))
	}
	for _, line := range content {
		lines = append(lines, s.Muted.Render(line))
	}
	return s.Panel.Render(strings.Join(lines, "\n"))
}

// PanelPadding creates a padded string for lipgloss box padding.
func PanelPadding(top, bottom, left, right int) string {
	if top == 0 && bottom == 0 && left == 0 && right == 0 {
		return ""
	}
	return fmt.Sprintf("%d,%d,%d,%d", top, bottom, left, right)
}

// ProgressBarChar returns the filled and empty characters for progress bars.
func ProgressBarChar() (filled, empty string) {
	return "█", "░"
}

// TokenColor returns the lipgloss color for a given theme token name.
func TokenColor(name string) lipgloss.Color {
	switch name {
	case "bg":
		return theme.SolarizedDark.BG
	case "panel":
		return theme.SolarizedDark.Panel
	case "border":
		return theme.SolarizedDark.Border
	case "text":
		return theme.SolarizedDark.Text
	case "muted":
		return theme.SolarizedDark.Muted
	case "title":
		return theme.SolarizedDark.Title
	case "accent":
		return theme.SolarizedDark.Accent
	case "cyan":
		return theme.SolarizedDark.Cyan
	case "green":
		return theme.SolarizedDark.Green
	case "yellow":
		return theme.SolarizedDark.Yellow
	case "orange":
		return theme.SolarizedDark.Orange
	case "red":
		return theme.SolarizedDark.Red
	case "magenta":
		return theme.SolarizedDark.Magenta
	case "blue":
		return theme.SolarizedDark.Blue
	default:
		return theme.SolarizedDark.Text
	}
}
