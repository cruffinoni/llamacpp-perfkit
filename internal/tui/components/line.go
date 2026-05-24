package components

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// LineBuilder constructs a single line of text with mixed styling and consistent background.
// Every segment — styled or raw — carries the line's background, so ANSI resets
// between segments never expose the terminal default.
type LineBuilder struct {
	bg  lipgloss.Color
	fg  lipgloss.Color
	buf strings.Builder
}

// NewLineBuilder returns a LineBuilder that uses bg as the background for every
// segment and fg as the default foreground for Raw segments.
func NewLineBuilder(bg, fg lipgloss.Color) *LineBuilder {
	return &LineBuilder{bg: bg, fg: fg}
}

// Styled appends text rendered with the given style. The style's foreground and
// attributes are preserved; the line's background overrides any background on
// the style.
func (b *LineBuilder) Styled(style lipgloss.Style, text string) *LineBuilder {
	b.buf.WriteString(style.Background(b.bg).Render(text))
	return b
}

// Raw appends unstyled text using the line's default foreground and background.
func (b *LineBuilder) Raw(text string) *LineBuilder {
	s := lipgloss.NewStyle().Foreground(b.fg).Background(b.bg).Render(text)
	b.buf.WriteString(s)
	return b
}

// Rawf appends unstyled formatted text using the line's default foreground
// and background.
func (b *LineBuilder) Rawf(format string, args ...any) *LineBuilder {
	return b.Raw(fmt.Sprintf(format, args...))
}

// Render returns the complete line with background continuity guaranteed.
func (b *LineBuilder) Render() string {
	return b.buf.String()
}
