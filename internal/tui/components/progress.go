package components

import (
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/theme"
)

// ProgressBar renders a themed progress bar string using the given style.
func ProgressBar(s theme.Styles, style ProgressBarStyle, done, total, width int) string {
	opts := ProgressBarOptions{
		Style:        style,
		Width:        width,
		ShowBrackets: true,
		FilledStyle:  s.ProgressFilled,
		EmptyStyle:   s.ProgressEmpty,
	}
	return NewProgressBarRenderer(opts).Render(done, total)
}
