package components

import (
	tuifmt "github.com/cruffinoni/llamacpp-perfkit/internal/tui/format"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/theme"
)

// ProgressBar renders a themed progress bar string.
func ProgressBar(s theme.Styles, done, total, width int) string {
	raw := tuifmt.FormatProgress(done, total, width)
	var result string
	for _, r := range raw {
		switch r {
		case '█':
			result += s.Info.Render(string(r))
		case '░':
			result += s.Muted.Render(string(r))
		default:
			result += string(r)
		}
	}
	return result
}
