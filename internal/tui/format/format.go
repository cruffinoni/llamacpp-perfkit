package format

import (
	"fmt"
	"math"
	"strings"
)

// FormatContextSize formats a token count as a human-readable string, using
// "k" suffixes for values of 1000 or more.
func FormatContextSize(tokens int) string {
	if tokens < 1000 {
		return fmt.Sprintf("%d", tokens)
	}
	return fmt.Sprintf("%.0fk", math.Round(float64(tokens)/1000))
}

// FormatGiBFromMiB converts a value in mebibytes to a GiB string.
func FormatGiBFromMiB(mib float64) string {
	return fmt.Sprintf("%.2f GiB", mib/1024.0)
}

// FormatDuration formats a duration in seconds as a fixed-width string.
func FormatDuration(seconds float64) string {
	return fmt.Sprintf("%.2fs", seconds)
}

// FormatElapsed formats an elapsed duration in seconds as HH:MM:SS or MM:SS.
func FormatElapsed(seconds float64) string {
	whole := int(seconds)
	minutes := whole / 60
	secs := whole % 60
	hours := minutes / 60
	mins := minutes % 60
	if hours > 0 {
		return fmt.Sprintf("%d:%02d:%02d", hours, mins, secs)
	}
	return fmt.Sprintf("%02d:%02d", mins, secs)
}

// FormatTokS formats a tokens-per-second value with appropriate precision for
// the given metric kind ("prompt" or other).
func FormatTokS(value float64, kind string) string {
	if kind == "prompt" {
		if value >= 100 || value == 0 {
			return fmt.Sprintf("%.0f", value)
		}
		return fmt.Sprintf("%.1f", value)
	}
	return fmt.Sprintf("%.1f", value)
}

// FormatProgress renders a progress bar string of the given width.
func FormatProgress(done int, total int, width int) string {
	if width < 0 {
		width = 0
	}
	if total <= 0 {
		return "[" + strings.Repeat("░", width) + "]"
	}
	if done < 0 {
		done = 0
	}
	if done > total {
		done = total
	}
	filled := int(math.Round((float64(done) / float64(total)) * float64(width)))
	if filled > width {
		filled = width
	}
	return "[" + strings.Repeat("█", filled) + strings.Repeat("░", width-filled) + "]"
}
