package format

import (
	"fmt"
	"math"
	"strings"
)

func FormatContextSize(tokens int) string {
	if tokens < 1000 {
		return fmt.Sprintf("%d", tokens)
	}
	return fmt.Sprintf("%.0fk", math.Round(float64(tokens)/1000))
}

func FormatGiBFromMiB(mib float64) string {
	return fmt.Sprintf("%.2f GiB", mib/1024.0)
}

func FormatDuration(seconds float64) string {
	return fmt.Sprintf("%.2fs", seconds)
}

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

func FormatTokS(value float64, kind string) string {
	if kind == "prompt" {
		if value >= 100 || value == 0 {
			return fmt.Sprintf("%.0f", value)
		}
		return fmt.Sprintf("%.1f", value)
	}
	return fmt.Sprintf("%.1f", value)
}

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

// Token represents a named color token used for formatting.
type Token int

const (
	TokenBg      Token = iota // #002b36
	TokenPanel                // #073642
	TokenBorder               // #586e75
	TokenText                 // #839496
	TokenMuted                // #657b83
	TokenTitle                // #eee8d5
	TokenAccent               // #b58900
	TokenCyan                 // #2aa198
	TokenGreen                // #859900
	TokenYellow               // #b58900
	TokenOrange               // #cb4b16
	TokenRed                  // #dc322f
	TokenMagenta              // #d33682
	TokenBlue                 // #268bd2

	// Semantic color tokens for values
	TokenSuccess    = TokenGreen
	TokenRunning    = TokenCyan
	TokenPending    = TokenMuted
	TokenTimeout    = TokenYellow
	TokenOOM        = TokenRed
	TokenFailed     = TokenRed
	TokenPrefill    = TokenBlue
	TokenGenerating = TokenCyan
	TokenDone       = TokenMuted
)

// ColorForStatus returns the color token for a given status.
func ColorForStatus(status string) Token {
	switch status {
	case "success":
		return TokenSuccess
	case "running":
		return TokenRunning
	case "pending":
		return TokenPending
	case "timeout":
		return TokenTimeout
	case "oom", "failed":
		return TokenFailed
	default:
		return TokenMuted
	}
}

// ColorForPhase returns the color token for a given phase.
func ColorForPhase(phase string) Token {
	switch phase {
	case "prefill", "generating", "starting":
		return TokenGenerating
	case "done", "pending", "-":
		return TokenDone
	case "timeout":
		return TokenTimeout
	case "oom", "failed":
		return TokenFailed
	default:
		return TokenMuted
	}
}
