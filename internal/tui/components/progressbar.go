package components

import (
	"fmt"
	"math"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// ProgressBarStyle enumerates available progress bar visual styles.
type ProgressBarStyle int

const (
	ProgressBarStyleBlock     ProgressBarStyle = iota // █ / ░
	ProgressBarStyleLine                              // ━ / ─
	ProgressBarStyleDot                               // • / ·
	ProgressBarStyleSegmented                         // ■ / □
	ProgressBarStyleBraille                           // braille 8 sub-levels per cell
)

var progressBarStyleNames = [...]string{"block", "line", "dot", "segmented", "braille"}

// String returns the canonical name of the style.
func (s ProgressBarStyle) String() string {
	if s < 0 || int(s) >= len(progressBarStyleNames) {
		return "block"
	}
	return progressBarStyleNames[s]
}

// ParseProgressBarStyle parses a style name string. Unknown values return an error.
func ParseProgressBarStyle(name string) (ProgressBarStyle, error) {
	switch strings.ToLower(name) {
	case "block":
		return ProgressBarStyleBlock, nil
	case "line":
		return ProgressBarStyleLine, nil
	case "dot":
		return ProgressBarStyleDot, nil
	case "segmented":
		return ProgressBarStyleSegmented, nil
	case "braille":
		return ProgressBarStyleBraille, nil
	default:
		return 0, fmt.Errorf("unknown progress bar style %q; valid values: block, line, dot, segmented, braille", name)
	}
}

// ProgressBarOptions controls the rendering of a progress bar.
type ProgressBarOptions struct {
	Style        ProgressBarStyle
	Width        int
	ShowBrackets bool
	FilledStyle  lipgloss.Style
	EmptyStyle   lipgloss.Style
}

// ProgressBarRenderer renders progress bars with pre-configured options.
type ProgressBarRenderer struct {
	opts ProgressBarOptions
}

// NewProgressBarRenderer creates a renderer for the given options. Width is
// clamped to 0 if negative.
func NewProgressBarRenderer(opts ProgressBarOptions) *ProgressBarRenderer {
	if opts.Width < 0 {
		opts.Width = 0
	}
	return &ProgressBarRenderer{opts: opts}
}

// Render produces a styled progress bar string for the given progress values.
func (r *ProgressBarRenderer) Render(done, total int) string {
	frac := fraction(done, total)
	if r.opts.Style == ProgressBarStyleBraille {
		return r.renderBraille(frac)
	}
	return r.renderDiscrete(frac)
}

// fraction computes done/total clamped to [0,1]. Returns 0 when total <= 0.
func fraction(done, total int) float64 {
	if total <= 0 {
		return 0
	}
	if done <= 0 {
		return 0
	}
	f := float64(done) / float64(total)
	if f > 1.0 {
		return 1.0
	}
	return f
}

type discreteCharSet struct {
	filled rune
	empty  rune
}

var discreteChars = map[ProgressBarStyle]discreteCharSet{
	ProgressBarStyleBlock:     {'█', '░'},
	ProgressBarStyleLine:      {'━', '─'},
	ProgressBarStyleDot:       {'•', '·'},
	ProgressBarStyleSegmented: {'■', '□'},
}

func (r *ProgressBarRenderer) renderDiscrete(frac float64) string {
	chars, ok := discreteChars[r.opts.Style]
	if !ok {
		chars = discreteChars[ProgressBarStyleBlock]
	}

	filled := int(math.Round(frac * float64(r.opts.Width)))
	if filled > r.opts.Width {
		filled = r.opts.Width
	}
	if filled < 0 {
		filled = 0
	}

	var buf strings.Builder
	if r.opts.ShowBrackets {
		buf.WriteByte('[')
	}
	for i := 0; i < filled; i++ {
		buf.WriteString(r.opts.FilledStyle.Render(string(chars.filled)))
	}
	for i := filled; i < r.opts.Width; i++ {
		buf.WriteString(r.opts.EmptyStyle.Render(string(chars.empty)))
	}
	if r.opts.ShowBrackets {
		buf.WriteByte(']')
	}
	return buf.String()
}

// brailleLevels maps fill level 0-8 to the low byte of the Unicode braille
// codepoint (U+2800 + offset). Level 0 is empty, level 8 is full (all 8 dots).
// Dot numbering (standard braille):
//
//	1 • 4
//	2 • 5
//	3 • 6
//	7 • 8
//
// Bits 0-7 in the low byte correspond to dots 1-8. Dots are filled from
// bottom to top, then left to right.
var brailleLevels = [9]rune{
	0x000, // 0 — no dots:          U+2800 (⠀)
	0x080, // 1 — dot 8:            U+2880 (⢀)
	0x0C0, // 2 — dots 7,8:         U+28C0 (⣀)
	0x0E0, // 3 — dots 6,7,8:       U+28E0 (⣠)
	0x0F0, // 4 — dots 5,6,7,8:     U+28F0 (⣰)
	0x0F8, // 5 — dots 4-8:         U+28F8 (⣸)
	0x0FC, // 6 — dots 3-8:         U+28FC (⣼)
	0x0FE, // 7 — dots 2-8:         U+28FE (⣾)
	0x0FF, // 8 — all 8 dots:       U+28FF (⣿)
}

// renderBraille renders a braille progress bar with 8 sub-levels per cell.
// Each braille cell has 8 dot positions, giving the bar 8× the resolution of
// a discrete bar. Characters may render differently across terminal emulators;
// some display them as dotted patterns, others as solid blocks.
func (r *ProgressBarRenderer) renderBraille(frac float64) string {
	totalSubUnits := r.opts.Width * 8
	filledSubUnits := int(math.Round(frac * float64(totalSubUnits)))
	if filledSubUnits > totalSubUnits {
		filledSubUnits = totalSubUnits
	}
	if filledSubUnits < 0 {
		filledSubUnits = 0
	}

	var buf strings.Builder
	if r.opts.ShowBrackets {
		buf.WriteByte('[')
	}
	for i := 0; i < r.opts.Width; i++ {
		cellFilled := filledSubUnits - i*8
		switch {
		case cellFilled >= 8:
			cellFilled = 8
		case cellFilled <= 0:
			cellFilled = 0
		}
		ch := rune(0x2800 + brailleLevels[cellFilled])
		if cellFilled == 0 {
			buf.WriteString(r.opts.EmptyStyle.Render(string(ch)))
		} else {
			buf.WriteString(r.opts.FilledStyle.Render(string(ch)))
		}
	}
	if r.opts.ShowBrackets {
		buf.WriteByte(']')
	}
	return buf.String()
}
