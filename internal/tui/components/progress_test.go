package components

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/theme"
)

func TestParseProgressBarStyle(t *testing.T) {
	tests := map[string]struct {
		input     string
		want      ProgressBarStyle
		wantError bool
	}{
		"block is parsed":     {input: "block", want: ProgressBarStyleBlock},
		"line is parsed":      {input: "line", want: ProgressBarStyleLine},
		"dot is parsed":       {input: "dot", want: ProgressBarStyleDot},
		"segmented is parsed": {input: "segmented", want: ProgressBarStyleSegmented},
		"braille is parsed":   {input: "braille", want: ProgressBarStyleBraille},
		"uppercase BLOCK is parsed case-insensitively": {input: "BLOCK", want: ProgressBarStyleBlock},
		"invalid style returns error":                  {input: "invalid", wantError: true},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := ParseProgressBarStyle(tc.input)
			if tc.wantError {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "unknown progress bar style")
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.want, got)
		})
	}
}

func TestProgressBarStyleString(t *testing.T) {
	tests := map[string]struct {
		style ProgressBarStyle
		want  string
	}{
		"block returns block":     {style: ProgressBarStyleBlock, want: "block"},
		"dot returns dot":         {style: ProgressBarStyleDot, want: "dot"},
		"braille returns braille": {style: ProgressBarStyleBraille, want: "braille"},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.want, tc.style.String())
		})
	}
}

func TestProgressBar(t *testing.T) {
	s := theme.NewStyles(theme.SolarizedDark)
	tests := map[string]struct {
		done, total, width int
		wantNotEmpty       bool
		wantPrefix         string
		wantSuffix         string
	}{
		"empty bar at 0 of 10":         {done: 0, total: 10, width: 10, wantNotEmpty: true, wantPrefix: "["},
		"full bar at 10 of 10":         {done: 10, total: 10, width: 10, wantNotEmpty: true, wantSuffix: "]"},
		"zero total does not panic":    {done: 0, total: 0, width: 10, wantNotEmpty: true},
		"zero width produces brackets": {done: 5, total: 10, width: 0, wantPrefix: "[", wantSuffix: "]"},
		"halfway bar at 5 of 10":       {done: 5, total: 10, width: 10, wantNotEmpty: true},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			rendered := ProgressBar(s, ProgressBarStyleBlock, tc.done, tc.total, tc.width)
			visible := stripANSI(rendered)
			if tc.wantNotEmpty {
				assert.NotEmpty(t, rendered)
			}
			if tc.wantPrefix != "" {
				assert.True(t, strings.HasPrefix(visible, tc.wantPrefix))
			}
			if tc.wantSuffix != "" {
				assert.True(t, strings.HasSuffix(visible, tc.wantSuffix))
			}
		})
	}
}

func TestProgressBarRenderer(t *testing.T) {
	s := theme.NewStyles(theme.SolarizedDark)
	discreteStyles := []ProgressBarStyle{
		ProgressBarStyleBlock, ProgressBarStyleLine,
		ProgressBarStyleDot, ProgressBarStyleSegmented,
	}

	tests := map[string]struct {
		style        ProgressBarStyle
		width        int
		showBrackets bool
		done         int
		total        int
		wantLen      int
		wantStr      string
		wantNoBrack  bool
	}{
		"block zero percent":                     {style: ProgressBarStyleBlock, width: 30, showBrackets: true, done: 0, total: 10, wantLen: 32},
		"block fifty percent":                    {style: ProgressBarStyleBlock, width: 30, showBrackets: true, done: 5, total: 10, wantLen: 32},
		"block full":                             {style: ProgressBarStyleBlock, width: 30, showBrackets: true, done: 10, total: 10, wantLen: 32},
		"block done greater than total clamps":   {style: ProgressBarStyleBlock, width: 10, showBrackets: true, done: 20, total: 10, wantLen: 12},
		"block negative done clamps to zero":     {style: ProgressBarStyleBlock, width: 10, showBrackets: true, done: -5, total: 10, wantLen: 12},
		"block zero total":                       {style: ProgressBarStyleBlock, width: 10, showBrackets: true, done: 0, total: 0, wantLen: 12},
		"block zero width":                       {style: ProgressBarStyleBlock, width: 0, showBrackets: true, done: 5, total: 10, wantStr: "[]"},
		"block no brackets":                      {style: ProgressBarStyleBlock, width: 10, showBrackets: false, done: 5, total: 10, wantLen: 10, wantNoBrack: true},
		"braille zero percent":                   {style: ProgressBarStyleBraille, width: 30, showBrackets: true, done: 0, total: 10, wantLen: 32},
		"braille full":                           {style: ProgressBarStyleBraille, width: 30, showBrackets: true, done: 10, total: 10, wantLen: 32},
		"braille done greater than total clamps": {style: ProgressBarStyleBraille, width: 5, showBrackets: true, done: 20, total: 10, wantLen: 7},
		"braille negative done clamps to zero":   {style: ProgressBarStyleBraille, width: 5, showBrackets: true, done: -5, total: 10, wantLen: 7},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			opts := ProgressBarOptions{
				Style: tc.style, Width: tc.width, ShowBrackets: tc.showBrackets,
				FilledStyle: s.ProgressFilled, EmptyStyle: s.ProgressEmpty,
			}
			r := NewProgressBarRenderer(opts)
			rendered := r.Render(tc.done, tc.total)
			visible := stripANSI(rendered)
			if tc.wantStr != "" {
				assert.Equal(t, tc.wantStr, visible)
				return
			}
			if tc.wantLen > 0 {
				assert.Len(t, []rune(visible), tc.wantLen)
			}
			if tc.wantNoBrack {
				assert.NotContains(t, visible, "[")
				assert.NotContains(t, visible, "]")
			}
		})
	}

	// Test all discrete styles at zero, fifty, and full
	for _, style := range discreteStyles {
		styleName := style.String()
		t.Run("all discrete styles "+styleName+" zero percent", func(t *testing.T) {
			opts := ProgressBarOptions{
				Style: style, Width: 30, ShowBrackets: true,
				FilledStyle: s.ProgressFilled, EmptyStyle: s.ProgressEmpty,
			}
			r := NewProgressBarRenderer(opts)
			visible := stripANSI(r.Render(0, 10))
			assert.Len(t, []rune(visible), 32)
		})
		t.Run("all discrete styles "+styleName+" fifty percent", func(t *testing.T) {
			opts := ProgressBarOptions{
				Style: style, Width: 30, ShowBrackets: true,
				FilledStyle: s.ProgressFilled, EmptyStyle: s.ProgressEmpty,
			}
			r := NewProgressBarRenderer(opts)
			visible := stripANSI(r.Render(5, 10))
			assert.Len(t, []rune(visible), 32)
		})
		t.Run("all discrete styles "+styleName+" full", func(t *testing.T) {
			opts := ProgressBarOptions{
				Style: style, Width: 30, ShowBrackets: true,
				FilledStyle: s.ProgressFilled, EmptyStyle: s.ProgressEmpty,
			}
			r := NewProgressBarRenderer(opts)
			visible := stripANSI(r.Render(10, 10))
			assert.Len(t, []rune(visible), 32)
		})
	}
}

// stripANSI removes ANSI escape sequences and returns visible text only.
func stripANSI(s string) string {
	var out strings.Builder
	esc := false
	for _, r := range s {
		if esc {
			if (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') {
				esc = false
			}
			continue
		}
		if r == '\x1b' {
			esc = true
			continue
		}
		out.WriteRune(r)
	}
	return out.String()
}
