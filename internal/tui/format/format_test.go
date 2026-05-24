package format

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFormatContextSize(t *testing.T) {
	tests := map[string]struct {
		input int
		want  string
	}{
		"values below 1000 return as is":      {input: 512, want: "512"},
		"4096 becomes 4k":                     {input: 4096, want: "4k"},
		"8192 becomes 8k":                     {input: 8192, want: "8k"},
		"16384 becomes 16k":                   {input: 16384, want: "16k"},
		"32768 becomes 33k":                   {input: 32768, want: "33k"},
		"65536 becomes 66k":                   {input: 65536, want: "66k"},
		"131072 becomes 131k":                 {input: 131072, want: "131k"},
		"exactly 1000 becomes 1k":             {input: 1000, want: "1k"},
		"zero returns zero":                   {input: 0, want: "0"},
		"negative values are formatted as is": {input: -100, want: "-100"},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.want, FormatContextSize(tc.input))
		})
	}
}

func TestFormatGiBFromMiB(t *testing.T) {
	tests := map[string]struct {
		input float64
		want  string
	}{
		"5040 MiB becomes 4.92 GiB":   {input: 5040, want: "4.92 GiB"},
		"5190 MiB becomes 5.07 GiB":   {input: 5190, want: "5.07 GiB"},
		"14115 MiB becomes 13.78 GiB": {input: 14115, want: "13.78 GiB"},
		"zero MiB becomes 0.00 GiB":   {input: 0, want: "0.00 GiB"},
		"1024 MiB becomes 1.00 GiB":   {input: 1024, want: "1.00 GiB"},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.want, FormatGiBFromMiB(tc.input))
		})
	}
}

func TestFormatDuration(t *testing.T) {
	tests := map[string]struct {
		input float64
		want  string
	}{
		"2.7 seconds formats with two decimals": {input: 2.7, want: "2.70s"},
		"integer formats with two decimals":     {input: 10, want: "10.00s"},
		"zero formats with two decimals":        {input: 0, want: "0.00s"},
		"fractional formats correctly":          {input: 0.5, want: "0.50s"},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.want, FormatDuration(tc.input))
		})
	}
}

func TestFormatElapsed(t *testing.T) {
	tests := map[string]struct {
		input float64
		want  string
	}{
		"522 seconds becomes 08:42":        {input: 522, want: "08:42"},
		"3862 seconds becomes 1:04:22":     {input: 3862, want: "1:04:22"},
		"zero seconds becomes 00:00":       {input: 0, want: "00:00"},
		"less than one minute":             {input: 45, want: "00:45"},
		"exactly one hour becomes 1:00:00": {input: 3600, want: "1:00:00"},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.want, FormatElapsed(tc.input))
		})
	}
}

func TestFormatTokS(t *testing.T) {
	tests := map[string]struct {
		value float64
		kind  string
		want  string
	}{
		"generation 78 returns 78.0":            {value: 78, kind: "gen", want: "78.0"},
		"prompt 812 large returns integer":      {value: 812, kind: "prompt", want: "812"},
		"prompt 87.4 small returns one decimal": {value: 87.4, kind: "prompt", want: "87.4"},
		"prompt zero returns integer":           {value: 0, kind: "prompt", want: "0"},
		"prompt 63.24 returns 63.2":             {value: 63.24, kind: "prompt", want: "63.2"},
		"unknown kind uses one decimal":         {value: 100, kind: "unknown", want: "100.0"},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.want, FormatTokS(tc.value, tc.kind))
		})
	}
}

func TestFormatProgress(t *testing.T) {
	tests := map[string]struct {
		done, total, width int
		want               string
	}{
		"half progress at width 10":                       {done: 5, total: 10, width: 10, want: "[█████░░░░░]"},
		"full progress at width 10":                       {done: 10, total: 10, width: 10, want: "[██████████]"},
		"zero total with width 4 shows empty":             {done: 0, total: 0, width: 4, want: "[░░░░]"},
		"done greater than total clamps to full":          {done: 15, total: 10, width: 10, want: "[██████████]"},
		"full progress at width 20":                       {done: 10, total: 10, width: 20, want: "[████████████████████]"},
		"negative done clamps to zero":                    {done: -5, total: 10, width: 10, want: "[░░░░░░░░░░]"},
		"zero width with brackets returns empty brackets": {done: 5, total: 10, width: 0, want: "[]"},
		"negative width clamps to zero":                   {done: 5, total: 10, width: -5, want: "[]"},
		"zero progress at width 10":                       {done: 0, total: 10, width: 10, want: "[░░░░░░░░░░]"},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.want, FormatProgress(tc.done, tc.total, tc.width))
		})
	}
}
