package format

import "testing"

func TestFormatContextSize(t *testing.T) {
	cases := []struct {
		input int
		want  string
	}{
		{512, "512"},
		{4096, "4k"},
		{8192, "8k"},
		{16384, "16k"},
		{32768, "33k"},
		{65536, "66k"},
		{131072, "131k"},
	}
	for _, tc := range cases {
		if got := FormatContextSize(tc.input); got != tc.want {
			t.Fatalf("FormatContextSize(%d) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestFormatGiBFromMiB(t *testing.T) {
	cases := []struct {
		input float64
		want  string
	}{
		{5040, "4.92 GiB"},
		{5190, "5.07 GiB"},
		{14115, "13.78 GiB"},
	}
	for _, tc := range cases {
		if got := FormatGiBFromMiB(tc.input); got != tc.want {
			t.Fatalf("FormatGiBFromMiB(%v) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestFormatDurationAndElapsed(t *testing.T) {
	if got := FormatDuration(2.7); got != "2.70s" {
		t.Fatalf("duration = %q", got)
	}
	if got := FormatDuration(10); got != "10.00s" {
		t.Fatalf("duration integer = %q", got)
	}
	if got := FormatElapsed(522); got != "08:42" {
		t.Fatalf("elapsed = %q", got)
	}
	if got := FormatElapsed(3862); got != "1:04:22" {
		t.Fatalf("elapsed hour = %q", got)
	}
	if got := FormatElapsed(0); got != "00:00" {
		t.Fatalf("elapsed zero = %q", got)
	}
}

func TestFormatTokS(t *testing.T) {
	if got := FormatTokS(78, "gen"); got != "78.0" {
		t.Fatalf("gen tok/s = %q", got)
	}
	if got := FormatTokS(812, "prompt"); got != "812" {
		t.Fatalf("prompt tok/s large = %q", got)
	}
	if got := FormatTokS(87.4, "prompt"); got != "87.4" {
		t.Fatalf("prompt tok/s small = %q", got)
	}
	if got := FormatTokS(0, "prompt"); got != "0" {
		t.Fatalf("prompt tok/s zero = %q", got)
	}
}

func TestFormatProgress(t *testing.T) {
	if got := FormatProgress(5, 10, 10); got != "[█████░░░░░]" {
		t.Fatalf("progress = %q", got)
	}
	if got := FormatProgress(10, 10, 10); got != "[██████████]" {
		t.Fatalf("full progress = %q", got)
	}
	if got := FormatProgress(0, 0, 4); got != "[░░░░]" {
		t.Fatalf("zero total progress = %q", got)
	}
	if got := FormatProgress(15, 10, 10); got != "[██████████]" {
		t.Fatalf("clamped progress = %q", got)
	}
	if got := FormatProgress(10, 10, 20); got != "[████████████████████]" {
		t.Fatalf("full progress wide = %q", got)
	}
	if got := FormatProgress(-5, 10, 10); got != "[░░░░░░░░░░]" {
		t.Fatalf("negative done clamped = %q", got)
	}
}
