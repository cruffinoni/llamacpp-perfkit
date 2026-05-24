package format

import "testing"

func TestFormatContextSize(t *testing.T) {
	cases := map[int]string{512: "512", 4096: "4k", 8192: "8k", 16384: "16k", 32768: "33k", 65536: "66k", 131072: "131k"}
	for input, want := range cases {
		if got := FormatContextSize(input); got != want {
			t.Fatalf("FormatContextSize(%d) = %q, want %q", input, got, want)
		}
	}
}

func TestFormatGiBFromMiB(t *testing.T) {
	cases := map[float64]string{5040: "4.92 GiB", 5190: "5.07 GiB", 14115: "13.78 GiB"}
	for input, want := range cases {
		if got := FormatGiBFromMiB(input); got != want {
			t.Fatalf("FormatGiBFromMiB(%v) = %q, want %q", input, got, want)
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
		got := FormatProgress(10, 10, 20)
		if got != "[██████████████]" {
			t.Fatalf("full progress wide: expected \"%q\", got %q", "[██████████████]", got)
		t.Fatalf("full progress wide = %q", got)
	}
	if got := FormatProgress(-5, 10, 10); got != "[░░░░░░░░░░]" {
		t.Fatalf("negative done clamped = %q", got)
	}
}

func TestColorForStatus(t *testing.T) {
	cases := map[string]Token{
		"success": TokenSuccess,
		"running": TokenRunning,
		"pending": TokenPending,
		"timeout": TokenTimeout,
		"oom":     TokenFailed,
		"failed":  TokenFailed,
		"unknown": TokenMuted,
	}
	for status, want := range cases {
		if got := ColorForStatus(status); got != want {
			t.Fatalf("ColorForStatus(%q) = %v, want %v", status, got, want)
		}
	}
}

func TestColorForPhase(t *testing.T) {
	cases := map[string]Token{
		"prefill":    TokenGenerating,
		"generating": TokenGenerating,
		"starting":   TokenGenerating,
		"done":       TokenDone,
		"pending":    TokenDone,
		"-":          TokenDone,
		"timeout":    TokenTimeout,
		"oom":        TokenFailed,
		"failed":     TokenFailed,
		"unknown":    TokenMuted,
	}
	for phase, want := range cases {
		if got := ColorForPhase(phase); got != want {
			t.Fatalf("ColorForPhase(%q) = %v, want %v", phase, got, want)
		}
	}
}

func TestTokenConstants(t *testing.T) {
	// Verify semantic token constants map to expected palette tokens
	if TokenSuccess != TokenGreen {
		t.Errorf("TokenSuccess should alias TokenGreen")
	}
	if TokenRunning != TokenCyan {
		t.Errorf("TokenRunning should alias TokenCyan")
	}
	if TokenPending != TokenMuted {
		t.Errorf("TokenPending should alias TokenMuted")
	}
	if TokenTimeout != TokenYellow {
		t.Errorf("TokenTimeout should alias TokenYellow")
	}
	if TokenFailed != TokenRed {
		t.Errorf("TokenFailed should alias TokenRed")
	}
	if TokenPrefill != TokenBlue {
		t.Errorf("TokenPrefill should alias TokenBlue")
	}
	if TokenGenerating != TokenCyan {
		t.Errorf("TokenGenerating should alias TokenCyan")
	}
}
